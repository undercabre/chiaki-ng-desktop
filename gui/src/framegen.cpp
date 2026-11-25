// Frame generation: see framegen.h.
//
// Pipeline for each frame the queue hands us (render thread only):
//   1. render it with libplacebo into an RGBA16F texture in the target's color space
//      (same size and crop as the swapchain image, so presenting it later is an identity copy)
//   2. derive a small luma image and compare it with the previous distinct frame. A frame
//      that barely differs is a duplicate (a 30 fps game inside a 60 fps stream) and dropped
//   3. for each new distinct frame, run the NVIDIA optical flow engine on (previous, new) in
//      both directions and copy the flow fields into textures libplacebo can sample
//   4. at every vsync, pick the pair of distinct frames around (display time - delay) and
//      warp both towards the in-between phase with a shader, or present a real frame
//
// Raw Vulkan work (copies, optical flow) is serialized on one timeline semaphore. Textures
// owned by libplacebo are handed over with pl_vulkan_hold_ex / pl_vulkan_release_ex.

#include "framegen.h"

#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/gpu.h>

#include <QLoggingCategory>
#include <QtGlobal>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

Q_LOGGING_CATEGORY(chiakiFrameGen, "chiaki.gui.framegen", QtInfoMsg);

#if defined(VK_NV_optical_flow) && VK_NV_optical_flow
#define FG_HAVE_OF 1
#else
#define FG_HAVE_OF 0
#endif

const char *FrameGenerator::extensionName()
{
#if FG_HAVE_OF
    return VK_NV_OPTICAL_FLOW_EXTENSION_NAME;
#else
    return "VK_NV_optical_flow";
#endif
}

bool FrameGenerator::compiledIn()
{
    return FG_HAVE_OF != 0;
}

VkQueueFlags FrameGenerator::queueFlag()
{
#if FG_HAVE_OF
    return VK_QUEUE_OPTICAL_FLOW_BIT_NV;
#else
    return 0;
#endif
}

bool FrameGenerator::instanceHasCapableDevice(VkInstance instance, PFN_vkGetInstanceProcAddr gipa)
{
#if FG_HAVE_OF
    if (!instance || !gipa)
        return false;
    auto enumDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
    auto enumExts = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(gipa(instance, "vkEnumerateDeviceExtensionProperties"));
    if (!enumDevices || !enumExts)
        return false;
    uint32_t count = 0;
    if (enumDevices(instance, &count, nullptr) != VK_SUCCESS || !count)
        return false;
    std::vector<VkPhysicalDevice> devices(count);
    if (enumDevices(instance, &count, devices.data()) != VK_SUCCESS)
        return false;
    for (VkPhysicalDevice dev : devices) {
        uint32_t num = 0;
        if (enumExts(dev, nullptr, &num, nullptr) != VK_SUCCESS || !num)
            continue;
        std::vector<VkExtensionProperties> exts(num);
        if (enumExts(dev, nullptr, &num, exts.data()) != VK_SUCCESS)
            continue;
        for (const auto &e : exts) {
            if (strcmp(e.extensionName, VK_NV_OPTICAL_FLOW_EXTENSION_NAME) == 0)
                return true;
        }
    }
    return false;
#else
    Q_UNUSED(instance);
    Q_UNUSED(gipa);
    return false;
#endif
}

#if !FG_HAVE_OF

struct FrameGenerator::Impl {};

FrameGenerator::FrameGenerator(pl_log, pl_vulkan)
{
    status_text = QStringLiteral("not compiled in (Vulkan headers without VK_NV_optical_flow)");
}
FrameGenerator::~FrameGenerator() {}
void FrameGenerator::reset() {}
bool FrameGenerator::process(const pl_frame_mix *, double, double, const pl_render_params *, const pl_frame *, pl_frame *) { return false; }
bool FrameGenerator::takeStats(FrameGenStats *, qint64) { return false; }

#else

namespace {

constexpr int kSlots = 5;          // distinct frames kept (history uses kSlots - 1, one is free)
constexpr int kCmdRing = 8;
constexpr int kDiffW = 64;
constexpr int kDiffH = 36;
constexpr VkFormat kFlowFormat = static_cast<VkFormat>(1000464000); // VK_FORMAT_R16G16_SFIXED5_NV

qint64 nowUs()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

double envDouble(const char *name, double def)
{
    bool ok = false;
    const double v = qEnvironmentVariable(name).toDouble(&ok);
    return ok ? v : def;
}

int envInt(const char *name, int def)
{
    bool ok = false;
    const int v = qEnvironmentVariable(name).toInt(&ok);
    return ok ? v : def;
}

double median(std::deque<double> values)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

const char *kLumaShader = R"(
    vec2 uv = gl_FragCoord.xy / fg_out_size;
    vec3 c = textureLod(fg_src, uv, 0.0).rgb;
    color = vec4(dot(c, vec3(0.2126, 0.7152, 0.0722)), 0.0, 0.0, 1.0);
)";

const char *kDiffShader = R"(
    vec2 blk = vec2(1.0 / float(FG_DIFF_W), 1.0 / float(FG_DIFF_H));
    vec2 base = floor(gl_FragCoord.xy) * blk;
    float s = 0.0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            vec2 uv = base + (vec2(float(x), float(y)) + 0.5) * (blk / 8.0);
            s += abs(textureLod(fg_a, uv, 0.0).r - textureLod(fg_b, uv, 0.0).r);
        }
    }
    color = vec4(s / 64.0, 0.0, 0.0, 1.0);
)";

// fg_flow_f: displacement prev -> cur, sampled at prev positions
// fg_flow_b: displacement cur -> prev, sampled at cur positions
// Both are stored as snorm and scaled to uv units by fg_fscale.
const char *kWarpHeader = R"(
vec2 fg_F(vec2 uv) { return textureLod(fg_flow_f, uv, 0.0).xy * fg_fscale; }
vec2 fg_B(vec2 uv) { return textureLod(fg_flow_b, uv, 0.0).xy * fg_fscale; }
)";

const char *kWarpShader = R"(
    vec2 uv = gl_FragCoord.xy / fg_out_size;
    float t = fg_t;
    // where the pixel at uv was in the previous frame and will be in the next one
    vec2 p0 = uv - t * fg_F(uv);
    p0 = uv - t * fg_F(p0);
    vec2 p1 = uv - (1.0 - t) * fg_B(uv);
    p1 = uv - (1.0 - t) * fg_B(p1);
    // forward/backward consistency (in output pixels), large error = occlusion
    vec2 f0 = fg_F(p0);
    float e0 = length((f0 + fg_B(p0 + f0)) * fg_out_size);
    vec2 b1 = fg_B(p1);
    float e1 = length((b1 + fg_F(p1 + b1)) * fg_out_size);
    float w0 = (1.0 - t) / (1.0 + 0.25 * e0 * e0);
    float w1 = t / (1.0 + 0.25 * e1 * e1);
    vec4 c0 = textureLod(fg_prev, p0, 0.0);
    vec4 c1 = textureLod(fg_cur, p1, 0.0);
    color = (c0 * w0 + c1 * w1) / max(w0 + w1, 1e-5);
    if (fg_debug == 1) {
        vec2 m = fg_F(uv) * fg_out_size * 0.05;
        color = vec4(0.5 + m.x, 0.5 + m.y, 0.5, 1.0);
    }
    color.a = 1.0;
)";

} // namespace

struct FrameGenerator::Impl
{
    pl_log log = nullptr;
    pl_vulkan vk = nullptr;
    pl_gpu gpu = nullptr;
    pl_dispatch dp = nullptr;
    pl_renderer ingest = nullptr;
    pl_timer warp_timer = nullptr;

    VkDevice dev = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;

#define FG_FN(name) PFN_##name name = nullptr;
    FG_FN(vkGetDeviceQueue)
    FG_FN(vkCreateCommandPool)
    FG_FN(vkDestroyCommandPool)
    FG_FN(vkAllocateCommandBuffers)
    FG_FN(vkBeginCommandBuffer)
    FG_FN(vkEndCommandBuffer)
    FG_FN(vkResetCommandBuffer)
    FG_FN(vkQueueSubmit)
    FG_FN(vkCreateImage)
    FG_FN(vkDestroyImage)
    FG_FN(vkGetImageMemoryRequirements)
    FG_FN(vkAllocateMemory)
    FG_FN(vkFreeMemory)
    FG_FN(vkBindImageMemory)
    FG_FN(vkCreateImageView)
    FG_FN(vkDestroyImageView)
    FG_FN(vkCreateBuffer)
    FG_FN(vkDestroyBuffer)
    FG_FN(vkGetBufferMemoryRequirements)
    FG_FN(vkBindBufferMemory)
    FG_FN(vkCmdPipelineBarrier)
    FG_FN(vkCmdCopyImage)
    FG_FN(vkCmdCopyImageToBuffer)
    FG_FN(vkCmdCopyBufferToImage)
    FG_FN(vkCreateSemaphore)
    FG_FN(vkDestroySemaphore)
    FG_FN(vkWaitSemaphores)
    FG_FN(vkGetSemaphoreCounterValue)
    FG_FN(vkCreateQueryPool)
    FG_FN(vkDestroyQueryPool)
    FG_FN(vkCmdResetQueryPool)
    FG_FN(vkCmdWriteTimestamp)
    FG_FN(vkGetQueryPoolResults)
    FG_FN(vkDeviceWaitIdle)
    FG_FN(vkCreateOpticalFlowSessionNV)
    FG_FN(vkDestroyOpticalFlowSessionNV)
    FG_FN(vkBindOpticalFlowSessionImageNV)
    FG_FN(vkCmdOpticalFlowExecuteNV)
    FG_FN(vkGetPhysicalDeviceQueueFamilyProperties)
    FG_FN(vkGetPhysicalDeviceMemoryProperties)
    FG_FN(vkGetPhysicalDeviceProperties2)
    FG_FN(vkGetPhysicalDeviceOpticalFlowImageFormatsNV)
#undef FG_FN

    uint32_t gfx_qf = 0;
    uint32_t gfx_qidx = 0;
    uint32_t of_qf = UINT32_MAX;
    VkQueue gfx_q = VK_NULL_HANDLE;
    VkQueue of_q = VK_NULL_HANDLE;
    VkCommandPool gfx_pool = VK_NULL_HANDLE;
    VkCommandPool of_pool = VK_NULL_HANDLE;
    VkCommandBuffer gfx_cmd[kCmdRing] = {};
    VkCommandBuffer of_cmd[kCmdRing] = {};
    uint64_t gfx_cmd_val[kCmdRing] = {};
    uint64_t of_cmd_val[kCmdRing] = {};
    int gfx_next = 0;
    int of_next = 0;

    VkSemaphore fg_sem = VK_NULL_HANDLE;
    uint64_t fg_val = 0;
    uint64_t last_of_val = 0;

    VkPhysicalDeviceMemoryProperties memprops = {};

    // optical flow
    bool bidir = false;
    uint32_t grid = 4;
    VkOpticalFlowPerformanceLevelNV perf = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV;
    uint32_t of_min_w = 1, of_min_h = 1, of_max_w = 8192, of_max_h = 8192;
    VkOpticalFlowSessionNV sess = VK_NULL_HANDLE;      // input = cur, reference = prev
    VkOpticalFlowSessionNV sess_rev = VK_NULL_HANDLE;  // input = prev, reference = cur (no bidir)

    // timestamps on the optical flow queue
    bool timestamps = false;
    double ts_period_ns = 1.0;
    uint64_t ts_mask = ~0ull;
    VkQueryPool qpool = VK_NULL_HANDLE;
    bool query_pending[kCmdRing] = {};
    uint64_t query_val[kCmdRing] = {};

    // formats
    pl_fmt fmt_rgb = nullptr;
    pl_fmt fmt_luma = nullptr;
    pl_fmt fmt_diff = nullptr;
    pl_fmt fmt_flow = nullptr;

    struct Tex
    {
        pl_tex tex = nullptr;
        VkImage img = VK_NULL_HANDLE;
        VkSemaphore sem = VK_NULL_HANDLE;
        uint64_t val = 0;
    };

    struct Img
    {
        VkImage img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    struct Slot
    {
        pl_tex rgb = nullptr;
        Tex luma;
        Tex flow_f;
        Tex flow_b;
        Img ofin;
        double pts = 0.0;
        bool flow_ok = false;
        bool cut = false;
    };

    Slot fslots[kSlots];
    std::deque<int> hist; // slot indices of distinct frames, oldest first
    Img flowout_f;
    Img flowout_b;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    pl_tex diff_tex = nullptr;
    pl_tex warp_out = nullptr;

    int W = 0, H = 0;     // output (swapchain) size
    int LW = 0, LH = 0;   // luma / optical flow input size
    int FW = 0, FH = 0;   // flow field size
    pl_rect2df crop = {};
    bool resources_ok = false;

    // ingest state
    uint64_t seen[24] = {};
    int seen_pos = 0;
    double last_ingest_pts = -1e18;
    int consecutive_dups = 0;
    double motion_ref = 0.02;
    std::deque<double> intervals;
    std::deque<double> leads;
    double delay_s = 0.0;

    // tuning (environment overrides, for testing on the user's machine)
    double dup_min = 0.0015;
    double dup_max = 0.008;
    double dup_block = 0.03;
    double cut_thr = 0.12;
    double margin_s = 0.003;
    bool flip = false;
    int debug = 0;

    // stats over the current period
    qint64 period_start_us = 0;
    int st_stream = 0, st_distinct = 0, st_real = 0, st_gen = 0, st_late = 0, st_cuts = 0;
    double st_flow_ns = 0.0; int st_flow_n = 0;
    double st_warp_ns = 0.0; int st_warp_n = 0;
    double st_cpu_us = 0.0; int st_cpu_n = 0;
    double last_diff = 0.0;

    ~Impl() { destroy(); }

    bool loadFunctions(QString *why);
    bool init(QString *why);
    void destroyResources();
    void destroy();
    bool allocate(int w, int h, QString *why);
    bool createImage(Img &img, VkFormat format, int w, int h, VkImageUsageFlags usage, VkOpticalFlowUsageFlagsNV of_usage);
    void destroyImage(Img &img);
    bool createTex(Tex &t, pl_fmt fmt, int w, int h, bool blit_src, bool host_writable);
    void destroyTex(Tex &t);
    int findMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const;

    VkCommandBuffer beginCmd(bool of_queue, int *idx);
    bool submit(bool of_queue, VkCommandBuffer cmd, int idx, const std::vector<std::pair<VkSemaphore, uint64_t>> &waits);
    bool hold(Tex &t, VkImageLayout layout);
    void release(Tex &t, VkImageLayout layout, bool signaled);
    void imageBarrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to);

    bool dispatch(const char *header, const char *body, const std::vector<pl_shader_desc> &descs,
                  const std::vector<pl_shader_var> &vars, pl_tex target, pl_timer timer);
    bool renderLuma(pl_tex rgb, pl_tex luma);
    bool measureDiff(pl_tex a, pl_tex b, double *mean, double *max_block);
    bool copyLumaToFlowInput(Slot &s);
    bool runFlow(Slot &prev, Slot &cur);
    bool warp(Slot &prev, Slot &cur, double phase);

    pl_frame rgbFrame(pl_tex tex, const pl_frame *target) const;
    void ingestFrame(const pl_frame *frame, double pts, const pl_render_params *params, const pl_frame *target);
    void resetHistory();
    void pollTimers();
};

int FrameGenerator::Impl::findMemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const
{
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & flags) == flags)
            return static_cast<int>(i);
    }
    return -1;
}

bool FrameGenerator::Impl::loadFunctions(QString *why)
{
    PFN_vkGetInstanceProcAddr gipa = vk->get_proc_addr;
    VkInstance inst = vk->instance;
    auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(inst, "vkGetDeviceProcAddr"));
    if (!gdpa) {
        *why = QStringLiteral("vkGetDeviceProcAddr missing");
        return false;
    }
#define FG_DEV(name) name = reinterpret_cast<PFN_##name>(gdpa(dev, #name)); \
    if (!name) { *why = QStringLiteral("missing Vulkan function " #name); return false; }
#define FG_INST(name) name = reinterpret_cast<PFN_##name>(gipa(inst, #name)); \
    if (!name) { *why = QStringLiteral("missing Vulkan function " #name); return false; }
    FG_DEV(vkGetDeviceQueue)
    FG_DEV(vkCreateCommandPool)
    FG_DEV(vkDestroyCommandPool)
    FG_DEV(vkAllocateCommandBuffers)
    FG_DEV(vkBeginCommandBuffer)
    FG_DEV(vkEndCommandBuffer)
    FG_DEV(vkResetCommandBuffer)
    FG_DEV(vkQueueSubmit)
    FG_DEV(vkCreateImage)
    FG_DEV(vkDestroyImage)
    FG_DEV(vkGetImageMemoryRequirements)
    FG_DEV(vkAllocateMemory)
    FG_DEV(vkFreeMemory)
    FG_DEV(vkBindImageMemory)
    FG_DEV(vkCreateImageView)
    FG_DEV(vkDestroyImageView)
    FG_DEV(vkCreateBuffer)
    FG_DEV(vkDestroyBuffer)
    FG_DEV(vkGetBufferMemoryRequirements)
    FG_DEV(vkBindBufferMemory)
    FG_DEV(vkCmdPipelineBarrier)
    FG_DEV(vkCmdCopyImage)
    FG_DEV(vkCmdCopyImageToBuffer)
    FG_DEV(vkCmdCopyBufferToImage)
    FG_DEV(vkCreateSemaphore)
    FG_DEV(vkDestroySemaphore)
    FG_DEV(vkWaitSemaphores)
    FG_DEV(vkGetSemaphoreCounterValue)
    FG_DEV(vkCreateQueryPool)
    FG_DEV(vkDestroyQueryPool)
    FG_DEV(vkCmdResetQueryPool)
    FG_DEV(vkCmdWriteTimestamp)
    FG_DEV(vkGetQueryPoolResults)
    FG_DEV(vkDeviceWaitIdle)
    FG_DEV(vkCreateOpticalFlowSessionNV)
    FG_DEV(vkDestroyOpticalFlowSessionNV)
    FG_DEV(vkBindOpticalFlowSessionImageNV)
    FG_DEV(vkCmdOpticalFlowExecuteNV)
    FG_INST(vkGetPhysicalDeviceQueueFamilyProperties)
    FG_INST(vkGetPhysicalDeviceMemoryProperties)
    FG_INST(vkGetPhysicalDeviceProperties2)
    FG_INST(vkGetPhysicalDeviceOpticalFlowImageFormatsNV)
#undef FG_DEV
#undef FG_INST
    return true;
}

bool FrameGenerator::Impl::init(QString *why)
{
    gpu = vk->gpu;
    dev = vk->device;
    phys = vk->phys_device;

    bool ext_enabled = false;
    for (int i = 0; i < vk->num_extensions; i++) {
        if (vk->extensions[i] && strcmp(vk->extensions[i], VK_NV_OPTICAL_FLOW_EXTENSION_NAME) == 0)
            ext_enabled = true;
    }
    if (!ext_enabled) {
        *why = QStringLiteral("this GPU or driver does not offer VK_NV_optical_flow (an NVIDIA RTX 20 series or newer with a recent driver is needed)");
        return false;
    }
    bool feature_enabled = false;
    for (const VkBaseInStructure *s = reinterpret_cast<const VkBaseInStructure *>(vk->features); s; s = s->pNext) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV) {
            feature_enabled = reinterpret_cast<const VkPhysicalDeviceOpticalFlowFeaturesNV *>(s)->opticalFlow == VK_TRUE;
            break;
        }
    }
    if (!feature_enabled) {
        *why = QStringLiteral("the opticalFlow device feature is not available");
        return false;
    }
    if (!loadFunctions(why))
        return false;

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qf_count, qfs.data());
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) {
            of_qf = i;
            break;
        }
    }
    if (of_qf == UINT32_MAX) {
        *why = QStringLiteral("no optical flow queue family");
        return false;
    }
    if (vk->queue_graphics.index < 0 || vk->queue_graphics.count <= 0) {
        *why = QStringLiteral("no graphics queue");
        return false;
    }
    gfx_qf = static_cast<uint32_t>(vk->queue_graphics.index);
    gfx_qidx = 0;
    vkGetDeviceQueue(dev, gfx_qf, gfx_qidx, &gfx_q);
    vkGetDeviceQueue(dev, of_qf, 0, &of_q);
    if (!gfx_q || !of_q) {
        *why = QStringLiteral("could not get the graphics or optical flow queue");
        return false;
    }
    timestamps = qfs[of_qf].timestampValidBits > 0 && envInt("CHIAKI_FG_NO_TIMESTAMPS", 0) == 0;
    if (timestamps && qfs[of_qf].timestampValidBits < 64)
        ts_mask = (1ull << qfs[of_qf].timestampValidBits) - 1;

    VkPhysicalDeviceOpticalFlowPropertiesNV of_props = {};
    of_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV;
    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &of_props;
    vkGetPhysicalDeviceProperties2(phys, &props2);
    ts_period_ns = props2.properties.limits.timestampPeriod > 0 ? props2.properties.limits.timestampPeriod : 1.0;
    bidir = of_props.bidirectionalFlowSupported == VK_TRUE && envInt("CHIAKI_FG_NO_BIDIR", 0) == 0;
    of_min_w = qMax<uint32_t>(1, of_props.minWidth);
    of_min_h = qMax<uint32_t>(1, of_props.minHeight);
    of_max_w = of_props.maxWidth ? of_props.maxWidth : 4096;
    of_max_h = of_props.maxHeight ? of_props.maxHeight : 4096;
    const uint32_t wanted_grid = static_cast<uint32_t>(envInt("CHIAKI_FG_GRID", 4));
    const uint32_t grids = of_props.supportedOutputGridSizes;
    grid = 0;
    const uint32_t prefs[] = { wanted_grid, 4, 2, 8, 1 };
    for (uint32_t g : prefs) {
        if (g && (grids & g) && (g & (g - 1)) == 0) {
            grid = g;
            break;
        }
    }
    if (!grid) {
        *why = QStringLiteral("no supported optical flow grid size");
        return false;
    }
    switch (envInt("CHIAKI_FG_PERF", 1)) {
    case 0: perf = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_NV; break;
    case 2: perf = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_NV; break;
    default: perf = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV; break;
    }

    // the input and output formats this implementation uses must be supported
    auto hasOfFormat = [this](VkOpticalFlowUsageFlagsNV usage, VkFormat format) {
        VkOpticalFlowImageFormatInfoNV info = {};
        info.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV;
        info.usage = usage;
        uint32_t n = 0;
        if (vkGetPhysicalDeviceOpticalFlowImageFormatsNV(phys, &info, &n, nullptr) != VK_SUCCESS || !n)
            return false;
        std::vector<VkOpticalFlowImageFormatPropertiesNV> fmts(n);
        for (auto &f : fmts) {
            f = {};
            f.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_NV;
        }
        if (vkGetPhysicalDeviceOpticalFlowImageFormatsNV(phys, &info, &n, fmts.data()) != VK_SUCCESS)
            return false;
        for (const auto &f : fmts) {
            if (f.format == format)
                return true;
        }
        return false;
    };
    if (!hasOfFormat(VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV, VK_FORMAT_R8_UNORM)) {
        *why = QStringLiteral("optical flow does not accept R8 input images");
        return false;
    }
    if (!hasOfFormat(VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV, kFlowFormat)) {
        *why = QStringLiteral("optical flow does not output R16G16_SFIXED5 vectors");
        return false;
    }

    vkGetPhysicalDeviceMemoryProperties(phys, &memprops);

    // libplacebo formats
    fmt_rgb = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 16,
                          static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR | PL_FMT_CAP_RENDERABLE));
    fmt_luma = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 8, 8,
                           static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR | PL_FMT_CAP_RENDERABLE));
    fmt_diff = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 32,
                           static_cast<pl_fmt_caps>(PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE));
    fmt_flow = pl_find_fmt(gpu, PL_FMT_SNORM, 2, 16, 16,
                           static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR));
    if (!fmt_rgb || !fmt_luma || !fmt_diff || !fmt_flow) {
        *why = QStringLiteral("missing texture formats (rgba16f=%1 r8=%2 r32f=%3 rg16snorm=%4)")
            .arg(fmt_rgb ? 1 : 0).arg(fmt_luma ? 1 : 0).arg(fmt_diff ? 1 : 0).arg(fmt_flow ? 1 : 0);
        return false;
    }

    VkSemaphoreTypeCreateInfo tinfo = {};
    tinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    tinfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tinfo.initialValue = 0;
    VkSemaphoreCreateInfo sinfo = {};
    sinfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sinfo.pNext = &tinfo;
    if (vkCreateSemaphore(dev, &sinfo, nullptr, &fg_sem) != VK_SUCCESS) {
        *why = QStringLiteral("could not create a timeline semaphore");
        return false;
    }

    auto makePool = [this](uint32_t qf, VkCommandPool *pool, VkCommandBuffer *cmds) {
        VkCommandPoolCreateInfo pinfo = {};
        pinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pinfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pinfo.queueFamilyIndex = qf;
        if (vkCreateCommandPool(dev, &pinfo, nullptr, pool) != VK_SUCCESS)
            return false;
        VkCommandBufferAllocateInfo ainfo = {};
        ainfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ainfo.commandPool = *pool;
        ainfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ainfo.commandBufferCount = kCmdRing;
        return vkAllocateCommandBuffers(dev, &ainfo, cmds) == VK_SUCCESS;
    };
    if (!makePool(gfx_qf, &gfx_pool, gfx_cmd) || !makePool(of_qf, &of_pool, of_cmd)) {
        *why = QStringLiteral("could not create command pools");
        return false;
    }

    if (timestamps) {
        VkQueryPoolCreateInfo qinfo = {};
        qinfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qinfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qinfo.queryCount = 2 * kCmdRing;
        if (vkCreateQueryPool(dev, &qinfo, nullptr, &qpool) != VK_SUCCESS)
            timestamps = false;
    }

    dp = pl_dispatch_create(log, gpu);
    ingest = pl_renderer_create(log, gpu);
    warp_timer = pl_timer_create(gpu);
    if (!dp || !ingest) {
        *why = QStringLiteral("could not create libplacebo objects");
        return false;
    }

    dup_min = envDouble("CHIAKI_FG_DUP_MIN", dup_min);
    dup_max = envDouble("CHIAKI_FG_DUP_MAX", dup_max);
    dup_block = envDouble("CHIAKI_FG_DUP_BLOCK", dup_block);
    cut_thr = envDouble("CHIAKI_FG_CUT", cut_thr);
    margin_s = envDouble("CHIAKI_FG_MARGIN_MS", margin_s * 1000.0) / 1000.0;
    flip = envInt("CHIAKI_FG_FLIP", 0) != 0;
    debug = envInt("CHIAKI_FG_DEBUG", 0);

    qCInfo(chiakiFrameGen).nospace()
        << "Frame generation ready: of_queue_family=" << of_qf
        << " bidirectional=" << bidir << " grid=" << grid
        << " perf=" << static_cast<int>(perf)
        << " timestamps=" << timestamps
        << " max=" << of_max_w << "x" << of_max_h;
    return true;
}

bool FrameGenerator::Impl::createImage(Img &img, VkFormat format, int w, int h, VkImageUsageFlags usage,
                                       VkOpticalFlowUsageFlagsNV of_usage)
{
    VkOpticalFlowImageFormatInfoNV of_info = {};
    of_info.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV;
    of_info.usage = of_usage;
    const uint32_t families[] = { gfx_qf, of_qf };
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.pNext = &of_info;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    if (gfx_qf != of_qf) {
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = 2;
        info.pQueueFamilyIndices = families;
    } else {
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev, &info, nullptr, &img.img) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req = {};
    vkGetImageMemoryRequirements(dev, img.img, &req);
    const int type = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0)
        return false;
    VkMemoryAllocateInfo ainfo = {};
    ainfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ainfo.allocationSize = req.size;
    ainfo.memoryTypeIndex = static_cast<uint32_t>(type);
    if (vkAllocateMemory(dev, &ainfo, nullptr, &img.mem) != VK_SUCCESS)
        return false;
    if (vkBindImageMemory(dev, img.img, img.mem, 0) != VK_SUCCESS)
        return false;
    VkImageViewCreateInfo vinfo = {};
    vinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vinfo.image = img.img;
    vinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vinfo.format = format;
    vinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vinfo.subresourceRange.levelCount = 1;
    vinfo.subresourceRange.layerCount = 1;
    return vkCreateImageView(dev, &vinfo, nullptr, &img.view) == VK_SUCCESS;
}

void FrameGenerator::Impl::destroyImage(Img &img)
{
    if (img.view)
        vkDestroyImageView(dev, img.view, nullptr);
    if (img.img)
        vkDestroyImage(dev, img.img, nullptr);
    if (img.mem)
        vkFreeMemory(dev, img.mem, nullptr);
    img = Img();
}

bool FrameGenerator::Impl::createTex(Tex &t, pl_fmt fmt, int w, int h, bool blit_src, bool host_writable)
{
    pl_tex_params params = {};
    params.w = w;
    params.h = h;
    params.format = fmt;
    params.sampleable = true;
    params.renderable = (fmt->caps & PL_FMT_CAP_RENDERABLE) != 0;
    params.storable = (fmt->caps & PL_FMT_CAP_STORABLE) != 0;
    params.blit_src = blit_src && (fmt->caps & PL_FMT_CAP_BLITTABLE);
    params.host_readable = blit_src && !params.blit_src;
    params.host_writable = host_writable;
    t.tex = pl_tex_create(gpu, &params);
    if (!t.tex)
        return false;
    VkFormat vk_fmt = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    t.img = pl_vulkan_unwrap(gpu, t.tex, &vk_fmt, &usage);
    if (!t.img)
        return false;
    if (blit_src && !(usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        return false;
    if (host_writable && !(usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return false;
    pl_vulkan_sem_params sp = {};
    sp.type = VK_SEMAPHORE_TYPE_TIMELINE;
    sp.initial_value = 0;
    t.sem = pl_vulkan_sem_create(gpu, &sp);
    t.val = 0;
    return t.sem != VK_NULL_HANDLE;
}

void FrameGenerator::Impl::destroyTex(Tex &t)
{
    if (t.tex)
        pl_tex_destroy(gpu, &t.tex);
    if (t.sem)
        pl_vulkan_sem_destroy(gpu, &t.sem);
    t = Tex();
}

void FrameGenerator::Impl::destroyResources()
{
    if (dev && fg_val > 0 && vkWaitSemaphores) {
        VkSemaphoreWaitInfo winfo = {};
        winfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        winfo.semaphoreCount = 1;
        winfo.pSemaphores = &fg_sem;
        winfo.pValues = &fg_val;
        vkWaitSemaphores(dev, &winfo, 1000000000ull);
    }
    if (gpu)
        pl_gpu_finish(gpu);
    for (auto &s : fslots) {
        if (s.rgb)
            pl_tex_destroy(gpu, &s.rgb);
        destroyTex(s.luma);
        destroyTex(s.flow_f);
        destroyTex(s.flow_b);
        destroyImage(s.ofin);
        s = Slot();
    }
    destroyImage(flowout_f);
    destroyImage(flowout_b);
    if (staging)
        vkDestroyBuffer(dev, staging, nullptr);
    if (staging_mem)
        vkFreeMemory(dev, staging_mem, nullptr);
    staging = VK_NULL_HANDLE;
    staging_mem = VK_NULL_HANDLE;
    if (diff_tex)
        pl_tex_destroy(gpu, &diff_tex);
    if (warp_out)
        pl_tex_destroy(gpu, &warp_out);
    if (sess)
        vkDestroyOpticalFlowSessionNV(dev, sess, nullptr);
    if (sess_rev)
        vkDestroyOpticalFlowSessionNV(dev, sess_rev, nullptr);
    sess = VK_NULL_HANDLE;
    sess_rev = VK_NULL_HANDLE;
    hist.clear();
    W = H = LW = LH = FW = FH = 0;
    resources_ok = false;
}

void FrameGenerator::Impl::destroy()
{
    if (!dev)
        return;
    destroyResources();
    if (gfx_pool)
        vkDestroyCommandPool(dev, gfx_pool, nullptr);
    if (of_pool)
        vkDestroyCommandPool(dev, of_pool, nullptr);
    if (qpool)
        vkDestroyQueryPool(dev, qpool, nullptr);
    if (fg_sem)
        vkDestroySemaphore(dev, fg_sem, nullptr);
    gfx_pool = of_pool = VK_NULL_HANDLE;
    qpool = VK_NULL_HANDLE;
    fg_sem = VK_NULL_HANDLE;
    if (warp_timer)
        pl_timer_destroy(gpu, &warp_timer);
    if (ingest)
        pl_renderer_destroy(&ingest);
    if (dp)
        pl_dispatch_destroy(&dp);
    dev = VK_NULL_HANDLE;
}

bool FrameGenerator::Impl::allocate(int w, int h, QString *why)
{
    destroyResources();
    W = w;
    H = h;
    // optical flow input: at most 1920x1088, keeping the aspect ratio
    double scale = 1.0;
    const double max_w = qMin<double>(1920.0, of_max_w);
    const double max_h = qMin<double>(1088.0, of_max_h);
    scale = qMin(scale, max_w / w);
    scale = qMin(scale, max_h / h);
    LW = qMax<int>(static_cast<int>(of_min_w), static_cast<int>(std::lround(w * scale)));
    LH = qMax<int>(static_cast<int>(of_min_h), static_cast<int>(std::lround(h * scale)));
    LW = (LW + 1) & ~1;
    LH = (LH + 1) & ~1;
    FW = static_cast<int>((LW + grid - 1) / grid);
    FH = static_cast<int>((LH + grid - 1) / grid);

    auto fail = [&](const QString &msg) {
        *why = msg;
        destroyResources();
        return false;
    };

    for (auto &s : fslots) {
        pl_tex_params rp = {};
        rp.w = W;
        rp.h = H;
        rp.format = fmt_rgb;
        rp.sampleable = true;
        rp.renderable = true;
        rp.storable = (fmt_rgb->caps & PL_FMT_CAP_STORABLE) != 0;
        s.rgb = pl_tex_create(gpu, &rp);
        if (!s.rgb)
            return fail(QStringLiteral("could not create RGB textures"));
        if (!createTex(s.luma, fmt_luma, LW, LH, true, false))
            return fail(QStringLiteral("could not create luma textures"));
        if (!createTex(s.flow_f, fmt_flow, FW, FH, false, true) || !createTex(s.flow_b, fmt_flow, FW, FH, false, true))
            return fail(QStringLiteral("could not create flow textures"));
        VkFormat luma_fmt = VK_FORMAT_UNDEFINED, flow_fmt = VK_FORMAT_UNDEFINED;
        pl_vulkan_unwrap(gpu, s.luma.tex, &luma_fmt, nullptr);
        pl_vulkan_unwrap(gpu, s.flow_f.tex, &flow_fmt, nullptr);
        if (luma_fmt != VK_FORMAT_R8_UNORM || flow_fmt != VK_FORMAT_R16G16_SNORM)
            return fail(QStringLiteral("unexpected texture formats (%1, %2)").arg(static_cast<int>(luma_fmt)).arg(static_cast<int>(flow_fmt)));
        if (!createImage(s.ofin, VK_FORMAT_R8_UNORM, LW, LH,
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV))
            return fail(QStringLiteral("could not create optical flow input images"));
    }
    const VkImageUsageFlags out_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    if (!createImage(flowout_f, kFlowFormat, FW, FH, out_usage, VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV) ||
        !createImage(flowout_b, kFlowFormat, FW, FH, out_usage, VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV))
        return fail(QStringLiteral("could not create optical flow output images"));

    VkBufferCreateInfo binfo = {};
    binfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    binfo.size = static_cast<VkDeviceSize>(FW) * FH * 4 * 2;
    binfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    binfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &binfo, nullptr, &staging) != VK_SUCCESS)
        return fail(QStringLiteral("could not create the staging buffer"));
    VkMemoryRequirements breq = {};
    vkGetBufferMemoryRequirements(dev, staging, &breq);
    const int btype = findMemoryType(breq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (btype < 0)
        return fail(QStringLiteral("no memory type for the staging buffer"));
    VkMemoryAllocateInfo bainfo = {};
    bainfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bainfo.allocationSize = breq.size;
    bainfo.memoryTypeIndex = static_cast<uint32_t>(btype);
    if (vkAllocateMemory(dev, &bainfo, nullptr, &staging_mem) != VK_SUCCESS ||
        vkBindBufferMemory(dev, staging, staging_mem, 0) != VK_SUCCESS)
        return fail(QStringLiteral("could not allocate the staging buffer"));

    pl_tex_params dpar = {};
    dpar.w = kDiffW;
    dpar.h = kDiffH;
    dpar.format = fmt_diff;
    dpar.renderable = true;
    dpar.host_readable = true;
    diff_tex = pl_tex_create(gpu, &dpar);
    pl_tex_params wpar = {};
    wpar.w = W;
    wpar.h = H;
    wpar.format = fmt_rgb;
    wpar.sampleable = true;
    wpar.renderable = true;
    wpar.storable = (fmt_rgb->caps & PL_FMT_CAP_STORABLE) != 0;
    warp_out = pl_tex_create(gpu, &wpar);
    if (!diff_tex || !warp_out)
        return fail(QStringLiteral("could not create the work textures"));

    auto makeSession = [this](VkOpticalFlowSessionNV *out, bool both) {
        VkOpticalFlowSessionCreateInfoNV sinfo = {};
        sinfo.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV;
        sinfo.width = static_cast<uint32_t>(LW);
        sinfo.height = static_cast<uint32_t>(LH);
        sinfo.imageFormat = VK_FORMAT_R8_UNORM;
        sinfo.flowVectorFormat = kFlowFormat;
        sinfo.costFormat = VK_FORMAT_UNDEFINED;
        sinfo.outputGridSize = static_cast<VkOpticalFlowGridSizeFlagsNV>(grid);
        sinfo.hintGridSize = 0;
        sinfo.performanceLevel = perf;
        sinfo.flags = both ? VK_OPTICAL_FLOW_SESSION_CREATE_BOTH_DIRECTIONS_BIT_NV : 0;
        return vkCreateOpticalFlowSessionNV(dev, &sinfo, nullptr, out) == VK_SUCCESS;
    };
    if (!makeSession(&sess, bidir))
        return fail(QStringLiteral("could not create the optical flow session (%1x%2)").arg(LW).arg(LH));
    if (!bidir && !makeSession(&sess_rev, false))
        return fail(QStringLiteral("could not create the reverse optical flow session"));
    // flow outputs never change, bind them once
    if (vkBindOpticalFlowSessionImageNV(dev, sess, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV,
                                        flowout_b.view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS)
        return fail(QStringLiteral("could not bind the flow output"));
    if (bidir) {
        if (vkBindOpticalFlowSessionImageNV(dev, sess, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_BACKWARD_FLOW_VECTOR_NV,
                                            flowout_f.view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS)
            return fail(QStringLiteral("could not bind the backward flow output"));
    } else {
        if (vkBindOpticalFlowSessionImageNV(dev, sess_rev, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV,
                                            flowout_f.view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS)
            return fail(QStringLiteral("could not bind the reverse flow output"));
    }

    resources_ok = true;
    qCInfo(chiakiFrameGen).nospace() << "Frame generation buffers: output=" << W << "x" << H
        << " flow_input=" << LW << "x" << LH << " flow=" << FW << "x" << FH;
    return true;
}

VkCommandBuffer FrameGenerator::Impl::beginCmd(bool of_queue, int *idx)
{
    int &next = of_queue ? of_next : gfx_next;
    uint64_t *vals = of_queue ? of_cmd_val : gfx_cmd_val;
    VkCommandBuffer *cmds = of_queue ? of_cmd : gfx_cmd;
    const int i = next;
    next = (next + 1) % kCmdRing;
    if (vals[i] > 0) {
        VkSemaphoreWaitInfo winfo = {};
        winfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        winfo.semaphoreCount = 1;
        winfo.pSemaphores = &fg_sem;
        winfo.pValues = &vals[i];
        if (vkWaitSemaphores(dev, &winfo, 1000000000ull) != VK_SUCCESS)
            return VK_NULL_HANDLE;
    }
    if (of_queue && query_pending[i]) {
        // results of the previous use of this ring entry are ready now
        uint64_t ts[2] = {};
        if (vkGetQueryPoolResults(dev, qpool, 2 * i, 2, sizeof(ts), ts, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const uint64_t delta = ((ts[1] & ts_mask) - (ts[0] & ts_mask)) & ts_mask;
            st_flow_ns += static_cast<double>(delta) * ts_period_ns;
            st_flow_n++;
        }
        query_pending[i] = false;
    }
    VkCommandBuffer cmd = cmds[i];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo binfo = {};
    binfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    binfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &binfo) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    *idx = i;
    return cmd;
}

bool FrameGenerator::Impl::submit(bool of_queue, VkCommandBuffer cmd, int idx,
                                  const std::vector<std::pair<VkSemaphore, uint64_t>> &extra_waits)
{
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        return false;
    std::vector<VkSemaphore> wait_sems;
    std::vector<uint64_t> wait_vals;
    std::vector<VkPipelineStageFlags> wait_stages;
    if (fg_val > 0) {
        wait_sems.push_back(fg_sem);
        wait_vals.push_back(fg_val);
        wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    for (const auto &w : extra_waits) {
        wait_sems.push_back(w.first);
        wait_vals.push_back(w.second);
        wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    const uint64_t signal_val = fg_val + 1;
    VkTimelineSemaphoreSubmitInfo tl = {};
    tl.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tl.waitSemaphoreValueCount = static_cast<uint32_t>(wait_vals.size());
    tl.pWaitSemaphoreValues = wait_vals.data();
    tl.signalSemaphoreValueCount = 1;
    tl.pSignalSemaphoreValues = &signal_val;
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tl;
    si.waitSemaphoreCount = static_cast<uint32_t>(wait_sems.size());
    si.pWaitSemaphores = wait_sems.data();
    si.pWaitDstStageMask = wait_stages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &fg_sem;
    VkResult res;
    if (of_queue) {
        res = vkQueueSubmit(of_q, 1, &si, VK_NULL_HANDLE);
    } else {
        vk->lock_queue(vk, gfx_qf, gfx_qidx);
        res = vkQueueSubmit(gfx_q, 1, &si, VK_NULL_HANDLE);
        vk->unlock_queue(vk, gfx_qf, gfx_qidx);
    }
    if (res != VK_SUCCESS) {
        qCWarning(chiakiFrameGen) << "vkQueueSubmit failed:" << static_cast<int>(res);
        return false;
    }
    fg_val = signal_val;
    (of_queue ? of_cmd_val : gfx_cmd_val)[idx] = fg_val;
    return true;
}

bool FrameGenerator::Impl::hold(Tex &t, VkImageLayout layout)
{
    pl_vulkan_hold_params hp = {};
    hp.tex = t.tex;
    hp.layout = layout;
    hp.qf = gfx_qf;
    hp.semaphore.sem = t.sem;
    hp.semaphore.value = t.val + 1;
    if (!pl_vulkan_hold_ex(gpu, &hp))
        return false;
    t.val++;
    return true;
}

void FrameGenerator::Impl::release(Tex &t, VkImageLayout layout, bool signaled)
{
    pl_vulkan_release_params rp = {};
    rp.tex = t.tex;
    rp.layout = layout;
    rp.qf = gfx_qf;
    if (signaled) {
        rp.semaphore.sem = fg_sem;
        rp.semaphore.value = fg_val;
    }
    pl_vulkan_release_ex(gpu, &rp);
}

void FrameGenerator::Impl::imageBarrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

bool FrameGenerator::Impl::dispatch(const char *header, const char *body, const std::vector<pl_shader_desc> &descs,
                                    const std::vector<pl_shader_var> &vars, pl_tex target, pl_timer timer)
{
    pl_shader sh = pl_dispatch_begin(dp);
    if (!sh)
        return false;
    pl_custom_shader cs = {};
    cs.header = header;
    cs.body = body;
    cs.description = "chiaki frame generation";
    cs.input = PL_SHADER_SIG_NONE;
    cs.output = PL_SHADER_SIG_COLOR;
    cs.descriptors = descs.data();
    cs.num_descriptors = static_cast<int>(descs.size());
    cs.variables = vars.data();
    cs.num_variables = static_cast<int>(vars.size());
    if (!pl_shader_custom(sh, &cs)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    pl_dispatch_params p = {};
    p.shader = &sh;
    p.target = target;
    p.timer = timer;
    return pl_dispatch_finish(dp, &p);
}

static pl_shader_desc texDesc(const char *name, pl_tex tex, bool linear)
{
    pl_shader_desc d = {};
    d.desc.name = name;
    d.desc.type = PL_DESC_SAMPLED_TEX;
    d.binding.object = tex;
    d.binding.address_mode = PL_TEX_ADDRESS_CLAMP;
    d.binding.sample_mode = linear ? PL_TEX_SAMPLE_LINEAR : PL_TEX_SAMPLE_NEAREST;
    return d;
}

static pl_shader_var vec2Var(const char *name, const float *data)
{
    pl_shader_var v = {};
    v.var = pl_var_vec2(name);
    v.data = data;
    v.dynamic = true;
    return v;
}

static pl_shader_var floatVar(const char *name, const float *data)
{
    pl_shader_var v = {};
    v.var = pl_var_float(name);
    v.data = data;
    v.dynamic = true;
    return v;
}

static pl_shader_var intVar(const char *name, const int *data)
{
    pl_shader_var v = {};
    v.var = pl_var_int(name);
    v.data = data;
    v.dynamic = false;
    return v;
}

bool FrameGenerator::Impl::renderLuma(pl_tex rgb, pl_tex luma)
{
    const float size[2] = { static_cast<float>(LW), static_cast<float>(LH) };
    return dispatch(nullptr, kLumaShader, { texDesc("fg_src", rgb, true) }, { vec2Var("fg_out_size", size) }, luma, nullptr);
}

bool FrameGenerator::Impl::measureDiff(pl_tex a, pl_tex b, double *mean, double *max_block)
{
    static const char *header = "#define FG_DIFF_W 64\n#define FG_DIFF_H 36\n";
    if (!dispatch(header, kDiffShader, { texDesc("fg_a", a, false), texDesc("fg_b", b, false) }, {}, diff_tex, nullptr))
        return false;
    float data[kDiffW * kDiffH] = {};
    pl_tex_transfer_params tp = {};
    tp.tex = diff_tex;
    tp.ptr = data;
    if (!pl_tex_download(gpu, &tp))
        return false;
    double sum = 0.0, mx = 0.0;
    for (float v : data) {
        sum += v;
        mx = qMax<double>(mx, v);
    }
    *mean = sum / (kDiffW * kDiffH);
    *max_block = mx;
    return true;
}

bool FrameGenerator::Impl::copyLumaToFlowInput(Slot &s)
{
    if (!hold(s.luma, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL))
        return false;
    int idx = 0;
    VkCommandBuffer cmd = beginCmd(false, &idx);
    if (!cmd) {
        release(s.luma, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, false);
        return false;
    }
    imageBarrier(cmd, s.ofin.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy region = {};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource = region.srcSubresource;
    region.extent = { static_cast<uint32_t>(LW), static_cast<uint32_t>(LH), 1 };
    vkCmdCopyImage(cmd, s.luma.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   s.ofin.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    imageBarrier(cmd, s.ofin.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    const bool ok = submit(false, cmd, idx, { { s.luma.sem, s.luma.val } });
    release(s.luma, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ok);
    return ok;
}

bool FrameGenerator::Impl::runFlow(Slot &prev, Slot &cur)
{
    // the session images can only be rebound once the previous execution finished
    if (last_of_val > 0) {
        VkSemaphoreWaitInfo winfo = {};
        winfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        winfo.semaphoreCount = 1;
        winfo.pSemaphores = &fg_sem;
        winfo.pValues = &last_of_val;
        if (vkWaitSemaphores(dev, &winfo, 100000000ull) != VK_SUCCESS)
            return false;
    }
    // input = cur, reference = prev: the flow vectors point from cur to prev (flow_b) and the
    // backward vectors from prev to cur (flow_f)
    if (vkBindOpticalFlowSessionImageNV(dev, sess, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
                                        cur.ofin.view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS ||
        vkBindOpticalFlowSessionImageNV(dev, sess, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV,
                                        prev.ofin.view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS)
        return false;
    if (sess_rev) {
        if (vkBindOpticalFlowSessionImageNV(dev, sess_rev, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
                                            prev.ofin.view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS ||
            vkBindOpticalFlowSessionImageNV(dev, sess_rev, VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV,
                                            cur.ofin.view, VK_IMAGE_LAYOUT_GENERAL) != VK_SUCCESS)
            return false;
    }

    int idx = 0;
    VkCommandBuffer cmd = beginCmd(true, &idx);
    if (!cmd)
        return false;
    imageBarrier(cmd, flowout_f.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    imageBarrier(cmd, flowout_b.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    if (timestamps) {
        vkCmdResetQueryPool(cmd, qpool, 2 * idx, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 2 * idx);
    }
    VkOpticalFlowExecuteInfoNV einfo = {};
    einfo.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV;
    vkCmdOpticalFlowExecuteNV(cmd, sess, &einfo);
    if (sess_rev)
        vkCmdOpticalFlowExecuteNV(cmd, sess_rev, &einfo);
    if (timestamps)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 2 * idx + 1);
    if (!submit(true, cmd, idx, {}))
        return false;
    last_of_val = fg_val;
    if (timestamps) {
        query_pending[idx] = true;
        query_val[idx] = fg_val;
    }

    // copy both flow fields into the libplacebo textures of `cur`
    if (!hold(cur.flow_f, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL))
        return false;
    if (!hold(cur.flow_b, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)) {
        release(cur.flow_f, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, false);
        return false;
    }
    cmd = beginCmd(false, &idx);
    if (!cmd) {
        release(cur.flow_f, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, false);
        release(cur.flow_b, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, false);
        return false;
    }
    imageBarrier(cmd, flowout_f.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    imageBarrier(cmd, flowout_b.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const VkDeviceSize half = static_cast<VkDeviceSize>(FW) * FH * 4;
    VkBufferImageCopy bic = {};
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount = 1;
    bic.imageExtent = { static_cast<uint32_t>(FW), static_cast<uint32_t>(FH), 1 };
    bic.bufferOffset = 0;
    vkCmdCopyImageToBuffer(cmd, flowout_f.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &bic);
    bic.bufferOffset = half;
    vkCmdCopyImageToBuffer(cmd, flowout_b.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &bic);
    VkBufferMemoryBarrier bb = {};
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = staging;
    bb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &bb, 0, nullptr);
    Tex &dst_f = flip ? cur.flow_b : cur.flow_f;
    Tex &dst_b = flip ? cur.flow_f : cur.flow_b;
    bic.bufferOffset = 0;
    vkCmdCopyBufferToImage(cmd, staging, dst_f.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
    bic.bufferOffset = half;
    vkCmdCopyBufferToImage(cmd, staging, dst_b.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
    const bool ok = submit(false, cmd, idx, { { cur.flow_f.sem, cur.flow_f.val }, { cur.flow_b.sem, cur.flow_b.val } });
    release(cur.flow_f, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ok);
    release(cur.flow_b, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ok);
    return ok;
}

bool FrameGenerator::Impl::warp(Slot &prev, Slot &cur, double phase)
{
    const float out_size[2] = { static_cast<float>(W), static_cast<float>(H) };
    // snorm value * 32767 = S10.5 raw value, / 32 = pixels of the flow input, / size = uv
    const float fscale[2] = {
        static_cast<float>(32767.0 / 32.0 / LW),
        static_cast<float>(32767.0 / 32.0 / LH),
    };
    const float t = static_cast<float>(phase);
    const int dbg = debug;
    return dispatch(kWarpHeader, kWarpShader,
                    {
                        texDesc("fg_prev", prev.rgb, true),
                        texDesc("fg_cur", cur.rgb, true),
                        texDesc("fg_flow_f", cur.flow_f.tex, true),
                        texDesc("fg_flow_b", cur.flow_b.tex, true),
                    },
                    {
                        vec2Var("fg_out_size", out_size),
                        vec2Var("fg_fscale", fscale),
                        floatVar("fg_t", &t),
                        intVar("fg_debug", &dbg),
                    },
                    warp_out, warp_timer);
}

pl_frame FrameGenerator::Impl::rgbFrame(pl_tex tex, const pl_frame *target) const
{
    pl_frame f = {};
    f.num_planes = 1;
    f.planes[0].texture = tex;
    f.planes[0].components = 4;
    f.planes[0].component_mapping[0] = PL_CHANNEL_R;
    f.planes[0].component_mapping[1] = PL_CHANNEL_G;
    f.planes[0].component_mapping[2] = PL_CHANNEL_B;
    f.planes[0].component_mapping[3] = PL_CHANNEL_A;
    f.repr = pl_color_repr_rgb;
    f.repr.alpha = PL_ALPHA_NONE;
    f.color = target->color;
    f.crop = target->crop;
    return f;
}

void FrameGenerator::Impl::resetHistory()
{
    hist.clear();
    for (auto &s : fslots) {
        s.flow_ok = false;
        s.cut = false;
        s.pts = 0.0;
    }
    last_ingest_pts = -1e18;
    consecutive_dups = 0;
    intervals.clear();
    leads.clear();
    delay_s = 0.0;
}

void FrameGenerator::Impl::ingestFrame(const pl_frame *frame, double pts, const pl_render_params *params, const pl_frame *target)
{
    // the slot after the history is free: the history keeps at most kSlots - 1 entries
    int slot = -1;
    for (int i = 0; i < kSlots; i++) {
        if (std::find(hist.begin(), hist.end(), i) == hist.end()) {
            slot = i;
            break;
        }
    }
    if (slot < 0)
        return;
    Slot &s = fslots[slot];

    pl_frame t = rgbFrame(s.rgb, target);
    pl_render_params p = *params;
    p.frame_mixer = nullptr;
    if (!pl_render_image(ingest, frame, &t, &p)) {
        qCWarning(chiakiFrameGen) << "Failed to render a frame for frame generation";
        return;
    }
    if (!renderLuma(s.rgb, s.luma.tex)) {
        qCWarning(chiakiFrameGen) << "Failed to compute luma";
        return;
    }

    Slot *prev = hist.empty() ? nullptr : &fslots[hist.back()];
    bool cut = false;
    if (prev) {
        double mean = 0.0, max_block = 0.0;
        const bool measured = measureDiff(s.luma.tex, prev->luma.tex, &mean, &max_block);
        if (measured) {
            last_diff = mean;
            const double thr = qBound(dup_min, 0.3 * motion_ref, dup_max);
            const bool dup = mean < thr && max_block < dup_block && consecutive_dups < 3 && (pts - prev->pts) < 0.07;
            if (dup) {
                consecutive_dups++;
                return;
            }
            cut = mean > cut_thr;
            if (!cut)
                motion_ref = 0.9 * motion_ref + 0.1 * mean;
        }
        intervals.push_back(pts - prev->pts);
        while (intervals.size() > 8)
            intervals.pop_front();
    }
    consecutive_dups = 0;
    st_distinct++;

    s.pts = pts;
    s.cut = cut;
    s.flow_ok = false;
    if (!copyLumaToFlowInput(s)) {
        qCWarning(chiakiFrameGen) << "Failed to prepare optical flow input";
        return;
    }
    if (prev && !cut)
        s.flow_ok = runFlow(*prev, s);
    if (cut)
        st_cuts++;
    hist.push_back(slot);
    while (static_cast<int>(hist.size()) > kSlots - 1)
        hist.pop_front();
}

void FrameGenerator::Impl::pollTimers()
{
    if (warp_timer) {
        uint64_t ns;
        while ((ns = pl_timer_query(gpu, warp_timer)) != 0) {
            st_warp_ns += static_cast<double>(ns);
            st_warp_n++;
        }
    }
}

FrameGenerator::FrameGenerator(pl_log log, pl_vulkan vk)
    : d(new Impl)
{
    d->log = log;
    d->vk = vk;
    QString why;
    if (!vk) {
        status_text = QStringLiteral("needs the Vulkan renderer");
        return;
    }
    init_ok = d->init(&why);
    status_text = init_ok ? QStringLiteral("ready") : why;
    if (!init_ok)
        qCWarning(chiakiFrameGen) << "Frame generation disabled:" << why;
}

FrameGenerator::~FrameGenerator()
{
    delete d;
}

void FrameGenerator::reset()
{
    if (d)
        d->resetHistory();
}

bool FrameGenerator::process(const pl_frame_mix *mix, double display_pts, double vsync_s,
                             const pl_render_params *ingest_params, const pl_frame *target, pl_frame *out)
{
    if (!init_ok || !mix || mix->num_frames <= 0 || !target || target->num_planes < 1 || !target->planes[0].texture)
        return false;
    Impl &x = *d;
    const qint64 begin_us = nowUs();
    x.pollTimers();

    const pl_tex target_tex = target->planes[0].texture;
    const int tw = target_tex->params.w, th = target_tex->params.h;
    if (!x.resources_ok || tw != x.W || th != x.H) {
        QString why;
        if (!x.allocate(tw, th, &why)) {
            qCWarning(chiakiFrameGen) << "Frame generation disabled:" << why;
            status_text = why;
            init_ok = false;
            return false;
        }
        x.resetHistory();
        x.crop = target->crop;
    }
    if (std::fabs(x.crop.x0 - target->crop.x0) > 0.5f || std::fabs(x.crop.y0 - target->crop.y0) > 0.5f ||
        std::fabs(x.crop.x1 - target->crop.x1) > 0.5f || std::fabs(x.crop.y1 - target->crop.y1) > 0.5f) {
        x.resetHistory();
        x.crop = target->crop;
    }

    const double frame_dur = mix->vsync_duration > 0.0 ? vsync_s / mix->vsync_duration : 1.0 / 60.0;

    // ingest the frames we have not seen yet, oldest first
    std::vector<int> order(mix->num_frames);
    for (int i = 0; i < mix->num_frames; i++)
        order[i] = i;
    std::sort(order.begin(), order.end(), [mix](int a, int b) { return mix->timestamps[a] < mix->timestamps[b]; });
    for (int i : order) {
        const uint64_t sig = mix->signatures[i];
        if (std::find(std::begin(x.seen), std::end(x.seen), sig) != std::end(x.seen))
            continue;
        x.seen[x.seen_pos] = sig;
        x.seen_pos = (x.seen_pos + 1) % static_cast<int>(std::size(x.seen));
        const double pts = display_pts + mix->timestamps[i] * frame_dur;
        if (pts <= x.last_ingest_pts + 1e-4)
            continue;
        x.last_ingest_pts = pts;
        x.st_stream++;
        x.leads.push_back(pts - display_pts);
        while (x.leads.size() > 30)
            x.leads.pop_front();
        x.ingestFrame(mix->frames[i], pts, ingest_params, target);
    }
    if (!init_ok || x.hist.empty())
        return false;

    // display delay: the next distinct frame must have arrived when the target time passes
    // the current one
    const double interval = x.intervals.empty() ? 1.0 / 30.0 : qBound(1.0 / 240.0, median(x.intervals), 0.1);
    double lead_min = 0.0;
    if (!x.leads.empty())
        lead_min = *std::min_element(x.leads.begin(), x.leads.end());
    const double want = qBound(0.0, interval - lead_min + x.margin_s, 0.1);
    if (want > x.delay_s)
        x.delay_s += 0.25 * (want - x.delay_s);
    else
        x.delay_s += 0.02 * (want - x.delay_s);
    const double target_time = display_pts - x.delay_s;

    Impl::Slot *a = nullptr;
    Impl::Slot *b = nullptr;
    const int n = static_cast<int>(x.hist.size());
    if (target_time <= x.fslots[x.hist.front()].pts) {
        a = &x.fslots[x.hist.front()];
    } else {
        for (int k = 0; k + 1 < n; k++) {
            Impl::Slot &s0 = x.fslots[x.hist[k]];
            Impl::Slot &s1 = x.fslots[x.hist[k + 1]];
            if (s0.pts <= target_time && target_time < s1.pts) {
                a = &s0;
                b = &s1;
                break;
            }
        }
        if (!a) {
            a = &x.fslots[x.hist.back()];
            if (target_time > a->pts + interval * 1.05)
                x.st_late++;
        }
    }

    pl_tex show = a->rgb;
    bool generated = false;
    if (b) {
        const double phase = (target_time - a->pts) / qMax(1e-6, b->pts - a->pts);
        if (b->cut || !b->flow_ok) {
            show = phase < 0.5 ? a->rgb : b->rgb;
        } else if (phase > 0.98) {
            show = b->rgb;
        } else if (phase > 0.02) {
            if (x.warp(*a, *b, phase)) {
                show = x.warp_out;
                generated = true;
            }
        }
    }
    if (generated)
        x.st_gen++;
    else
        x.st_real++;
    *out = x.rgbFrame(show, target);

    const qint64 end_us = nowUs();
    x.st_cpu_us += static_cast<double>(end_us - begin_us);
    x.st_cpu_n++;
    return true;
}

bool FrameGenerator::takeStats(FrameGenStats *out, qint64 now_us)
{
    if (!d)
        return false;
    Impl &x = *d;
    if (x.period_start_us == 0) {
        x.period_start_us = now_us;
        return false;
    }
    const qint64 elapsed = now_us - x.period_start_us;
    if (elapsed < 1000000)
        return false;
    const double secs = elapsed / 1e6;
    FrameGenStats s;
    s.active = init_ok;
    s.stream_fps = x.st_stream / secs;
    s.content_fps = x.st_distinct / secs;
    s.real_fps = x.st_real / secs;
    s.generated_fps = x.st_gen / secs;
    s.flow_ms = x.st_flow_n ? x.st_flow_ns / x.st_flow_n / 1e6 : -1.0;
    s.warp_ms = x.st_warp_n ? x.st_warp_ns / x.st_warp_n / 1e6 : -1.0;
    s.cpu_ms = x.st_cpu_n ? x.st_cpu_us / x.st_cpu_n / 1e3 : 0.0;
    s.delay_ms = x.delay_s * 1000.0;
    s.diff = x.last_diff;
    s.late = x.st_late;
    s.cuts = x.st_cuts;
    *out = s;
    x.period_start_us = now_us;
    x.st_stream = x.st_distinct = x.st_real = x.st_gen = x.st_late = x.st_cuts = 0;
    x.st_flow_ns = x.st_warp_ns = x.st_cpu_us = 0.0;
    x.st_flow_n = x.st_warp_n = x.st_cpu_n = 0;
    return true;
}

#endif // FG_HAVE_OF
