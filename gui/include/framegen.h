// Frame generation for the desktop client: motion-compensated interpolation between decoded
// frames, using the NVIDIA optical flow hardware through VK_NV_optical_flow and a warp shader
// run by libplacebo. Everything here runs on the render thread.
#ifndef CHIAKI_GUI_FRAMEGEN_H
#define CHIAKI_GUI_FRAMEGEN_H

#include <libplacebo/vulkan.h>
#include <libplacebo/renderer.h>

#include <QString>

struct FrameGenStats
{
    bool active = false;
    double stream_fps = 0.0;     // frames received from the queue
    double content_fps = 0.0;    // distinct game frames (duplicates removed)
    double real_fps = 0.0;       // presented frames that are real game frames
    double generated_fps = 0.0;  // presented frames that were generated
    double flow_ms = 0.0;        // optical flow GPU time per pair
    double warp_ms = 0.0;        // interpolation shader GPU time per generated frame
    double cpu_ms = 0.0;         // render thread time spent in the generator per presented frame
    double delay_ms = 0.0;       // extra display delay added by the generator
    double diff = 0.0;           // last measured luma difference (duplicate detection)
    int late = 0;                // presents where the next real frame had not arrived
    int cuts = 0;                // scene cuts (no interpolation)
};

class FrameGenerator
{
public:
    // The device extension to request at device creation time
    static const char *extensionName();
    // Whether this Vulkan headers build knows the extension at all
    static bool compiledIn();
    // The queue flag to add to pl_vulkan_params.extra_queues
    static VkQueueFlags queueFlag();
    // Whether any physical device of `instance` exposes the extension
    static bool instanceHasCapableDevice(VkInstance instance, PFN_vkGetInstanceProcAddr gipa);

    FrameGenerator(pl_log log, pl_vulkan vk);
    ~FrameGenerator();

    bool ok() const { return init_ok; }
    QString status() const { return status_text; }

    // Drops all history (queue reset, resolution change, stream restart)
    void reset();

    // Called once per presented vsync with the queue's frame mix. When it returns true,
    // `out` holds the frame to present in place of the mix (already in the target's color
    // space, with the target's crop). `display_pts` and `vsync_s` are in queue seconds.
    bool process(const struct pl_frame_mix *mix, double display_pts, double vsync_s,
                 const struct pl_render_params *ingest_params, const struct pl_frame *target,
                 struct pl_frame *out);

    // Returns true about once per second with the stats of the last period
    bool takeStats(FrameGenStats *out, qint64 now_us);

private:
    struct Impl;
    Impl *d = nullptr;
    bool init_ok = false;
    QString status_text;
};

#endif // CHIAKI_GUI_FRAMEGEN_H
