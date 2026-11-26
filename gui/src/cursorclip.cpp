// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include "cursorclip.h"

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static bool clip_active = false;

void CursorClipToWindow(QWindow *window)
{
	HWND hwnd = reinterpret_cast<HWND>(window->winId());
	RECT client;
	if (!hwnd || !GetClientRect(hwnd, &client))
		return;
	POINT tl = { client.left, client.top };
	POINT br = { client.right, client.bottom };
	ClientToScreen(hwnd, &tl);
	ClientToScreen(hwnd, &br);
	// Central half of the window, in physical pixels: the cursor never
	// reaches an edge, so it cannot leave the window between warps.
	const LONG w = br.x - tl.x, h = br.y - tl.y;
	if (w < 8 || h < 8)
		return;
	RECT clip = { tl.x + w / 4, tl.y + h / 4, br.x - w / 4, br.y - h / 4 };
	RECT current;
	if (clip_active && GetClipCursor(&current) && EqualRect(&current, &clip))
		return;
	ClipCursor(&clip);
	clip_active = true;
}

void CursorClipRelease()
{
	if (!clip_active)
		return;
	ClipCursor(nullptr);
	clip_active = false;
}

#else

void CursorClipToWindow(QWindow *) {}
void CursorClipRelease() {}

#endif
