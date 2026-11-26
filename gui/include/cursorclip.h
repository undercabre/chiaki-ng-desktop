// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#ifndef CHIAKI_CURSORCLIP_H
#define CHIAKI_CURSORCLIP_H

#include <QWindow>

// Keep the mouse cursor inside the central part of a window (Windows only;
// no-op elsewhere). Used by the keyboard and mouse game mode.
void CursorClipToWindow(QWindow *window);
void CursorClipRelease();

#endif
