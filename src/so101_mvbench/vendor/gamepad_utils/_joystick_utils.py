# SPDX-License-Identifier: MIT
"""Shared Linux joystick API utilities.

Centralises the raw /dev/input/js* constants and helpers that are used by
both JointGamepad and Se3GamepadPygame to avoid duplication.
"""

from __future__ import annotations

import fcntl
import os
import struct

# ---------------------------------------------------------------------------
# Linux joystick event struct: __u32 time, __s16 value, __u8 type, __u8 number
# ---------------------------------------------------------------------------
JS_EVENT_FMT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FMT)

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80  # OR'd with type on initial state reports

# ---------------------------------------------------------------------------
# ioctl request code for reading the device name (JSIOCGNAME).
# Encoding: _IOC(_IOC_READ=0x80, type='j'=0x6a, nr=0x13, size=256)
# → 0x80006a13 | (256 << 16)
# ---------------------------------------------------------------------------
_MAX_NAME_LEN = 256
_JSIOCGNAME = 0x80006a13 + (_MAX_NAME_LEN << 16)

# Maximum number of /dev/input/js* devices to probe.
_MAX_JS_DEVICES = 4


# Virtual input devices that expose a js* node but are NOT physical gamepads.
# Sunshine/Moonlight game streaming creates "Mouse passthrough (absolute)" as
# js0 — naively taking the first js* then feeds MOUSE motion into the teleop
# joint mapping (observed 2026-06-11: arm slammed into the table instantly).
_VIRTUAL_NAME_MARKERS = ("passthrough", "mouse", "keyboard", "touchpad")


def _device_name(path: str) -> str:
    """Read a js device's name without keeping the fd open."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return "Unknown Joystick"
    try:
        return read_device_name(fd)
    finally:
        os.close(fd)


def find_js_device() -> str | None:
    """Return the path of the best available joystick device, or None.

    Selection order:
      1. ``$GAMEPAD_JS_DEVICE`` if set and existing (explicit override).
      2. The first js* whose device name does NOT look like a virtual
         passthrough device (mouse/keyboard emulation from game streaming).
      3. Fallback: the first existing js* (previous behavior).
    """
    override = os.environ.get("GAMEPAD_JS_DEVICE")
    if override and os.path.exists(override):
        return override

    first_existing: str | None = None
    for i in range(_MAX_JS_DEVICES):
        path = f"/dev/input/js{i}"
        if not os.path.exists(path):
            continue
        if first_existing is None:
            first_existing = path
        name = _device_name(path).lower()
        if not any(marker in name for marker in _VIRTUAL_NAME_MARKERS):
            return path
    return first_existing


def read_device_name(fd: int) -> str:
    """Read the device name from an open joystick file descriptor via ioctl.

    Returns 'Unknown Joystick' if the ioctl call fails.
    """
    buf = bytearray(_MAX_NAME_LEN)
    try:
        fcntl.ioctl(fd, _JSIOCGNAME, buf)
        return buf.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    except OSError:
        return "Unknown Joystick"
