# Copyright (C) 2026 Michael Cobb
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frozen-app PATH so PortAudio / Pedalboard DLLs resolve next to the exe."""

from __future__ import annotations

import os
import sys

if getattr(sys, "frozen", False):
    root = os.path.dirname(sys.executable)
    os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
