# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The shared deprecation helper for the TTNN Python API.

See .github/instructions/ttnn-python-deprecation.instructions.md. Imports only the standard library, so the
C++ wrapper (ttnn/cpp/ttnn-nanobind/deprecation.hpp) can import it at any time without an import cycle.
"""

import os
import sys
import warnings

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep


def _is_inside_ttnn_package(frame) -> bool:
    filename = frame.f_code.co_filename
    # Pseudo-filenames such as "<string>" or "<frozen ...>" have no source file, so they are never TTNN's code.
    return not filename.startswith("<") and os.path.realpath(filename).startswith(_PACKAGE_DIR)


def _is_skipped_by_warnings(frame) -> bool:
    # warnings.warn doesn't count importlib's bootstrap frames toward stacklevel; this is its test for them.
    filename = frame.f_code.co_filename
    return "importlib" in filename and "_bootstrap" in filename


def warn_deprecated(what: str, instead: str) -> None:
    """Warn that ``what`` is deprecated and that the caller should use ``instead``.

    Emits a ``FutureWarning`` attributed to the first stack frame outside the TTNN package, which is the user's
    code, whether the call comes from pure Python or from a binding through TTNN's operation wrappers. If every
    frame is inside the package, the warning is attributed to the outermost one.
    """
    if not what:
        raise ValueError(f"warn_deprecated needs a non-empty 'what'; got what={what!r}, instead={instead!r}")
    if not instead:
        raise ValueError(f"warn_deprecated needs a non-empty 'instead'; got what={what!r}, instead={instead!r}")

    # Python 3.10 is supported, so warnings.warn's skip_file_prefixes (3.12+) isn't available: count the frames
    # the way warnings.warn walks them. stacklevel 1 is this function's own frame. When a binding is called from
    # C++ with no Python frame above it, the walk finds nothing and the warning stays at stacklevel 1.
    stacklevel = 1
    frame = sys._getframe(0).f_back
    while frame is not None:
        if not _is_skipped_by_warnings(frame):
            stacklevel += 1
            if not _is_inside_ttnn_package(frame):
                break
        frame = frame.f_back
    warnings.warn(
        f"{what} is deprecated and will be removed; use {instead} instead.", FutureWarning, stacklevel=stacklevel
    )
