# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Re-export the registry declarations from the implementation submodule so the
# package `ttnn.operations.point_to_point` exposes them directly — this is what the
# golden harness (`from ttnn.operations.point_to_point import EXCLUSIONS,
# INPUT_TAGGERS, SUPPORTED`) and `eval.verify_supported` introspect.
#
# EXCLUSIONS / INPUT_TAGGERS / validate are eager (no ttnn-enum references at module
# scope). SUPPORTED references ttnn.* enums that are not registered when this package
# is imported during ttnn init, so it stays lazy and is forwarded via __getattr__
# (PEP 562) — `hasattr(package, "SUPPORTED")` triggers it.
from .point_to_point import EXCLUSIONS, INPUT_TAGGERS, point_to_point, validate

__all__ = ["point_to_point", "validate", "SUPPORTED", "EXCLUSIONS", "INPUT_TAGGERS"]


def __getattr__(name):
    if name == "SUPPORTED":
        from .point_to_point import _supported

        return _supported()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
