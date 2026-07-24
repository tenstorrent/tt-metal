# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Isolated perf bake-off for rms_norm pass-1: BATCH the square + local reduce across the C
# tile-rows of a cross-core round (blocking-perf-part-optimizer artifact; NOT the real op,
# uncommitted). Deliberately imports NOTHING at package-init time: ttnn.operations.__init__
# walk-imports every package here at `import ttnn`, so a heavy or fragile __init__ would break
# the whole library import for every sibling. The bench module is imported lazily by the test.
