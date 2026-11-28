# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: worst per-core RT words of the host move list for every TILE INPUTS entry + a size ramp."""
import math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ttnn
import hml
from eval.golden_tests.g6_reshape import feature_spec as S

ES = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.int32: 4, ttnn.uint32: 4}
for var in ("hml_bank_split", "hml_bank_split_zc"):
    for e in S.INPUTS:
        sc = e[0]
        if sc["layout"] != ttnn.TILE_LAYOUT:
            continue
        out = hml.BASE._resolve_shape(math.prod(sc["input_shape"]), sc["shape"])
        st = hml.plan_stats(sc["input_shape"], out, ES[sc["dtype"]], var)
        print(var, sc["input_shape"], "->", out, st)
# ramp: (1, n, 6, 256) -> (1, 1, n, 1536) (the q553 family) until the RT cap
for n in (64, 256, 1024, 2048, 4096, 8192):
    st = hml.plan_stats((1, n, 6, 256), (1, 1, n, 1536), 2, "hml_bank_split")
    print("ramp", n, st, "feasible" if st["worst_words"] <= hml.MAX_RT_WORDS else "OVER CAP")
