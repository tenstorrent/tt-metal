# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The routed-expert shapes of the close-out set (EXPERTS_COMBINE_CLOSEOUT.md) and their per-chip sizing on a stage:
experts per chip, home tokens per chip and tokens per expert (uniform routing) for a chunk of T tokens over the stage's
chips. H is the width of the rows the experts and the combine see (K3: the latent width)."""

MODELS = {
    "glm53": dict(name="GLM-5.3", H=6144, I=2048, E=256, K=8, act="silu"),
    "glm53_flash": dict(name="GLM-5.3-Flash", H=4096, I=2048, E=288, K=8, act="silu"),
    "k3": dict(name="Kimi K3", H=3584, I=3072, E=896, K=16, act="situ"),
    "k27": dict(name="Kimi K2.7", H=7168, I=2048, E=384, K=8, act="silu"),
    "m3": dict(name="MiniMax M3", H=6144, I=3072, E=128, K=4, act="swigluoai"),
    "dsv41_flash": dict(name="DSv4.1-Flash", H=5120, I=2304, E=384, K=6, act="clamped_silu"),
}
STAGES = {"8x4": (8, 4), "4x4": (4, 4), "2x4": (2, 4)}  # (SP, TP) chips
CHUNKS = (5120, 16384)
GALAXY = "8x4"


def per_chip(model, stage, chunk):
    """dict(chips, epc (experts per chip), S (home tokens per chip), M (tokens per expert, uniform))."""
    m = MODELS[model]
    sp, tp = STAGES[stage]
    n = sp * tp
    assert m["E"] % n == 0 and chunk % n == 0, (model, stage, chunk)
    return dict(chips=n, epc=m["E"] // n, S=chunk // n, M=chunk * m["K"] / m["E"])


def fabric_payload(H):
    """Fabric payload bytes for the combine's L1 hand-off: one piece a row (H bf16) plus its trailer, 64-B aligned."""
    return max(7232, (2 * H + 32 + 63) // 64 * 64)
