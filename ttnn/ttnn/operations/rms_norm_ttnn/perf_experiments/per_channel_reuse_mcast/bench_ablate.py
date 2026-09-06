# Perf experiment `per_channel_reuse_mcast` -- STEP 1: the UPPER BOUND.
#
# The idea under test is "read the per-channel vector ONCE on an injector core and
# multicast it to the reuse group".  Before wiring any multicast, measure the
# ceiling that idea can ever reach: a build in which the per-channel DRAM read
# COSTS NOTHING (the reads are deleted, the CB handshake is kept verbatim).
#
#   base   = the shipped reader (k_base, a byte-for-byte copy of kernels/)
#   ablate = k_ablate, stage_per_channel_chunk's TILE branch with the
#            noc_async_read calls removed.  OUTPUT IS INTENTIONALLY WRONG --
#            this is a perf ablation, never a correctness run.
#
# No multicast can beat `ablate`, because a multicast still costs one DRAM read
# plus a broadcast.  If (base - ablate) is inside the noise band on a shape, the
# GAMMA_MCAST deferral is CLOSED for that shape with a number.
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes"))

import bench_r3  # noqa: E402

NAMES = [
    "G4_blk7168",  # BLOCK 8x8, gamma+bias+residual -- reuse 8, TWO operands
    "G3_blk8192",  # BLOCK 8x8, gamma -- reuse 8
    "P1_int1024_g",  # INTERLEAVED row split -- reuse == num_cores (whole vector/core)
    "P2_int1024_gb",  # same, two operands
    "S1_stream_gbr",  # STREAM: per-channel re-read per chunk per block
    "P6_int7168_g",  # DRAM-saturated prefill
    "G1_w7168_g28",  # focus shape: WIDTH shard, NO reuse at all
]

if __name__ == "__main__":
    names = sys.argv[1:] or NAMES
    bench_r3.sweep(
        [
            ("base", {"KERNEL_DIR": HERE / "k_base"}),
            ("ablate", {"KERNEL_DIR": HERE / "k_ablate"}),
        ],
        names,
    )
