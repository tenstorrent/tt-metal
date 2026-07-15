# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only benchmark: matmul OUTPUT-SUBBLOCK shape → SRC-register argument reuse.

A blocked tiled matmul `C[M,N] = A[M,K] @ B[K,N]` computed with the `matmul_block` kernel-lib helper.
The helper produces the output one `sb_h x sb_w` subblock at a time; each subblock is a single block
matmul on the Matrix Unit. The load-bearing knob is the **subblock shape**, because of how operands
are reused in the SRC registers within one block-matmul call:

  - A WIDE subblock (`sb_w > sb_h`, e.g. 1x8): one A row-tile is loaded into a SRC register ONCE and
    multiplied against all `sb_w` B column-tiles — A is reused `sb_w` times instead of re-fetched per
    output tile.
  - A TALL subblock (`sb_h > sb_w`, e.g. 8x1): one B column-tile is loaded once and reused across all
    `sb_h` A row-tiles.
  - A SQUARE subblock (`sb_h == sb_w`, e.g. 2x2) still reuses: the LLK condition is `reuse_a =
    ct_dim >= rt_dim` (`llk_unpack_AB_matmul.h`), so the tie takes the reuse-A path — A is held and B
    streamed across the `sb_w` columns. Any subblock with a dimension > 1 reuses an operand.
  - Only a 1x1 subblock reuses NOTHING: every output tile re-loads both its A and B operand into SRC.

So a bigger subblock amortizes SRC-register operand loads (and the per-subblock init/handshake) over
more output tiles. Everything is sharded in L1 on one Tensix core — no DRAM in the fast path — so the
measured delta is that operand-reuse effect.

To isolate the output-block reuse cleanly the contraction is a single K-tile (`Kt=1`, K=32): the win
is entirely in how each subblock is produced, not in K accumulation. That also frees the DEST budget
(no fp32 accumulator needed for a single K-tile), so a subblock may fill all 8 DEST tiles — enough for
1x8 / 8x1. The output is packed row-major (`OutputCBLayout::TileRowMajor`) so every variant produces
the identical, correct C. See README.md.
"""

import ttnn

TILE = 32
CB_A = 0  # A, sharded L1 (resident) -> matmul in0 (SrcB)
CB_B = 1  # B, sharded L1 (resident) -> matmul in1 (SrcA)
CB_C = 16  # C, sharded L1 (output)

# Baseline first: 1x1 reuses nothing. The rest reuse an operand across the subblock:
# wide (1x8, 2x4) reuse A across B tiles; tall (8x1, 4x2) reuse B across A rows; 2x2 is balanced.
VARIANTS = ("sb_1x1", "sb_1x8", "sb_8x1", "sb_2x4", "sb_4x2", "sb_2x2")

# variant -> (out_subblock_h, out_subblock_w) in tiles.
SUBBLOCK = {
    "sb_1x1": (1, 1),
    "sb_1x8": (1, 8),
    "sb_8x1": (8, 1),
    "sb_2x4": (2, 4),
    "sb_4x2": (4, 2),
    "sb_2x2": (2, 2),
}

# fp16 DEST holds 8 tiles (half-sync, no fp32 accumulator). A subblock must fit: sb_h * sb_w <= 8.
DST_TILES = 8

_MATMUL_KERNEL = r"""
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace ckl = compute_kernel_lib;

// C[M,N] = A[M,K] @ B[K,N] via the matmul_block helper, single K-block. The output-subblock shape
// (sb_h x sb_w) sets how operands are reused in SRC within each block-matmul call. Inputs are
// retained across the in-kernel iterations (sharded, resident); output is packed row-major so C is
// correct for any subblock shape.
void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t sb_h = get_compile_time_arg_val(3);
    constexpr uint32_t sb_w = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 16;
    constexpr uint32_t m_sb = Mt / sb_h;  // output subblocks along M
    constexpr uint32_t n_sb = Nt / sb_w;  // output subblocks along N

    // Sharded inputs already resident in L1 — mark them available once; the matmul retains them
    // (WaitAndRetainOnLastBlock) so they stay fronted across every iteration.
    cb_reserve_back(cb_a, Mt * Kt); cb_push_back(cb_a, Mt * Kt);
    cb_reserve_back(cb_b, Kt * Nt); cb_push_back(cb_b, Kt * Nt);

    // Matmul maps in0->SrcB, in1->SrcA: boot the one-time hw_configure with SrcOrder::Reverse, then
    // the short matmul unpack/math init (no hw_configure -- mm_block_init is deprecated and would
    // redundantly re-run the whole hw_configure MMIO that compute_kernel_hw_startup already did).
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_c);
    matmul_block_init(cb_a, cb_b, /*transpose=*/0, /*ct=*/sb_w, /*rt=*/sb_h, /*kt=*/Kt);

    CircularBuffer a_buf(cb_a), b_buf(cb_b), c_buf(cb_c);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor,
            ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndRetainOnLastBlock,
            ckl::InputPolicy::WaitAndRetainOnLastBlock>(
            a_buf, b_buf, c_buf, c_buf, ckl::MatmulBlockShape::of(m_sb, n_sb, sb_h, sb_w, Kt, 1));
        // Drain the row-major output between iterations (the last iteration's C is left for readback).
        if (iter + 1 < kernel_iters) { cb_wait_front(cb_c, Mt * Nt); cb_pop_front(cb_c, Mt * Nt); }
    }
    cb_pop_front(cb_a, Mt * Kt);
    cb_pop_front(cb_b, Kt * Nt);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(h_tiles, w_tiles):
    """The whole [h_tiles x w_tiles] tile matrix as one shard on a single core (tiles row-major)."""
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def validate(a, b, variant):
    if variant not in VARIANTS:
        raise ValueError(f"matmul_output_subblock: variant must be one of {VARIANTS}, got {variant!r}")
    for name, t in (("A", a), ("B", b)):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"matmul_output_subblock: {name} must be bfloat16 TILE_LAYOUT")
    if list(a.shape)[1] != list(b.shape)[0]:
        raise ValueError(f"matmul_output_subblock: inner dims must match, got A={a.shape} B={b.shape}")
    sb_h, sb_w = SUBBLOCK[variant]
    if sb_h * sb_w > DST_TILES:
        raise ValueError(f"matmul_output_subblock: subblock {sb_h}x{sb_w} exceeds DEST budget {DST_TILES}")
    Mt, Nt = (list(a.shape)[0] // TILE, list(b.shape)[1] // TILE)
    if Mt % sb_h or Nt % sb_w:
        raise ValueError(f"matmul_output_subblock: Mt={Mt},Nt={Nt} must divide subblock {sb_h}x{sb_w}")


def create_program_descriptor(a, b, c, *, variant, kernel_iters):
    validate(a, b, variant)
    sb_h, sb_w = SUBBLOCK[variant]
    Mt, Kt = (list(a.shape)[0] // TILE, list(a.shape)[1] // TILE)
    Nt = list(b.shape)[1] // TILE

    compute = ttnn.KernelDescriptor(
        kernel_source=_MATMUL_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[Mt, Nt, Kt, sb_h, sb_w, kernel_iters],
        # Single K-tile contraction: no K accumulation, so no fp32 DEST needed -> the 8-tile fp16
        # DEST budget is free for wide/tall subblocks (up to 8). HiFi2 is correct for bf16 here.
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.HiFi2),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_A, a),
        ttnn.cb_descriptor_from_sharded_tensor(CB_B, b),
        ttnn.cb_descriptor_from_sharded_tensor(CB_C, c),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def matmul_output_subblock(a, b, *, variant="sb_2x2", kernel_iters=1):
    """C = A @ B on one core, all operands sharded in L1. `variant` sets the output-subblock shape
    (see VARIANTS / SUBBLOCK). Output is the identical, correct C for every variant."""
    if kernel_iters < 1:
        raise ValueError("matmul_output_subblock: kernel_iters must be >= 1")
    validate(a, b, variant)
    Mt = list(a.shape)[0] // TILE
    Nt = list(b.shape)[1] // TILE
    c = ttnn.allocate_tensor_on_device(
        ttnn.Shape([Mt * TILE, Nt * TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        a.device(),
        create_sharded_memory_config(Mt, Nt),
    )
    descriptor = create_program_descriptor(a, b, c, variant=variant, kernel_iters=kernel_iters)
    return ttnn.generic_op([a, b, c], descriptor)
