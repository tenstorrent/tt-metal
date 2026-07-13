# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only benchmark: how BIG a block each compute helper processes per call.

The op is `out = (A + B) @ C`, built from five compute-helper phases that are all parallel over
the M (row) dimension:

  1. tilize A   (row-major A -> tiled)
  2. tilize B   (row-major B -> tiled)
  3. add        (A_tiled + B_tiled -> interm)
  4. matmul     (interm @ C -> mm_out)
  5. untilize   (mm_out -> row-major out)

Because every phase is row-parallel, we are free to chop M into `num_blocks` chunks and run the
WHOLE five-phase chain on one chunk at a time. The total work is identical no matter how we chop;
only the *granularity* changes:

  block_rows = M_tiles           (num_blocks = 1)          -> one go, whole M in a single pass
  block_rows = 2 or 4            (num_blocks = M_tiles/br)  -> in-between
  block_rows = 1                 (num_blocks = M_tiles)     -> tile-row by tile-row (degenerate)

Everything else is fixed (shapes, dtype bf16, math fidelity, subblock shape). The measured delta
is therefore pure compute: each helper call carries fixed per-call overhead — the phase-boundary
data-format reconfig + the LLK init/uninit + the unpack/math/pack pipeline fill and drain. Bigger
blocks amortize that overhead over more tiles; a tile-row-by-tile-row loop pays it M_tiles times.

Everything lives in sharded L1 on one Tensix core — there is no DRAM movement, so nothing but the
on-core compute pipeline is being timed.
"""

import ttnn

TILE = 32
TILE_SIZE = ttnn.tile_size(ttnn.bfloat16)  # 2048 bytes

# CB assignment.
CB_A_RM = 0  # A, row-major (tensor-backed)
CB_B_RM = 1  # B, row-major (tensor-backed)
CB_C = 2  # C weights, tiled (tensor-backed)
CB_A_TILED = 3  # scratch: A tilized
CB_B_TILED = 4  # scratch: B tilized
CB_INTERM = 5  # scratch: A + B
CB_MM_OUT = 6  # scratch: matmul result (tiled)
CB_OUT_RM = 16  # out, row-major (tensor-backed)

# variant name -> semantic block-row policy. block_rows is derived from M_tiles at build time.
#   per_tile_row : block_rows = 1        (num_blocks = M_tiles) — the naive baseline
#   block2       : block_rows = 2
#   block4       : block_rows = 4
#   one_block    : block_rows = M_tiles  (num_blocks = 1)       — everything in one go
VARIANTS = ("per_tile_row", "block2", "block4", "one_block")
BASELINE = "per_tile_row"

_FIXED_BLOCK_ROWS = {"per_tile_row": 1, "block2": 2, "block4": 4}


def block_rows_for(variant, m_tiles):
    """Tile-rows processed per five-phase pass for `variant` at a given M height (in tiles)."""
    if variant == "one_block":
        return m_tiles
    if variant in _FIXED_BLOCK_ROWS:
        return _FIXED_BLOCK_ROWS[variant]
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def variant_is_valid(variant, m_tiles):
    """A variant is runnable only when its block height evenly divides M (in tiles)."""
    br = block_rows_for(variant, m_tiles)
    return 1 <= br <= m_tiles and m_tiles % br == 0


# =============================================================================
# Compute kernel — the SAME source for every variant; block_rows / num_blocks are compile-time
# args, so the only thing that changes between variants is how many tiles each helper call chews.
# CT args: [block_rows, num_blocks, Kt, Nt, out_subblock_w, in1_num_subblocks, kernel_iters]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

// out = (A + B) @ C, all phases parallel over M. The M height is chopped into `num_blocks`
// chunks of `block_rows` tile-rows each; the whole tilize/tilize/add/matmul/untilize chain runs
// on one chunk per outer iteration.
void kernel_main() {
    constexpr uint32_t cb_a_rm = 0, cb_b_rm = 1, cb_c = 2;
    constexpr uint32_t cb_a_tiled = 3, cb_b_tiled = 4, cb_interm = 5, cb_mm_out = 6;
    constexpr uint32_t cb_out_rm = 16;

    constexpr uint32_t block_rows = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(6);
    constexpr uint32_t reconfig_on = get_compile_time_arg_val(7);

    constexpr uint32_t Mt = block_rows * num_blocks;
    constexpr uint32_t block_ab_tiles = block_rows * Kt;   // A/B/interm tiles per pass
    constexpr uint32_t block_out_tiles = block_rows * Nt;  // matmul-out tiles per pass

    using namespace compute_kernel_lib;

    // Per-phase data-format reconfig selection. Every CB here is bf16, so the format never changes
    // through the op — the reconfig each helper issues at its phase boundary is then pure wasted
    // MMIO. reconfig_on == 0 turns it OFF (the inits stay: each phase is still a different op).
    constexpr auto tilize_rc = reconfig_on
                                   ? tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                                   : tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure;
    constexpr auto untilize_rc = reconfig_on
                                     ? untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                                     : untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure;
    constexpr auto add_rc = reconfig_on ? BinaryDataFormatReconfig::Input : BinaryDataFormatReconfig::None;
    constexpr auto add_pack_rc = reconfig_on ? PackTileReconfig::Output : PackTileReconfig::None;
    constexpr auto mm_rc =
        reconfig_on ? matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT : matmul_config::DataFormatReconfig::NONE;

    // One hw_configure at boot (unsafe mid-kernel). The matmul boot init is NOT needed here — the
    // matmul_block helper (InitMode::Short) issues its own matmul init before every call. Every
    // other helper likewise issues its own per-phase init, so interleaving them per block is safe.
    compute_kernel_hw_startup(cb_a_rm, cb_a_tiled);

    // C weights are resident and reused by every matmul — mark them available once and let the
    // matmul retain (never pop) them.
    cb_reserve_back(cb_c, Kt * Nt);
    cb_push_back(cb_c, Kt * Nt);

    CircularBuffer in0(cb_interm), in1(cb_c), mm_out(cb_mm_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // A and B are tilize-consumed (popped), so re-mark the resident shards each iteration.
        cb_reserve_back(cb_a_rm, Mt * Kt);
        cb_push_back(cb_a_rm, Mt * Kt);
        cb_reserve_back(cb_b_rm, Mt * Kt);
        cb_push_back(cb_b_rm, Mt * Kt);

        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            // 1 + 2. tilize A and B for this chunk (row-major -> tiled).
            tilize<Kt, cb_a_rm, cb_a_tiled, tilize_config::InitUninitMode::InitAndUninit,
                   tilize_config::WaitMode::WaitBlock, tilize_rc>(block_rows);
            tilize<Kt, cb_b_rm, cb_b_tiled, tilize_config::InitUninitMode::InitAndUninit,
                   tilize_config::WaitMode::WaitBlock, tilize_rc>(block_rows);

            // 3. interm = A + B (FPU eltwise add over the chunk's tiles).
            add<cb_a_tiled, cb_b_tiled, cb_interm, BroadcastDim::None, InputLifecycle::Streaming,
                InputLifecycle::Streaming, OutputLifecycle::Streaming, add_rc, add_pack_rc>(
                EltwiseShape::tiles(block_ab_tiles));

            // 4. mm_out = interm @ C. num_k_blocks == 1 (K fits in one block); retain C on the
            //    last (only) K-block so the next chunk's matmul reuses the same weights.
            matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/false,
                LastBlockTarget::Out,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndRetainOnLastBlock,
                NoPostCompute, NoPreKBlock, NoPostKBlock,
                /*untilize_block_ct_dim=*/0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset,
                /*caller_owns_pack_target=*/false, NoneActivation, mm_rc>(
                in0, in1, mm_out, mm_out,
                MatmulBlockShape::of(block_rows, in1_num_subblocks, /*out_subblock_h=*/1,
                                     out_subblock_w, /*in0_block_k=*/Kt, /*num_k_blocks=*/1));

            // 5. untilize mm_out -> row-major out.
            untilize<Nt, cb_mm_out, cb_out_rm, untilize_config::InitUninitMode::InitAndUninit,
                     untilize_config::WaitMode::WaitBlock, untilize_rc>(block_rows);
        }

        // Drain the output between steady-state iterations; leave the last pass resident for readback.
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out_rm, Mt * Nt);
            cb_pop_front(cb_out_rm, Mt * Nt);
        }
    }
}
"""


# =============================================================================
# Host-side sharded-L1 layout + program descriptor
# =============================================================================


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(shape):
    """Whole `shape` as a single-core height shard (row-major orientation)."""
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _tile_paged_backed_cb(cb_id, tensor):
    """Alias a sharded tensor's L1 to a CB with TILE-sized pages.

    `cb_descriptor_from_sharded_tensor` inherits the tensor's page size — tile-sized for a TILE
    tensor, but one-row-per-page for a ROW_MAJOR tensor. The tilize/untilize helpers account in
    whole tiles, so we override the page size to a tile: the row-major bytes then sit in the same
    L1 exactly as a tile-paged buffer (this is the same aliasing a sharded tilize factory does),
    keeping the whole chain zero-copy while the helpers see clean tile pages.
    """
    cb = ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor)
    fds = cb.format_descriptors
    fds[0].page_size = TILE_SIZE
    cb.format_descriptors = fds
    return cb


def _scratch_cb(cb_id, num_tiles):
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat16, page_size=TILE_SIZE)
    return ttnn.CBDescriptor(total_size=TILE_SIZE * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])


# Matmul output subblock: out_subblock_h == 1, so out_subblock_w is bounded by DEST capacity.
# We fix fp32_dest_acc_en=True at half-sync -> DEST holds 4 fp32 tiles (DEST_AUTO_LIMIT).
_DEST_LIMIT = 4


def _matmul_subblock_w(n_tiles):
    """Largest divisor of Nt that fits the fp32 half-sync DEST (out_subblock_h == 1 -> w <= 4)."""
    for w in range(min(n_tiles, _DEST_LIMIT), 0, -1):
        if n_tiles % w == 0:
            return w
    return 1


def create_program_descriptor(input_tensors, output_tensor, *, block_rows, kernel_iters=1, reconfig=True):
    if len(input_tensors) != 3:
        raise ValueError("compute_block_size needs 3 input tensors: [A, B, C]")
    a, b, c = input_tensors
    for t in (a, b):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("A and B must be bfloat16 ROW_MAJOR_LAYOUT")
    if c.dtype != ttnn.bfloat16 or c.layout != ttnn.TILE_LAYOUT:
        raise ValueError("C must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.bfloat16 or output_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("output must be bfloat16 ROW_MAJOR_LAYOUT")

    m, k = a.shape[-2], a.shape[-1]
    kc, n = c.shape[-2], c.shape[-1]
    if k != kc:
        raise ValueError(f"inner dims disagree: A K={k}, C K={kc}")
    m_tiles, k_tiles, n_tiles = m // TILE, k // TILE, n // TILE
    if m % TILE or k % TILE or n % TILE:
        raise ValueError("M, K, N must all be tile-aligned (multiples of 32)")
    if block_rows < 1 or m_tiles % block_rows:
        raise ValueError(f"block_rows={block_rows} must divide M_tiles={m_tiles}")
    if kernel_iters < 1:
        raise ValueError("kernel_iters must be positive")

    num_blocks = m_tiles // block_rows
    out_subblock_w = _matmul_subblock_w(n_tiles)
    in1_num_subblocks = n_tiles // out_subblock_w

    compile_time_args = [
        block_rows,
        num_blocks,
        k_tiles,
        n_tiles,
        out_subblock_w,
        in1_num_subblocks,
        kernel_iters,
        int(reconfig),
    ]

    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        # bf16 inputs with K-tiles > 1 accumulate best in fp32 DEST at HiFi2 (fixed across variants).
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True),
    )

    cbs = [
        _tile_paged_backed_cb(CB_A_RM, a),
        _tile_paged_backed_cb(CB_B_RM, b),
        ttnn.cb_descriptor_from_sharded_tensor(CB_C, c),
        _scratch_cb(CB_A_TILED, block_rows * k_tiles),
        _scratch_cb(CB_B_TILED, block_rows * k_tiles),
        _scratch_cb(CB_INTERM, block_rows * k_tiles),
        _scratch_cb(CB_MM_OUT, block_rows * n_tiles),
        _tile_paged_backed_cb(CB_OUT_RM, output_tensor),
    ]

    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_op(input_tensors, *, block_rows, kernel_iters=1, reconfig=True):
    """Allocate the row-major sharded output and run one variant."""
    a, c = input_tensors[0], input_tensors[2]
    m = a.shape[-2]
    n = c.shape[-1]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m, n]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        a.device(),
        create_sharded_memory_config((m, n)),
    )
    descriptor = create_program_descriptor(
        input_tensors, output, block_rows=block_rows, kernel_iters=kernel_iters, reconfig=reconfig
    )
    return ttnn.generic_op([*input_tensors, output], descriptor)
