// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::experimental::prim {

namespace {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> determine_default_block_sizes(
    uint32_t M, uint32_t K, uint32_t N, bool fp32_dest_acc_en) {
    (void)K;  // K not used for determining defaults currently
    uint32_t M_block_tiles = 8;
    uint32_t K_block_tiles = 8;
    uint32_t N_block_tiles = 8;

    // 8-tile subblocks (2x4 / 4x2) halve the per-subblock acquire/commit/pack overhead vs 2x2, but
    // they only FIT in half-sync DST for bf16 acc (half-sync DST = 8 tiles). With fp32_dest_acc the
    // half-sync DST is only 4 tiles, so an 8-tile subblock OVERFLOWS and silently corrupts the output
    // (verified PCC 0.28-0.79); keep fp32 at 2x2 (<=4 tiles). Orient the longer subblock dim along
    // the larger output dim for the bf16 win.
    uint32_t subblock_h = 2;
    uint32_t subblock_w = 2;
    if (!fp32_dest_acc_en) {
        if (N >= M) {
            subblock_h = 2;
            subblock_w = 4;
        } else {
            subblock_h = 4;
            subblock_w = 2;
        }
    }

    return {M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w};
}

// Build a linear order of cores along one axis for data movement, plus index of the current core
std::pair<std::vector<CoreCoord>, uint32_t> build_core_order_for_axis(
    const CoreCoord& core,
    bool transpose_core_grid,
    uint32_t axis_length,
    tt::tt_metal::NOC noc,
    bool axis_is_x_when_not_transposed,
    const CoreCoord& initial_endpoint,
    uint32_t axis_base = 0) {
    // The forwarding chain covers axis positions [axis_base, axis_base + axis_length). axis_base != 0
    // is used by core-grid slicing to bound the big-input (down-rows) chain to a single row-group;
    // initial_endpoint must sit at axis_base. At axis_base == 0 this is the original full-axis chain.
    std::vector<CoreCoord> order;
    order.reserve(axis_length);
    order.push_back(initial_endpoint);

    // Determine which coordinate of the current core defines its position along this axis
    const size_t current_axis_value = transpose_core_grid ? (axis_is_x_when_not_transposed ? core.y : core.x)
                                                          : (axis_is_x_when_not_transposed ? core.x : core.y);

    // Direction along the axis: increasing for NOC_0, decreasing for NOC_1
    const bool increasing = (noc == tt::tt_metal::NOC::NOC_0);

    uint32_t index_of_current = 0;  // default to 0 if axis_length == 1
    for (uint32_t worker_idx = 1; worker_idx < axis_length; ++worker_idx) {
        CoreCoord worker_core = core;
        size_t& coord_to_modify = transpose_core_grid ? (axis_is_x_when_not_transposed ? worker_core.y : worker_core.x)
                                                      : (axis_is_x_when_not_transposed ? worker_core.x : worker_core.y);

        coord_to_modify = axis_base + (increasing ? worker_idx : (axis_length - worker_idx));
        if (coord_to_modify == current_axis_value) {
            index_of_current = worker_idx;
        }
        order.push_back(worker_core);
    }
    return {order, index_of_current};
}

CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index) {
    return order.at(index == 0 ? 0 : index - 1);
}

CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index) {
    const uint32_t last = static_cast<uint32_t>(order.size() - 1);
    return order.at(index >= last ? last : index + 1);
}

// Append tensor accessors in a consistent order
void append_accessors(
    std::vector<uint32_t>& args,
    const Tensor& main_tensor,
    const std::vector<Tensor>& output_tensors,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<const Tensor>& ag_input_tensor = std::nullopt,
    const std::optional<const Tensor>& ternary_a_tensor = std::nullopt,
    const std::optional<const Tensor>& ternary_b_tensor = std::nullopt) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    for (const auto& output_tensor : output_tensors) {
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
    }
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(args);
    }
    // AG input must come before ternary to match kernel accessor order
    if (ag_input_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ag_input_tensor.value().buffer()).append_to(args);
    }
    if (ternary_a_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_a_tensor.value().buffer()).append_to(args);
    }
    if (ternary_b_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_b_tensor.value().buffer()).append_to(args);
    }
}

}  // namespace

// SHARED IMPLEMENTATION - works with vector of output tensors (exposed for minimal_matmul_split)
MinimalMatmulProgramFactory::shared_variables_t minimal_matmul_factory_helper_common(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const std::vector<Tensor>& output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler,
    uint32_t N_chunks,
    std::optional<float> fused_ternary_scalar,
    const std::optional<const Tensor>& fused_ternary_input_a,
    const std::optional<const Tensor>& fused_ternary_input_b,
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler) {
    (void)fused_ternary_scalar;  // Scalar not needed in dataflow kernel, only in compute kernel
    auto* device = input_tensor.device();

    bool fuse_op = fused_op_signaler.has_value();

    if (!config.has_value()) {
        log_debug(tt::LogOp, "No config provided, using default block sizes and core grid");
    }

    auto grid_size =
        config.has_value() ? config.value().compute_with_storage_grid_size : device->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    bool use_bias = bias_tensor.has_value();
    bool use_fused_ternary = fused_ternary_input_a.has_value() && fused_ternary_input_b.has_value();

    /**
     * Determine dataformats, compute kernel config
     */
    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensors[0].dtype());
    auto out_tile_size = tt::tile_size(output_data_format);

    auto in2_data_format =
        use_bias ? tt::tt_metal::datatype_to_dataformat_converter(bias_tensor.value().dtype()) : in1_data_format;
    auto in2_tile_size = tt::tile_size(in2_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // Intermediate CB dataformat is the same datatype as DST register.
    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    /**
     * in0: M_tiles x K_tiles
     * in0 is divided into blocks, which are M_block_tiles x K_block_tiles
     *
     * in1: K_tiles x N_tiles
     * in1 is divided into blocks, which are K_block_tiles x N_block_tiles
     *
     * output: M_tiles x N_tiles
     * output is divided into blocks, which are M_block_tiles x N_block_tiles
     *
     * Blocks are further subdivided into subblocks. The output block is subdivided into subblock_h x subblock_w
     * subblocks. The in0 and in1 blocks are accordingly subdivided on M and N.
     */

    auto in0_tensor_shape = input_tensor.padded_shape();
    auto in1_tensor_shape = weight_tensor.padded_shape();
    // Fold activation (LHS) upper dimensions into rows: M_total = prod(upper dims) * M
    uint32_t K = in0_tensor_shape[-1];
    uint32_t M = input_tensor.physical_volume() / K;
    uint32_t N = in1_tensor_shape[-1];

    uint32_t M_tiles = M / tt::constants::TILE_HEIGHT;
    uint32_t K_tiles = K / tt::constants::TILE_WIDTH;
    uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    // Compute N_tiles_per_chunk for splitting
    const uint32_t N_tiles_per_chunk = N_tiles / N_chunks;

    auto [default_M_block_tiles, default_K_block_tiles, default_N_block_tiles, default_subblock_h, default_subblock_w] =
        determine_default_block_sizes(M, K, N, fp32_dest_acc_en);

    /**
     * TODO: Pick optimal subblock sizes. Currently a simple default is used.
     */
    uint32_t subblock_h = config.has_value() ? config.value().subblock_h : default_subblock_h;
    uint32_t subblock_w = config.has_value() ? config.value().subblock_w : default_subblock_w;

    uint32_t M_block_tiles = config.has_value() ? config.value().M_block_size : default_M_block_tiles;
    uint32_t K_block_tiles = config.has_value() ? config.value().K_block_size : default_K_block_tiles;
    uint32_t N_block_tiles = config.has_value() ? config.value().N_block_size : default_N_block_tiles;

    /**
     * We originally saw that for non-square outputs, N > M was significantly faster than M > N.
     * This is because originally, the in0 DM kernel was responsible for reading in0 and writing output.
     * When M > N, the in0 DM kernel has more data to read on top of its responsibility to write output.
     *
     * An optimization is to have the DM kernel with less data to read handle writes, and transpose the core_grid
     * to keep NOC usage consistent. With this optimization, N > M performance is symmetric with M > N.
     *
     * The smaller input read and mcast is always across a row of cores (x, y): (0, core_y) -> (grid_size.x-1, core_y)
     * The larger input read and mcast is always across a column of cores (x, y): (core_x, 0) -> (core_x. grid_size.y-1)
     *
     * Output is always written by DM reading the smaller input.
     *
     * Small input + output DM always runs on RISCV_1, NOC_1
     * Large input DM always runs on RISCV_0, NOC_0
     */

    // Blackhole's larger grid (e.g. 11x10 = 110 cores vs WH 8x8 = 64) gives much smaller per-core M/N
    // tile counts, which flips the block-sizing optimum from "reuse" (big blocks, K=8, fewest blocks —
    // best on WH) to "pipelining" (even-dividing smaller blocks + finer K). The block sizer and K default
    // below branch on this; see minimal_matmul_blackhole_heuristic_analysis.md.
    const bool is_blackhole = device->arch() == tt::ARCH::BLACKHOLE;

    auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;

    // Transpose core grid if the output is wide (M > N)
    // If transpose core grid, we parallelize M on cores_x and N on cores_y and swap the NOCs and RISCVs
    // When fusing with strided reduce scatter, transposing is disabled
    // because it resulted in slightly lower performance on a case of interest.
    // (This can be revisited if needed.)
    const bool fuse_srs = srs_fused_op_signaler.has_value();
    bool transpose_core_grid = M > N && !fuse_srs;

    // Core-grid SLICING (see minimal-matmul-nslicing-plan): split the physical rows (grid.y) into
    // `num_slices` groups, each an independent sub-matmul on the FULL smaller output dim x a slice of
    // the bigger dim, to fill cores that idle on skewed shapes. Canonical frame: the smaller dim is
    // always parallelized over the rows (grid.y) and the bigger over the cols (grid.x) — transpose
    // canonicalizes this — so slicing always groups grid.y and slices grid.x, regardless of transpose.
    // Effective parallelism: small dim over (grid.y / S) rows-per-group; big dim over (S * grid.x).
    // S=1 reduces exactly to the un-sliced partition. (Step 1: partition only; S>1 also needs the
    // per-group forwarding change before it is correct. Auto-derivation of S is deferred to step 5.)
    uint32_t num_slices = 1;
    // Auto-derive S from the M:N aspect ratio. Engage on the same delivery-bound condition as the
    // prefetch gate — min(M,N) tiles-per-core <= 2 at S=1, i.e. min(M_tiles,N_tiles) <= 2*grid.y — and
    // pick S = nearest power-of-two to sqrt(max/min) so each sub-grid is roughly square. Only when the
    // caller didn't pin a config and there are no fused ops (matches the gate). Measured: this lifts
    // skinny shapes by up to ~2.3x (see minimal-matmul-nslicing-plan); near-square shapes get S=1.
    if (!config.has_value() && !fuse_op && !fuse_srs) {
        const uint32_t mn = std::min(M_tiles, N_tiles);
        const uint32_t mx = std::max(M_tiles, N_tiles);
        const uint32_t min_tpc_s1 = (mn + grid_size.y - 1) / grid_size.y;  // ceil(min_tiles / grid.y), S=1
        if (mn > 0 && min_tpc_s1 <= 2) {
            const double r = std::sqrt(static_cast<double>(mx) / static_cast<double>(mn));
            uint32_t best = 1;
            double best_dist = std::abs(1.0 - r);
            // Only consider slice counts that are a power of 2 AND divide grid.y (the TT_FATAL below
            // requires both). On a power-of-2 grid (WH 8x8) every power-of-2 c<=grid.y divides it, but on
            // a non-power-of-2 grid (Blackhole grid.y=10) candidates like 4/8 don't divide grid.y and
            // would trip the assert -> only 1,2 are legal. Skip the indivisible ones here.
            for (uint32_t c = 2; c <= grid_size.y; c *= 2) {
                if (grid_size.y % c != 0) {
                    continue;
                }
                const double d = std::abs(static_cast<double>(c) - r);
                if (d < best_dist) {
                    best_dist = d;
                    best = c;
                }
            }
            num_slices = best;
        }
    }
    // Env override (also enables forcing S for fused-off experiments / sweeps).
    if (const char* s = std::getenv("TT_MM_NUM_SLICES"); s != nullptr && !fuse_op && !fuse_srs) {
        num_slices = std::max(1u, static_cast<uint32_t>(std::atoi(s)));
    }
    TT_FATAL(
        grid_size.y % num_slices == 0 && (num_slices & (num_slices - 1)) == 0,
        "TT_MM_NUM_SLICES ({}) must be a power of 2 dividing grid.y ({})",
        num_slices,
        grid_size.y);

    // Core-grid K-PARALLELISM (split-K): split the physical rows into `num_k_slices` (Pk) OUTER bands,
    // each computing the FULL M/N output over a 1/Pk slice of the K reduction. Two ways to engage:
    //   * EXPLICIT (env): TT_MM_K_SLICES=Pk [+ TT_MM_K_FUSED=1 -> plan B fused [M,N]; else plan A2
    //     host-summed [Pk*M,N]]. Full manual control, sweepable alongside TT_MM_NUM_SLICES.
    //   * AUTO (default): when neither TT_MM_NUM_SLICES nor TT_MM_K_SLICES is pinned, a heuristic jointly
    //     picks (num_slices S, num_k_slices Pk) spending the grid.y row budget S*Pk=grid.y between
    //     N/M-slicing and K-reduction. Auto K-par ALWAYS uses the fused reduction (plan B -> single
    //     [M,N]), so the output shape is identical to the no-K-par path and compute_output_specs needs no
    //     change. S*Pk=grid.y => rows_per_group=1 => no M-padding. Validated on 28 "FLUX" shapes: geomean
    //     1.30x, 0 regressions, 96.4% of oracle-best (see minimal_matmul_nslice_kpar_flux.md).
    const bool num_slices_pinned = std::getenv("TT_MM_NUM_SLICES") != nullptr;
    const bool k_slices_pinned = std::getenv("TT_MM_K_SLICES") != nullptr;
    const bool kpar_auto_ok = !config.has_value() && !fuse_op && !fuse_srs;

    uint32_t num_k_slices = 1;
    bool num_k_fused = false;
    if (k_slices_pinned && kpar_auto_ok) {
        num_k_slices = std::max(1u, static_cast<uint32_t>(std::atoi(std::getenv("TT_MM_K_SLICES"))));
        const char* kf = std::getenv("TT_MM_K_FUSED");
        num_k_fused = (kf != nullptr && std::atoi(kf) != 0);
    } else if (kpar_auto_ok && !num_slices_pinned && std::getenv("TT_MM_NO_AUTO_KPAR") == nullptr) {
        // Joint (S, Pk) auto heuristic. Fires only where the N-slicer already engages (num_slices>1) —
        // the delivery-bound/skewed regime the heuristic was fit on. K-dominance score
        //   D = K_tiles * num_cores / out_tiles
        // measures deep reduction vs how output-saturated the grid already is; bigger D => more K-bands.
        // Tunable thresholds (override via env, re-fit per architecture).
        struct KParParams {
            uint32_t d8, d4, d2;  // D thresholds selecting Pk = 8 / 4 / 2 (else 1)
            uint32_t nwide;       // N_tiles >= this -> in1 DRAM-bandwidth bound, disable K-par
            uint32_t min_kt;      // keep K_tiles/Pk >= this (deep-K availability)
        };
        // WH 8x8 back-tested defaults. To make these architecture-specific, branch on device->arch()
        // here and assign a re-fit KParParams (e.g. BH grid.y=10 changes the row budget and core count);
        // until then every arch reuses these and can be tuned live via the TT_MM_KPAR_* env vars below.
        KParParams kp{280u, 40u, 20u, 256u, 8u};
        auto envov = [](const char* n, uint32_t& f) {
            if (const char* s = std::getenv(n)) {
                f = static_cast<uint32_t>(std::atoi(s));
            }
        };
        envov("TT_MM_KPAR_D8", kp.d8);
        envov("TT_MM_KPAR_D4", kp.d4);
        envov("TT_MM_KPAR_D2", kp.d2);
        envov("TT_MM_KPAR_NWIDE", kp.nwide);
        envov("TT_MM_KPAR_MINKB", kp.min_kt);

        // Safety gate: the budget split sets num_slices = grid.y / Pk, which is only guaranteed to be a
        // valid power-of-2 slice count when grid.y is a power of 2 (true on WH 8x8). On a non-pow2 grid
        // (e.g. BH grid.y=10) Pk=2 would give num_slices=5 — untested — so auto K-par stays OFF there
        // until the heuristic is re-fit and validated for that grid. Explicit TT_MM_K_SLICES still works.
        const bool grid_y_pow2 = (grid_size.y & (grid_size.y - 1)) == 0;
        if (num_slices > 1 && grid_y_pow2) {
            const uint32_t cores = grid_size.x * grid_size.y;
            const uint32_t out_tiles = M_tiles * N_tiles;
            const double D = out_tiles ? static_cast<double>(K_tiles) * cores / out_tiles : 0.0;
            uint32_t Pk = D >= kp.d8 ? 8u : D >= kp.d4 ? 4u : D >= kp.d2 ? 2u : 1u;
            if (N_tiles >= kp.nwide) {
                Pk = 1u;  // wide-N DRAM guard
            }
            while (Pk > 1 && grid_size.y % Pk != 0) {
                Pk /= 2;  // fit the row budget
            }
            while (Pk > 1 && (K_tiles % Pk != 0 || K_tiles / Pk < kp.min_kt)) {
                Pk /= 2;  // deep-K availability
            }
            if (Pk > 1) {
                num_slices = grid_size.y / Pk;  // rows_per_group = 1 (no M-padding); S*Pk = grid.y
                num_k_slices = Pk;
                num_k_fused = true;
                log_debug(
                    tt::LogOp,
                    "minimal_matmul auto K-par: D={:.0f} -> num_slices={} num_k_slices={} (fused)",
                    D,
                    num_slices,
                    num_k_slices);
            }
        }
    }
    TT_FATAL(
        grid_size.y % (num_slices * num_k_slices) == 0 && (num_k_slices & (num_k_slices - 1)) == 0,
        "num_k_slices ({}): must be a power of 2 and num_slices*num_k_slices ({}) must divide grid.y ({})",
        num_k_slices,
        num_slices * num_k_slices,
        grid_size.y);
    TT_FATAL(K_tiles % num_k_slices == 0, "num_k_slices ({}) must divide K_tiles ({})", num_k_slices, K_tiles);
    // Plan B (fused on-device column reduction) needs >1 K-band; A2 (host-summed) is the explicit-env
    // alternative. The reduction reuses the store-and-forward semaphore pattern, so it stays on the
    // plain unicast path (incompatible with the mcast prototypes / N-chunk split outputs).
    TT_FATAL(
        !(num_k_fused && num_k_slices == 1),
        "fused split-K (plan B) requires num_k_slices > 1 (the fused reduction needs >1 K-band).");
    // Sub-grid K_block refinement: a sliced sub-grid has small per-core M/N, so the default K_block=8
    // makes the per-k-block work too coarse — finer K (4) pipelines read/forward/compute better.
    // Measured ~+8%/+6%/+3.6% on sliced 4864x4096x512 / 4864x4096x32 / 32x2048x2048. Only on the auto
    // path (no pinned config); the caller's K_block is respected when a config is given.
    if (num_slices > 1 && !config.has_value()) {
        K_block_tiles = std::min(K_block_tiles, 4u);
    }
    // Small-K: never let K_block exceed K_tiles — otherwise round_up pads K (e.g. K=128 -> 4 tiles
    // padded to K_block=8 = 2x wasted K work). Pure win; only reduces K_block when K is tiny.
    if (!config.has_value()) {
        K_block_tiles = std::min(K_block_tiles, std::max(1u, K_tiles));
    }
    // Rows are partitioned as num_k_slices (K-bands, outer) x num_slices (N-slice groups) x
    // rows_per_group (small-dim parallelism, innermost). K-bands do NOT add output parallelism (they
    // reduce K and are summed), so x_axis_parallel keeps only the N-slice factor.
    const uint32_t rows_per_group = grid_size.y / (num_slices * num_k_slices);  // small-dim parallelism (innermost)
    const uint32_t y_axis_parallel = rows_per_group;            // dim on grid.y is grouped
    const uint32_t x_axis_parallel = num_slices * grid_size.x;  // dim on grid.x is sliced (cols x slices)

    auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    uint32_t in0_parallel_axis_cores = transpose_core_grid ? x_axis_parallel : y_axis_parallel;

    auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    uint32_t in1_parallel_axis_cores = transpose_core_grid ? y_axis_parallel : x_axis_parallel;

    /**
     * We pad the input dimensions to the nearest multiple of the parallelization factor.
     *
     * Each core is assigned a certain number of tiles in M and N to compute.
     * Within a core, tiles are blocked by M_block_tiles and N_block_tiles.
     * Most output blocks are the full block size, but the last block in M or N can be partial.
     */
    uint32_t padded_M_tiles = tt::round_up(M_tiles, in0_parallel_axis_cores);
    uint32_t padded_N_tiles = tt::round_up(N_tiles, in1_parallel_axis_cores);
    // Split-K (A2) writes each band's full M x N partial into an M-stripe of the [num_k_slices * M, N]
    // output, stacked by the LOGICAL M tiles. That is only correct without M-padding (otherwise a band
    // writes padded_M_tiles rows and stripes would overlap). Require it for now; for the starved shapes
    // K-par targets, choose num_k_slices so grid.y/(num_slices*num_k_slices) divides M_tiles (e.g. M=1
    // tile with num_k_slices=grid.y). Padded-M support is a follow-up.
    TT_FATAL(
        num_k_slices == 1 || padded_M_tiles == M_tiles,
        "TT_MM_K_SLICES ({}): M is padded ({} -> {} tiles by {}-way M parallelism); split-K stacking "
        "requires no M-padding. Pick num_slices*num_k_slices so grid.y/(that) divides M_tiles.",
        num_k_slices,
        M_tiles,
        padded_M_tiles,
        in0_parallel_axis_cores);

    uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;
    uint32_t N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;

    // Blackhole finer-K refinement (Change 2). On BH the per-core M/N tiles are small, so the default
    // K_block=8 makes each per-k-block too coarse and (by doubling the in0/in1/intermediate L1 footprint)
    // pushes the block sizer over its L1 budget, forcing uneven M/N splits. Dropping to K=4 pipelines
    // better and frees L1 for an even block — the same reasoning as the sliced sub-grid K=4 refinement
    // above, generalized to the non-sliced BH path. Must run BEFORE padded_K_tiles so the K padding uses
    // the final K_block. (num_slices==1 here: the sliced path already clamped K to 4 above.)
    //
    // Gate = min per-core tiles >= 4 AND per-core output tiles <= 128. Mined+measured on the 82-shape BH
    // sweep:
    //  - min(per_core) <= 3 prefers K=8 (skinny, mcast/forwarding-bound: 1024x6144x768 +6%, 4096x6144x768
    //    +3.5% at K=8), so route those to K=8 (the >=4 floor).
    //  - The Mpc*Npc <= 128 cap is LOAD-BEARING, not cosmetic: although the baseline sweep's best K=4 block
    //    beats its best K=8 block on LARGE per-core shapes too, the AUTO chooser's K=4 blocking there is
    //    much worse than its K=8 blocking, so forcing K=4 on large per-core regressed shapes the branch was
    //    winning at K=8 (16384x2304x6144 1.04x->0.92x, 8192x6144x4608 1.07x->0.99x when the cap was
    //    removed). Keep K=4 to the small/mid per-core regime where the chooser tracks the optimum.
    // Must run BEFORE padded_K_tiles so the K padding uses the final K_block.
    if (is_blackhole && !config.has_value() && !fuse_op && !fuse_srs && num_slices == 1 &&
        std::min(M_tiles_per_core, N_tiles_per_core) >= 4 &&
        static_cast<uint64_t>(M_tiles_per_core) * N_tiles_per_core <= 128) {
        K_block_tiles = std::min(K_block_tiles, 4u);
    }

    // Per-K-band reduction depth: each split-K band reduces K_tiles/num_k_slices. This drives the kernels'
    // K-block COUNT (K_num_blocks) and the compute accumulation depth; in0/in1 STRIDING still uses the full
    // K_tiles (passed separately), and each band's absolute K-block start is a per-core runtime arg. Uses
    // the FINAL K_block_tiles (after the BH change-2 refinement above), so the K padding matches the chosen
    // block. On BH num_k_slices stays 1 (auto K-par gated off on non-pow2 grid.y until re-fit) -> == K_tiles.
    const uint32_t K_tiles_per_band = K_tiles / num_k_slices;
    uint32_t padded_K_tiles = tt::round_up(K_tiles_per_band, K_block_tiles);

    // Default subblock selection (only when the caller did not pin a config). Maximize DST utilization:
    // among power-of-2 (subblock_h, subblock_w) pairs where each dim divides its per-core tile count (so
    // it also divides the default-8 block and any block the sizer below picks) and the product fits the
    // HALF-SYNC DST depth (4 fp32 / 8 bf16 tiles), pick the largest area, tie-broken toward a balanced
    // (squarer) subblock. The kernel always runs half-sync (the factory never wires dst_full_sync_en to
    // ComputeConfig), so exceeding the DST depth silently corrupts the output — the product cap enforces
    // safety. The earlier heuristic capped one dim at 2, which UNDER-filled the DST when the other dim
    // was odd (e.g. M_pc=4, N_pc=9 -> 2x1 = 2 tiles instead of 4x1 = 4) and cost ~5% on those shapes; the
    // balanced tiebreak keeps the well-utilized cases unchanged (e.g. M_pc=N_pc=16 stays 2x2, not 1x4).
    if (!config.has_value()) {
        const uint32_t max_dst = fp32_dest_acc_en ? 4u : 8u;
        uint32_t best_h = 1, best_w = 1;
        for (uint32_t sh = 1; sh <= max_dst; sh <<= 1) {
            if (M_tiles_per_core == 0 || M_tiles_per_core % sh != 0) {
                continue;
            }
            for (uint32_t sw = 1; sh * sw <= max_dst; sw <<= 1) {
                if (N_tiles_per_core == 0 || N_tiles_per_core % sw != 0) {
                    continue;
                }
                const uint32_t area = sh * sw;
                const uint32_t best_area = best_h * best_w;
                if (area > best_area || (area == best_area && std::max(sh, sw) < std::max(best_h, best_w))) {
                    best_h = sh;
                    best_w = sw;
                }
            }
        }
        subblock_h = best_h;
        subblock_w = best_w;
    }

    // Auto-gate (productization): mcast+prefetch beats the unicast baseline when the skinny output
    // dimension is delivery-bound, i.e. has a small per-core tile count. Measured win for
    // min(M,N) tiles-per-core <= 2 (+1.6% to +30%, larger for smaller shapes); a loss above that. When
    // it fires we ALSO clamp the compile-time block to the per-core tiles — the win requires
    // M_block == M_tiles_per_core; leaving the default 8 makes the kernel reserve/pipeline 8x-too-large
    // CB blocks (~+50% slower even though runtime byte counts are clamped). Fires only when the caller
    // didn't pin a config and there are no fused ops (mcast prototype). Disable via TT_MM_NO_AUTO_PREFETCH;
    // TT_MM_MCAST_PREFETCH forces the dataflow on regardless (handled at the mcast-flags block below).
    const uint32_t min_tiles_per_core = std::min(M_tiles_per_core, N_tiles_per_core);
    const char* no_auto_prefetch = std::getenv("TT_MM_NO_AUTO_PREFETCH");
    // Gate is computed on the SUB-GRID per-core counts (M/N_tiles_per_core already reflect slicing),
    // so it composes with slicing: each sub-grid is squarer, min_tiles_per_core is the sub-grid value.
    const bool prefetch_gate = !config.has_value() && !fuse_op && !fuse_srs && min_tiles_per_core <= 2 &&
                               !(no_auto_prefetch != nullptr && std::string(no_auto_prefetch) != "0");
    if (prefetch_gate) {
        M_block_tiles = std::min(M_block_tiles, M_tiles_per_core);
        N_block_tiles = std::min(N_block_tiles, N_tiles_per_core);
    }

    // Default block sizer (no pinned config / no fused ops). The fixed default block (8) fragments
    // large per-core dims — e.g. N_tiles_per_core=18 split as 8+8+2 — which costs ~10-30% on large
    // shapes vs a per-core-sized block (measured: 1024x6144x4608 956->830us). Clamp each block to its
    // per-core count (never useful to exceed it), then re-pick N (the free dim, most fragmentation-
    // sensitive) and M via choose() below, bounded by an L1 circular-buffer budget. K is fixed
    // (K_block_tiles already = min(default, K_tiles)). Reproduces the swept-best blocks on the large
    // non-gated shapes (e.g. 4096x6144x4608 -> 4/8/18, matching the hand-tuned optimum); geomean over a
    // 65-shape sweep improved 1.25x -> 1.35x vs main, losses (vs swept-best) cut from 20 to 3.
    if (!config.has_value() && !fuse_op && !fuse_srs) {
        M_block_tiles = std::min(M_block_tiles, std::max(1u, M_tiles_per_core));
        N_block_tiles = std::min(N_block_tiles, std::max(1u, N_tiles_per_core));
        if (!prefetch_gate) {
            // CB footprint in bytes: in0/in1/out double-buffered, intermediate (fp32 partials) single.
            auto footprint = [&](uint32_t mb, uint32_t nb) -> uint64_t {
                return (uint64_t)mb * K_block_tiles * 2 * in0_tile_size +
                       (uint64_t)K_block_tiles * nb * 2 * in1_tile_size + (uint64_t)mb * nb * 2 * out_tile_size +
                       (uint64_t)mb * nb * intermediate_tile_size;
            };
            // ~1.25 MiB. The default 8/8/8 fp32 block uses 1.0 MiB; the largest block this admits
            // (e.g. 4/8/18 = 1.31 MiB) is proven safe on-device with headroom for kernels/args/sems.
            const uint64_t L1_CB_BUDGET = 1310720;
            // Pick the block that best amortizes the fixed default-8 fragmentation. CONSTRAINT (hard, for
            // correctness): the block must be a multiple of its subblock — the kernel requires
            // block % subblock == 0 (the host validator asserts this for explicit configs; the auto path
            // skips that check, so a non-multiple silently corrupts the output). Among multiples of the
            // subblock that are <= the per-core count and fit the L1 budget, choose the one with the
            // FEWEST blocks (ceil(per_core/block)), tie-broken toward an even split (block divides
            // per_core, avoiding a tiny tail like 24=10+10+4) and then a larger block. Removes the
            // default-8 fragmentation (e.g. N_pc=18 -> 10+8 instead of 8+8+2; M_pc=9 -> one block of 9
            // instead of 8+1) while staying within the subblock and L1 constraints.
            auto choose = [&](uint32_t per_core, uint32_t sb, uint32_t cur, auto fits) -> uint32_t {
                sb = std::max(1u, sb);
                // Blackhole even-split objective (Change 1), ONLY for the un-sliced path (num_slices==1).
                // Sliced sub-grids already had a well-tuned blocking that wins big; the new chooser
                // regressed them (576x6144x6144 1.45x->1.09x), so leave sliced shapes on the WH chooser.
                if (is_blackhole && num_slices == 1) {
                    // On BH the per-core counts are small and an uneven tail costs far more (measured +31%
                    // from one uneven dim) than the reuse gained by a big block. Search the FULL set of
                    // subblock-multiple blocks (NOT floored at the default 8) and rank by
                    //   (1) padding waste = ceil(per_core/b)*b - per_core   (kill the uneven tail first),
                    //   (2) |b - 8|                                         (target the reuse sweet-spot),
                    //   (3) fewest blocks.
                    // Waste is primary so a clean even split is never traded for an uneven one even when
                    // the subblock constraint blocks the ideal divisor (per_core=12, sb=4 -> 12 not 8=8+4).
                    // A floor of b>=3 plus the L1 `fits` cap keeps it prime-safe: when per_core is prime
                    // (47) the only waste-0 blocks are 1 and per_core; the floor kills b=1 and L1 usually
                    // excludes per_core, so it settles on the min-waste block near 8 (47 -> 8). Where
                    // per_core % 8 == 0, b=8 is waste-0 and dist-0 -> returns 8, matching the WH path.
                    const uint32_t lo = std::max(sb, 3u);
                    uint32_t best = 0, best_waste = 0, best_dist = 0, best_blocks = 0;
                    for (uint32_t b = (lo + sb - 1) / sb * sb; b <= per_core; b += sb) {
                        if (!fits(b)) {
                            continue;
                        }
                        const uint32_t blocks = (per_core + b - 1) / b;
                        const uint32_t waste = blocks * b - per_core;
                        const uint32_t dist = b > 8 ? b - 8 : 8 - b;
                        if (best == 0 || waste < best_waste || (waste == best_waste && dist < best_dist) ||
                            (waste == best_waste && dist == best_dist && blocks < best_blocks)) {
                            best = b;
                            best_waste = waste;
                            best_dist = dist;
                            best_blocks = blocks;
                        }
                    }
                    return best == 0 ? std::min(per_core, sb) : best;
                }
                // Wormhole path (unchanged). If the default block already tiles the per-core count evenly,
                // leave it: there is no partial tail to remove, and merging into fewer-but-larger blocks
                // only inflates the L1 footprint without a throughput gain (measured: M_pc=64 grown
                // 8->16 regressed ~12%).
                if (per_core % cur == 0) {
                    return cur;
                }
                uint32_t best = cur;
                uint32_t best_blocks = (per_core + cur - 1) / cur;
                for (uint32_t b = (per_core / sb) * sb; b > cur; b -= sb) {
                    if (!fits(b)) {
                        continue;
                    }
                    uint32_t blocks = (per_core + b - 1) / b;
                    bool even = (per_core % b) == 0;
                    bool best_even = (per_core % best) == 0;
                    if (blocks < best_blocks || (blocks == best_blocks && even && !best_even) ||
                        (blocks == best_blocks && even == best_even && b > best)) {
                        best = b;
                        best_blocks = blocks;
                    }
                }
                return best;
            };
            const uint32_t sbw = std::max(1u, subblock_w);
            const uint32_t sbh = std::max(1u, subblock_h);
            N_block_tiles = choose(N_tiles_per_core, sbw, N_block_tiles, [&](uint32_t nb) {
                return footprint(M_block_tiles, nb) <= L1_CB_BUDGET;
            });
            M_block_tiles = choose(M_tiles_per_core, sbh, M_block_tiles, [&](uint32_t mb) {
                return footprint(mb, N_block_tiles) <= L1_CB_BUDGET;
            });
        }
    }

    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);
    uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    log_debug(tt::LogOp, "M_tiles_per_core: {}", M_tiles_per_core);
    log_debug(tt::LogOp, "N_tiles_per_core: {}", N_tiles_per_core);
    log_debug(tt::LogOp, "M_blocks_per_core: {}", M_blocks_per_core);
    log_debug(tt::LogOp, "N_blocks_per_core: {}", N_blocks_per_core);

    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    uint32_t in2_block_num_tiles = N_block_tiles;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    // TODO: consider not double buffering the output
    uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered
    uint32_t in2_cb_num_tiles = in2_block_num_tiles;     // not double buffered

    auto core_0_0 = CoreCoord{0, 0};
    auto core_0_1 = CoreCoord{0, 1};
    auto core_1_0 = CoreCoord{1, 0};
    auto core_endx_0 = CoreCoord{grid_size.x - 1, 0};
    auto core_0_endy = CoreCoord{0, grid_size.y - 1};
    auto core_endx_endy = CoreCoord{grid_size.x - 1, grid_size.y - 1};

    auto in0_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_endx_0 : core_0_endy);
    auto in0_receiver_cores = CoreRange(transpose_core_grid ? core_0_1 : core_1_0, core_endx_endy);
    auto in1_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_0_endy : core_endx_0);
    auto in1_receiver_cores = CoreRange(transpose_core_grid ? core_1_0 : core_0_1, core_endx_endy);

    // Slicing: the BIG input (forwarded down rows) gets an injector at EACH group's top row, not just
    // row 0 — one independent forwarding chain per row-group. The SMALL input (across cols) is
    // unchanged (injector at col 0 of every row). At num_slices==1 the group-top set is just row 0,
    // so these specs match the original single-range sender/receiver cores.
    std::vector<CoreRange> rows_sender_ranges;
    for (uint32_t g = 0; g < num_slices * num_k_slices; g++) {
        std::size_t r = static_cast<std::size_t>(g * rows_per_group);
        rows_sender_ranges.push_back(CoreRange(CoreCoord{0, r}, CoreCoord{grid_size.x - 1, r}));
    }
    CoreRangeSet rows_sender_spec(rows_sender_ranges);                // big input: group-top rows x all cols
    CoreRangeSet cols_sender_spec(CoreRange(core_0_0, core_0_endy));  // small input: col 0 x all rows
    CoreRangeSet in0_sender_spec = transpose_core_grid ? rows_sender_spec : cols_sender_spec;
    CoreRangeSet in1_sender_spec = transpose_core_grid ? cols_sender_spec : rows_sender_spec;
    CoreRangeSet in0_receiver_spec = CoreRangeSet(core_grid).subtract(in0_sender_spec);
    CoreRangeSet in1_receiver_spec = CoreRangeSet(core_grid).subtract(in1_sender_spec);

    auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

    // Split-K plan B (fused L1 column reduction): two semaphores for the vertical running-sum handshake.
    //  - reduce_ready: lives on the band-BELOW (sender); the band-above increments it to signal "my
    //    cb_reduce slot is free, you may write" (mirror of in0/in1 sender_semaphore).
    //  - reduce_recv:  lives on the band-ABOVE (receiver); the band-below sets it VALID after writing the
    //    partial into the receiver's cb_reduce (mirror of in0/in1 receiver_semaphore). The VALID source
    //    reuses the writer's existing in{0,1}_valid_semaphore (a constant VALID cell).
    auto reduce_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto reduce_recv_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);

    uint32_t in0_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::create_cb(in0_cb_id, program, core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format);

    uint32_t in1_cb_id = tt::CBIndex::c_1;
    tt::tt_metal::create_cb(in1_cb_id, program, core_grid, in1_tile_size, in1_cb_num_tiles, in1_data_format);

    uint32_t out_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::create_cb(out_cb_id, program, core_grid, out_tile_size, out_cb_num_tiles, output_data_format);

    uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    tt::tt_metal::create_cb(
        intermediate_cb_id, program, core_grid, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    // Split-K plan B: cb_reduce (c_7) holds the running sum forwarded UP the column from the band below.
    // The band-below's output writer NoC-writes its out_cb (output_data_format) here, then compute adds it
    // to this band's own partial. Single block (1 slot) => its L1 write pointer is constant on every core,
    // so the remote writer can target it with its own local write pointer (no cross-core ptr tracking).
    if (num_k_fused) {
        uint32_t reduce_cb_id = tt::CBIndex::c_7;
        tt::tt_metal::create_cb(
            reduce_cb_id, program, core_grid, out_tile_size, out_block_num_tiles, output_data_format);
    }

    if (use_bias) {
        uint32_t in2_cb_id = tt::CBIndex::c_4;
        tt::tt_metal::create_cb(in2_cb_id, program, core_grid, in2_tile_size, in2_cb_num_tiles, in2_data_format);
    }

    // Create circular buffers for fused ternary inputs
    if (use_fused_ternary) {
        uint32_t ternary_a_cb_id = tt::CBIndex::c_5;
        uint32_t ternary_c_cb_id = tt::CBIndex::c_6;

        // Fused ternary input A - circular buffer c_5
        auto ternary_a_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_a.value().dtype());
        auto ternary_a_tile_size = tt::tile_size(ternary_a_data_format);

        TT_FATAL(ternary_a_tile_size == in1_tile_size, "ternary_a_tile_size must be equal to in1_tile_size");
        TT_FATAL(ternary_a_data_format == in1_data_format, "ternary_a_data_format must be equal to in1_data_format");
        uint32_t ternary_a_cb_num_tiles = out_block_num_tiles;  // Same as output block, not double buffered

        tt::tt_metal::create_cb(
            ternary_a_cb_id, program, core_grid, ternary_a_tile_size, ternary_a_cb_num_tiles, ternary_a_data_format);

        // Fused ternary input C - circular buffer c_6
        auto ternary_c_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_b.value().dtype());
        auto ternary_c_tile_size = tt::tile_size(ternary_c_data_format);
        uint32_t ternary_c_cb_num_tiles = N_block_tiles;  // Single row (like bias), broadcast across M

        tt::tt_metal::create_cb(
            ternary_c_cb_id, program, core_grid, ternary_c_tile_size, ternary_c_cb_num_tiles, ternary_c_data_format);

        log_debug(tt::LogOp, "ternary_a_cb_id: {}", ternary_a_cb_id);
        log_debug(tt::LogOp, "ternary_c_cb_id: {}", ternary_c_cb_id);
    }

    log_debug(tt::LogOp, "in0_cb_id: {}", in0_cb_id);
    log_debug(tt::LogOp, "in1_cb_id: {}", in1_cb_id);
    log_debug(tt::LogOp, "out_cb_id: {}", out_cb_id);
    log_debug(tt::LogOp, "intermediate_cb_id: {}", intermediate_cb_id);
    log_debug(tt::LogOp, "M_tiles: {}", M_tiles);
    log_debug(tt::LogOp, "padded_M_tiles: {}", padded_M_tiles);
    log_debug(tt::LogOp, "K_tiles: {}", K_tiles);
    log_debug(tt::LogOp, "padded_K_tiles: {}", padded_K_tiles);
    log_debug(tt::LogOp, "N_tiles: {}", N_tiles);
    log_debug(tt::LogOp, "padded_N_tiles: {}", padded_N_tiles);
    log_debug(tt::LogOp, "M_block_tiles: {}", M_block_tiles);
    log_debug(tt::LogOp, "K_block_tiles: {}", K_block_tiles);
    log_debug(tt::LogOp, "N_block_tiles: {}", N_block_tiles);
    log_debug(tt::LogOp, "subblock_h: {}", subblock_h);
    log_debug(tt::LogOp, "subblock_w: {}", subblock_w);
    log_debug(tt::LogOp, "in0_tile_size: {}", in0_tile_size);
    log_debug(tt::LogOp, "in1_tile_size: {}", in1_tile_size);
    log_debug(tt::LogOp, "out_tile_size: {}", out_tile_size);
    log_debug(tt::LogOp, "in2_tile_size: {}", in2_tile_size);
    log_debug(tt::LogOp, "intermediate_tile_size: {}", intermediate_tile_size);
    log_debug(tt::LogOp, "intermediate_data_format: {}", intermediate_data_format);
    log_debug(tt::LogOp, "in0_cb_num_tiles: {}", in0_cb_num_tiles);
    log_debug(tt::LogOp, "in1_cb_num_tiles: {}", in1_cb_num_tiles);
    log_debug(tt::LogOp, "out_cb_num_tiles: {}", out_cb_num_tiles);
    log_debug(tt::LogOp, "interm_cb_num_tiles: {}", interm_cb_num_tiles);

    std::map<std::string, std::string> defines;
    std::map<std::string, std::string> in0_injector_defines;
    if (use_bias) {
        defines["FUSE_BIAS"] = "1";
    }

    if (use_fused_ternary) {
        defines["FUSE_TERNARY"] = "1";

        // Workaround for LLK bug (https://github.com/tenstorrent/tt-llk/issues/1338)
        // - If ternary_b / gate is float32 then use unary_bcast (row broadcast) + mul_binary_tile (accurate)
        // - If ternary_b / gate is bfloat16 then use mul_tiles_bcast (row broadcast) (workaround)
        if (fused_ternary_input_b.value().dtype() == DataType::FLOAT32) {
            defines["TERNARY_B_IS_FLOAT32"] = "1";
        }
    }

    // Output-contention DRAM levers. Both relieve the same bottleneck — output writes contending with
    // in0 reads on NOC_1 — so they only pay off when the output is WIDE (large N). Trigger on N, not
    // max(M,K,N): the latter also fires on skewed shapes where M or K is large but N is small, where
    // both levers REGRESS (e.g. 4864x2048x1024: both -11%, output-split -6.8%, in0-barrier -3.7%).
    // Measured by output width: N=4096 helps (+0.6% to +2.6%), N<=3072 hurts. (Verified per-lever via
    // matched-blocking ablation; both levers track N.)
    //   - IN0_READ_BARRIER_THRESHOLD: cap outstanding in0 DRAM reads (un-staggered bursts share NOC_1
    //     with output writes); only worth it when there are heavy output writes to contend with.
    //   - OUTPUT_WRITE_NOC0_PCT: route ~40% of output writes onto the otherwise-idle NOC_0 (needs
    //     DM_DYNAMIC_NOC); parallelism gain scales with the per-row write run length, i.e. N.
    auto dm_noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    constexpr uint32_t WIDE_OUTPUT_DIM = 4096;  // N threshold for the output-contention levers
    // The in0-read barrier and prefetch are fundamentally opposed: the barrier caps outstanding in0
    // reads (every IN0_READ_BARRIER_THRESHOLD tiles), while prefetch's whole purpose is to keep the
    // next block's read in flight. Applying both negates prefetch (~+14% slowdown, worst on the
    // transpose orientation where in0 is the large input). So skip the in0 barrier whenever prefetch
    // is active. The split-NOC output write is independent (minor effect) and kept.
    const char* force_prefetch = std::getenv("TT_MM_MCAST_PREFETCH");
    const bool prefetch_active = prefetch_gate || (force_prefetch != nullptr && std::string(force_prefetch) != "0");
    // TT_MM_NO_LARGE_LEVERS: ablation toggle to isolate the large-shape DRAM-contention levers
    // (in0-read barrier + split-NOC output write) from the rest of the dataflow.
    if (N >= WIDE_OUTPUT_DIM && std::getenv("TT_MM_NO_LARGE_LEVERS") == nullptr) {
        if (std::getenv("TT_MM_NO_IN0_BARRIER") == nullptr && !prefetch_active) {
            defines["IN0_READ_BARRIER_THRESHOLD"] = "10";
        }
        if (std::getenv("TT_MM_NO_OUTPUT_NOC_SPLIT") == nullptr) {
            defines["OUTPUT_WRITE_NOC0_PCT"] = "40";
        }
        dm_noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC;
    }

    // Experimental dataflow selectors (env-gated, off by default).
    auto env_flag_set = [](const char* name) {
        const char* v = std::getenv(name);
        return v != nullptr && v[0] != '\0' && std::string(v) != "0";
    };
    // Prefetch: software-pipeline the injector k-loop so block k+1's DRAM read is issued before block
    // k's multicast, overlapping read latency with mcast transit on the single injector DM RISC. The
    // auto-gate (prefetch_gate) was computed earlier with the block sizes (it also clamps the
    // compile-time block). TT_MM_MCAST_PREFETCH forces the dataflow on regardless of the gate.
    const bool mcast_prefetch = env_flag_set("TT_MM_MCAST_PREFETCH") || prefetch_gate;
    const bool mcast_broadcast = env_flag_set("TT_MM_MCAST_BROADCAST") || mcast_prefetch;
    if (mcast_broadcast) {
        // Replace the store-and-forward chain with a single NoC multicast per block. The injector reads
        // DRAM once and mcasts the block to all cores along its broadcast axis (one hardware-replicated
        // transaction instead of N serialized unicast+handshake hops).
        TT_FATAL(!fuse_op && !fuse_srs, "MCAST_BROADCAST does not support fused ops");
        defines["MCAST_BROADCAST"] = "1";
    }
    if (mcast_prefetch) {
        defines["MCAST_PREFETCH"] = "1";
    }

    if (fuse_op) {
        // Create semaphores
        fused_op_signaler->init_fused_op(program, device, in0_sender_cores);
        defines["FUSE_AG"] = "1";
        if (fused_op_signaler->read_local_slice_from_input) {
            in0_injector_defines = defines;
            in0_injector_defines["READ_FROM_LOCAL_INPUT"] = "1";
        }
    }

    uint32_t srs_fuse_signaler_sync_semaphore_id = 0;
    if (fuse_srs) {
        defines["SRS_FUSE_OP_SIGNALER"] = "1";
        srs_fuse_signaler_sync_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, 0);
    }

    std::vector<CoreCoord> all_worker_cores_noc;
    if (fuse_srs) {
        all_worker_cores_noc.reserve(num_cores);
        auto all_cores_tmp = corerange_to_cores(core_grid, num_cores, true);
        for (const auto& c : all_cores_tmp) {
            all_worker_cores_noc.push_back(device->worker_core_from_logical_core(c));
        }
    }

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t in2_addr = use_bias ? bias_tensor.value().buffer()->address() : 0;
    // Note: Dataflow kernels can take a variable number of output tensors.
    // They are appended as a variable-length array at the end of the runtime-args:
    //   - for in0 output-writer cores the first output address is at index 13
    //   - for in1 output-writer cores the first output address is at index 12
    uint32_t in3_addr = (fuse_op && fused_op_signaler->read_local_slice_from_input)
                            ? fused_op_signaler->ag_input.value().buffer()->address()
                            : 0;
    auto in3_data_format =
        (fuse_op && fused_op_signaler->read_local_slice_from_input)
            ? tt::tt_metal::datatype_to_dataformat_converter(fused_op_signaler->ag_input.value().dtype())
            : in1_data_format;

    auto in3_tile_size = tt::tile_size(in3_data_format);

    /**
     * Create kernels
     */

    bool in0_is_output_writer = !transpose_core_grid;
    bool in1_is_output_writer = transpose_core_grid;

    // Split-K plan B: REDUCE_K is defined only on the OUTPUT-WRITER DM kernel (the one that owns out_cb)
    // and on compute. The non-writer DM keeps the plain defines. The reduction dataflow lives in the
    // writer; compute emits the running sum (copy on the bottom band, add on the rest).
    std::map<std::string, std::string> in0_defines = defines;
    std::map<std::string, std::string> in1_defines = defines;
    if (num_k_fused) {
        (in0_is_output_writer ? in0_defines : in1_defines)["REDUCE_K"] = "1";
    }

    std::vector<uint32_t> in0_sender_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        true,               // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        in3_tile_size,
    };
    append_accessors(
        in0_sender_compile_time_args,
        input_tensor,
        output_tensors,
        bias_tensor,
        (fuse_op && fused_op_signaler->read_local_slice_from_input) ? fused_op_signaler->ag_input : std::nullopt,
        fused_ternary_input_a,
        fused_ternary_input_b);
    auto in0_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        in0_sender_spec,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .noc_mode = dm_noc_mode,
            .compile_args = in0_sender_compile_time_args,
            .defines =
                (fuse_op && fused_op_signaler->read_local_slice_from_input) ? in0_injector_defines : in0_defines});

    std::vector<uint32_t> in0_receiver_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        false,              // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        in3_tile_size,
    };
    append_accessors(
        in0_receiver_compile_time_args,
        input_tensor,
        output_tensors,
        bias_tensor,
        std::nullopt,  // no ag_input for in0_receiver
        fused_ternary_input_a,
        fused_ternary_input_b);

    auto in0_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        in0_receiver_spec,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .noc_mode = dm_noc_mode,
            .compile_args = in0_receiver_compile_time_args,
            .defines = in0_defines});

    std::vector<uint32_t> in1_sender_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        true,               // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
    };
    append_accessors(
        in1_sender_compile_time_args,
        weight_tensor,
        output_tensors,
        bias_tensor,
        std::nullopt,  // no ag_input for in1_sender
        fused_ternary_input_a,
        fused_ternary_input_b);

    auto in1_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_sender_spec,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc,
            .noc = in1_noc,
            .noc_mode = dm_noc_mode,
            .compile_args = in1_sender_compile_time_args,
            .defines = in1_defines});

    std::vector<uint32_t> in1_receiver_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        false,              // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
    };
    append_accessors(
        in1_receiver_compile_time_args,
        weight_tensor,
        output_tensors,
        bias_tensor,
        std::nullopt,  // no ag_input for in1_receiver
        fused_ternary_input_a,
        fused_ternary_input_b);

    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_receiver_spec,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc,
            .noc = in1_noc,
            .noc_mode = dm_noc_mode,
            .compile_args = in1_receiver_compile_time_args,
            .defines = in1_defines});

    std::vector<uint32_t> compute_compile_time_args = {
        K_blocks,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        subblock_h,
        subblock_w};

    auto compute_defines = defines;
    if (num_k_fused) {
        compute_defines["REDUCE_K"] = "1";
    }
    std::map<std::string, std::string> compute_activation_defines;
    if (fused_activation.has_value()) {
        compute_activation_defines = ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type,
            fused_activation.value().params,
            "ACTIVATION",
            "fused_act_dst_id",
            output_tensors[0].dtype());
    }
    compute_defines.merge(compute_activation_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, compute_defines, ttnn::get_throttle_level(compute_kernel_config));
    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines});

    /**
     * The receiver writer cores defer their writes in order to reduce NOC congestion.
     * Further, the amount of K_blocks they defer by depends on their core coordinate.
     * If we have core_grid.x cores, we'd want to evenly stride the K_blocks they defer by.
     * For first pass, it's easy enough to use core_grid.x
     */
    uint32_t k_blocks_per_core =
        tt::div_up(K_blocks, (transpose_core_grid ? in1_parallel_axis_cores : in0_parallel_axis_cores));

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    uint32_t max_defer_write_k_block = 0;
    for (const auto& c : cores) {
        uint32_t dwk = std::min(static_cast<uint32_t>(c.y) * k_blocks_per_core, K_blocks - 1);
        max_defer_write_k_block = std::max(max_defer_write_k_block, dwk);
    }

    // NOTE: Uniform per-core M/N ranges are required for DM forward handshakes to match across links.
    // If neighboring cores along a forwarding chain iterate different (M,N) counts, the sender can wait
    // for requests that the receiver will never issue, leading to deadlock. Keep the original uniform
    // div_up-based ranges for M and N.

    // Index where the output addresses begin in the in0/in1 sender/receiver arg vectors. Captured below
    // right before the outputs are pushed, so it accounts for ALL conditionally-inserted args (the +3
    // split-K args, the +8 plan-B reduce args, and the +3 ternary args). override_runtime_arguments()
    // needs this to rewrite the output buffer addresses on program-cache REUSE; a hardcoded index here
    // silently corrupted reuse after split-K added args before the outputs. Uniform across cores.
    uint32_t in0_out_addr_start_idx = 0;
    uint32_t in1_out_addr_start_idx = 0;

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        // Slicing: group physical rows (grid.y). The dim on grid.y is grouped (parallel index = row
        // within group); the dim on grid.x is sliced (parallel index = group * grid.x + col). At
        // num_slices==1 this is exactly the original (group=0, y_idx=core.y, x_idx=core.x).
        const uint32_t group = core.y / rows_per_group;   // innermost group: [0, num_slices*num_k_slices)
        const uint32_t kband = group / num_slices;        // K-band index (outer)
        const uint32_t slice_group = group % num_slices;  // N-slice group within the band
        const uint32_t y_idx = core.y % rows_per_group;
        // N-slice index uses ONLY slice_group (K-bands share the full N, they differ in K not output cols).
        const uint32_t x_idx = slice_group * grid_size.x + core.x;
        uint32_t in0_idx = transpose_core_grid ? x_idx : y_idx;
        uint32_t in1_idx = transpose_core_grid ? y_idx : x_idx;
        // Split-K per-core: which K-block this band starts at, and where its partial lands in the
        // [num_k_slices * M, N] output (offset down by kband full-M stripes). num_k_slices=1 => all 0.
        // Plan B (num_k_fused) instead reduces in L1 to a single [M, N], so every band targets the same
        // output (offset 0, total M_tiles) and only the top band actually writes it.
        const uint32_t k_block_start = kband * (padded_K_tiles / K_block_tiles);
        const uint32_t out_m_tile_offset = num_k_fused ? 0u : kband * M_tiles;
        const uint32_t out_M_tiles_total = num_k_fused ? M_tiles : num_k_slices * M_tiles;
        // Split-K plan B: bottom K-band has no incoming running sum (emits its own partial). Consumed by
        // the compute kernel only under REDUCE_K; harmless otherwise.
        const uint32_t is_reduce_bottom = (kband == num_k_slices - 1) ? 1u : 0u;
        const uint32_t is_reduce_top = (kband == 0) ? 1u : 0u;
        // Plan B vertical reduction neighbors: the band ABOVE (kband-1, where this band forwards its
        // running sum) and the band BELOW (kband+1, which forwards into this band's cb_reduce). Same
        // grid.x, +/- one K-band of rows. Clamped to self at the ends (unused there: top never sends,
        // bottom never receives). Physical NoC coords are what the kernel writes to.
        const uint32_t band_rows = num_slices * rows_per_group;  // rows spanning one K-band
        CoreCoord reduce_up_l = {(std::size_t)core.x, (std::size_t)(is_reduce_top ? core.y : core.y - band_rows)};
        CoreCoord reduce_down_l = {(std::size_t)core.x, (std::size_t)(is_reduce_bottom ? core.y : core.y + band_rows)};
        auto reduce_up_p = device->worker_core_from_logical_core(reduce_up_l);
        auto reduce_down_p = device->worker_core_from_logical_core(reduce_down_l);

        // Injector identification by PHYSICAL position (not the sliced idx): the across-cols (small)
        // injector is at col 0 of every row; the down-rows (big) injector is at each group's top row.
        const bool is_cols_injector = (core.x == 0);
        const bool is_rows_injector = (core.y % rows_per_group == 0);
        const bool is_in0_injector = transpose_core_grid ? is_rows_injector : is_cols_injector;
        const bool is_in1_injector = transpose_core_grid ? is_cols_injector : is_rows_injector;

        // Forwarding chains: the SMALL input is forwarded across cols (full grid.x, base 0); the BIG
        // input is forwarded down rows but BOUNDED to the row-group (length rows_per_group, base =
        // group's top row). The big-input injector endpoint (top_core) sits at the group's top row.
        const uint32_t base_row = group * rows_per_group;
        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)base_row};

        // in0: across-cols (small) when !transpose; down-rows (big, group-bounded) when transpose.
        auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            transpose_core_grid ? rows_per_group : grid_size.x,
            in0_noc,
            /*axis_is_x_when_not_transposed=*/true,
            /*initial_endpoint=*/(transpose_core_grid ? top_core : left_core),
            /*axis_base=*/(transpose_core_grid ? base_row : 0));

        // in1: down-rows (big, group-bounded) when !transpose; across-cols (small) when transpose.
        auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            transpose_core_grid ? grid_size.x : rows_per_group,
            in1_noc,
            /*axis_is_x_when_not_transposed=*/false,
            /*initial_endpoint=*/(transpose_core_grid ? left_core : top_core),
            /*axis_base=*/(transpose_core_grid ? 0 : base_row));

        auto in0_prev_core = clamped_prev(in0_core_order, in0_core_order_index);
        auto in0_next_core = clamped_next(in0_core_order, in0_core_order_index);
        auto in1_prev_core = clamped_prev(in1_core_order, in1_core_order_index);
        auto in1_next_core = clamped_next(in1_core_order, in1_core_order_index);

        auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);

        /**
         * NOTE: Some cores are doing unnecessary work, on blocks which are processed just to make
         * the total number of blocks divisible by the number of cores.
         * We can't yet get rid of these blocks, since the receiver cores must ack
         * all blocks that sender cores are expected to send.
         */
        uint32_t M_start_tile = M_tiles_per_core * in0_idx;
        uint32_t M_end_tile = M_tiles_per_core * (in0_idx + 1);
        uint32_t N_start_tile = N_tiles_per_core * in1_idx;
        uint32_t N_end_tile = N_tiles_per_core * (in1_idx + 1);

        // Defer write to K block with same coordinate as core
        // The writer receiver cores always have core.x > 0
        uint32_t defer_write_k_block = std::min(static_cast<uint32_t>(core.y) * k_blocks_per_core, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        // Multicast-broadcast prototype (TT_MM_MCAST_BROADCAST): per core, compute the broadcast
        // rectangle (the receiver cores along each input's broadcast axis), num receivers, and the
        // injector's physical coords (receivers signal readiness to it). in0 broadcasts perpendicular
        // to in0_idx, in1 perpendicular to in1_idx; transpose swaps which axis is which.
        CoreCoord in0_inj_l, in0_rf_l, in0_rl_l, in1_inj_l, in1_rf_l, in1_rl_l;
        uint32_t num_in0_recv = 0, num_in1_recv = 0;
        // The big input (down rows) mcasts only within its row-group: rect = [base_row+1, group end],
        // injector at the group's top row. The small input (across cols) mcasts the full row. num_recv
        // for the big input is rows_per_group-1 (== 0 when a group is a single row → no big-input mcast,
        // the kernel skips it). At num_slices==1 (base_row=0, rows_per_group=grid.y) this is the original.
        const std::size_t grp_last_row = base_row + rows_per_group - 1;
        if (!transpose_core_grid) {
            in0_inj_l = {(std::size_t)0, (std::size_t)core.y};  // small: across cols
            in0_rf_l = {(std::size_t)1, (std::size_t)core.y};
            in0_rl_l = {(std::size_t)grid_size.x - 1, (std::size_t)core.y};
            num_in0_recv = grid_size.x - 1;
            in1_inj_l = {(std::size_t)core.x, (std::size_t)base_row};  // big: down group rows
            in1_rf_l = {(std::size_t)core.x, (std::size_t)(base_row + 1)};
            in1_rl_l = {(std::size_t)core.x, grp_last_row};
            num_in1_recv = rows_per_group - 1;
        } else {
            in0_inj_l = {(std::size_t)core.x, (std::size_t)base_row};  // big: down group rows
            in0_rf_l = {(std::size_t)core.x, (std::size_t)(base_row + 1)};
            in0_rl_l = {(std::size_t)core.x, grp_last_row};
            num_in0_recv = rows_per_group - 1;
            in1_inj_l = {(std::size_t)0, (std::size_t)core.y};  // small: across cols
            in1_rf_l = {(std::size_t)1, (std::size_t)core.y};
            in1_rl_l = {(std::size_t)grid_size.x - 1, (std::size_t)core.y};
            num_in1_recv = grid_size.x - 1;
        }
        auto in0_inj_p = device->worker_core_from_logical_core(in0_inj_l);
        auto in0_mc_s = device->worker_core_from_logical_core(in0_rf_l);
        auto in0_mc_e = device->worker_core_from_logical_core(in0_rl_l);
        // The multicast-rect start/end ordering must match the mcast NoC (swap iff NOC_1).
        if (in0_noc == tt::tt_metal::NOC::NOC_1) {
            std::swap(in0_mc_s, in0_mc_e);
        }
        auto in1_inj_p = device->worker_core_from_logical_core(in1_inj_l);
        auto in1_mc_s = device->worker_core_from_logical_core(in1_rf_l);
        auto in1_mc_e = device->worker_core_from_logical_core(in1_rl_l);
        if (in1_noc == tt::tt_metal::NOC::NOC_1) {
            std::swap(in1_mc_s, in1_mc_e);
        }

        std::vector<uint32_t> in0_args = {
            in0_addr,
            in2_addr,
            in3_addr,
            is_in0_sink,
            (std::uint32_t)in0_next_core_physical.x,  // in0_dest_noc_x
            (std::uint32_t)in0_next_core_physical.y,  // in0_dest_noc_y
            (std::uint32_t)in0_prev_core_physical.x,  // in0_sender_noc_x
            (std::uint32_t)in0_prev_core_physical.y,  // in0_sender_noc_y
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
            max_defer_write_k_block,
            k_block_start,      // split-K: absolute first K-block this band reduces (0 unless K-par)
            out_m_tile_offset,  // split-K: M-stripe offset of this band's partial in [Pk*M, N]
            out_M_tiles_total,  // split-K: total M tiles of the [Pk*M, N] output (= num_k_slices * M)
        };
        // Split-K plan B reduction args (read by the writer kernel under REDUCE_K, right after the
        // split-K args and before any ternary/output args — must match the kernel's read order).
        if (num_k_fused && in0_is_output_writer) {
            in0_args.push_back((std::uint32_t)reduce_up_p.x);
            in0_args.push_back((std::uint32_t)reduce_up_p.y);
            in0_args.push_back((std::uint32_t)reduce_down_p.x);
            in0_args.push_back((std::uint32_t)reduce_down_p.y);
            in0_args.push_back(is_reduce_top);
            in0_args.push_back(is_reduce_bottom);
            in0_args.push_back(reduce_ready_semaphore_id);
            in0_args.push_back(reduce_recv_semaphore_id);
        }
        // Add ternary addresses if present (after defer_write_k_block, before output addresses)
        if (use_fused_ternary) {
            in0_args.push_back(fused_ternary_input_a.value().buffer()->address());
            in0_args.push_back(fused_ternary_input_b.value().buffer()->address());
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            in0_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        // Add output addresses at the end (unified layout for both regular and split)
        in0_out_addr_start_idx = static_cast<uint32_t>(in0_args.size());
        for (const auto& output_tensor : output_tensors) {
            in0_args.push_back(output_tensor.buffer()->address());
        }
        if (mcast_broadcast) {
            // Appended after outputs (mcast prototype is incompatible with fused ops, so nothing else
            // follows). Kernel reads these at out_addr_rt_arg_idx + N_chunks under MCAST_BROADCAST.
            in0_args.push_back((std::uint32_t)in0_mc_s.x);
            in0_args.push_back((std::uint32_t)in0_mc_s.y);
            in0_args.push_back((std::uint32_t)in0_mc_e.x);
            in0_args.push_back((std::uint32_t)in0_mc_e.y);
            in0_args.push_back(num_in0_recv);
            in0_args.push_back((std::uint32_t)in0_inj_p.x);
            in0_args.push_back((std::uint32_t)in0_inj_p.y);
            // Pipelined: this core's index within the in0 broadcast group (= in1_idx). Injector
            // (idx 0) polls slots 1..num; receiver incs its own slot on the injector's credit CB.
            in0_args.push_back(in1_idx);
        }
        if (fuse_op) {
            fused_op_signaler->push_matmul_fused_op_rt_args(in0_args, padded_K_tiles / K_block_tiles, K_block_tiles);
        }
        if (fuse_srs) {
            in0_args.push_back(static_cast<uint32_t>(num_cores));
            in0_args.push_back(static_cast<uint32_t>(core_id));
            in0_args.push_back(static_cast<uint32_t>(srs_fuse_signaler_sync_semaphore_id));
            for (const auto& noc_core : all_worker_cores_noc) {
                in0_args.push_back(static_cast<uint32_t>(noc_core.x));
                in0_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in0_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->num_fused_op_cores_to_signal));
            for (const auto& noc_core : srs_fused_op_signaler->fused_op_receiver_cores_noc) {
                in0_args.push_back(static_cast<uint32_t>(noc_core.x));
                in0_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in0_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->fused_op_receiver_signal_semaphore));
            in0_args.push_back(1);  // mcast_signal_op_cores
        }
        if (is_in0_injector) {
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
        } else {
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_args);
        }

        std::vector<uint32_t> in1_args = {
            in1_addr,
            in2_addr,
            is_in1_sink,
            (std::uint32_t)in1_next_core_physical.x,  // in1_dest_noc_x
            (std::uint32_t)in1_next_core_physical.y,  // in1_dest_noc_y
            (std::uint32_t)in1_prev_core_physical.x,  // in1_sender_noc_x
            (std::uint32_t)in1_prev_core_physical.y,  // in1_sender_noc_y
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
            max_defer_write_k_block,
            k_block_start,      // split-K: absolute first K-block this band reduces (0 unless K-par)
            out_m_tile_offset,  // split-K: M-stripe offset of this band's partial in [Pk*M, N]
            out_M_tiles_total,  // split-K: total M tiles of the [Pk*M, N] output (= num_k_slices * M)
        };
        // Split-K plan B reduction args (read by the writer kernel under REDUCE_K, right after the
        // split-K args and before any ternary/output args — must match the kernel's read order).
        if (num_k_fused && in1_is_output_writer) {
            in1_args.push_back((std::uint32_t)reduce_up_p.x);
            in1_args.push_back((std::uint32_t)reduce_up_p.y);
            in1_args.push_back((std::uint32_t)reduce_down_p.x);
            in1_args.push_back((std::uint32_t)reduce_down_p.y);
            in1_args.push_back(is_reduce_top);
            in1_args.push_back(is_reduce_bottom);
            in1_args.push_back(reduce_ready_semaphore_id);
            in1_args.push_back(reduce_recv_semaphore_id);
        }
        // Add ternary addresses if present (after defer_write_k_block, before output addresses)
        if (use_fused_ternary) {
            in1_args.push_back(fused_ternary_input_a.value().buffer()->address());
            in1_args.push_back(fused_ternary_input_b.value().buffer()->address());
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            in1_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        // Add output addresses at the end (unified layout for both regular and split)
        in1_out_addr_start_idx = static_cast<uint32_t>(in1_args.size());
        for (const auto& output_tensor : output_tensors) {
            in1_args.push_back(output_tensor.buffer()->address());
        }
        if (mcast_broadcast) {
            in1_args.push_back((std::uint32_t)in1_mc_s.x);
            in1_args.push_back((std::uint32_t)in1_mc_s.y);
            in1_args.push_back((std::uint32_t)in1_mc_e.x);
            in1_args.push_back((std::uint32_t)in1_mc_e.y);
            in1_args.push_back(num_in1_recv);
            in1_args.push_back((std::uint32_t)in1_inj_p.x);
            in1_args.push_back((std::uint32_t)in1_inj_p.y);
            in1_args.push_back(in0_idx);
        }
        if (fuse_op) {
            fused_op_signaler->push_matmul_fused_op_rt_args(in1_args, padded_K_tiles / K_block_tiles, K_block_tiles);
        }
        if (fuse_srs) {
            in1_args.push_back(static_cast<uint32_t>(num_cores));
            in1_args.push_back(static_cast<uint32_t>(core_id));
            in1_args.push_back(static_cast<uint32_t>(srs_fuse_signaler_sync_semaphore_id));
            for (const auto& noc_core : all_worker_cores_noc) {
                in1_args.push_back(static_cast<uint32_t>(noc_core.x));
                in1_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in1_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->num_fused_op_cores_to_signal));
            for (const auto& noc_core : srs_fused_op_signaler->fused_op_receiver_cores_noc) {
                in1_args.push_back(static_cast<uint32_t>(noc_core.x));
                in1_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in1_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->fused_op_receiver_signal_semaphore));
            in1_args.push_back(1);  // mcast_signal_op_cores
        }
        if (is_in1_injector) {
            // in1 sender
            SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_args);
        } else {
            // in1 receiver
            SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_args);
        }

        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            is_reduce_bottom,  // split-K B (used by compute only under REDUCE_K)
        };
        if (use_fused_ternary) {
            compute_runtime_args.push_back(*reinterpret_cast<const uint32_t*>(&fused_ternary_scalar.value()));
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            compute_runtime_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    return MinimalMatmulProgramFactory::shared_variables_t{
        num_cores,
        cores,
        in0_sender_kernels_id,
        in0_receiver_kernels_id,
        in1_sender_kernels_id,
        in1_receiver_kernels_id,
        compute_kernels_id,
        transpose_core_grid,
        fuse_op && fused_op_signaler->read_local_slice_from_input,
        rows_per_group,
        in0_out_addr_start_idx,
        in1_out_addr_start_idx};
}

MinimalMatmulProgramFactory::shared_variables_t minimal_matmul_factory_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler,
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler>& srs_fused_op_signaler) {
    std::vector<Tensor> output_tensors = {output_tensor};
    return minimal_matmul_factory_helper_common(
        program,
        input_tensor,
        weight_tensor,
        bias_tensor,
        fused_activation,
        config,
        output_tensors,
        compute_kernel_config,
        fused_op_signaler,
        1,  // N_chunks = 1 for regular minimal_matmul
        std::nullopt,
        std::nullopt,
        std::nullopt,
        srs_fused_op_signaler);
}

MinimalMatmulProgramFactory::cached_program_t MinimalMatmulProgramFactory::create(
    const MinimalMatmulParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> empty_fused_op_signaler;
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> empty_srs_fused_op_signaler;

    auto shared_vars = minimal_matmul_factory_helper_common(
        program,
        tensor_args.input_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias_tensor,
        operation_attributes.fused_activation,
        operation_attributes.config,
        tensor_return_value,
        operation_attributes.compute_kernel_config,
        empty_fused_op_signaler,
        static_cast<uint32_t>(operation_attributes.chunks),
        operation_attributes.fused_ternary_scalar,
        tensor_args.fused_ternary_input_a,
        tensor_args.fused_ternary_input_b,
        empty_srs_fused_op_signaler);

    return {std::move(program), std::move(shared_vars)};
}

// Common helper for override_runtime_arguments - works with both single and multiple output tensors
void MinimalMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MinimalMatmulParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& override_variables = cached_program.shared_variables;

    auto& in0_sender_runtime_args = GetRuntimeArgs(program, override_variables.in0_sender_kernels_id);
    auto& in0_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in0_receiver_kernels_id);
    auto& in1_sender_runtime_args = GetRuntimeArgs(program, override_variables.in1_sender_kernels_id);
    auto& in1_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in1_receiver_kernels_id);
    auto& compute_runtime_args = GetRuntimeArgs(program, override_variables.compute_kernels_id);

    // RT args layout for in0: [in0_addr, in2_addr, in3_addr, is_sink, noc_coords(4), tile_ranges(4),
    //   defer_write_k_block, max_defer_write_k_block, split_k_args(3), [optional reduce(8)],
    //   [optional ternary_a_addr, ternary_b_addr, broadcast_ternary_b], out_addrs(N)...]
    // The in0/in0_in2/in0_in3 indices are fixed at the front; the OUTPUT (and ternary) addresses move as
    // conditional args are inserted, so their start index is captured in create() (shared_variables) rather
    // than hardcoded — a hardcoded index silently corrupted program-cache reuse once split-K added 3 args
    // before the outputs (cached re-runs wrote the new output address into the split-K-arg slot -> nan).
    constexpr uint32_t in0_in0_addr_idx = 0;
    constexpr uint32_t in0_in2_addr_idx = 1;
    constexpr uint32_t in0_in3_addr_idx = 2;

    constexpr uint32_t in1_in0_addr_idx = 0;
    constexpr uint32_t in1_bias_addr_idx = 1;

    // Check if ternary addresses are present
    bool has_fused_ternary =
        tensor_args.fused_ternary_input_a.has_value() && tensor_args.fused_ternary_input_b.has_value();
    // Output addresses begin where create() put them; ternary (when present) is the 3 args right before.
    const uint32_t in0_out_addr_start_idx = override_variables.in0_out_addr_start_idx;
    const uint32_t in1_out_addr_start_idx = override_variables.in1_out_addr_start_idx;
    const uint32_t in0_ternary_a_addr_idx = in0_out_addr_start_idx - 3;
    const uint32_t in0_ternary_b_addr_idx = in0_out_addr_start_idx - 2;
    const uint32_t in1_ternary_a_addr_idx = in1_out_addr_start_idx - 3;
    const uint32_t in1_ternary_b_addr_idx = in1_out_addr_start_idx - 2;

    for (uint32_t i = 0; i < override_variables.num_cores; ++i) {
        CoreCoord core = override_variables.cores.at(i);
        // Injector identification by physical position, slicing-aware (must match create()): across-cols
        // (small) injector at col 0; down-rows (big) injector at each group's top row.
        const bool is_cols_injector = (core.x == 0);
        const bool is_rows_injector = (core.y % override_variables.rows_per_group == 0);
        const bool is_in0_injector = override_variables.transpose_core_grid ? is_rows_injector : is_cols_injector;
        const bool is_in1_injector = override_variables.transpose_core_grid ? is_cols_injector : is_rows_injector;

        if (is_in0_injector) {
            auto& in0_sender_args = in0_sender_runtime_args[core.x][core.y];

            in0_sender_args[in0_in0_addr_idx] = tensor_args.input_tensor.buffer()->address();
            in0_sender_args[in0_in2_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            in0_sender_args[in0_in3_addr_idx] = tensor_args.optional_input_tensor.has_value() &&
                                                        cached_program.shared_variables.read_local_slice_from_input
                                                    ? tensor_args.optional_input_tensor.value().buffer()->address()
                                                    : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in0_sender_args[in0_ternary_a_addr_idx] = tensor_args.fused_ternary_input_a.value().buffer()->address();
                in0_sender_args[in0_ternary_b_addr_idx] = tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in0_sender_args[in0_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        } else {
            auto& in0_receiver_args = in0_receiver_runtime_args[core.x][core.y];
            in0_receiver_args[in0_in2_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in0_receiver_args[in0_ternary_a_addr_idx] =
                    tensor_args.fused_ternary_input_a.value().buffer()->address();
                in0_receiver_args[in0_ternary_b_addr_idx] =
                    tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in0_receiver_args[in0_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        }

        if (is_in1_injector) {
            auto& in1_sender_args = in1_sender_runtime_args[core.x][core.y];
            in1_sender_args[in1_in0_addr_idx] = tensor_args.weight_tensor.buffer()->address();
            in1_sender_args[in1_bias_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in1_sender_args[in1_ternary_a_addr_idx] = tensor_args.fused_ternary_input_a.value().buffer()->address();
                in1_sender_args[in1_ternary_b_addr_idx] = tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in1_sender_args[in1_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        } else {
            auto& in1_receiver_args = in1_receiver_runtime_args[core.x][core.y];
            in1_receiver_args[in1_bias_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in1_receiver_args[in1_ternary_a_addr_idx] =
                    tensor_args.fused_ternary_input_a.value().buffer()->address();
                in1_receiver_args[in1_ternary_b_addr_idx] =
                    tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in1_receiver_args[in1_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        }
    }

    // Update compute kernel runtime args for scalar
    for (uint32_t i = 0; i < override_variables.num_cores; ++i) {
        CoreCoord core = override_variables.cores.at(i);
        auto& compute_args = compute_runtime_args[core.x][core.y];

        // Compute RT args: [M_start, M_end, N_start, N_end, [optional: scalar]]
        // If ternary is present and scalar arg exists, update it at index 4
        if (has_fused_ternary && operation_attributes.fused_ternary_scalar.has_value()) {
            float scalar = operation_attributes.fused_ternary_scalar.value();
            uint32_t scalar_as_uint = *reinterpret_cast<const uint32_t*>(&scalar);
            compute_args[4] = scalar_as_uint;
        }
    }
}

}  // namespace ttnn::experimental::prim
