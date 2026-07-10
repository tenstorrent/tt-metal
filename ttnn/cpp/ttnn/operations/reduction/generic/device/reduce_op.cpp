// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"

#include <optional>
#include <string>

#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace reduce_op_utils {

std::map<std::string, std::string> get_defines(
    tt::tt_metal::ReduceOpMath reduce_op, tt::tt_metal::ReduceOpDim reduce_dim) {
    std::map<std::string, std::string> defines;
    // TODO(AP): need a sync with Reduce::Max from HLK headers
    std::string reduce_dim_str;
    switch (reduce_dim) {
        case tt::tt_metal::ReduceOpDim::W: reduce_dim_str = "ckernel::ReduceDim::REDUCE_ROW"; break;
        case tt::tt_metal::ReduceOpDim::H: reduce_dim_str = "ckernel::ReduceDim::REDUCE_COL"; break;
        case tt::tt_metal::ReduceOpDim::HW: reduce_dim_str = "ckernel::ReduceDim::REDUCE_SCALAR"; break;
        default: TT_THROW("Invalid reduce_dim!");
    }
    switch (reduce_op) {
        case tt::tt_metal::ReduceOpMath::MAX: defines["REDUCE_OP"] = "ckernel::PoolType::MAX"; break;
        case tt::tt_metal::ReduceOpMath::AVG: defines["REDUCE_OP"] = "ckernel::PoolType::AVG"; break;
        default: defines["REDUCE_OP"] = "ckernel::PoolType::SUM"; break;
    }
    defines["REDUCE_DIM"] = reduce_dim_str;
    return defines;
}

}  // namespace reduce_op_utils
namespace ttnn::operations::reduction::generic::detail {

// reduce min
// reduce min = - reduce_max( -x )
Tensor reduce_min(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler = 1.0f,
    const tt::tt_metal::MemoryConfig& output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt) {
    Tensor input = input_tensor;
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
        input.storage_type() == tt::tt_metal::StorageType::DEVICE) {
        // Changing layout to TILE with +inf padding
        auto pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(input.padded_shape());
        input = ttnn::tilize_with_val_padding(
            input,
            pad_shape,
            std::numeric_limits<float>::infinity(),
            output_mem_config,
            std::nullopt,
            true,
            sub_core_grids);
    }
    return detail::reduce(
        input,
        tt::tt_metal::ReduceOpMath::MAX,
        reduce_dim,
        scaler,
        output_mem_config,
        std::nullopt,
        compute_kernel_config,
        sub_core_grids,
        true);
}

Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    bool negate,
    bool use_row_major_support,
    bool fast_and_approximate_mode) {
    if (reduce_math == tt::tt_metal::ReduceOpMath::MIN) {
        return reduce_min(input_tensor, reduce_dim, scaler, output_mem_config, compute_kernel_config, sub_core_grids);
    }

    auto parallelization_strategy = ttnn::prim::get_parallelization_strategy(input_tensor, reduce_dim);
    auto is_multicore_hw = parallelization_strategy == tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    float pad_value = reduce_math == tt::tt_metal::ReduceOpMath::MAX ? -std::numeric_limits<float>::infinity() : 0;

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Expected input tensor to be on device");
    TT_FATAL(
        input_tensor.device() != nullptr,
        "input_tensor.device() == nullptr, No device found, move input_tensor to device");

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // fp32_dest_acc_en defaults to True here, so always use HiFi3 as default on Wormhole B0.
    const auto arch = input_tensor.device()->arch();
    const auto is_wormhole = arch == tt::ARCH::WORMHOLE_B0;

    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
        arch,
        std::nullopt,
        is_wormhole ? tt::tt_metal::MathFidelity::HiFi3 : tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true));
    ttnn::verify_numerical_configuration(arch, compute_kernel_config);

    // Accurate fp32 mean: SFPU AVG (lowered to SUM + 1/N below); FPU fallback without fp32_dest_acc_en or on Quasar.
    const bool use_sfpu_fp32_mean =
        !fast_and_approximate_mode && input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 &&
        reduce_math == tt::tt_metal::ReduceOpMath::AVG && arch != tt::ARCH::QUASAR && config.fp32_dest_acc_en;

    // Dense row-major reduce: a fast path that consumes ROW_MAJOR input directly (no host tilize)
    // and is currently restricted to mean (AVG) / sum (SUM) on 4D BF16/FLOAT32 tensors with
    // interleaved I/O on both sides. Anything else — MAX/MIN, HW reduce, other dtypes, sharded
    // input or output — falls back to the standard tilize + tile-reduce path.
    //
    // MAX/MIN are excluded because the RM compute kernel accumulates partial reductions via
    // Accumulate::at across chunks, and the cross-chunk fold uses SUM semantics. Wiring MAX
    // accumulation through that pipeline is doable but not yet done; for now they tilize.
    //
    // The path is opt-in via use_row_major_support: when false (the default), eligibility is forced
    // off and the op always tilizes through the classic tile-reduce kernels. Default-off because the
    // dense RM path currently regresses perf and can hang on tall (multi-H-tile) reduces; flip on
    // only once those are fixed.
    const bool both_interleaved =
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    const bool rm_base_eligible =
        use_row_major_support && input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
        input_tensor.logical_shape().rank() == 4 && both_interleaved &&
        (input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
         input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32) &&
        (reduce_math == tt::tt_metal::ReduceOpMath::AVG || reduce_math == tt::tt_metal::ReduceOpMath::SUM);
    const bool use_rm_dense_w = rm_base_eligible && reduce_dim == tt::tt_metal::ReduceOpDim::W;
    const bool use_rm_dense_h = rm_base_eligible && reduce_dim == tt::tt_metal::ReduceOpDim::H;
    const bool use_rm_dense = use_rm_dense_w || use_rm_dense_h;

    // High-level mean uses AVG with scaler (1/N). On the tiled path, GMPOOL AVG matches that intent. On the dense
    // row-major W/H path we tilize one logical row at a time from a narrow RM page; AVG applies an extra normalization
    // for full tile faces that does not match torch.mean together with partial-row tilize. Use SUM + the same scaler.
    // SUM and MAX pass through unchanged.
    // The accurate fp32 SFPU mean path also lowers AVG to SUM: the SFPU folds tiles with a plain
    // add (add_binary_tile) and normalizes via the 1/N post-mul below, so it needs SUM semantics.
    tt::tt_metal::ReduceOpMath prim_reduce_math = reduce_math;
    if ((use_rm_dense || use_sfpu_fp32_mean) && reduce_math == tt::tt_metal::ReduceOpMath::AVG) {
        prim_reduce_math = tt::tt_metal::ReduceOpMath::SUM;
    }

    // Reduce only works with tile layout on the classic path; dense path keeps row-major input.
    Tensor tilized_input = input_tensor;
    if (!use_rm_dense) {
        auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
        tilized_input = ttnn::tilize_with_val_padding(
            input_tensor, padded_shape, pad_value, input_tensor.memory_config(), std::nullopt, true, sub_core_grids);
    }

    // A non-unity scalar is applied after the reduction (see requires_post_mul() in common.hpp):
    // GMPOOL keeps only the scaler's exponent for MAX/MIN, and the Int32 SFPU path ignores the
    // scaler CB. Int32 post-mul rounds through fp32, so it is lossy for |result| > 2^24.
    // The accurate fp32 SFPU mean also post-muls (SFPU ignores the scaler CB), applying the 1/N here.
    const bool use_post_mul =
        ttnn::prim::requires_post_mul(reduce_math, tilized_input.dtype(), scaler, use_sfpu_fp32_mean);
    const float reduce_scaler = use_post_mul ? 1.0f : scaler;
    const float post_mul = use_post_mul ? scaler : 1.0f;

    // External-negate fallback for the H step when the fused-negate kernel's
    // CBs (reduce_h_neg.cpp uses Ht * lcm(Wt_g1, Wt_g2) tiles for both c_4 and
    // c_5) won't fit in L1.  Computes -reduce(MAX, H, -x) using the regular
    // reduce kernel.
    auto h_reduce_with_external_negate =
        [&](const Tensor& h_input, float h_scaler, float h_post_mul, tt::tt_metal::DataType h_out_dtype) {
            // Keep neg_input in h_input's memory config (pass std::nullopt) so the
            // pre-reduce negation stays in place; forcing output_mem_config here
            // could trigger a reshard (DRAM↔L1, interleaved↔sharded) before the
            // H-reduce.  Only the final neg enforces output_mem_config.
            Tensor neg_input = ttnn::neg(h_input, std::nullopt, std::nullopt, sub_core_grids);
            Tensor h_out = ttnn::prim::reduce(
                neg_input,
                reduce_math,
                tt::tt_metal::ReduceOpDim::H,
                h_scaler,
                output_mem_config,
                h_out_dtype,
                config,
                sub_core_grids,
                /*negate=*/false,
                /*post_mul_scaler=*/h_post_mul);
            return ttnn::neg(h_out, output_mem_config, std::nullopt, sub_core_grids);
        };

    // The single-core HW path uses REDUCE_SCALAR mode, which applies the
    // scaler twice internally (once per dimension).  The host compensates with
    // sqrt(scaler) in ReduceSingleCoreHwProgramFactory::create.
    // However, sqrt of a negative number is NaN, so negative scalers
    // must take the two-step W-then-H path where the scaler is applied once.
    //
    // INT32 SFPU reduce has no REDUCE_SCALAR primitive (ROW/COL only), so Int32 HW always uses
    // W-then-H. Float32 max HW can use single-core REDUCE_SCALAR (FPU) when num_tiles == 1;
    // multi-tile HW still uses W-then-H via is_multicore_hw. Applies to MAX/SUM and MIN (MIN via negate).
    // The accurate fp32 SFPU mean (AVG) likewise has no SFPU REDUCE_SCALAR, so it must decompose HW
    // into W-then-H regardless of tile count; forcing the two-step keeps it off the single-core path.
    const bool use_two_step_hw_sfpu_reduce =
        (reduce_dim == tt::tt_metal::ReduceOpDim::HW) &&
        ((tilized_input.dtype() == tt::tt_metal::DataType::INT32 &&
          (reduce_math == tt::tt_metal::ReduceOpMath::MAX || reduce_math == tt::tt_metal::ReduceOpMath::SUM)) ||
         use_sfpu_fp32_mean);

    if (is_multicore_hw || use_two_step_hw_sfpu_reduce ||
        (reduce_dim == tt::tt_metal::ReduceOpDim::HW && reduce_scaler < 0)) {
        // Multi-core HW reduction: first reduce W, then reduce H on the result.
        // Keep the W intermediate in FP32 (only H packs to BF16) to preserve accumulation
        // precision. Applies to SUM only:
        // - FP32 input after an earlier NC-stage reduction with a BF16 final pack (chain path), or
        // - BF16 input on a pure H+W reduction (e.g. dim=[-2,-1] on 8D tensors).
        // MAX/MIN must not use this path: MIN is lowered to MAX + negate, and the fused-negate
        // W step produces wrong results with an FP32 intermediate (issue #40854). They also gain
        // no precision from FP32 since they select, not accumulate.
        const auto out_final_dtype = output_dtype.value_or(input_tensor.dtype());
        const bool keep_w_fp32 =
            reduce_math == tt::tt_metal::ReduceOpMath::SUM &&
            ((output_dtype.has_value() && out_final_dtype == tt::tt_metal::DataType::BFLOAT16 &&
              tilized_input.dtype() == tt::tt_metal::DataType::FLOAT32) ||
             (tilized_input.dtype() == tt::tt_metal::DataType::BFLOAT16 && config.fp32_dest_acc_en));
        const auto out_w_dtype = keep_w_fp32 ? tt::tt_metal::DataType::FLOAT32 : out_final_dtype;

        const Tensor output_tensor = ttnn::prim::reduce(
            tilized_input,
            prim_reduce_math,
            tt::tt_metal::ReduceOpDim::W,
            1.0f,
            output_mem_config,
            out_w_dtype,
            config,
            sub_core_grids,
            negate,
            /*post_mul_scaler=*/1.0f,
            /*row_major_w_dense_path=*/false,
            /*row_major_h_dense_path=*/false,
            /*use_sfpu_reduce=*/use_sfpu_fp32_mean);

        if (negate && !ttnn::prim::h_reduce_negate_fits_in_l1(output_tensor, sub_core_grids)) {
            return h_reduce_with_external_negate(output_tensor, reduce_scaler, post_mul, out_final_dtype);
        }

        return ttnn::prim::reduce(
            output_tensor,
            prim_reduce_math,
            tt::tt_metal::ReduceOpDim::H,
            reduce_scaler,
            output_mem_config,
            out_final_dtype,
            config,
            sub_core_grids,
            negate,
            /*post_mul_scaler=*/post_mul,
            /*row_major_w_dense_path=*/false,
            /*row_major_h_dense_path=*/false,
            /*use_sfpu_reduce=*/use_sfpu_fp32_mean);
    }

    if (negate && reduce_dim == tt::tt_metal::ReduceOpDim::H &&
        !ttnn::prim::h_reduce_negate_fits_in_l1(tilized_input, sub_core_grids)) {
        return h_reduce_with_external_negate(
            tilized_input, reduce_scaler, post_mul, output_dtype.value_or(input_tensor.dtype()));
    }

    // H-axis split (see #46110): the un-split RM-H path uses only NC*Wt cores (one per output tile
    // column) and issues one narrow read per H row per core, so tall-H shapes starve the grid. When
    // that happens, split the reduction axis into S contiguous segments:
    //   stage 1 → (N,C,S,W) partial (pure SUM, FP32 for accumulation accuracy),
    //   stage 2 → collapse the S-row shard axis with the user scaler and final dtype.
    // Exact-sum decomposition: sum over H == sum of per-shard sums, so mean is applied once in stage 2.
    if (use_rm_dense_h) {
        const auto& logical = input_tensor.logical_shape();
        const auto& padded = input_tensor.padded_shape();
        const uint32_t tile_h = input_tensor.tensor_spec().tile().get_height();
        const uint32_t tile_w = input_tensor.tensor_spec().tile().get_width();
        const uint32_t NC = logical[0] * logical[1];
        const uint32_t Wt = (padded[3] + tile_w - 1) / tile_w;
        const uint32_t Ht_rm = (logical[2] + tile_h - 1) / tile_h;
        const uint32_t col_groups = NC * Wt;  // cores the un-split path would use

        const auto grid = input_tensor.device()->compute_with_storage_grid_size();
        const uint32_t grid_cores = sub_core_grids.has_value() ? sub_core_grids->num_cores() : (grid.x * grid.y);

        // Only split genuinely tall reduces: below this the un-split path already performs well and a
        // second stage would just add a dispatch + reduction rounding (small-shape precision noise).
        constexpr uint32_t k_min_ht_for_split = 16;  // ~H >= 512 rows
        uint32_t S = 1;
        if (col_groups > 0 && col_groups < grid_cores && Ht_rm >= k_min_ht_for_split) {
            const uint32_t cand = grid_cores / col_groups;  // shards that keep the grid filled
            S = (cand < Ht_rm) ? cand : Ht_rm;              // never more shards than H tiles
        }

        if (S >= 2) {
            const Tensor partials = ttnn::prim::reduce(
                input_tensor,
                tt::tt_metal::ReduceOpMath::SUM,
                tt::tt_metal::ReduceOpDim::H,
                /*scaler=*/1.0f,
                output_mem_config,
                tt::tt_metal::DataType::FLOAT32,
                config,
                sub_core_grids,
                /*negate=*/false,
                /*post_mul_scaler=*/1.0f,
                /*row_major_w_dense_path=*/false,
                /*row_major_h_dense_path=*/true,
                /*h_num_shards=*/S);

            return ttnn::prim::reduce(
                partials,
                tt::tt_metal::ReduceOpMath::SUM,
                tt::tt_metal::ReduceOpDim::H,
                reduce_scaler,
                output_mem_config,
                output_dtype.value_or(input_tensor.dtype()),
                config,
                sub_core_grids,
                /*negate=*/false,
                /*post_mul_scaler=*/post_mul,
                /*row_major_w_dense_path=*/false,
                /*row_major_h_dense_path=*/true,
                /*h_num_shards=*/1);
        }
    }

    return ttnn::prim::reduce(
        tilized_input,
        prim_reduce_math,
        reduce_dim,
        reduce_scaler,
        output_mem_config,
        output_dtype.value_or(input_tensor.dtype()),
        config,
        sub_core_grids,
        negate,
        /*post_mul_scaler=*/post_mul,
        /*row_major_w_dense_path=*/use_rm_dense_w,
        /*row_major_h_dense_path=*/use_rm_dense_h,
        /*use_sfpu_reduce=*/use_sfpu_fp32_mean);
}

}  // namespace ttnn::operations::reduction::generic::detail
