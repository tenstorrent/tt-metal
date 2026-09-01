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
        case tt::tt_metal::ReduceOpMath::MIN: defines["REDUCE_OP"] = "ckernel::PoolType::MIN"; break;
        case tt::tt_metal::ReduceOpMath::AVG: defines["REDUCE_OP"] = "ckernel::PoolType::AVG"; break;
        default: defines["REDUCE_OP"] = "ckernel::PoolType::SUM"; break;
    }
    defines["REDUCE_DIM"] = reduce_dim_str;
    return defines;
}

// Padding identity for the pre-reduce tilize, mirroring get_pad_value() in generic_reductions.cpp.
static ttnn::PadValue get_tilize_pad_value(tt::tt_metal::ReduceOpMath reduce_math, tt::tt_metal::DataType dtype) {
    if (dtype == tt::tt_metal::DataType::INT32) {
        switch (reduce_math) {
            case tt::tt_metal::ReduceOpMath::MAX: return ttnn::PadValue{uint32_t{0x80000001}};  // INT32_MIN + 1
            case tt::tt_metal::ReduceOpMath::MIN: return ttnn::PadValue{uint32_t{0x7FFFFFFF}};  // INT32_MAX
            default: return ttnn::PadValue{uint32_t{0}};
        }
    }
    return ttnn::PadValue{ttnn::prim::get_reduce_pad_value(reduce_math)};
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
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR && input.storage_type() == ttnn::StorageType::DEVICE) {
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
    bool fast_and_approximate_mode,
    const std::optional<tt::tt_metal::Layout>& output_layout,
    uint32_t reduce_batch_size) {
    TT_FATAL(
        reduce_batch_size == 1 || reduce_dim == tt::tt_metal::ReduceOpDim::H,
        "reduce_batch_size {} (NC grouping) is only defined for an H reduce",
        reduce_batch_size);
    // Only ttnn::sum / ttnn::mean expose output_layout and convert when the device path can't emit it.
    // Checked before the MIN branch, which would drop the request silently.
    TT_FATAL(
        !output_layout.has_value() || reduce_math == tt::tt_metal::ReduceOpMath::SUM ||
            reduce_math == tt::tt_metal::ReduceOpMath::AVG,
        "output_layout is only supported for sum (SUM) and mean (AVG) reductions, got {}",
        reduce_math);

    auto parallelization_strategy = ttnn::prim::get_parallelization_strategy(input_tensor, reduce_dim);
    auto is_multicore_hw = parallelization_strategy == tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_HW;
    const ttnn::PadValue pad_value = reduce_op_utils::get_tilize_pad_value(reduce_math, input_tensor.dtype());

    TT_FATAL(input_tensor.storage_type() == ttnn::StorageType::DEVICE, "Expected input tensor to be on device");
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

    // Dense row-major reduce: a fast path that consumes ROW_MAJOR input directly (no host tilize)
    // and is currently restricted to mean (AVG) / sum (SUM) on 4D BF16/FLOAT32 tensors with
    // interleaved I/O on both sides. Anything else — MAX/MIN, HW reduce, other dtypes, sharded
    // input or output — falls back to the standard tilize + tile-reduce path.
    //
    // MAX/MIN are excluded because the RM compute kernel accumulates partial reductions via
    // Accumulate::at across chunks, and the cross-chunk fold uses SUM semantics. Wiring MAX
    // accumulation through that pipeline is doable but not yet done; for now they tilize.
    const bool both_interleaved =
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    const bool rm_base_eligible =
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR && input_tensor.logical_shape().rank() == 4 &&
        both_interleaved &&
        (input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
         input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32) &&
        (reduce_math == tt::tt_metal::ReduceOpMath::AVG || reduce_math == tt::tt_metal::ReduceOpMath::SUM);

    // A TILE request must be explicit: std::nullopt means "whatever the selected path emits".
    // The RM H path can emit both layouts, so it stays eligible; the RM W path can emit only
    // ROW_MAJOR, so a TILE request sends it back to the tilize + tile-reduce path.
    const bool explicit_tile_request = output_layout == tt::tt_metal::Layout::TILE;
    const bool use_rm_dense_w =
        rm_base_eligible && reduce_dim == tt::tt_metal::ReduceOpDim::W && !explicit_tile_request;
    const bool use_rm_dense_h = rm_base_eligible && reduce_dim == tt::tt_metal::ReduceOpDim::H;
    const bool use_rm_dense = use_rm_dense_w || use_rm_dense_h;

    // Accurate fp32 SFPU path (the FPU truncates fp32 to tf32 on the way into SrcA/SrcB). Falls back
    // to the FPU without fp32_dest_acc_en or on Quasar (no SFPU reduce LLKs). The dense RM kernels
    // share the same SFPU fold via CT arg 6. negate=true marks the FPU's -MAX(-x) min lowering,
    // which has no SFPU fold, so MAX/MIN exclude it.
    const bool fp32_sfpu_eligible = !fast_and_approximate_mode &&
                                    input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 &&
                                    arch != tt::ARCH::QUASAR && config.fp32_dest_acc_en;

    const bool use_sfpu_fp32_sum = fp32_sfpu_eligible && reduce_math == tt::tt_metal::ReduceOpMath::SUM;
    const bool use_sfpu_fp32_mean = fp32_sfpu_eligible && reduce_math == tt::tt_metal::ReduceOpMath::AVG;
    const bool use_sfpu_fp32_max = fp32_sfpu_eligible && reduce_math == tt::tt_metal::ReduceOpMath::MAX && !negate;
    const bool use_sfpu_fp32_min = fp32_sfpu_eligible && reduce_math == tt::tt_metal::ReduceOpMath::MIN && !negate;

    const bool use_sfpu_fp32_reduce = use_sfpu_fp32_sum || use_sfpu_fp32_mean || use_sfpu_fp32_max || use_sfpu_fp32_min;

    // The FPU has no float/bf16 MIN primitive, so fast-mode MIN lowers to -MAX(-x) via the fused
    // negate kernels. Accurate fp32 MIN drives the LLK MIN reduce directly (like Int32 MIN) and must
    // skip that lowering.
    if (reduce_math == tt::tt_metal::ReduceOpMath::MIN && input_tensor.dtype() != tt::tt_metal::DataType::INT32 &&
        !use_sfpu_fp32_min) {
        return reduce_min(input_tensor, reduce_dim, scaler, output_mem_config, compute_kernel_config, sub_core_grids);
    }

    // Layout the dense RM path will be asked to produce; ROW_MAJOR unless TILE was requested.
    const auto rm_dense_out_layout =
        explicit_tile_request ? tt::tt_metal::Layout::TILE : tt::tt_metal::Layout::ROW_MAJOR;

    // High-level mean uses AVG with scaler (1/N). On the tiled path, GMPOOL AVG matches that intent. On the dense
    // row-major W/H path we tilize one logical row at a time from a narrow RM page; AVG applies an extra normalization
    // for full tile faces that does not match torch.mean together with partial-row tilize. Use SUM + the same scaler.
    // SUM, MAX and MIN pass through unchanged.
    // The accurate fp32 SFPU mean also lowers AVG to SUM: the SFPU folds tiles with a plain
    // add (add_binary_tile) and normalizes via the 1/N post-mul below, so it needs SUM semantics.
    tt::tt_metal::ReduceOpMath prim_reduce_math = reduce_math;
    if ((use_rm_dense || use_sfpu_fp32_mean) && reduce_math == tt::tt_metal::ReduceOpMath::AVG) {
        prim_reduce_math = tt::tt_metal::ReduceOpMath::SUM;
    }

    // Reduce only works with tile layout on the classic path; dense path keeps row-major input.
    Tensor prepared_input = input_tensor;
    if (!use_rm_dense) {
        auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
        prepared_input = ttnn::tilize_with_val_padding(
            input_tensor, padded_shape, pad_value, input_tensor.memory_config(), std::nullopt, true, sub_core_grids);
    }

    // A non-unity scalar is applied after the reduction (see requires_post_mul() in common.hpp):
    // GMPOOL keeps only the scaler's exponent for MAX/MIN, and the Int32 SFPU path ignores the
    // scaler CB. Int32 post-mul rounds through fp32, so it is lossy for |result| > 2^24.
    // The accurate fp32 SFPU path also post-muls (the SFPU ignores the scaler CB): mean applies its
    // 1/N here, the rest their user scalar.
    const bool use_post_mul =
        ttnn::prim::requires_post_mul(reduce_math, prepared_input.dtype(), scaler, use_sfpu_fp32_reduce);
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
    // W-then-H. Fast-mode Float32 max HW can use single-core REDUCE_SCALAR (FPU) when num_tiles == 1;
    // multi-tile HW still uses W-then-H via is_multicore_hw. Applies to Int32 MAX/SUM/MIN.
    // The accurate fp32 SFPU path likewise has no SFPU REDUCE_SCALAR, so it decomposes HW into
    // W-then-H regardless of tile count; every op does so exactly (sum of sums, max of maxes, ...).
    const bool use_two_step_hw_sfpu_reduce =
        (reduce_dim == tt::tt_metal::ReduceOpDim::HW) &&
        ((prepared_input.dtype() == tt::tt_metal::DataType::INT32 &&
          (reduce_math == tt::tt_metal::ReduceOpMath::MAX || reduce_math == tt::tt_metal::ReduceOpMath::SUM ||
           reduce_math == tt::tt_metal::ReduceOpMath::MIN)) ||
         use_sfpu_fp32_reduce);

    if (is_multicore_hw || use_two_step_hw_sfpu_reduce ||
        (reduce_dim == tt::tt_metal::ReduceOpDim::HW && reduce_scaler < 0)) {
        // Multi-core HW reduction: first reduce W, then reduce H on the result.
        // Keep the W intermediate in FP32 (only H packs to BF16) to preserve accumulation
        // precision. Applies to SUM only:
        // - FP32 input after an earlier NC-stage reduction with a BF16 final pack (chain path), or
        // - BF16 input on a pure H+W reduction (e.g. dim=[-2,-1] on 8D tensors).
        // MAX/MIN must not use this path: float/bf16 MIN uses -MAX(-x), and the fused-negate
        // W step produces wrong results with an FP32 intermediate (issue #40854). They also gain
        // no precision from FP32 since they select, not accumulate.
        const auto out_final_dtype = output_dtype.value_or(input_tensor.dtype());
        const bool keep_w_fp32 =
            reduce_math == tt::tt_metal::ReduceOpMath::SUM &&
            ((output_dtype.has_value() && out_final_dtype == tt::tt_metal::DataType::BFLOAT16 &&
              prepared_input.dtype() == tt::tt_metal::DataType::FLOAT32) ||
             (prepared_input.dtype() == tt::tt_metal::DataType::BFLOAT16 && config.fp32_dest_acc_en));
        const auto out_w_dtype = keep_w_fp32 ? tt::tt_metal::DataType::FLOAT32 : out_final_dtype;

        const Tensor output_tensor = ttnn::prim::reduce(
            prepared_input,
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
            /*use_sfpu_reduce=*/use_sfpu_fp32_reduce);

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
            /*use_sfpu_reduce=*/use_sfpu_fp32_reduce);
    }

    if (negate && reduce_dim == tt::tt_metal::ReduceOpDim::H &&
        !ttnn::prim::h_reduce_negate_fits_in_l1(prepared_input, sub_core_grids)) {
        return h_reduce_with_external_negate(
            prepared_input, reduce_scaler, post_mul, output_dtype.value_or(input_tensor.dtype()));
    }

    // H-axis split: the un-split RM-H path uses only NC*Wt cores, so tall-H shapes starve the grid.
    // Split the reduction axis into S segments — stage 1 → (N,C,S,W) FP32 partials, stage 2 →
    // collapse the slice axis with the user scaler and final dtype. Sum over H is the sum of the
    // per-slice sums, so mean's 1/N applies once, in stage 2.
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

        // Splitting an Ht_rm below this costs more (extra dispatch + rounding) than it gains.
        constexpr uint32_t k_min_ht_for_split = 16;  // ~H >= 512 rows
        uint32_t num_h_slices = 1;
        // A grouped reduce has nowhere to put per-slice partials — its output row already stands for
        // reduce_batch_size dim-1 slices — so the two are mutually exclusive.
        if (reduce_batch_size == 1 && col_groups > 0 && col_groups < grid_cores && Ht_rm >= k_min_ht_for_split) {
            const uint32_t slices_to_fill_grid = grid_cores / col_groups;  // slices that keep the grid filled
            num_h_slices =
                (slices_to_fill_grid < Ht_rm) ? slices_to_fill_grid : Ht_rm;  // never more slices than H tiles
        }

        if (num_h_slices >= 2) {
            // Stage 1 emits ROW_MAJOR partials; only stage 2 honors the requested layout.
            const Tensor partials = ttnn::prim::reduce(
                prepared_input,
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
                /*use_sfpu_reduce=*/use_sfpu_fp32_reduce,
                /*num_h_slices=*/num_h_slices,
                /*output_layout=*/tt::tt_metal::Layout::ROW_MAJOR);

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
                /*use_sfpu_reduce=*/use_sfpu_fp32_reduce,
                /*num_h_slices=*/1,
                /*output_layout=*/rm_dense_out_layout);
        }
    }

    return ttnn::prim::reduce(
        prepared_input,
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
        /*use_sfpu_reduce=*/use_sfpu_fp32_reduce,
        /*num_h_slices=*/1,
        /*output_layout=*/use_rm_dense ? rm_dense_out_layout : tt::tt_metal::Layout::TILE,
        reduce_batch_size);
}

}  // namespace ttnn::operations::reduction::generic::detail
