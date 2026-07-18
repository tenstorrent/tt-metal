// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/experimental/quasar/reduction/generic/device/reduce_op_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reduction/generic/device/common.hpp"

#include <optional>
#include <string>

#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/experimental/quasar/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace reduce_op_utils_qsr {

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

}  // namespace reduce_op_utils_qsr
namespace ttnn::operations::experimental::quasar::generic::detail {

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
        input = ttnn::operations::experimental::quasar::tilize_with_val_padding(
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
    bool use_row_major_support) {
    if (reduce_math == tt::tt_metal::ReduceOpMath::MIN) {
        return reduce_min(input_tensor, reduce_dim, scaler, output_mem_config, compute_kernel_config, sub_core_grids);
    }

    auto parallelization_strategy = ttnn::prim::qsr::get_parallelization_strategy(input_tensor, reduce_dim);
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
    tt::tt_metal::ReduceOpMath prim_reduce_math = reduce_math;
    if (use_rm_dense && reduce_math == tt::tt_metal::ReduceOpMath::AVG) {
        prim_reduce_math = tt::tt_metal::ReduceOpMath::SUM;
    }

    // Reduce only works with tile layout on the classic path; dense path keeps row-major input.
    Tensor tilized_input = input_tensor;
    if (!use_rm_dense) {
        auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
        tilized_input = ttnn::operations::experimental::quasar::tilize_with_val_padding(
            input_tensor, padded_shape, pad_value, input_tensor.memory_config(), std::nullopt, true, sub_core_grids);
    }

    // GMPOOL applies exp2(floor(log2(|s|))) of the scalar (only the exponent), so for
    // MAX/MIN with non-unity scalar we instead reduce with scaler=1.0 and apply the user
    // scalar after reduction via post-multiplication. See issue #40498. The flag also
    // covers reduce_min (math_op=MAX with negate=true) since high-level dispatch lowers
    // min through reduce_min before reaching here.
    //
    // Quasar-specific: the SUM/AVG pool path (GAPOOL) on Quasar also fails to apply the
    // full-precision scaler from the scaler CB — it loses scaler-mantissa precision (a
    // coarse unit-mantissa rounding: e.g. avg_pool2d's 1/49, bf16 1.3047*2^-6, is applied
    // as ~1.5*2^-6 = 3/128, a fixed ~1.15x inflation independent of the input value).
    // WH/BH GAPOOL applies the full scaler correctly (they pass), so restrict the SUM/AVG
    // post-multiplication to Quasar. As with MAX/MIN we reduce with scaler=1.0 (bf16-exact,
    // so GAPOOL sums with no scaler-precision loss) and apply the user scalar afterwards via
    // SFPU post-multiplication (mul_unary_tile, full precision).
    const bool is_quasar = arch == tt::ARCH::QUASAR;
    const bool use_post_mul = (scaler != 1.0f) && ((reduce_math == tt::tt_metal::ReduceOpMath::MAX) ||
                                                   (is_quasar && (reduce_math == tt::tt_metal::ReduceOpMath::SUM ||
                                                                  reduce_math == tt::tt_metal::ReduceOpMath::AVG)));
    const float reduce_scaler = use_post_mul ? 1.0f : scaler;
    const float post_mul = use_post_mul ? scaler : 1.0f;

    // External-negate fallback for the H step: the fused-negate compute kernel is unported on Quasar
    // (negative_tile stub) and removed, so MIN H-reduce always takes this path. Computes
    // -reduce(MAX, H, -x) using the regular reduce kernel.
    auto h_reduce_with_external_negate =
        [&](const Tensor& h_input, float h_scaler, float h_post_mul, tt::tt_metal::DataType h_out_dtype) {
            // Keep neg_input in h_input's memory config (pass std::nullopt) so the
            // pre-reduce negation stays in place; forcing output_mem_config here
            // could trigger a reshard (DRAM↔L1, interleaved↔sharded) before the
            // H-reduce.  Only the final neg enforces output_mem_config.
            Tensor neg_input = ttnn::neg(h_input, std::nullopt, std::nullopt, sub_core_grids);
            Tensor h_out = ttnn::prim::qsr::reduce(
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
    // INT32 SFPU max/min has no REDUCE_SCALAR primitive (ROW/COL only), so Int32 HW always uses
    // W-then-H. Float32 max HW can use single-core REDUCE_SCALAR (FPU) when num_tiles == 1;
    // multi-tile HW still uses W-then-H via is_multicore_hw. Applies to MAX and MIN (MIN via negate).
    const bool use_two_step_hw_sfpu_reduce = (reduce_dim == tt::tt_metal::ReduceOpDim::HW) &&
                                             (tilized_input.dtype() == tt::tt_metal::DataType::INT32) &&
                                             (reduce_math == tt::tt_metal::ReduceOpMath::MAX);

    if (is_multicore_hw || use_two_step_hw_sfpu_reduce ||
        (reduce_dim == tt::tt_metal::ReduceOpDim::HW && reduce_scaler < 0)) {
        // Multi-core HW reduction: first reduce W, then reduce H on the result.
        // For the Sum chain's terminal fp32->bf16 stage, keep W in fp32 so only H packs to bf16.
        const auto out_final_dtype = output_dtype.value_or(input_tensor.dtype());
        // Port of tt-metal 4eed5a8b8c7 (#48578): keep the W intermediate FP32 (only H packs to BF16) to
        // preserve accumulation precision — SUM only. MAX/MIN must NOT (MIN = MAX+negate; the fused-negate
        // W step gives wrong results with an FP32 intermediate, #40854; and select ops gain no precision).
        const bool keep_w_fp32 =
            reduce_math == tt::tt_metal::ReduceOpMath::SUM &&
            ((output_dtype.has_value() && out_final_dtype == tt::tt_metal::DataType::BFLOAT16 &&
              tilized_input.dtype() == tt::tt_metal::DataType::FLOAT32) ||
             (tilized_input.dtype() == tt::tt_metal::DataType::BFLOAT16 && config.fp32_dest_acc_en));
        const auto out_w_dtype = keep_w_fp32 ? tt::tt_metal::DataType::FLOAT32 : out_final_dtype;

        const Tensor output_tensor = ttnn::prim::qsr::reduce(
            tilized_input,
            reduce_math,
            tt::tt_metal::ReduceOpDim::W,
            1.0f,
            output_mem_config,
            out_w_dtype,
            config,
            sub_core_grids,
            negate,
            /*post_mul_scaler=*/1.0f,
            /*row_major_w_dense_path=*/false,
            /*row_major_h_dense_path=*/false);

        if (negate && !ttnn::prim::qsr::h_reduce_negate_fits_in_l1(output_tensor, sub_core_grids)) {
            return h_reduce_with_external_negate(output_tensor, reduce_scaler, post_mul, out_final_dtype);
        }

        return ttnn::prim::qsr::reduce(
            output_tensor,
            reduce_math,
            tt::tt_metal::ReduceOpDim::H,
            reduce_scaler,
            output_mem_config,
            out_final_dtype,
            config,
            sub_core_grids,
            negate,
            /*post_mul_scaler=*/post_mul,
            /*row_major_w_dense_path=*/false,
            /*row_major_h_dense_path=*/false);
    }

    if (negate && reduce_dim == tt::tt_metal::ReduceOpDim::H &&
        !ttnn::prim::qsr::h_reduce_negate_fits_in_l1(tilized_input, sub_core_grids)) {
        return h_reduce_with_external_negate(
            tilized_input, reduce_scaler, post_mul, output_dtype.value_or(input_tensor.dtype()));
    }

    return ttnn::prim::qsr::reduce(
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
        /*row_major_h_dense_path=*/use_rm_dense_h);
}

}  // namespace ttnn::operations::experimental::quasar::generic::detail
