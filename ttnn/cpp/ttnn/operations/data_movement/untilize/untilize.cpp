// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize.hpp"

#include "codegen/untilize_codegen_device_operation.hpp"
#include "codegen/untilize_codegen_supported.hpp"
#include "device/untilize_device_operation.hpp"
#include "untilize_force.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {
using OwnedUntilizeArgs = std::tuple<ttnn::Tensor>;
using BaseUntilizeType = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

using MassagedUntilize = MassagedOperation<ttnn::Tensor, const ttnn::Tensor&>;
using MassagedUntilizeParams = MassagedOperationParams<ttnn::Tensor, const ttnn::Tensor&>;

MassagedUntilize build_ndiml_untilize(BaseUntilizeType base_untilize) {
    auto original_shape = std::make_shared<std::pair<ttnn::Shape, ttnn::Shape>>();
    return MassagedUntilize(MassagedUntilizeParams{
        .predicate = [](const ttnn::Tensor& input_tensor) -> bool { return input_tensor.logical_shape().rank() > 4; },
        .pre_transform = [=](const ttnn::Tensor& input_tensor) -> OwnedUntilizeArgs {
            *original_shape = std::make_pair(input_tensor.logical_shape(), input_tensor.padded_shape());
            ttnn::Tensor squeezed_tensor = squeeze_from_ND_to_4D(input_tensor);
            return std::make_tuple(squeezed_tensor);
        },
        .post_transform = [=](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto unsqueezed_tensor = ttnn::reshape(output, original_shape->first, original_shape->second);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_untilize)});
}

// The existing native implementation, unconditionally. Calls the native prim directly rather than
// re-entering ttnn::untilize, so a call that has already been routed to native cannot be routed a
// second time and land on codegen partway through.
ttnn::Tensor untilize_native(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // If the input tensor is not sharded, on DRAM and logical shape != padded shape, then unpad the input tensor.
    // conv op_slicing logic requires the padding information to be present in the input tensor.
    if (!input_tensor.is_sharded() && input_tensor.memory_config().is_dram() &&
        input_tensor.logical_shape() != input_tensor.padded_shape()) {
        ttnn::Shape output_tensor_end(ttsl::SmallVector<uint32_t>(input_tensor.logical_shape().rank(), 0));
        int logical_rank = input_tensor.logical_shape().rank();
        for (int index = -1; index >= -logical_rank; --index) {
            output_tensor_end[index] = input_tensor.logical_shape()[index] - 1;
        }
        return ttnn::untilize_with_unpadding(
            input_tensor, output_tensor_end, memory_config, use_multicore, sub_core_grids);
    }
    bool fp32_dest_acc_en = input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                            input_tensor.dtype() == DataType::FLOAT32;
    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_single_tile_size = input_single_tile_size;

    uint32_t num_tiles_per_row = input_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // Reserve the output buffer's per-core L1 up front: it is allocated after this check
    // but before the CBs are placed, so leaving it out overestimates the CB budget and can
    // pick a factory whose static CBs then clash with it (issue #21358).
    const uint32_t pending_l1_output_bytes = operations::data_movement::get_pending_l1_output_reservation(
        input_tensor,
        input_tensor.padded_shape(),
        memory_config.value_or(input_tensor.memory_config()),
        input_tensor.dtype(),
        Layout::ROW_MAJOR);

    bool enough_space_height = operations::data_movement::is_enough_space(
        input_tensor,
        input_single_tile_size,
        output_single_tile_size,
        num_tiles_per_row,
        /*staging_bytes_per_tile=*/0,
        /*fixed_staging_bytes=*/0,
        pending_l1_output_bytes);

    auto base_untilize = [=](const ttnn::Tensor& input_tensor) {
        auto pf_type = ttnn::operations::data_movement::get_pf_type(
            memory_config.has_value() ? memory_config.value().is_sharded() : input_tensor.is_sharded(), input_tensor);

        return ttnn::prim::untilize(
            input_tensor,
            memory_config.value_or(input_tensor.memory_config()),
            use_multicore,
            fp32_dest_acc_en,
            sub_core_grids,
            enough_space_height,
            pf_type);
    };

    return operations::data_movement::build_ndiml_untilize(base_untilize)(input_tensor);
}

namespace detail {

ttnn::Tensor untilize_force_native(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return untilize_native(input_tensor, memory_config, use_multicore, sub_core_grids);
}

ttnn::Tensor untilize_force_codegen(
    const ttnn::Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    using ttnn::operations::data_movement::untilize_codegen::supported_by_codegen;

    // Normalize exactly as the routed entry does, so a rank>4 input reaches supported_by_codegen and
    // the prim as the squeezed 4D tensor rather than the raw one.
    auto dispatch = [=](const ttnn::Tensor& normalized_input) -> ttnn::Tensor {
        const auto output_mem_config = memory_config.value_or(normalized_input.memory_config());
        TT_FATAL(
            supported_by_codegen(normalized_input, output_mem_config),
            "untilize_force_codegen invoked for a case the codegen prim does not support (requires "
            "TILE-layout, interleaved (non-sharded) input and output, dtype bfloat16 or bfloat8_b "
            "(bfloat8_b additionally requires a tile-aligned logical shape), and a width within the L1 "
            "chunking threshold). This entry never falls back to native, because a forced leg that "
            "quietly served native would make any comparison against native vacuous. Use ttnn::untilize "
            "if you want the case routed.");
        return ttnn::prim::untilize_codegen(normalized_input, output_mem_config);
    };

    return build_ndiml_untilize(dispatch)(input_tensor);
}

}  // namespace detail

}  // namespace ttnn::operations::data_movement

namespace ttnn {

ttnn::Tensor untilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using ttnn::operations::data_movement::untilize_codegen::is_demoted;
    using ttnn::operations::data_movement::untilize_codegen::supported_by_codegen;
    using ttnn::operations::data_movement::untilize_codegen::supported_execution_controls;

    const bool controls_ok = supported_execution_controls(use_multicore, sub_core_grids);

    // Route on the same normalized (squeeze_from_ND_to_4D'd) attributes untilize_native applies
    // via build_ndiml_untilize -- otherwise a logical-rank>4 input reaches supported_by_codegen /
    // is_demoted / the codegen dispatch itself on the raw, un-squeezed tensor, while the native
    // path's equivalent decisions run on the squeezed 4D tensor.
    auto dispatch = [=](const ttnn::Tensor& normalized_input) -> ttnn::Tensor {
        const auto output_mem_config = memory_config.value_or(normalized_input.memory_config());

        if (controls_ok && supported_by_codegen(normalized_input, output_mem_config) &&
            !is_demoted(normalized_input, output_mem_config)) {
            return ttnn::prim::untilize_codegen(normalized_input, output_mem_config);
        }

        return operations::data_movement::untilize_native(
            normalized_input, memory_config, use_multicore, sub_core_grids);
    };

    return operations::data_movement::build_ndiml_untilize(dispatch)(input_tensor);
}

}  // namespace ttnn
