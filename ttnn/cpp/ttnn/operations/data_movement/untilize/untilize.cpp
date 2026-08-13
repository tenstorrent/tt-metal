// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize.hpp"

#include <tt_stl/assert.hpp>

#include "codegen/untilize_codegen_device_operation.hpp"
#include "codegen/untilize_codegen_supported.hpp"
#include "device/untilize_device_operation.hpp"
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

}  // namespace ttnn::operations::data_movement

namespace ttnn {

ttnn::Tensor untilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::string& implementation) {
    namespace untilize_codegen = operations::data_movement::untilize_codegen;
    // Validate before any early return so invalid values fail consistently.
    const auto sel = untilize_codegen::parse_implementation(implementation);

    if (sel != untilize_codegen::ImplementationSelector::Native) {
        // Execution-control overrides (use_multicore=false, sub_core_grids) select
        // placement/scheduling the codegen builders never honour -- they always
        // dispatch over the full compute_with_storage_grid_size(). Neither "auto"
        // nor "codegen" may silently ignore that; kept out of supported_by_codegen()
        // per the task's own guidance since it is not a correctness/scope question.
        const bool has_override = untilize_codegen::has_execution_control_override(use_multicore, sub_core_grids);
        MemoryConfig output_mem_config = memory_config.value_or(input_tensor.memory_config());

        // A sharded *destination* is served by build_untilize_sharded / build_untilize_i2s, separate
        // builder entry points outside this manifest's scope, and the writer this port does build
        // addresses the destination as one page per logical row -- an identity that fails the moment
        // a ROW_MAJOR page becomes a shard width. Gated here rather than in supported_by_codegen(),
        // which is about the input side, exactly as the merged `repeat` port gates it.
        const bool codegen_output_ok = !output_mem_config.is_sharded();

        if (sel == untilize_codegen::ImplementationSelector::Codegen) {
            TT_FATAL(
                !has_override,
                "untilize: implementation=\"codegen\" does not support use_multicore=false or sub_core_grids "
                "overrides -- codegen always dispatches over the full compute grid");
            TT_FATAL(
                codegen_output_ok && untilize_codegen::supported_by_codegen(input_tensor),
                "untilize: implementation=\"codegen\" requires a supported input and an interleaved output memory "
                "configuration");
            return ttnn::prim::untilize_codegen(input_tensor, ttnn::prim::UntilizeCodegenParams{output_mem_config});
        }

        // Auto: codegen iff supported, not perf-demoted, and no execution-control override; else fall
        // through to the native path below (including its own DRAM+padding redirect).
        if (!has_override && codegen_output_ok && untilize_codegen::supported_by_codegen(input_tensor) &&
            !untilize_codegen::is_demoted(input_tensor)) {
            return ttnn::prim::untilize_codegen(input_tensor, ttnn::prim::UntilizeCodegenParams{output_mem_config});
        }
    }

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

    bool enough_space_height = operations::data_movement::is_enough_space(
        input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_row);

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

}  // namespace ttnn
