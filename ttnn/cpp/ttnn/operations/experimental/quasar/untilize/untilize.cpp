// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize.hpp"

#include "device/untilize_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/quasar/untilize_with_unpadding/untilize_with_unpadding.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::quasar {
// Shared data-movement helpers (declared in data_movement/common/common.hpp) used bare by this op;
// they previously resolved via the enclosing data_movement namespace.
using ttnn::operations::data_movement::MassagedOperation;
using ttnn::operations::data_movement::MassagedOperationParams;
using ttnn::operations::data_movement::squeeze_from_ND_to_4D;

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
            auto unsqueezed_tensor =
                ttnn::operations::experimental::quasar::reshape(output, original_shape->first, original_shape->second);
            return unsqueezed_tensor;
        },
        .operation = std::move(base_untilize)});
}

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor untilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // If the input tensor is not sharded and logical shape != padded shape, then unpad the input tensor.
    // conv op_slicing logic requires the padding information to be present in the input tensor.
    if (!input_tensor.is_sharded() && input_tensor.logical_shape() != input_tensor.padded_shape()) {
        ttnn::Shape output_tensor_end(ttsl::SmallVector<uint32_t>(input_tensor.logical_shape().rank(), 0));
        int logical_rank = input_tensor.logical_shape().rank();
        for (int index = -1; index >= -logical_rank; --index) {
            output_tensor_end[index] = input_tensor.logical_shape()[index] - 1;
        }
        return ttnn::operations::experimental::quasar::untilize_with_unpadding(
            input_tensor, output_tensor_end, memory_config, use_multicore, sub_core_grids);
    }

    bool fp32_dest_acc_en = input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                            input_tensor.dtype() == DataType::FLOAT32;

    // The prim emits BFLOAT16 for a BFLOAT8_B input, so size the output CB estimate and the pending
    // output buffer by the output dtype rather than the input's tile size.
    const DataType output_dtype = operations::data_movement::untilize_output_dtype(input_tensor.dtype());
    auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_tiles_per_row = input_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // The interleaved L1 output is allocated after this check but before the CBs are placed; reserve its
    // per-core footprint so the routing doesn't pick a factory whose static CBs clash with it.
    const uint32_t pending_l1_output_bytes = operations::data_movement::get_pending_l1_output_reservation(
        input_tensor,
        input_tensor.padded_shape(),
        memory_config.value_or(input_tensor.memory_config()),
        output_dtype,
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
        auto pf_type = ttnn::operations::experimental::quasar::get_pf_type(
            memory_config.has_value() ? memory_config.value().is_sharded() : input_tensor.is_sharded(), input_tensor);

        return ttnn::prim::qsr::untilize(
            input_tensor,
            memory_config.value_or(input_tensor.memory_config()),
            use_multicore,
            fp32_dest_acc_en,
            sub_core_grids,
            enough_space_height,
            pf_type);
    };

    return build_ndiml_untilize(base_untilize)(input_tensor);
}

}  // namespace ttnn::operations::experimental::quasar
