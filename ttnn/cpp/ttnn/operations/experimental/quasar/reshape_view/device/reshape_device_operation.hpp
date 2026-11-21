// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_device_operation_types.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/device/reshape_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim::qsr {

struct ReshapeViewDeviceOperation {
    using operation_attributes_t = ReshapeViewParams;
    using tensor_args_t = ReshapeViewInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ReshapeViewRowMajorProgramFactory, ReshapeViewTiledProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

// Reshape `input` (rank 2 for ROW_MAJOR, rank 3 for TILE: the views the host entry builds) into
// `logical_output_shape` of the same rank and volume. `num_threads` DM harts per node move the data.
ttnn::Tensor reshape_view(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const ttnn::Shape& padded_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grid,
    uint32_t num_threads = RESHAPE_MAX_DM_THREADS);

// The default hart count, or the TTNN_QSR_RESHAPE_THREADS override (experiments only).
uint32_t reshape_threads_from_env();

}  // namespace ttnn::prim::qsr
