// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/types.hpp"
#include "full_device_operation_types.hpp"
#include "full_program_factory_interleaved.hpp"
#include "full_program_factory_sharded.hpp"
#include "full_program_factory_nd_sharded.hpp"

namespace ttnn::operations::full {

struct FullDeviceOperation {
    using operation_attributes_t = ttnn::operations::full::operation_attributes_t;
    using tensor_args_t = ttnn::operations::full::tensor_args_t;
    using spec_return_value_t = ttnn::operations::full::spec_return_value_t;
    using tensor_return_value_t = ttnn::operations::full::tensor_return_value_t;
    using program_factory_t =
        std::variant<FullInterleavedProgramFactory, FullShardedProgramFactory, FullNDShardedProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::full

namespace ttnn::prim {
ttnn::operations::full::FullDeviceOperation::tensor_return_value_t full(
    ttnn::SmallVector<uint32_t> shape,
    std::variant<float, int> fill_value,
    ttnn::MeshDevice* mesh_device,
    const DataType& dtype,
    const Layout& layout,
    const MemoryConfig& memory_config);
}  // namespace ttnn::prim
