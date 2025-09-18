// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "typecast_program_factory.hpp"
#include "typecast_sharded_program_factory.hpp"
#include "typecast_device_op_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::copy {

struct TypecastDeviceOperation {
    using operation_attributes_t = copy::operation_attributes_t;
    using tensor_args_t = copy::tensor_args_t;
    using spec_return_value_t = copy::spec_return_value_t;
    using tensor_return_value_t = copy::tensor_return_value_t;

    using program_factory_t = std::variant<
        program::TypecastProgramFactory,
        program::TypecastShardedProgramFactory,
        program::TypecastSubgridProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        DataType output_dtype,
        const MemoryConfig& output_memory_config,
        bool fp32_dest_acc_en,
        bool preserve_fp32_precision,
        bool bfp8_pack_precise,
        const std::optional<Tensor>& preallocated_output,
        const std::optional<CoreRangeSet>& sub_core_grids);
};

}  // namespace ttnn::operations::copy

namespace ttnn::prim {
constexpr auto typecast =
    ttnn::register_operation<"ttnn::prim::typecast", ttnn::operations::copy::TypecastDeviceOperation>();
}  // namespace ttnn::prim
