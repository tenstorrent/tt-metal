// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <variant>
#include "ttnn/decorators.hpp"

#include <tt_metal/common/core_coord.hpp>
#include <tt_metal/impl/kernels/kernel_types.hpp>
#include <tt_metal/hostdevcommon/api/hostdevcommon/kernel_structs.h>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
namespace ttnn::operations::generic {

struct circular_buffer_attributes_t {
    CoreRangeSet core_spec;
    uint32_t total_size;
    uint32_t page_size;
    // uint8_t buffer_index;
    tt::DataFormat data_format;

    // this needs better solution as we now have input tensors (std::vector) and output tensor so index is not great
    std::optional<int> set_globally_allocated_address = std::nullopt; // an index to io_tensors that will set globally allocated address on CB
};

struct data_movement_attributes_t {
    CoreRangeSet core_spec;
    std::string kernel_path;
    tt::tt_metal::DataMovementConfig config;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core = {};

    // std::variant<CoreCoord, CoreRange, CoreRangeSet> core_spec;
    // std::shared_ptr<RuntimeArgs> runtime_args;
    // std::vector<std::shared_ptr<RuntimeArgs>> runtime_args;

};

struct compute_attributes_t {
    CoreRangeSet core_spec;
    std::string  kernel_path;
    tt::tt_metal::ComputeConfig config;
    // std::vector<uint32_t> runtime_args = {};
    std::unordered_map<CoreCoord, std::vector<uint32_t>> runtime_args_per_core = {};

};

using cb_attr_map = std::unordered_map<tt::CBIndex, circular_buffer_attributes_t>;
struct GenericOpDeviceOperation {

    struct operation_attributes_t {
        cb_attr_map circular_buffer_attributes;

        std::vector<data_movement_attributes_t> data_movement_attributes;

        // Ethernet not supported at the moment.
        // std::optional<tt::metal::EthernetConfig> ethernetConfig;

        std::vector<compute_attributes_t> compute_attributes;
    };


    // Define the return types for the shape(s) of the operation
    using shape_return_value_t = std::vector<SimpleShape>;

    // Define the return types for the tensor(s) of the operation
    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = Tensor;

    using spec_return_value_t = TensorSpec;
    struct tensor_args_t {
        const Tensor& input_tensor;
        // std::vector<std::reference_wrapper<Tensor>> io_tensors;
        std::vector<Tensor> io_tensors;
        // std::vector<Tensor> input_tensors; // this might not be the best thing?
        // const std::vector<Tensor>& input_tensors;
        // const Tensor& input_tensor;
        // Tensor& output_tensor;
        // reflections assume there are no two params of the same type?
        // note: in instantiation of function template specialization 'reflect::size<ttnn::operations::generic::GenericOpDeviceOperation::tensor_args_t>' requested here
    };

    // Program factories
    struct GenericProgram {
        // to refactor this when we implement caching
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // where/when is this used?
        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<GenericProgram>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor&, const operation_attributes_t&, const std::vector<Tensor>& = {});

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};  // struct GenericOpDeviceOperation

}   // namespace ttnn::operations::generic

namespace ttnn::prim {
constexpr auto generic =
    ttnn::register_operation<"ttnn::prim::generic", ttnn::operations::generic::GenericOpDeviceOperation>();
}  // namespace ttnn::prim
