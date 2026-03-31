// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "generic_op_program_factory.hpp"
#include "generic_op_device_operation_types.hpp"

namespace ttnn::operations::generic {

// Post-compilation metadata for a generic_op program.
// Populated by querying the cached, finalized Program object.
struct ProgramCompileInfo {
    uint32_t rta_offset = 0;
    uint32_t sem_offset = 0;
    uint32_t sem_size = 0;
    uint32_t cb_offset = 0;
    uint32_t cb_size = 0;
    uint32_t dfb_offset = 0;
    uint32_t dfb_size = 0;
    uint32_t local_cb_size = 0;
    uint32_t kernel_text_offset = 0;
    uint32_t kernel_text_size = 0;
    std::vector<uint32_t> program_config_sizes;
    std::vector<tt::tt_metal::detail::KernelMeta> kernel_metas;
};

// Query the kernel config layout of a compiled generic_op program.
// Must be called after ttnn.generic_op(io_tensors, program_descriptor) has executed at least once.
ProgramCompileInfo get_program_compile_info(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor);

struct GenericOpDeviceOperation {
    using operation_attributes_t = generic::operation_attributes_t;
    using tensor_args_t = generic::tensor_args_t;
    using spec_return_value_t = generic::spec_return_value_t;
    using tensor_return_value_t = generic::tensor_return_value_t;
    using program_factory_t = std::variant<program::GenericMeshProgramFactory>;

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_inputs(const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Note: will either compute a program hash, or simply return user provided custom program hash
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};  // struct GenericOpDeviceOperation

}  // namespace ttnn::operations::generic

namespace ttnn::prim {
ttnn::operations::generic::tensor_return_value_t generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::operation_attributes_t& operation_attributes);
}  // namespace ttnn::prim
