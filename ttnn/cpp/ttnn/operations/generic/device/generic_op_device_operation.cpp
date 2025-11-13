// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op_device_operation.hpp"

#include <tt_stl/reflection.hpp>

namespace ttnn::operations::generic {

using namespace tt::tt_metal;
GenericOpDeviceOperation::program_factory_t GenericOpDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return GenericProgram{};
}

void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void GenericOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

GenericOpDeviceOperation::spec_return_value_t GenericOpDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // User has to do this. Just referencing last element (preallocated output tensor).
    return tensor_args.output_tensor.tensor_spec();
}

GenericOpDeviceOperation::tensor_return_value_t GenericOpDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Don't create anything, user is passing output tensor.
    return tensor_args.output_tensor;
}

tt::stl::hash::hash_t GenericOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.custom_program_hash) {
        return *operation_attributes.custom_program_hash;
    }

    auto hash_kernel = [&](const KernelDescriptor& kernel) -> size_t {
        size_t hash = std::hash<std::string>()(kernel.kernel_source);
        ttsl::hash::hash_combine(hash, static_cast<size_t>(kernel.source_type));

        ttsl::hash::hash_combine(hash, kernel.core_ranges.size());
        for (const auto& core_range : kernel.core_ranges.ranges()) {
            ttsl::hash::hash_combine(hash, core_range);
        }

        ttsl::hash::hash_combine(hash, kernel.compile_time_args.size());
        for (const auto& compile_time_arg : kernel.compile_time_args) {
            ttsl::hash::hash_combine(hash, compile_time_arg);
        }

        ttsl::hash::hash_combine(hash, kernel.defines.size());
        for (const auto& [key, value] : kernel.defines) {
            ttsl::hash::hash_combine(hash, key);
            ttsl::hash::hash_combine(hash, value);
        }

        ttsl::hash::hash_combine(hash, kernel.common_runtime_args.size());

        ttsl::hash::hash_combine(hash, kernel.runtime_args.size());
        for (const auto& runtime_args_row : kernel.runtime_args) {
            ttsl::hash::hash_combine(hash, runtime_args_row.size());
            for (const auto& core_runtime_args : runtime_args_row) {
                ttsl::hash::hash_combine(hash, core_runtime_args.size());
            }
        }

        size_t hash_config = std::visit(
            tt::stl::overloaded{
                [&](const ReaderConfigDescriptor& reader_config) -> size_t { return 0; },
                [&](const WriterConfigDescriptor& writer_config) -> size_t { return 0; },
                [&](const DataMovementConfigDescriptor& data_movement_config) -> size_t {
                    size_t hash = static_cast<size_t>(data_movement_config.processor);
                    ttsl::hash::hash_combine(hash, static_cast<size_t>(data_movement_config.noc));
                    ttsl::hash::hash_combine(hash, static_cast<size_t>(data_movement_config.noc_mode));
                    return hash;
                },
                [&](const ComputeConfigDescriptor& compute_config) -> size_t {
                    size_t hash = static_cast<size_t>(compute_config.math_fidelity);
                    ttsl::hash::hash_combine(hash, compute_config.fp32_dest_acc_en);
                    ttsl::hash::hash_combine(hash, compute_config.dst_full_sync_en);
                    ttsl::hash::hash_combine(hash, compute_config.bfp8_pack_precise);
                    ttsl::hash::hash_combine(hash, compute_config.math_approx_mode);
                    ttsl::hash::hash_combine(hash, compute_config.unpack_to_dest_mode.size());
                    for (auto unpack_to_dest_mode : compute_config.unpack_to_dest_mode) {
                        ttsl::hash::hash_combine(hash, static_cast<size_t>(unpack_to_dest_mode));
                    }
                    return hash;
                },
                [&](const EthernetConfigDescriptor& ethernet_config) -> size_t {
                    size_t hash = static_cast<size_t>(ethernet_config.eth_mode);
                    ttsl::hash::hash_combine(hash, static_cast<size_t>(ethernet_config.noc));
                    ttsl::hash::hash_combine(hash, static_cast<size_t>(ethernet_config.processor));
                    return hash;
                }},
            kernel.config);
        ttsl::hash::hash_combine(hash, kernel.config.index());
        ttsl::hash::hash_combine(hash, hash_config);
        return hash;
    };

    auto hash_cb_format_descriptor = [&](const CBFormatDescriptor& format_descriptor) -> size_t {
        size_t hash = format_descriptor.buffer_index;
        ttsl::hash::hash_combine(hash, static_cast<size_t>(format_descriptor.data_format));
        ttsl::hash::hash_combine(hash, format_descriptor.page_size);
        return hash;
    };

    auto hash_circular_buffer = [&](const CBDescriptor& cb) -> size_t {
        size_t hash = cb.core_ranges.size();
        for (const auto& core_range : cb.core_ranges.ranges()) {
            ttsl::hash::hash_combine(hash, core_range);
        }
        ttsl::hash::hash_combine(hash, cb.format_descriptors.size());
        for (const auto& format_descriptor : cb.format_descriptors) {
            ttsl::hash::hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        ttsl::hash::hash_combine(hash, cb.remote_format_descriptors.size());
        for (const auto& format_descriptor : cb.remote_format_descriptors) {
            ttsl::hash::hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        ttsl::hash::hash_combine(hash, cb.buffer != nullptr);
        ttsl::hash::hash_combine(hash, cb.global_circular_buffer != nullptr);
        return hash;
    };

    auto hash_semaphore = [&](const SemaphoreDescriptor& semaphore) -> size_t {
        size_t hash = semaphore.core_ranges.size();
        for (const auto& core_range : semaphore.core_ranges.ranges()) {
            ttsl::hash::hash_combine(hash, core_range);
        }
        ttsl::hash::hash_combine(hash, semaphore.core_type);
        ttsl::hash::hash_combine(hash, semaphore.initial_value);
        return hash;
    };

    size_t hash = 0;
    for (const auto& kernel : operation_attributes.kernels) {
        ttsl::hash::hash_combine(hash, hash_kernel(kernel));
    }
    for (const auto& cb : operation_attributes.cbs) {
        ttsl::hash::hash_combine(hash, hash_circular_buffer(cb));
    }
    for (const auto& semaphore : operation_attributes.semaphores) {
        ttsl::hash::hash_combine(hash, hash_semaphore(semaphore));
    }
    return hash;
}

std::tuple<GenericOpDeviceOperation::operation_attributes_t, GenericOpDeviceOperation::tensor_args_t>
GenericOpDeviceOperation::invoke(
    const std::vector<Tensor>& io_tensors, const operation_attributes_t& operation_attributes) {
    TT_FATAL(
        io_tensors.size() >= 2,
        "io_tensors must contain at least one input tensor and one output tensor, got {} tensors.",
        io_tensors.size());

    // NOTE: The output tensor is the last one in the vector, the rest are input tensors
    // Passing in output_tensors into tensor_args_t like this for clarity reasons.
    return {operation_attributes, tensor_args_t{.io_tensors = io_tensors, .output_tensor = io_tensors.back()}};
}

}  // namespace ttnn::operations::generic
