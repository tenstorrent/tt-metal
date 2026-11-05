// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::tensor_prefetcher {

// Raises one Tensor prefetcher op signal on every device of the mesh: a single data-movement kernel on
// one worker core increments the signal's counter in every DRAM bank. Side effect only; it reads and
// writes no tensor.
struct SignalTensorPrefetcherDeviceOperation {
    struct operation_attributes_t {
        // The signal's L1 address on the DRAM cores (GetTensorPrefetcherSignalAddress). A runtime arg, so
        // it stays out of the program hash; a cache hit rebuilds the descriptor to pick up the new value.
        uint32_t signal_addr = 0;
        // Logical worker core that issues the increments.
        tt::tt_metal::CoreCoord core;
        ttnn::MeshDevice* mesh_device = nullptr;
    };

    struct tensor_args_t {};

    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output);
};

}  // namespace ttnn::operations::experimental::tensor_prefetcher

namespace ttnn::prim {
void signal_tensor_prefetcher(ttnn::MeshDevice* mesh_device, uint32_t signal_addr, const tt::tt_metal::CoreCoord& core);
}  // namespace ttnn::prim
