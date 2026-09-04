// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/operation.hpp"

#include "recurrent_chunk_scan_device_operation_types.hpp"
#include "recurrent_chunk_scan_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RecurrentChunkScanOperation {
    using operation_attributes_t = RecurrentChunkScanParams;
    using tensor_args_t = RecurrentChunkScanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RecurrentChunkScanProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> recurrent_chunk_scan(
    const Tensor& v_beta,
    const Tensor& kd,
    const Tensor& q_decay,
    const Tensor& intra,
    const Tensor& k_dec_t,
    const Tensor& final_decay,
    const Tensor& t_inv,
    const std::optional<Tensor>& initial_state,
    RecurrentChunkScanMode mode,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::experimental::prim
