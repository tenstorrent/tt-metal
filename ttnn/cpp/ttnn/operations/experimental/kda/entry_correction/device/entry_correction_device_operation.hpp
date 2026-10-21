// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct EntryCorrectionParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    // The same number of cores per head, head by head: core j of a head takes its j-th share of the chunks and forms
    // every cores_per_head-th tile row of the final state from row j.
    std::vector<tt::tt_metal::CoreCoord> cores;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct EntryCorrectionInputs {
    // Per chunk, the output for a zero seed F(0) and an identity seed F(I), height-sharded over the cores so each
    // core's chunk rows are in its own L1.
    Tensor zero_output;
    Tensor identity_output;
    // final = A @ entry + B.
    Tensor final_a;
    Tensor final_b;
    // The group's entry states, and the split tail's.
    Tensor entry;
    Tensor tail_entry;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
};

struct EntryCorrectionProgramFactory {
    // One program per coordinate: the chronology depends on the sequence-parallel rank.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const EntryCorrectionParams&,
        const EntryCorrectionInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

struct EntryCorrectionOperation {
    using operation_attributes_t = EntryCorrectionParams;
    using tensor_args_t = EntryCorrectionInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<EntryCorrectionProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return EntryCorrectionProgramFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim
