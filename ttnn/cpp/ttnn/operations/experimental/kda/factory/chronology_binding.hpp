// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
namespace ttnn::experimental::prim::kda_factory_detail {
inline void bind_chronology(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run,
    const std::optional<Tensor>& chronology,
    const Tensor& fallback,
    bool has_writer) {
    using namespace tt::tt_metal::experimental;
    const TensorParamName name{"chronology"};
    const auto& tensor = (chronology ? *chronology : fallback).mesh_tensor();
    spec.tensor_parameters.push_back({.unique_id = name, .spec = tensor.tensor_spec()});
    run.tensor_args.emplace(name, tensor);
    const DFBSpecName cc{"chronology_compute"}, wc{"chronology_writer"};
    spec.dataflow_buffers.push_back(
        {.unique_id = cc, .entry_size = 32, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt32});
    if (has_writer) {
        spec.dataflow_buffers.push_back(
            {.unique_id = wc, .entry_size = 32, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt32});
    }
    for (auto& kernel : spec.kernels) {
        kernel.compile_time_args.insert({"dynamic_chronology", uint32_t(chronology.has_value())});
        if (kernel.unique_id == KernelSpecName{"compute"}) {
            kernel.dfb_bindings.push_back(ConsumerOf(cc, "chronology_compute"));
        } else if (kernel.unique_id == KernelSpecName{"writer"}) {
            kernel.dfb_bindings.push_back(ConsumerOf(wc, "chronology_writer"));
        } else {
            kernel.tensor_bindings.push_back({name, "chronology"});
            kernel.dfb_bindings.push_back(ProducerOf(cc, "chronology_compute"));
            if (has_writer) {
                kernel.dfb_bindings.push_back(ProducerOf(wc, "chronology_writer"));
            }
        }
    }
}
}  // namespace ttnn::experimental::prim::kda_factory_detail
