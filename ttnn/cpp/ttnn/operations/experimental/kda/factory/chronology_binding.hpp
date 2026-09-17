// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
namespace ttnn::experimental::prim::kda_factory_detail {
// Rank belongs to mesh placement, not to caller-owned tensor data. The mesh
// adapter retains each coordinate's scalar specialization on program-cache hits.
inline ttnn::device_operation::MeshWorkloadArtifacts chronology_workload(
    const ttnn::device_operation::ProgramArtifacts& artifacts,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::MeshDevice& device,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows) {
    using namespace tt::tt_metal::experimental;
    TT_FATAL(sequence_parallel_axis < device.shape().dims(), "KDA: invalid sequence_parallel_axis");
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& coordinate : tensor_coords.coords()) {
        auto spec = artifacts.spec;
        for (auto& kernel : spec.kernels) {
            if (kernel.unique_id != KernelSpecName{"compute"} && kernel.unique_id != KernelSpecName{"writer"}) {
                kernel.compile_time_args.insert({"sp_rank", coordinate[sequence_parallel_axis]});
                kernel.compile_time_args.insert({"sp_size", device.shape()[sequence_parallel_axis]});
                kernel.compile_time_args.insert({"local_rows", local_rows});
            }
        }
        workload.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coordinate),
            .spec = std::move(spec),
            .run_params = artifacts.run_params,
        });
    }
    return workload;
}

inline void bind_chronology(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run,
    const std::optional<Tensor>& actual_start,
    const Tensor& fallback,
    bool has_writer) {
    using namespace tt::tt_metal::experimental;
    const TensorParamName name{"actual_start"};
    const auto& tensor = (actual_start ? *actual_start : fallback).mesh_tensor();
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
        kernel.compile_time_args.insert({"dynamic_chronology", uint32_t(actual_start.has_value())});
        if (kernel.unique_id == KernelSpecName{"compute"}) {
            kernel.dfb_bindings.push_back(ConsumerOf(cc, "chronology_compute"));
        } else if (kernel.unique_id == KernelSpecName{"writer"}) {
            kernel.dfb_bindings.push_back(ConsumerOf(wc, "chronology_writer"));
        } else {
            kernel.tensor_bindings.push_back({name, "actual_start"});
            kernel.dfb_bindings.push_back(ProducerOf(cc, "chronology_compute"));
            if (has_writer) {
                kernel.dfb_bindings.push_back(ProducerOf(wc, "chronology_writer"));
            }
        }
    }
}
}  // namespace ttnn::experimental::prim::kda_factory_detail
