// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
namespace ttnn::experimental::prim::kda_factory_detail {
// Rank belongs to mesh placement, not to caller-owned tensor data. The mesh
// adapter retains each coordinate's scalar specialization on program-cache hits.
inline ttnn::device_operation::MeshWorkloadArtifacts chronology_workload(
    const ttnn::device_operation::ProgramArtifacts& artifacts,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::MeshDevice& device,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    const tt::tt_metal::experimental::KernelSpecName& chronology_reader) {
    using namespace tt::tt_metal::experimental;
    TT_FATAL(sequence_parallel_axis < device.shape().dims(), "KDA: invalid sequence_parallel_axis");
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& coordinate : tensor_coords.coords()) {
        auto spec = artifacts.spec;
        auto reader = std::find_if(spec.kernels.begin(), spec.kernels.end(), [&](const auto& kernel) {
            return kernel.unique_id == chronology_reader;
        });
        TT_FATAL(reader != spec.kernels.end(), "KDA: chronology reader is absent from the program");
        reader->compile_time_args.insert({"sp_rank", coordinate[sequence_parallel_axis]});
        reader->compile_time_args.insert({"sp_size", device.shape()[sequence_parallel_axis]});
        reader->compile_time_args.insert({"local_rows", local_rows});
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
    const Tensor& actual_start,
    tt::tt_metal::experimental::KernelSpec& reader,
    tt::tt_metal::experimental::KernelSpec& compute) {
    using namespace tt::tt_metal::experimental;
    const TensorParamName name{"actual_start"};
    const auto& tensor = actual_start.mesh_tensor();
    spec.tensor_parameters.push_back({.unique_id = name, .spec = tensor.tensor_spec()});
    run.tensor_args.emplace(name, tensor);
    const DFBSpecName channel{"chronology_compute"};
    spec.dataflow_buffers.push_back(
        {.unique_id = channel, .entry_size = 32, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt32});
    reader.tensor_bindings.push_back({name, "actual_start"});
    reader.dfb_bindings.push_back(ProducerOf(channel, "chronology_compute"));
    compute.dfb_bindings.push_back(ConsumerOf(channel, "chronology_compute"));
}
}  // namespace ttnn::experimental::prim::kda_factory_detail
