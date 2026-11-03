// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <initializer_list>
#include <optional>
#include <vector>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
namespace ttnn::experimental::prim::kda_factory_detail {
// How finely chronology_workload splits the mesh into programs.
enum class ChronologyPrograms {
    // One program per SP rank of each tensor range. The programs differ only in sp_rank, so the TP
    // ranks of an SP rank share one.
    per_sp_rank,
    // One program per device, for an op that then gives each program device-specific run args
    // (its fabric connections).
    per_device,
};

// Rank belongs to mesh placement, not to caller-owned tensor data. The mesh
// adapter retains each range's scalar specialization on program-cache hits.
inline ttnn::device_operation::MeshWorkloadArtifacts chronology_workload(
    const ttnn::device_operation::ProgramArtifacts& artifacts,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::MeshDevice& device,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    std::initializer_list<tt::tt_metal::experimental::KernelSpecName> chronology_readers,
    ChronologyPrograms programs = ChronologyPrograms::per_sp_rank) {
    using namespace tt::tt_metal::experimental;
    TT_FATAL(sequence_parallel_axis < device.shape().dims(), "KDA: invalid sequence_parallel_axis");
    std::vector<ttnn::MeshCoordinateRange> ranges;
    if (programs == ChronologyPrograms::per_device) {
        for (const auto& coordinate : tensor_coords.coords()) {
            ranges.emplace_back(coordinate);
        }
    } else {
        // A slice of a range at one SP index is itself a range, covering every TP rank of that SP rank.
        for (const auto& range : tensor_coords.ranges()) {
            const auto first = range.start_coord()[sequence_parallel_axis];
            const auto last = range.end_coord()[sequence_parallel_axis];
            for (uint32_t rank = first; rank <= last; ++rank) {
                std::vector<uint32_t> start(range.start_coord().coords().begin(), range.start_coord().coords().end());
                std::vector<uint32_t> end(range.end_coord().coords().begin(), range.end_coord().coords().end());
                start[sequence_parallel_axis] = end[sequence_parallel_axis] = rank;
                ranges.emplace_back(ttnn::MeshCoordinate(start), ttnn::MeshCoordinate(end));
            }
        }
    }
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& range : ranges) {
        auto spec = artifacts.spec;
        for (const auto& chronology_reader : chronology_readers) {
            auto reader = std::find_if(spec.kernels.begin(), spec.kernels.end(), [&](const auto& kernel) {
                return kernel.unique_id == chronology_reader;
            });
            TT_FATAL(reader != spec.kernels.end(), "KDA: chronology reader is absent from the program");
            reader->compile_time_args.insert({"sp_rank", range.start_coord()[sequence_parallel_axis]});
            reader->compile_time_args.insert({"sp_size", device.shape()[sequence_parallel_axis]});
            reader->compile_time_args.insert({"local_rows", local_rows});
        }
        workload.programs.push_back({
            .range = range,
            .spec = std::move(spec),
            .run_params = artifacts.run_params,
        });
    }
    return workload;
}
inline ttnn::device_operation::MeshWorkloadArtifacts chronology_workload(
    const ttnn::device_operation::ProgramArtifacts& artifacts,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::MeshDevice& device,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    const tt::tt_metal::experimental::KernelSpecName& chronology_reader,
    ChronologyPrograms programs = ChronologyPrograms::per_sp_rank) {
    return chronology_workload(
        artifacts, tensor_coords, device, sequence_parallel_axis, local_rows, {chronology_reader}, programs);
}

inline void bind_chronology_channel(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::KernelSpec& reader,
    tt::tt_metal::experimental::KernelSpec& compute) {
    using namespace tt::tt_metal::experimental;
    const DFBSpecName channel{"chronology_compute"};
    spec.dataflow_buffers.push_back(
        {.unique_id = channel, .entry_size = 32, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt32});
    reader.dfb_bindings.push_back(ProducerOf(channel, "chronology_compute"));
    compute.dfb_bindings.push_back(ConsumerOf(channel, "chronology_compute"));
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
    reader.tensor_bindings.push_back({name, "actual_start"});
    bind_chronology_channel(spec, reader, compute);
}
// Leave actual_start unbound when absent; kernels resolve it with get_token_if_present.
inline void bind_chronology(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run,
    const std::optional<Tensor>& actual_start,
    tt::tt_metal::experimental::KernelSpec& reader,
    tt::tt_metal::experimental::KernelSpec& compute) {
    if (actual_start) {
        bind_chronology(spec, run, *actual_start, reader, compute);
    } else {
        bind_chronology_channel(spec, reader, compute);
    }
}
inline void bind_actual_end(
    tt::tt_metal::experimental::ProgramSpec& spec,
    tt::tt_metal::experimental::ProgramRunArgs& run,
    const std::optional<Tensor>& actual_end,
    tt::tt_metal::experimental::KernelSpec& reader) {
    using namespace tt::tt_metal::experimental;
    reader.compile_time_args.insert({"has_actual_end", uint32_t(actual_end.has_value())});
    // Leave the parameter unbound when absent; kernels resolve it with get_token_if_present.
    if (!actual_end) {
        return;
    }
    const TensorParamName name{"actual_end"};
    const auto& tensor = actual_end->mesh_tensor();
    spec.tensor_parameters.push_back({.unique_id = name, .spec = tensor.tensor_spec()});
    run.tensor_args.emplace(name, tensor);
    reader.tensor_bindings.push_back({name, "actual_end"});
}
}  // namespace ttnn::experimental::prim::kda_factory_detail
