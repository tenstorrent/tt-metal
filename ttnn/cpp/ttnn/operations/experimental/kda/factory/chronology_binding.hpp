// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <initializer_list>
#include <optional>
#include <vector>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/workload/run_args/program_run_args.hpp>
namespace ttnn::experimental::prim::kda_factory_detail {
// How finely chronology_workload splits the mesh into programs.
enum class ChronologyPrograms {
    // One program per tensor range: each device's sp_rank is a per-device run arg.
    per_range,
    // One program per device, for an op whose programs differ beyond sp_rank (its fabric connections).
    per_device,
};

// sp_rank, set on each chronology reader as a common run arg.
inline void set_sp_rank(
    tt::tt_metal::ProgramRunArgs& run,
    std::initializer_list<tt::tt_metal::KernelSpecName> chronology_readers,
    uint32_t sp_rank) {
    for (const auto& chronology_reader : chronology_readers) {
        auto kernel = std::find_if(run.kernel_run_args.begin(), run.kernel_run_args.end(), [&](const auto& args) {
            return args.kernel == chronology_reader;
        });
        if (kernel == run.kernel_run_args.end()) {
            run.kernel_run_args.push_back({.kernel = chronology_reader});
            kernel = std::prev(run.kernel_run_args.end());
        }
        kernel->common_runtime_arg_values["sp_rank"] = sp_rank;
    }
}

// Every device of `range` whose sp_rank differs from the range's first device's, with its sp_rank.
inline std::vector<ttnn::device_operation::DeviceRunArgs> sp_rank_per_device(
    const ttnn::MeshCoordinateRange& range,
    uint32_t sequence_parallel_axis,
    std::initializer_list<tt::tt_metal::KernelSpecName> chronology_readers) {
    std::vector<ttnn::device_operation::DeviceRunArgs> per_device;
    const uint32_t base_rank = range.start_coord()[sequence_parallel_axis];
    for (const auto& coordinate : range) {
        if (coordinate[sequence_parallel_axis] != base_rank) {
            tt::tt_metal::ProgramRunArgs values;
            set_sp_rank(values, chronology_readers, coordinate[sequence_parallel_axis]);
            per_device.push_back({.device = coordinate, .values = std::move(values)});
        }
    }
    return per_device;
}

// Rank belongs to mesh placement, not to caller-owned tensor data: sp_size and local_rows are compile-time args of
// every chronology reader, sp_rank a common run arg set per device. A factory using this declares
// `uses_per_device_run_args` and `fixed_per_device_run_args`: sp_rank depends only on the device, so it is set
// once, at the miss.
inline ttnn::device_operation::MeshWorkloadArtifacts chronology_workload(
    const ttnn::device_operation::ProgramArtifacts& artifacts,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::MeshDevice& device,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    std::initializer_list<tt::tt_metal::KernelSpecName> chronology_readers,
    ChronologyPrograms programs = ChronologyPrograms::per_range) {
    using namespace tt::tt_metal;
    TT_FATAL(sequence_parallel_axis < device.shape().dims(), "KDA: invalid sequence_parallel_axis");
    std::vector<ttnn::MeshCoordinateRange> ranges;
    if (programs == ChronologyPrograms::per_device) {
        for (const auto& coordinate : tensor_coords.coords()) {
            ranges.emplace_back(coordinate);
        }
    } else {
        ranges = tensor_coords.ranges();
    }
    auto spec = artifacts.spec;
    for (const auto& chronology_reader : chronology_readers) {
        auto reader = std::find_if(spec.kernels.begin(), spec.kernels.end(), [&](const auto& kernel) {
            return kernel.unique_id == chronology_reader;
        });
        TT_FATAL(reader != spec.kernels.end(), "KDA: chronology reader is absent from the program");
        reader->compile_time_args.insert({"sp_size", device.shape()[sequence_parallel_axis]});
        reader->compile_time_args.insert({"local_rows", local_rows});
        reader->runtime_arg_schema.common_runtime_arg_names.push_back("sp_rank");
    }
    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& range : ranges) {
        auto run_params = artifacts.run_params;
        set_sp_rank(run_params, chronology_readers, range.start_coord()[sequence_parallel_axis]);
        workload.programs.push_back({
            .range = range,
            .spec = spec,
            .run_params = std::move(run_params),
            .per_device_run_params = sp_rank_per_device(range, sequence_parallel_axis, chronology_readers),
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
    const tt::tt_metal::KernelSpecName& chronology_reader,
    ChronologyPrograms programs = ChronologyPrograms::per_range) {
    return chronology_workload(
        artifacts, tensor_coords, device, sequence_parallel_axis, local_rows, {chronology_reader}, programs);
}

inline void bind_chronology_channel(
    tt::tt_metal::ProgramSpec& spec, tt::tt_metal::KernelSpec& reader, tt::tt_metal::KernelSpec& compute) {
    using namespace tt::tt_metal;
    const DFBSpecName channel{"chronology_compute"};
    spec.dataflow_buffers.push_back(
        {.unique_id = channel, .entry_size = 32, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt32});
    reader.dfb_bindings.push_back(ProducerOf(channel, "chronology_compute"));
    compute.dfb_bindings.push_back(ConsumerOf(channel, "chronology_compute"));
}
inline void bind_chronology(
    tt::tt_metal::ProgramSpec& spec,
    tt::tt_metal::ProgramRunArgs& run,
    const Tensor& actual_start,
    tt::tt_metal::KernelSpec& reader,
    tt::tt_metal::KernelSpec& compute) {
    using namespace tt::tt_metal;
    const TensorParamName name{"actual_start"};
    const auto& tensor = actual_start.mesh_tensor();
    spec.tensor_parameters.push_back({.unique_id = name, .spec = tensor.tensor_spec()});
    run.tensor_args.emplace(name, tensor);
    reader.tensor_bindings.push_back({name, "actual_start"});
    bind_chronology_channel(spec, reader, compute);
}
// Leave actual_start unbound when absent; kernels resolve it with get_token_if_present.
inline void bind_chronology(
    tt::tt_metal::ProgramSpec& spec,
    tt::tt_metal::ProgramRunArgs& run,
    const std::optional<Tensor>& actual_start,
    tt::tt_metal::KernelSpec& reader,
    tt::tt_metal::KernelSpec& compute) {
    if (actual_start) {
        bind_chronology(spec, run, *actual_start, reader, compute);
    } else {
        bind_chronology_channel(spec, reader, compute);
    }
}
inline void bind_actual_end(
    tt::tt_metal::ProgramSpec& spec,
    tt::tt_metal::ProgramRunArgs& run,
    const std::optional<Tensor>& actual_end,
    tt::tt_metal::KernelSpec& reader) {
    using namespace tt::tt_metal;
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
