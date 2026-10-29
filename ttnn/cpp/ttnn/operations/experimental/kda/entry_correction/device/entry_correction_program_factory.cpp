// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/entry_correction/device/entry_correction_program_factory.hpp"

#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

namespace {

using tt::tt_metal::CoreRangeSet;

constexpr uint32_t cb_seed = 0;
constexpr uint32_t cb_tail = 1;
constexpr uint32_t cb_m = 2;
constexpr uint32_t cb_base = 3;
constexpr uint32_t cb_out = 4;
constexpr uint32_t cb_a = 5;
constexpr uint32_t cb_b = 6;
constexpr uint32_t cb_final = 7;
constexpr uint32_t cb_ctl = 8;
constexpr uint32_t cb_ctl_writer = 9;
constexpr uint32_t cb_diff = 10;
// On every core: the head's seeds have landed.
constexpr uint32_t seed_semaphore_id = 0;
constexpr const char* kernel_dir = "ttnn/cpp/ttnn/operations/experimental/kda/entry_correction/device/kernels/";

tt::tt_metal::ProgramDescriptor build_program(
    const EntryCorrectionParams& attrs,
    const EntryCorrectionInputs& in,
    const std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinate& coord) {
    auto* mesh = in.final_b.device();
    const auto& output = outputs[0];
    const auto& final_state = outputs[1];
    const uint32_t sp_size = mesh->shape()[attrs.sequence_parallel_axis];
    const uint32_t sp_rank = coord[attrs.sequence_parallel_axis];
    const uint32_t batch_heads = in.final_b.logical_shape()[0];
    const uint32_t kt = in.final_b.logical_shape()[1] / tt::constants::TILE_HEIGHT;
    const uint32_t state_tiles = kt * kt;
    const uint32_t num_chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    const uint32_t cores_per_head = attrs.cores.size() / batch_heads;
    const uint32_t per_core = num_chunks / cores_per_head;
    const uint32_t fp32_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t bf16_tile = tt::constants::TILE_HW * sizeof(uint16_t);
    const CoreRangeSet cores = kda_factory_detail::merged_core_set(attrs.cores);
    const auto& end = in.actual_end.value_or(in.actual_start);

    tt::tt_metal::ProgramDescriptor desc;
    desc.semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = seed_semaphore_id, .core_ranges = cores, .initial_value = 0});
    desc.cbs = {
        kda_factory_detail::cb_descriptor(cb_seed, state_tiles, fp32_tile, tt::DataFormat::Float32, cores),
        kda_factory_detail::cb_descriptor(cb_tail, state_tiles, fp32_tile, tt::DataFormat::Float32, cores),
        kda_factory_detail::sharded_cb_descriptor(cb_m, in.identity_output),
        kda_factory_detail::sharded_cb_descriptor(cb_base, in.zero_output),
        kda_factory_detail::cb_descriptor(cb_diff, per_core * kt, fp32_tile, tt::DataFormat::Float32, cores),
        kda_factory_detail::cb_descriptor(cb_out, 2 * kt, bf16_tile, tt::DataFormat::Float16_b, cores),
        kda_factory_detail::cb_descriptor(cb_a, kt, fp32_tile, tt::DataFormat::Float32, cores),
        kda_factory_detail::cb_descriptor(cb_b, kt, fp32_tile, tt::DataFormat::Float32, cores),
        kda_factory_detail::cb_descriptor(cb_final, 2 * kt, fp32_tile, tt::DataFormat::Float32, cores),
        kda_factory_detail::cb_descriptor(cb_ctl, 1, 64, tt::DataFormat::UInt32, cores),
        kda_factory_detail::cb_descriptor(cb_ctl_writer, 1, 64, tt::DataFormat::UInt32, cores),
    };

    std::vector<uint32_t> reader_ct = {
        cb_seed,
        cb_tail,
        cb_m,
        cb_base,
        cb_a,
        cb_b,
        cb_ctl,
        cb_ctl_writer,
        seed_semaphore_id,
        sp_rank,
        sp_size,
        attrs.local_rows,
        kt,
        static_cast<uint32_t>(in.actual_end.has_value()),
        cores_per_head};
    for (const auto* tensor : {&in.final_a, &in.final_b, &in.entry, &in.tail_entry, &in.actual_start, &end}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = {cb_out, cb_final, cb_ctl_writer, num_chunks, kt};
    for (const auto* tensor : {&output, &final_state}) {
        tt::tt_metal::TensorAccessorArgs(tensor->buffer()).append_to(writer_ct);
    }

    tt::tt_metal::KernelDescriptor reader{
        .kernel_source = std::string(kernel_dir) + "dataflow/reader_entry_correction.cpp",
        .core_ranges = cores,
        .compile_time_args = std::move(reader_ct),
        .config = tt::tt_metal::ReaderConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor writer{
        .kernel_source = std::string(kernel_dir) + "dataflow/writer_entry_correction.cpp",
        .core_ranges = cores,
        .compile_time_args = std::move(writer_ct),
        .config = tt::tt_metal::WriterConfigDescriptor{},
    };
    tt::tt_metal::KernelDescriptor compute{
        .kernel_source = std::string(kernel_dir) + "compute/entry_correction.cpp",
        .core_ranges = cores,
        .compile_time_args = {cb_seed, cb_tail, cb_m, cb_base, cb_diff, cb_out, cb_a, cb_b, cb_final, cb_ctl, kt},
        // The products accumulate in the FP32 DST at the scan's fidelity.
        .config = kda_factory_detail::compute_config_descriptor(mesh->arch(), attrs.compute_kernel_config),
    };

    for (uint32_t index = 0; index < attrs.cores.size(); ++index) {
        const uint32_t head = index / cores_per_head;
        const uint32_t part = index % cores_per_head;
        const uint32_t first = part * per_core;
        const auto& core = attrs.cores[index];
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt;
        for (const auto* tensor : {&in.final_a, &in.final_b, &in.entry, &in.tail_entry, &in.actual_start, &end}) {
            reader_rt.push_back(tensor->buffer());
        }
        // The first tile row of the final state this core forms (kt: none); it forms every cores_per_head-th row on.
        for (uint32_t value :
             {head, first, first + per_core, part < kt ? part : kt, static_cast<uint32_t>(part == 0)}) {
            reader_rt.push_back(value);
        }
        // The head's first core writes the head's seeds to the head's other cores (NoC x | y << 16 each).
        reader_rt.push_back(part == 0 ? cores_per_head - 1 : 0);
        if (part == 0) {
            for (uint32_t other = 1; other < cores_per_head; ++other) {
                const auto noc = mesh->worker_core_from_logical_core(attrs.cores[index + other]);
                reader_rt.push_back(static_cast<uint32_t>(noc.x) | (static_cast<uint32_t>(noc.y) << 16));
            }
        }
        reader.emplace_runtime_args(core, reader_rt);
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt;
        writer_rt.push_back(output.buffer());
        writer_rt.push_back(final_state.buffer());
        writer_rt.push_back(head);
        writer_rt.push_back(first);
        writer.emplace_runtime_args(core, writer_rt);
    }
    desc.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor EntryCorrectionProgramFactory::create_workload_descriptor(
    const EntryCorrectionParams& attrs,
    const EntryCorrectionInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord), build_program(attrs, in, outputs, coord)});
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
