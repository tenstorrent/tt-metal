// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/entry_correction/device/entry_correction_program_factory.hpp"

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/workload/specification/dataflow_buffer_spec.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

ttnn::device_operation::MeshWorkloadArtifacts EntryCorrectionProgramFactory::create_mesh_workload_artifacts(
    const EntryCorrectionParams& attrs,
    const EntryCorrectionInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    namespace m2 = tt::tt_metal;
    const auto& terms = in.terms.mesh_tensor();
    const auto& final_a = in.final_a.mesh_tensor();
    const auto& final_b = in.final_b.mesh_tensor();
    const auto& entry = in.entry.mesh_tensor();
    const auto& tail_entry = in.tail_entry.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& final_state = outputs[1].mesh_tensor();
    const auto& device = final_b.device();

    const uint32_t batch_heads = in.final_b.logical_shape()[0];
    const uint32_t kt = in.final_b.logical_shape()[1] / tt::constants::TILE_HEIGHT;
    const uint32_t state_tiles = kt * kt;
    const uint32_t num_chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    const uint32_t cores_per_head = attrs.cores.size() / batch_heads;
    const uint32_t per_core = num_chunks / cores_per_head;
    const tt::tt_metal::CoreRangeSet cores = kda_factory_detail::merged_core_set(attrs.cores);
    // Sharded terms are read in place; interleaved ones stream a chunk row at a time.
    const bool streamed = !in.terms.memory_config().is_sharded();

    const m2::KernelSpecName reader_kernel{"reader"};
    const m2::KernelSpecName writer_kernel{"writer"};
    const m2::KernelSpecName compute_kernel{"compute"};
    const m2::DFBSpecName seed_dfb{"seed"};
    const m2::SemaphoreSpecName seed_landed{"seed_landed"};
    const m2::DFBSpecName tail_dfb{"tail"};
    const m2::DFBSpecName terms_dfb{"terms"};
    const m2::DFBSpecName diff_dfb{"diff"};
    const m2::DFBSpecName out_dfb{"out"};
    const m2::DFBSpecName a_dfb{"a"};
    const m2::DFBSpecName b_dfb{"b"};
    const m2::DFBSpecName final_dfb{"final_rows"};
    const m2::DFBSpecName ctl_dfb{"ctl"};
    const m2::DFBSpecName ctl_writer_dfb{"ctl_writer"};
    const m2::TensorParamName terms_name{"terms"};
    const m2::TensorParamName final_a_name{"final_a"};
    const m2::TensorParamName final_b_name{"final_b"};
    const m2::TensorParamName entry_name{"entry"};
    const m2::TensorParamName tail_entry_name{"tail_entry"};
    const m2::TensorParamName actual_start_name{"actual_start"};
    const m2::TensorParamName output_name{"output"};
    const m2::TensorParamName final_state_name{"final_state"};

    constexpr auto fp32 = tt::DataFormat::Float32;
    const auto make_dfb = [](const m2::DFBSpecName& name, uint32_t entries, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = entries,
            .data_format_metadata = format,
        };
    };
    // A chunk row of F(0) then of F(I). In place, the core's whole terms shard: its chunk rows of F(0), then of F(I).
    // Streamed, two chunks' rows.
    auto terms_buffer = make_dfb(terms_dfb, (streamed ? 2 : per_core) * 2 * kt, fp32);
    if (!streamed) {
        terms_buffer.borrowed_from = terms_name;
    }
    const auto control = [](const m2::DFBSpecName& name) {
        return m2::DataflowBufferSpec{
            .unique_id = name, .entry_size = 64, .num_entries = 1, .data_format_metadata = tt::DataFormat::UInt32};
    };
    m2::Group<m2::DataflowBufferSpec> dfbs = {
        make_dfb(seed_dfb, state_tiles, fp32),
        make_dfb(tail_dfb, state_tiles, fp32),
        std::move(terms_buffer),
        make_dfb(diff_dfb, (streamed ? 2 : per_core) * kt, fp32),
        make_dfb(out_dfb, 2 * kt, tt::DataFormat::Float16_b),
        make_dfb(a_dfb, kt, fp32),
        make_dfb(b_dfb, kt, fp32),
        make_dfb(final_dfb, 2 * kt, fp32),
        control(ctl_dfb),
        control(ctl_writer_dfb),
    };

    m2::KernelSpec reader{
        .unique_id = reader_kernel,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/entry_correction/device/kernels/dataflow/"
            "reader_entry_correction.cpp",
        .dfb_bindings =
            {
                m2::ProducerOf(seed_dfb, "seed"),
                m2::ProducerOf(tail_dfb, "tail"),
                m2::ProducerOf(terms_dfb, "terms"),
                m2::ProducerOf(a_dfb, "a"),
                m2::ProducerOf(b_dfb, "b"),
                m2::ProducerOf(ctl_dfb, "ctl"),
                m2::ProducerOf(ctl_writer_dfb, "ctl_writer"),
            },
        .semaphore_bindings = {m2::SemaphoreBinding{seed_landed, "seed_landed"}},
        .tensor_bindings =
            {
                m2::TensorBinding{final_a_name, "final_a"},
                m2::TensorBinding{final_b_name, "final_b"},
                m2::TensorBinding{entry_name, "entry"},
                m2::TensorBinding{tail_entry_name, "tail_entry"},
                m2::TensorBinding{actual_start_name, "actual_start"},
                m2::TensorBinding{terms_name, "terms"},
            },
        .compile_time_args =
            {{"kt", kt},
             {"seed_receivers", cores_per_head - 1},
             {"core_chunks", per_core},
             {"final_stride", cores_per_head},
             {"streamed", static_cast<uint32_t>(streamed)}},
        .runtime_arg_schema = {.runtime_arg_names = {"head", "first_chunk", "last_chunk", "final_row", "seed_sender"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    reader.advanced_options.num_runtime_varargs = cores_per_head - 1;
    m2::KernelSpec writer{
        .unique_id = writer_kernel,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/entry_correction/device/kernels/dataflow/"
            "writer_entry_correction.cpp",
        .dfb_bindings =
            {
                m2::ConsumerOf(out_dfb, "out"),
                m2::ConsumerOf(final_dfb, "final_rows"),
                m2::ConsumerOf(ctl_writer_dfb, "ctl"),
            },
        .tensor_bindings =
            {
                m2::TensorBinding{output_name, "output"},
                m2::TensorBinding{final_state_name, "final_state"},
            },
        .compile_time_args = {{"num_chunks", num_chunks}, {"kt", kt}},
        .runtime_arg_schema = {.runtime_arg_names = {"head", "first_chunk"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    auto compute_hw = ttnn::to_compute_hardware_config(attrs.compute_kernel_config);
    for (const auto& name : {seed_dfb, tail_dfb, terms_dfb, diff_dfb, a_dfb, b_dfb}) {
        compute_hw.unpack_modes[name] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }
    m2::KernelSpec compute{
        .unique_id = compute_kernel,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/entry_correction/device/kernels/compute/"
            "entry_correction.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::ConsumerOf(seed_dfb, "seed"),
                m2::ConsumerOf(tail_dfb, "tail"),
                m2::ConsumerOf(terms_dfb, "terms"),
                m2::ProducerOf(diff_dfb, "diff"),
                m2::ConsumerOf(diff_dfb, "diff"),
                m2::ProducerOf(out_dfb, "out"),
                m2::ConsumerOf(a_dfb, "a"),
                m2::ConsumerOf(b_dfb, "b"),
                m2::ProducerOf(final_dfb, "final_rows"),
                m2::ConsumerOf(ctl_dfb, "ctl"),
            },
        .compile_time_args = {{"kt", kt}, {"core_chunks", per_core}, {"streamed", static_cast<uint32_t>(streamed)}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_run{.kernel = reader_kernel};
    m2::KernelRunArgs writer_run{.kernel = writer_kernel};
    for (uint32_t index = 0; index < attrs.cores.size(); ++index) {
        const uint32_t head = index / cores_per_head;
        const uint32_t part = index % cores_per_head;
        const uint32_t first = part * per_core;
        const auto& core = attrs.cores[index];
        // The first final-state tile row this core forms (kt: none), then whether it sends the head's seeds.
        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"head", head},
             {"first_chunk", first},
             {"last_chunk", first + per_core},
             {"final_row", part < kt ? part : kt},
             {"seed_sender", static_cast<uint32_t>(part == 0)}});
        // The seed receivers' NoC x | y << 16, on the head's first core.
        std::vector<uint32_t> receivers(cores_per_head - 1, 0);
        if (part == 0) {
            for (uint32_t other = 1; other < cores_per_head; ++other) {
                const auto noc = device.worker_core_from_logical_core(attrs.cores[index + other]);
                receivers[other - 1] = static_cast<uint32_t>(noc.x) | (static_cast<uint32_t>(noc.y) << 16);
            }
        }
        if (!receivers.empty()) {
            reader_run.advanced_options.runtime_varargs[m2::NodeCoord(core)] = std::move(receivers);
        }
        m2::AddRuntimeArgsForNode(writer_run.runtime_arg_values, core, {{"head", head}, {"first_chunk", first}});
    }

    m2::ProgramSpec spec{
        .name = "entry_correction",
        .dataflow_buffers = std::move(dfbs),
        .semaphores = {m2::SemaphoreSpec{.unique_id = seed_landed, .target_nodes = cores}},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = terms_name, .spec = terms.tensor_spec()},
                m2::TensorParameter{.unique_id = final_a_name, .spec = final_a.tensor_spec()},
                m2::TensorParameter{.unique_id = final_b_name, .spec = final_b.tensor_spec()},
                m2::TensorParameter{.unique_id = entry_name, .spec = entry.tensor_spec()},
                m2::TensorParameter{.unique_id = tail_entry_name, .spec = tail_entry.tensor_spec()},
                m2::TensorParameter{
                    .unique_id = actual_start_name, .spec = in.actual_start.mesh_tensor().tensor_spec()},
                m2::TensorParameter{.unique_id = output_name, .spec = output.tensor_spec()},
                m2::TensorParameter{.unique_id = final_state_name, .spec = final_state.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {reader_kernel, writer_kernel, compute_kernel},
                    .target_nodes = cores,
                },
            },
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {terms_name, terms},
        {final_a_name, final_a},
        {final_b_name, final_b},
        {entry_name, entry},
        {tail_entry_name, tail_entry},
        {actual_start_name, in.actual_start.mesh_tensor()},
        {output_name, output},
        {final_state_name, final_state},
    };
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, reader);
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    return kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.local_rows,
        reader_kernel);
}

}  // namespace ttnn::experimental::prim
