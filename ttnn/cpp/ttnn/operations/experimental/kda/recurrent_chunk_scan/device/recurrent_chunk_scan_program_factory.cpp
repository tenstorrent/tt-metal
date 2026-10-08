// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"

#include "ttnn/operations/experimental/kda/recurrent_chunk_scan/device/recurrent_chunk_scan_program_factory.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {
ttnn::device_operation::MeshWorkloadArtifacts RecurrentChunkScanProgramFactory::create_mesh_workload_artifacts(
    const RecurrentChunkScanParams& attrs,
    const RecurrentChunkScanInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& v_beta_tensor = in.v_beta.mesh_tensor();
    const auto& kd_tensor = in.kd.mesh_tensor();
    const auto& q_decay_tensor = in.q_decay.mesh_tensor();
    const auto& intra_tensor = in.intra.mesh_tensor();
    const auto& k_dec_t_tensor = in.k_dec_t.mesh_tensor();
    const auto& final_decay_tensor = in.final_decay.mesh_tensor();
    const auto& t_inv_tensor = in.t_inv.mesh_tensor();
    const auto& device = v_beta_tensor.device();

    const uint32_t BH = attrs.batch_heads;
    const uint32_t NC = attrs.num_chunks;
    constexpr uint32_t Ct = 1;
    const uint32_t Kt = attrs.key_dim / tt::constants::TILE_WIDTH;
    const uint32_t Vt_full = attrs.value_dim / tt::constants::TILE_WIDTH;
    const bool summary = attrs.mode == RecurrentChunkScanMode::SUMMARY;
    const bool packed_head = attrs.packed_head;
    const auto distribution =
        kda_factory_detail::distribute_value_blocks(device.compute_with_storage_grid_size(), BH, Vt_full);
    const auto& cores = distribution.core_set;
    const uint32_t Vt = distribution.value_tiles_per_core;
    const uint32_t value_blocks = distribution.value_blocks;
    // A head's value blocks share every V-independent chunk input: value block 0 reads it once and multicasts it.
    const bool mcast_shared = value_blocks > 1;
    // The recurrence multicasts the state update's inputs (k_dec_t and dl) from the writer on the other NoC, in
    // parallel with the reader's; the summary keeps them on the reader, whose writer drains split-head rows.
    const bool late_on_writer = mcast_shared && !summary;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    // The summary advances its zero- and identity-seeded states side by side as one [Kt, 2 * Vt] state.
    const uint32_t paths = summary ? 2 : 1;
    const uint32_t state_tiles = paths * kv;
    const uint32_t scratch_entries = std::max({cc, ck, paths * cv, kv, kc});

    const tt::tt_metal::experimental::KernelSpecName reader_kernel_name{"reader"};
    const tt::tt_metal::experimental::KernelSpecName writer_kernel_name{"writer"};
    const tt::tt_metal::experimental::KernelSpecName compute_kernel_name{"compute"};
    const tt::tt_metal::experimental::DFBSpecName state_dfb_name{"state"};
    const tt::tt_metal::experimental::DFBSpecName t_inv_dfb_name{"t_inv"};
    const tt::tt_metal::experimental::DFBSpecName v_beta_dfb_name{"v_beta"};
    const tt::tt_metal::experimental::DFBSpecName kd_dfb_name{"kd"};
    const tt::tt_metal::experimental::DFBSpecName q_decay_dfb_name{"q_decay"};
    const tt::tt_metal::experimental::DFBSpecName intra_dfb_name{"intra"};
    const tt::tt_metal::experimental::DFBSpecName state_ring_dfb_name{"state_ring"};
    const tt::tt_metal::experimental::DFBSpecName value_new_dfb_name{"value_new"};
    const tt::tt_metal::experimental::DFBSpecName final_decay_dfb_name{"final_decay"};
    const tt::tt_metal::experimental::DFBSpecName output_dfb_name{"output"};
    const tt::tt_metal::experimental::DFBSpecName output_intermediate_dfb_name{"output_intermediate"};
    const tt::tt_metal::experimental::DFBSpecName k_decay_transposed_dfb_name{"k_decay_transposed"};
    const tt::tt_metal::experimental::DFBSpecName state_update_dfb_name{"state_update"};
    const tt::tt_metal::experimental::DFBSpecName state_temporary_dfb_name{"state_temporary"};
    const tt::tt_metal::experimental::DFBSpecName final_state_dfb_name{"final_state"};
    const tt::tt_metal::experimental::DFBSpecName scratch_dfb_name{"scratch"};
    const tt::tt_metal::experimental::DFBSpecName summary_head_output_dfb_name{"summary_head_output"};
    const tt::tt_metal::experimental::DFBSpecName summary_head_state_dfb_name{"summary_head_state"};
    const tt::tt_metal::experimental::DFBSpecName tail_entry_states_dfb_name{"tail_entry_states"};

    const tt::tt_metal::experimental::SemaphoreSpecName ready_semaphore_name{"ready"};
    const tt::tt_metal::experimental::SemaphoreSpecName valid_semaphore_name{"valid"};
    const tt::tt_metal::experimental::SemaphoreSpecName ready_late_semaphore_name{"ready_late"};
    const tt::tt_metal::experimental::SemaphoreSpecName valid_late_semaphore_name{"valid_late"};
    const tt::tt_metal::experimental::TensorParamName v_beta_tensor_name{"v_beta"};
    const tt::tt_metal::experimental::TensorParamName kd_tensor_name{"kd"};
    const tt::tt_metal::experimental::TensorParamName q_decay_tensor_name{"q_decay"};
    const tt::tt_metal::experimental::TensorParamName intra_tensor_name{"intra"};
    const tt::tt_metal::experimental::TensorParamName k_decay_transposed_tensor_name{"k_decay_transposed"};
    const tt::tt_metal::experimental::TensorParamName final_decay_tensor_name{"final_decay"};
    const tt::tt_metal::experimental::TensorParamName t_inv_tensor_name{"t_inv"};
    const tt::tt_metal::experimental::TensorParamName group_entry_states_tensor_name{"group_entry_states"};
    const tt::tt_metal::experimental::TensorParamName tail_entry_states_tensor_name{"tail_entry_states"};
    const tt::tt_metal::experimental::TensorParamName output_tensor_name{"output"};
    const tt::tt_metal::experimental::TensorParamName final_state_tensor_name{"final_state"};
    const tt::tt_metal::experimental::TensorParamName tail_output_tensor_name{"tail_output"};
    const tt::tt_metal::experimental::TensorParamName tail_final_state_tensor_name{"tail_final_state"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto output_format = tt::DataFormat::Float16_b;
    const tt::tt_metal::experimental::DFBSpecName transport_state_dfb_name{"transport_state"};
    const tt::tt_metal::experimental::DFBSpecName identity_scratch_dfb_name{"identity_scratch"};
    const auto input_format = [](const Tensor& tensor) {
        return tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    };
    const auto make_dfb =
        [](const tt::tt_metal::experimental::DFBSpecName& name, uint32_t entries, tt::DataFormat format) {
            return tt::tt_metal::experimental::DataflowBufferSpec{
                .unique_id = name,
                .entry_size = tt::tile_size(format),
                .num_entries = entries,
                .data_format_metadata = format};
        };
    const uint32_t split_head_tiles = summary ? Vt : 1;
    const uint32_t tail_entry_states_tiles = !summary ? kv : 1;
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dfbs = {
        make_dfb(state_dfb_name, state_tiles, fp32),
        make_dfb(t_inv_dfb_name, 2 * cc, input_format(in.t_inv)),
        make_dfb(v_beta_dfb_name, 2 * cv, input_format(in.v_beta)),
        make_dfb(kd_dfb_name, 2 * ck, input_format(in.kd)),
        make_dfb(q_decay_dfb_name, summary ? 1 : 2 * ck, summary ? fp32 : input_format(in.q_decay)),
        make_dfb(intra_dfb_name, summary ? 1 : 2 * cc, summary ? fp32 : input_format(in.intra)),
        make_dfb(state_ring_dfb_name, 2 * state_tiles, fp32),
        make_dfb(value_new_dfb_name, paths * cv, fp32),
        make_dfb(final_decay_dfb_name, 2 * Kt, input_format(in.final_decay)),
        make_dfb(output_dfb_name, summary ? kv : 2 * cv, output_format),
        make_dfb(output_intermediate_dfb_name, summary ? 1 : cv, fp32),
        make_dfb(k_decay_transposed_dfb_name, 2 * kc, input_format(in.k_dec_t)),
        make_dfb(state_update_dfb_name, state_tiles, fp32),
        make_dfb(state_temporary_dfb_name, state_tiles, fp32),
        make_dfb(final_state_dfb_name, state_tiles, fp32),
        make_dfb(transport_state_dfb_name, summary ? kv : 1, tt::DataFormat::Float16_b),
        make_dfb(scratch_dfb_name, scratch_entries, fp32),
        // ProgramSpec names must exist even when if-constexpr discards their
        // users. Give inactive-mode buffers one tile instead of reserving every
        // summary and recurrent restart payload simultaneously.
        make_dfb(summary_head_output_dfb_name, split_head_tiles, output_format),
        make_dfb(summary_head_state_dfb_name, split_head_tiles, output_format),
        make_dfb(tail_entry_states_dfb_name, tail_entry_states_tiles, fp32),
        // The packed head writer's identity tile for ranks without a head.
        make_dfb(identity_scratch_dfb_name, 1, output_format),
    };

    tt::tt_metal::experimental::KernelSpec reader{
        .unique_id = reader_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/"
            "reader_recurrent_chunk_scan.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::ProducerOf(state_dfb_name, "state"),
                tt::tt_metal::experimental::ProducerOf(t_inv_dfb_name, "t_inv"),
                tt::tt_metal::experimental::ProducerOf(v_beta_dfb_name, "v_beta"),
                tt::tt_metal::experimental::ProducerOf(kd_dfb_name, "kd"),
                tt::tt_metal::experimental::ProducerOf(q_decay_dfb_name, "q_decay"),
                tt::tt_metal::experimental::ProducerOf(intra_dfb_name, "intra"),
                tt::tt_metal::experimental::ProducerOf(tail_entry_states_dfb_name, "tail_entry_states"),
            },
        .tensor_bindings =
            {
                tt::tt_metal::experimental::TensorBinding{v_beta_tensor_name, "v_beta"},
                tt::tt_metal::experimental::TensorBinding{kd_tensor_name, "kd"},
                tt::tt_metal::experimental::TensorBinding{k_decay_transposed_tensor_name, "k_decay_transposed"},
                tt::tt_metal::experimental::TensorBinding{final_decay_tensor_name, "final_decay"},
                tt::tt_metal::experimental::TensorBinding{t_inv_tensor_name, "t_inv"},
            },
        .compile_time_args =
            {{"Ct", Ct},
             {"Kt", Kt},
             {"Vt", Vt},
             {"Vt_full", Vt_full},
             {"summary", static_cast<uint32_t>(summary)},
             {"groups_per_head", attrs.groups_per_head},
             {"mcast_shared", static_cast<uint32_t>(mcast_shared)},
             {"late_on_writer", static_cast<uint32_t>(late_on_writer)}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"head", "value_block", "num_chunks", "peer_x0", "peer_y0", "peer_x1", "peer_y1", "receivers"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    reader.semaphore_bindings.push_back(tt::tt_metal::experimental::SemaphoreBinding{ready_semaphore_name, "ready"});
    reader.semaphore_bindings.push_back(tt::tt_metal::experimental::SemaphoreBinding{valid_semaphore_name, "valid"});
    if (!summary) {
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{q_decay_tensor_name, "q_decay"});
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{intra_tensor_name, "intra"});
        reader.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{group_entry_states_tensor_name, "group_entry_states"});
        reader.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{tail_entry_states_tensor_name, "tail_entry_states"});
    } else {
        // The discarded recurrence branch is still parsed; aliases provide its binding names without extra parameters.
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{v_beta_tensor_name, "q_decay"});
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{t_inv_tensor_name, "intra"});
        reader.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{v_beta_tensor_name, "group_entry_states"});
        reader.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{v_beta_tensor_name, "tail_entry_states"});
    }

    tt::tt_metal::experimental::KernelSpec writer{
        .unique_id = writer_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/"
            "writer_recurrent_chunk_scan.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::ConsumerOf(output_dfb_name, "output"),
                tt::tt_metal::experimental::ConsumerOf(
                    summary ? transport_state_dfb_name : final_state_dfb_name, "final_state"),
                tt::tt_metal::experimental::ConsumerOf(summary_head_output_dfb_name, "summary_head_output"),
                tt::tt_metal::experimental::ConsumerOf(summary_head_state_dfb_name, "summary_head_state"),
                tt::tt_metal::experimental::ConsumerOf(
                    summary ? final_state_dfb_name : transport_state_dfb_name, "unused_state"),
                tt::tt_metal::experimental::ProducerOf(identity_scratch_dfb_name, "identity_scratch"),
                tt::tt_metal::experimental::ConsumerOf(identity_scratch_dfb_name, "identity_scratch"),
            },
        // The packed head is the only output; its writer still parses the other output names.
        .tensor_bindings =
            {tt::tt_metal::experimental::TensorBinding{output_tensor_name, "output"},
             tt::tt_metal::experimental::TensorBinding{
                 packed_head ? output_tensor_name : final_state_tensor_name, "final_state"}},
        .compile_time_args =
            {{"Ct", Ct},
             {"Kt", Kt},
             {"Vt", Vt},
             {"Vt_full", Vt_full},
             {"summary", static_cast<uint32_t>(summary)},
             {"packed_head", static_cast<uint32_t>(packed_head)},
             {"late_on_writer", static_cast<uint32_t>(late_on_writer)}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"head",
                  "value_block",
                  "num_chunks",
                  "group",
                  "peer_x0",
                  "peer_y0",
                  "peer_x1",
                  "peer_y1",
                  "receivers"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // The state update's inputs come from whichever kernel multicasts them.
    auto& late_producer = late_on_writer ? writer : reader;
    late_producer.dfb_bindings.push_back(
        tt::tt_metal::experimental::ProducerOf(k_decay_transposed_dfb_name, "k_decay_transposed"));
    late_producer.dfb_bindings.push_back(tt::tt_metal::experimental::ProducerOf(final_decay_dfb_name, "final_decay"));
    writer.tensor_bindings.push_back(
        tt::tt_metal::experimental::TensorBinding{k_decay_transposed_tensor_name, "k_decay_transposed"});
    writer.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{final_decay_tensor_name, "final_decay"});
    writer.semaphore_bindings.push_back(
        tt::tt_metal::experimental::SemaphoreBinding{ready_late_semaphore_name, "ready_late"});
    writer.semaphore_bindings.push_back(
        tt::tt_metal::experimental::SemaphoreBinding{valid_late_semaphore_name, "valid_late"});

    if (packed_head) {
        writer.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{output_tensor_name, "tail_output"});
        writer.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{output_tensor_name, "tail_final_state"});
    } else if (summary) {
        writer.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{tail_output_tensor_name, "tail_output"});
        writer.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{tail_final_state_tensor_name, "tail_final_state"});
    } else {
        // Parsed even when if-constexpr discards segmented summary emission.
        writer.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{output_tensor_name, "tail_output"});
        writer.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{final_state_tensor_name, "tail_final_state"});
    }

    auto compute_hw = ttnn::to_compute_hardware_config(attrs.compute_kernel_config);
    auto& unpack_modes = compute_hw.unpack_modes;
    for (const auto& name :
         {state_dfb_name,
          t_inv_dfb_name,
          v_beta_dfb_name,
          kd_dfb_name,
          q_decay_dfb_name,
          intra_dfb_name,
          state_ring_dfb_name,
          value_new_dfb_name,
          final_decay_dfb_name,
          output_dfb_name,
          output_intermediate_dfb_name,
          k_decay_transposed_dfb_name,
          state_update_dfb_name,
          state_temporary_dfb_name,
          final_state_dfb_name,
          scratch_dfb_name,
          summary_head_output_dfb_name,
          summary_head_state_dfb_name,
          tail_entry_states_dfb_name}) {
        unpack_modes[name] = tt::tt_metal::UnpackMode::UnpackToSrc;
    }
    tt::tt_metal::experimental::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/compute/"
            "recurrent_chunk_scan.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::ConsumerOf(state_dfb_name, "state"),
                tt::tt_metal::experimental::ConsumerOf(t_inv_dfb_name, "t_inv"),
                tt::tt_metal::experimental::ConsumerOf(v_beta_dfb_name, "v_beta"),
                tt::tt_metal::experimental::ConsumerOf(kd_dfb_name, "kd"),
                tt::tt_metal::experimental::ConsumerOf(q_decay_dfb_name, "q_decay"),
                tt::tt_metal::experimental::ConsumerOf(intra_dfb_name, "intra"),
                tt::tt_metal::experimental::ProducerOf(state_ring_dfb_name, "state_ring"),
                tt::tt_metal::experimental::ConsumerOf(state_ring_dfb_name, "state_ring"),
                tt::tt_metal::experimental::ProducerOf(value_new_dfb_name, "value_new"),
                tt::tt_metal::experimental::ConsumerOf(value_new_dfb_name, "value_new"),
                tt::tt_metal::experimental::ConsumerOf(final_decay_dfb_name, "final_decay"),
                tt::tt_metal::experimental::ProducerOf(output_dfb_name, "output"),
                tt::tt_metal::experimental::ProducerOf(output_intermediate_dfb_name, "output_intermediate"),
                tt::tt_metal::experimental::ConsumerOf(output_intermediate_dfb_name, "output_intermediate"),
                tt::tt_metal::experimental::ConsumerOf(k_decay_transposed_dfb_name, "k_decay_transposed"),
                tt::tt_metal::experimental::ProducerOf(state_update_dfb_name, "state_update"),
                tt::tt_metal::experimental::ConsumerOf(state_update_dfb_name, "state_update"),
                tt::tt_metal::experimental::ProducerOf(state_temporary_dfb_name, "state_temporary"),
                tt::tt_metal::experimental::ConsumerOf(state_temporary_dfb_name, "state_temporary"),
                tt::tt_metal::experimental::ProducerOf(final_state_dfb_name, "final_state"),
                tt::tt_metal::experimental::ProducerOf(transport_state_dfb_name, "transport_state"),
                tt::tt_metal::experimental::ProducerOf(scratch_dfb_name, "scratch"),
                tt::tt_metal::experimental::ConsumerOf(scratch_dfb_name, "scratch"),
                tt::tt_metal::experimental::ProducerOf(summary_head_output_dfb_name, "summary_head_output"),
                tt::tt_metal::experimental::ProducerOf(summary_head_state_dfb_name, "summary_head_state"),
                tt::tt_metal::experimental::ConsumerOf(tail_entry_states_dfb_name, "tail_entry_states"),
            },
        .compile_time_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}, {"summary", static_cast<uint32_t>(summary)}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_chunks", "group"}},
        .hw_config = std::move(compute_hw),
    };
    tt::tt_metal::experimental::KernelRunArgs reader_run_args{.kernel = reader_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs writer_run_args{.kernel = writer_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = compute_kernel_name};
    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t head = distribution.head[index];
        const uint32_t value_block = distribution.value_block[index];
        const uint32_t group = head % attrs.groups_per_head;
        // The sender (value block 0) addresses its siblings' row segment; receivers address the sender. The
        // reader runs on NoC 0, so the segment starts at its lowest coordinate.
        uint32_t peer_x0 = 0;
        uint32_t peer_y0 = 0;
        uint32_t peer_x1 = 0;
        uint32_t peer_y1 = 0;
        if (mcast_shared) {
            const uint32_t sender_index = index - value_block;
            const auto first = device.worker_core_from_logical_core(
                distribution.cores[value_block == 0 ? sender_index + 1 : sender_index]);
            const auto last = device.worker_core_from_logical_core(distribution.cores[sender_index + value_blocks - 1]);
            peer_x0 = first.x;
            peer_y0 = first.y;
            peer_x1 = last.x;
            peer_y1 = last.y;
        }
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"head", head},
             {"value_block", value_block},
             {"num_chunks", NC},
             {"peer_x0", peer_x0},
             {"peer_y0", peer_y0},
             {"peer_x1", peer_x1},
             {"peer_y1", peer_y1},
             {"receivers", value_blocks - 1}});
        // The writer multicasts on NoC 1, so its segment starts at the highest coordinate; receivers address the
        // sender.
        const bool sender = value_block == 0;
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"head", head},
             {"value_block", value_block},
             {"num_chunks", NC},
             {"group", group},
             {"peer_x0", sender ? peer_x1 : peer_x0},
             {"peer_y0", sender ? peer_y1 : peer_y0},
             {"peer_x1", sender ? peer_x0 : peer_x1},
             {"peer_y1", sender ? peer_y0 : peer_y1},
             {"receivers", value_blocks - 1}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"num_chunks", NC}, {"group", group}});
    }

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::TensorParameter> tensor_parameters = {
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = v_beta_tensor_name, .spec = v_beta_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = kd_tensor_name, .spec = kd_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = k_decay_transposed_tensor_name, .spec = k_dec_t_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = final_decay_tensor_name, .spec = final_decay_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = t_inv_tensor_name, .spec = t_inv_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = output_tensor_name, .spec = outputs[0].mesh_tensor().tensor_spec()},
    };
    if (!packed_head) {
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = final_state_tensor_name, .spec = outputs[1].mesh_tensor().tensor_spec()});
    }
    if (summary && !packed_head) {
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = tail_output_tensor_name, .spec = outputs[2].mesh_tensor().tensor_spec()});
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = tail_final_state_tensor_name, .spec = outputs[3].mesh_tensor().tensor_spec()});
    }
    if (!summary) {
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = q_decay_tensor_name, .spec = q_decay_tensor.tensor_spec()});
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = intra_tensor_name, .spec = intra_tensor.tensor_spec()});
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = group_entry_states_tensor_name, .spec = in.group_entry_states->mesh_tensor().tensor_spec()});
        {
            tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
                .unique_id = tail_entry_states_tensor_name, .spec = in.tail_entry_states->mesh_tensor().tensor_spec()});
        }
    }

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = summary ? "summarize_chunk_recurrence" : "recurrent_chunk_scan",
        .dataflow_buffers = std::move(dfbs),
        .semaphores =
            {
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = ready_semaphore_name, .target_nodes = cores},
                tt::tt_metal::experimental::SemaphoreSpec{.unique_id = valid_semaphore_name, .target_nodes = cores},
                tt::tt_metal::experimental::SemaphoreSpec{
                    .unique_id = ready_late_semaphore_name, .target_nodes = cores},
                tt::tt_metal::experimental::SemaphoreSpec{
                    .unique_id = valid_late_semaphore_name, .target_nodes = cores},
            },
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {tt::tt_metal::experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {reader_kernel_name, writer_kernel_name, compute_kernel_name},
            .target_nodes = cores}},
    };
    tt::tt_metal::experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {v_beta_tensor_name, v_beta_tensor},
        {kd_tensor_name, kd_tensor},
        {k_decay_transposed_tensor_name, k_dec_t_tensor},
        {final_decay_tensor_name, final_decay_tensor},
        {t_inv_tensor_name, t_inv_tensor},
        {output_tensor_name, outputs[0].mesh_tensor()},
    };
    if (!packed_head) {
        run_args.tensor_args.emplace(final_state_tensor_name, outputs[1].mesh_tensor());
    }
    if (!summary) {
        run_args.tensor_args.emplace(q_decay_tensor_name, q_decay_tensor);
        run_args.tensor_args.emplace(intra_tensor_name, intra_tensor);
        run_args.tensor_args.emplace(group_entry_states_tensor_name, in.group_entry_states->mesh_tensor());
        run_args.tensor_args.emplace(tail_entry_states_tensor_name, in.tail_entry_states->mesh_tensor());
    }

    if (summary && !packed_head) {
        run_args.tensor_args.emplace(tail_output_tensor_name, outputs[2].mesh_tensor());
        run_args.tensor_args.emplace(tail_final_state_tensor_name, outputs[3].mesh_tensor());
    }
    kda_factory_detail::bind_chronology(spec, run_args, in.actual_start, reader, compute);
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, reader);
    // The writer reads the chronology channel on the same condition as the reader publishes it.
    writer.compile_time_args.insert({"has_actual_end", static_cast<uint32_t>(in.actual_end.has_value())});
    if (summary || in.actual_end.has_value()) {
        const tt::tt_metal::experimental::DFBSpecName writer_chronology{"chronology_writer"};
        spec.dataflow_buffers.push_back({
            .unique_id = writer_chronology,
            .entry_size = 32,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
        });
        reader.dfb_bindings.push_back(tt::tt_metal::experimental::ProducerOf(writer_chronology, "chronology_writer"));
        writer.dfb_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(writer_chronology, "chronology_writer"));
    }
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    return kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.num_chunks * attrs.groups_per_head * tt::constants::TILE_HEIGHT,
        reader_kernel_name);
}

}  // namespace ttnn::experimental::prim
