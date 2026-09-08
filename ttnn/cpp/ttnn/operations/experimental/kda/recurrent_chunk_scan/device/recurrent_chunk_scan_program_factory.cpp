// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/recurrent_chunk_scan/device/recurrent_chunk_scan_program_factory.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {
namespace {

struct ScanWorkDistribution {
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> head;
    std::vector<uint32_t> value_block;
    uint32_t value_tiles_per_core = 1;
    tt::tt_metal::CoreRangeSet core_set;
};

ScanWorkDistribution distribute_scan(
    tt::tt_metal::CoreCoord grid, uint32_t batch_heads, uint32_t value_tiles, bool summary) {
    const uint32_t num_cores = grid.x * grid.y;
    TT_FATAL(batch_heads <= num_cores, "KDA recurrent scan heads {} exceed compute cores {}", batch_heads, num_cores);
    uint32_t value_blocks = 1;
    if (!summary) {
        for (uint32_t candidate = value_tiles; candidate >= 1; --candidate) {
            if (value_tiles % candidate == 0 && batch_heads * candidate <= num_cores) {
                value_blocks = candidate;
                break;
            }
        }
    }
    ScanWorkDistribution result;
    result.value_tiles_per_core = value_tiles / value_blocks;
    for (uint32_t index = 0; index < batch_heads * value_blocks; ++index) {
        const tt::tt_metal::CoreCoord core{index % grid.x, index / grid.x};
        result.cores.push_back(core);
        result.head.push_back(index / value_blocks);
        result.value_block.push_back(index % value_blocks);
    }
    result.core_set = tt::tt_metal::num_cores_to_corerangeset(batch_heads * value_blocks, grid, /*row_wise=*/true);
    return result;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts RecurrentChunkScanProgramFactory::create_program_artifacts(
    const RecurrentChunkScanParams& attrs, const RecurrentChunkScanInputs& in, std::vector<Tensor>& outputs) {
    const auto& v_beta_tensor = in.v_beta.mesh_tensor();
    const auto& kd_tensor = in.kd.mesh_tensor();
    const auto& q_decay_tensor = in.q_decay.mesh_tensor();
    const auto& intra_tensor = in.intra.mesh_tensor();
    const auto& k_dec_t_tensor = in.k_dec_t.mesh_tensor();
    const auto& final_decay_tensor = in.final_decay.mesh_tensor();
    const auto& t_inv_tensor = in.t_inv.mesh_tensor();
    const auto& device = v_beta_tensor.device();
    const auto arch = device.arch();

    const uint32_t BH = attrs.batch_heads;
    const uint32_t NC = attrs.num_chunks;
    constexpr uint32_t Ct = 1;
    const uint32_t Kt = attrs.key_dim / tt::constants::TILE_WIDTH;
    const uint32_t Vt_full = attrs.value_dim / tt::constants::TILE_WIDTH;
    const bool summary = attrs.mode == RecurrentChunkScanMode::SUMMARY;
    const auto distribution = distribute_scan(device.compute_with_storage_grid_size(), BH, Vt_full, summary);
    const auto& cores = distribution.core_set;
    const uint32_t Vt = distribution.value_tiles_per_core;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scratch_entries = std::max({cc, ck, cv, kv, kc});

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
    const tt::tt_metal::experimental::DFBSpecName summary_raw_dfb_name{"summary_raw"};
    const tt::tt_metal::experimental::DFBSpecName summary_seed_dfb_name{"summary_seed"};
    const tt::tt_metal::experimental::DFBSpecName summary_ring_dfb_name{"summary_ring"};

    const tt::tt_metal::experimental::TensorParamName v_beta_tensor_name{"v_beta"};
    const tt::tt_metal::experimental::TensorParamName kd_tensor_name{"kd"};
    const tt::tt_metal::experimental::TensorParamName q_decay_tensor_name{"q_decay"};
    const tt::tt_metal::experimental::TensorParamName intra_tensor_name{"intra"};
    const tt::tt_metal::experimental::TensorParamName k_decay_transposed_tensor_name{"k_decay_transposed"};
    const tt::tt_metal::experimental::TensorParamName final_decay_tensor_name{"final_decay"};
    const tt::tt_metal::experimental::TensorParamName t_inv_tensor_name{"t_inv"};
    const tt::tt_metal::experimental::TensorParamName initial_state_tensor_name{"initial_state"};
    const tt::tt_metal::experimental::TensorParamName output_tensor_name{"output"};
    const tt::tt_metal::experimental::TensorParamName final_state_tensor_name{"final_state"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto output_format = summary ? fp32 : tt::DataFormat::Float16_b;
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
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dfbs = {
        make_dfb(state_dfb_name, kv, fp32),
        make_dfb(t_inv_dfb_name, 2 * cc, input_format(in.t_inv)),
        make_dfb(v_beta_dfb_name, 2 * cv, input_format(in.v_beta)),
        make_dfb(kd_dfb_name, 2 * ck, input_format(in.kd)),
        make_dfb(q_decay_dfb_name, summary ? 1 : 2 * ck, summary ? fp32 : input_format(in.q_decay)),
        make_dfb(intra_dfb_name, summary ? 1 : 2 * cc, summary ? fp32 : input_format(in.intra)),
        make_dfb(state_ring_dfb_name, 2 * kv, fp32),
        make_dfb(value_new_dfb_name, cv, fp32),
        make_dfb(final_decay_dfb_name, 2 * Kt, input_format(in.final_decay)),
        make_dfb(output_dfb_name, summary ? kv : 2 * cv, output_format),
        make_dfb(output_intermediate_dfb_name, summary ? 1 : cv, fp32),
        make_dfb(k_decay_transposed_dfb_name, 2 * kc, input_format(in.k_dec_t)),
        make_dfb(state_update_dfb_name, kv, fp32),
        make_dfb(state_temporary_dfb_name, kv, fp32),
        make_dfb(final_state_dfb_name, kv, fp32),
        make_dfb(scratch_dfb_name, scratch_entries, fp32),
        make_dfb(summary_raw_dfb_name, kv, fp32),
        make_dfb(summary_seed_dfb_name, kv, fp32),
        make_dfb(summary_ring_dfb_name, 2 * kv, fp32),
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
                tt::tt_metal::experimental::ProducerOf(summary_seed_dfb_name, "summary_seed"),
                tt::tt_metal::experimental::ProducerOf(k_decay_transposed_dfb_name, "k_decay_transposed"),
                tt::tt_metal::experimental::ProducerOf(final_decay_dfb_name, "final_decay"),
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
             {"summary_pair", static_cast<uint32_t>(summary)}},
        .runtime_arg_schema = {.runtime_arg_names = {"head", "value_block", "num_chunks"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    if (!summary) {
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{q_decay_tensor_name, "q_decay"});
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{intra_tensor_name, "intra"});
        reader.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{initial_state_tensor_name, "initial_state"});
    } else {
        // The discarded recurrence branch is still parsed; aliases provide its binding names without extra parameters.
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{v_beta_tensor_name, "q_decay"});
        reader.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{t_inv_tensor_name, "intra"});
        reader.tensor_bindings.push_back(
            tt::tt_metal::experimental::TensorBinding{v_beta_tensor_name, "initial_state"});
    }

    tt::tt_metal::experimental::KernelSpec writer{
        .unique_id = writer_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/"
            "writer_recurrent_chunk_scan.cpp",
        .dfb_bindings =
            {tt::tt_metal::experimental::ConsumerOf(output_dfb_name, "output"),
             tt::tt_metal::experimental::ConsumerOf(final_state_dfb_name, "final_state")},
        .tensor_bindings =
            {tt::tt_metal::experimental::TensorBinding{output_tensor_name, "output"},
             tt::tt_metal::experimental::TensorBinding{final_state_tensor_name, "final_state"}},
        .compile_time_args =
            {{"Ct", Ct},
             {"Kt", Kt},
             {"Vt", Vt},
             {"Vt_full", Vt_full},
             {"summary_pair", static_cast<uint32_t>(summary)}},
        .runtime_arg_schema = {.runtime_arg_names = {"head", "value_block", "num_chunks"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = tt::tt_metal::experimental::unpack_modes(compute_hw);
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
          summary_raw_dfb_name,
          summary_seed_dfb_name,
          summary_ring_dfb_name}) {
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
                tt::tt_metal::experimental::ProducerOf(scratch_dfb_name, "scratch"),
                tt::tt_metal::experimental::ConsumerOf(scratch_dfb_name, "scratch"),
                tt::tt_metal::experimental::ProducerOf(summary_raw_dfb_name, "summary_raw"),
                tt::tt_metal::experimental::ConsumerOf(summary_raw_dfb_name, "summary_raw"),
                tt::tt_metal::experimental::ConsumerOf(summary_seed_dfb_name, "summary_seed"),
                tt::tt_metal::experimental::ProducerOf(summary_ring_dfb_name, "summary_ring"),
                tt::tt_metal::experimental::ConsumerOf(summary_ring_dfb_name, "summary_ring"),
            },
        .compile_time_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}, {"summary_pair", static_cast<uint32_t>(summary)}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_chunks"}},
        .hw_config = std::move(compute_hw),
    };

    tt::tt_metal::experimental::KernelRunArgs reader_run_args{.kernel = reader_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs writer_run_args{.kernel = writer_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = compute_kernel_name};
    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t head = distribution.head[index];
        const uint32_t value_block = distribution.value_block[index];
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"head", head}, {"value_block", value_block}, {"num_chunks", NC}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"head", head}, {"value_block", value_block}, {"num_chunks", NC}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"num_chunks", NC}});
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
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = final_state_tensor_name, .spec = outputs[1].mesh_tensor().tensor_spec()},
    };
    if (!summary) {
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = q_decay_tensor_name, .spec = q_decay_tensor.tensor_spec()});
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = intra_tensor_name, .spec = intra_tensor.tensor_spec()});
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = initial_state_tensor_name, .spec = in.initial_state->mesh_tensor().tensor_spec()});
    }
    tt::tt_metal::experimental::ProgramSpec spec{
        .name = summary ? "summarize_chunk_recurrence" : "recurrent_chunk_scan",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
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
        {final_state_tensor_name, outputs[1].mesh_tensor()},
    };
    if (!summary) {
        run_args.tensor_args.emplace(q_decay_tensor_name, q_decay_tensor);
        run_args.tensor_args.emplace(intra_tensor_name, intra_tensor);
        run_args.tensor_args.emplace(initial_state_tensor_name, in.initial_state->mesh_tensor());
    }
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
