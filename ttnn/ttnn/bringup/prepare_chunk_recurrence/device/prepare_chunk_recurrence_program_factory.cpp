// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"

#include "prepare_chunk_recurrence_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/workload/specification/dataflow_buffer_spec.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim::bringup {
namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::MeshWorkloadArtifacts PrepareChunkRecurrenceProgramFactory::create_mesh_workload_artifacts(
    const PrepareChunkRecurrenceParams& attrs,
    const PrepareChunkRecurrenceInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& q = in.q.mesh_tensor();
    const auto& k = in.k.mesh_tensor();
    const auto& v = in.v.mesh_tensor();
    const auto& g = in.g.mesh_tensor();
    const auto& beta = in.beta.mesh_tensor();
    const auto& device = q.device();

    const uint32_t num_heads = attrs.num_heads;
    const uint32_t num_chunks = attrs.num_chunks;
    constexpr uint32_t Ct = 1;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scratch = std::max({cc, ck, cv, kv, kc});
    const auto distribution = kda_factory_detail::distribute_prep(
        device.compute_with_storage_grid_size(), num_heads * num_chunks, std::numeric_limits<uint32_t>::max());
    const auto& cores = distribution.core_set;

    const tt::tt_metal::KernelSpecName READER{"reader"};
    const tt::tt_metal::KernelSpecName WRITER{"writer"};
    const tt::tt_metal::KernelSpecName COMPUTE{"compute"};
    const tt::tt_metal::DFBSpecName q_dfb{"q"};
    const tt::tt_metal::DFBSpecName k_dfb{"k"};
    const tt::tt_metal::DFBSpecName v_dfb{"v"};
    const tt::tt_metal::DFBSpecName g_dfb{"g"};
    const tt::tt_metal::DFBSpecName beta_dfb{"beta"};
    // Sigmoid of the beta logits.
    const tt::tt_metal::DFBSpecName beta_activated_dfb{"beta_activated"};
    const tt::tt_metal::DFBSpecName eye_dfb{"eye"};
    const tt::tt_metal::DFBSpecName tril_dfb{"tril"};
    const tt::tt_metal::DFBSpecName ones_dfb{"ones"};
    const tt::tt_metal::DFBSpecName gate_tril_dfb{"gate_tril"};
    const tt::tt_metal::DFBSpecName gate_ones_dfb{"gate_ones"};
    const tt::tt_metal::DFBSpecName block_masks_dfb{"block_masks"};
    const tt::tt_metal::DFBSpecName workspace_0_dfb{"workspace_0"};
    const tt::tt_metal::DFBSpecName centered_inverse_decay_dfb{"centered_inverse_decay"};
    const tt::tt_metal::DFBSpecName akk_dfb{"akk"};
    const tt::tt_metal::DFBSpecName t_inv_dfb{"t_inv"};
    const tt::tt_metal::DFBSpecName v_beta_dfb{"v_beta"};
    const tt::tt_metal::DFBSpecName kd_dfb{"kd"};
    const tt::tt_metal::DFBSpecName q_decay_dfb{"q_decay"};
    const tt::tt_metal::DFBSpecName intra_dfb{"intra"};
    const tt::tt_metal::DFBSpecName workspace_1_dfb{"workspace_1"};
    const tt::tt_metal::DFBSpecName final_decay_dfb{"final_decay"};
    const tt::tt_metal::DFBSpecName k_decay_transposed_dfb{"k_decay_transposed"};
    const tt::tt_metal::DFBSpecName anchor_decay_dfb{"anchor_decay"};
    const tt::tt_metal::DFBSpecName normalized_q_dfb{"normalized_q"};
    const tt::tt_metal::DFBSpecName normalized_k_dfb{"normalized_k"};
    const tt::tt_metal::DFBSpecName tile_workspace_0_dfb{"tile_workspace_0"};
    const tt::tt_metal::DFBSpecName tile_workspace_1_dfb{"tile_workspace_1"};
    const tt::tt_metal::DFBSpecName tile_workspace_2_dfb{"tile_workspace_2"};

    const tt::tt_metal::DFBSpecName workspace_3_dfb{"workspace_3"};
    const tt::tt_metal::DFBSpecName workspace_2_dfb{"workspace_2"};
    // precise_gate_factors: masks scale*([j<=i] - 1/2), its negation and scale/2, filled once by the reader
    const tt::tt_metal::DFBSpecName gate_center_dfb{"gate_center"};
    const tt::tt_metal::DFBSpecName gate_center_neg_dfb{"gate_center_neg"};
    const tt::tt_metal::DFBSpecName gate_half_dfb{"gate_half"};
    const tt::tt_metal::TensorParamName Q_TENSOR{"q"};
    const tt::tt_metal::TensorParamName K_TENSOR{"k"};
    const tt::tt_metal::TensorParamName V_TENSOR{"v"};
    const tt::tt_metal::TensorParamName G_TENSOR{"g"};
    const tt::tt_metal::TensorParamName BETA_TENSOR{"beta"};
    const tt::tt_metal::TensorParamName V_BETA_OUTPUT{"v_beta_output"};
    const tt::tt_metal::TensorParamName KD_OUTPUT{"kd_output"};
    const tt::tt_metal::TensorParamName Q_DECAY_OUTPUT{"q_decay_output"};
    const tt::tt_metal::TensorParamName INTRA_OUTPUT{"intra_output"};
    const tt::tt_metal::TensorParamName K_DECAY_TRANSPOSED_OUTPUT{"k_decay_transposed_output"};
    const tt::tt_metal::TensorParamName FINAL_DECAY_OUTPUT{"final_decay_output"};
    const tt::tt_metal::TensorParamName T_INV_OUTPUT{"t_inv_output"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto gate_format = datatype_to_dataformat_converter(in.g.dtype());
    std::vector<tt::DataFormat> output_formats;
    output_formats.reserve(outputs.size());
    for (const auto& output : outputs) {
        output_formats.push_back(datatype_to_dataformat_converter(output.dtype()));
    }
    auto make_dfb = [](const tt::tt_metal::DFBSpecName& name, uint32_t entries, tt::DataFormat format) {
        return tt::tt_metal::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = entries,
            .data_format_metadata = format};
    };
    tt::tt_metal::Group<tt::tt_metal::DataflowBufferSpec> dfb_specs = {
        make_dfb(q_dfb, 2 * ck, bf16),
        make_dfb(k_dfb, 2 * ck, bf16),
        make_dfb(v_dfb, 2 * cv, bf16),
        make_dfb(g_dfb, 2 * ck, gate_format),
        make_dfb(beta_dfb, 2 * Ct, bf16),
        make_dfb(beta_activated_dfb, Ct, fp32),
        make_dfb(eye_dfb, cc, fp32),
        make_dfb(tril_dfb, cc, fp32),
        make_dfb(ones_dfb, cc, fp32),
        make_dfb(gate_tril_dfb, cc, fp32),
        make_dfb(gate_ones_dfb, cc, fp32),
        make_dfb(block_masks_dfb, 3, fp32),
        make_dfb(workspace_0_dfb, ck, fp32),
        make_dfb(centered_inverse_decay_dfb, ck, fp32),
        make_dfb(akk_dfb, cc, fp32),
        make_dfb(t_inv_dfb, 2 * cc, output_formats[6]),
        make_dfb(v_beta_dfb, 2 * cv, output_formats[0]),
        make_dfb(kd_dfb, 2 * ck, output_formats[1]),
        make_dfb(q_decay_dfb, 2 * ck, output_formats[2]),
        make_dfb(intra_dfb, 2 * cc, output_formats[3]),
        make_dfb(workspace_1_dfb, ck, fp32),
        make_dfb(final_decay_dfb, 2 * Kt, output_formats[5]),
        make_dfb(k_decay_transposed_dfb, 2 * kc, output_formats[4]),
        make_dfb(anchor_decay_dfb, kv, fp32),
        make_dfb(normalized_q_dfb, ck, fp32),
        make_dfb(normalized_k_dfb, ck, fp32),
        // Row workspaces publish whole Kt-tile rows. Keep one-tile reductions and
        // inverse intermediates separate so no transaction can cross a ring end.
        make_dfb(tile_workspace_0_dfb, 1, fp32),
        // The nested inverse reads its current level while packing the next one.
        make_dfb(tile_workspace_1_dfb, 2, fp32),
        make_dfb(tile_workspace_2_dfb, 1, fp32),

        make_dfb(workspace_3_dfb, scratch, fp32),
        make_dfb(workspace_2_dfb, ck, fp32),
    };
    if (attrs.precise_gate_factors) {
        dfb_specs.push_back(make_dfb(gate_center_dfb, cc, fp32));
        dfb_specs.push_back(make_dfb(gate_center_neg_dfb, cc, fp32));
        dfb_specs.push_back(make_dfb(gate_half_dfb, cc, fp32));
    }
    tt::tt_metal::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/ttnn/bringup/prepare_chunk_recurrence/device/kernels/dataflow/"
            "reader_prepare_chunk_recurrence.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::DFBBinding{q_dfb, "q", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{k_dfb, "k", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{v_dfb, "v", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{g_dfb, "g", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{beta_dfb, "beta", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{eye_dfb, "eye", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{tril_dfb, "tril", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{ones_dfb, "ones", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{gate_tril_dfb, "gate_tril", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{gate_ones_dfb, "gate_ones", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{block_masks_dfb, "block_masks", tt::tt_metal::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                tt::tt_metal::TensorBinding{Q_TENSOR, "q"},
                tt::tt_metal::TensorBinding{K_TENSOR, "k"},
                tt::tt_metal::TensorBinding{V_TENSOR, "v"},
                tt::tt_metal::TensorBinding{G_TENSOR, "g"},
                tt::tt_metal::TensorBinding{BETA_TENSOR, "beta"},
            },
        .compile_time_args =
            {{"Ct", Ct},
             {"Kt", Kt},
             {"Vt", Vt},
             {"beta_width_tiles", static_cast<uint32_t>(in.beta.padded_shape()[-1]) / tt::constants::TILE_WIDTH},
             {"beta_offset_tiles", attrs.beta_logits_column_offset / tt::constants::TILE_WIDTH},
             {"GATE_SCALE_BITS", std::bit_cast<uint32_t>(attrs.gate_scale)}},
        .runtime_arg_schema = {.runtime_arg_names = {"work_item_start", "work_item_count", "num_chunks", "num_heads"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    tt::tt_metal::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/ttnn/bringup/prepare_chunk_recurrence/device/kernels/dataflow/"
            "writer_prepare_chunk_recurrence.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::DFBBinding{v_beta_dfb, "v_beta", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{t_inv_dfb, "t_inv", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{kd_dfb, "kd", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{q_decay_dfb, "q_decay", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{intra_dfb, "intra", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{k_decay_transposed_dfb, "k_decay_transposed", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{final_decay_dfb, "final_decay", tt::tt_metal::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                tt::tt_metal::TensorBinding{V_BETA_OUTPUT, "v_beta_output"},
                tt::tt_metal::TensorBinding{KD_OUTPUT, "kd_output"},
                tt::tt_metal::TensorBinding{Q_DECAY_OUTPUT, "q_decay_output"},
                tt::tt_metal::TensorBinding{INTRA_OUTPUT, "intra_output"},
                tt::tt_metal::TensorBinding{K_DECAY_TRANSPOSED_OUTPUT, "k_decay_transposed_output"},
                tt::tt_metal::TensorBinding{FINAL_DECAY_OUTPUT, "final_decay_output"},
                tt::tt_metal::TensorBinding{T_INV_OUTPUT, "t_inv_output"},
            },
        .compile_time_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"work_item_start", "work_item_count", "num_chunks"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(attrs.compute_kernel_config);
    auto& unpack_modes = compute_hw.unpack_modes;
    unpack_modes[q_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[k_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[v_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[g_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[beta_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[beta_activated_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[eye_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[tril_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[ones_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[gate_tril_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[gate_ones_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[block_masks_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[workspace_0_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[centered_inverse_decay_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[akk_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[t_inv_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[v_beta_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[kd_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[q_decay_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[intra_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[workspace_1_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[final_decay_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[k_decay_transposed_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[anchor_decay_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[normalized_q_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[normalized_k_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[tile_workspace_0_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[tile_workspace_1_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[tile_workspace_2_dfb] = UnpackMode::UnpackToSrc;

    unpack_modes[workspace_3_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[workspace_2_dfb] = UnpackMode::UnpackToSrc;
    if (attrs.precise_gate_factors) {
        unpack_modes[gate_center_dfb] = UnpackMode::UnpackToSrc;
        unpack_modes[gate_center_neg_dfb] = UnpackMode::UnpackToSrc;
        unpack_modes[gate_half_dfb] = UnpackMode::UnpackToSrc;
    }
    tt::tt_metal::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/ttnn/bringup/prepare_chunk_recurrence/device/kernels/compute/"
            "prepare_chunk_recurrence.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::DFBBinding{q_dfb, "q", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{k_dfb, "k", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{v_dfb, "v", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{g_dfb, "g", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{beta_dfb, "beta", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{beta_activated_dfb, "beta_activated", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{beta_activated_dfb, "beta_activated", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{eye_dfb, "eye", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{tril_dfb, "tril", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{ones_dfb, "ones", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{gate_tril_dfb, "gate_tril", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{gate_ones_dfb, "gate_ones", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{block_masks_dfb, "block_masks", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{workspace_0_dfb, "workspace_0", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{workspace_0_dfb, "workspace_0", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{centered_inverse_decay_dfb, "centered_inverse_decay", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{centered_inverse_decay_dfb, "centered_inverse_decay", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{akk_dfb, "akk", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{akk_dfb, "akk", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{workspace_1_dfb, "workspace_1", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{workspace_1_dfb, "workspace_1", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{anchor_decay_dfb, "anchor_decay", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{anchor_decay_dfb, "anchor_decay", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{normalized_q_dfb, "normalized_q", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{normalized_q_dfb, "normalized_q", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{normalized_k_dfb, "normalized_k", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{normalized_k_dfb, "normalized_k", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{tile_workspace_0_dfb, "tile_workspace_0", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{tile_workspace_0_dfb, "tile_workspace_0", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{tile_workspace_1_dfb, "tile_workspace_1", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{tile_workspace_1_dfb, "tile_workspace_1", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{tile_workspace_2_dfb, "tile_workspace_2", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{tile_workspace_2_dfb, "tile_workspace_2", tt::tt_metal::DFBEndpointType::CONSUMER},

                tt::tt_metal::DFBBinding{workspace_3_dfb, "workspace_3", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{workspace_3_dfb, "workspace_3", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{workspace_2_dfb, "workspace_2", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{workspace_2_dfb, "workspace_2", tt::tt_metal::DFBEndpointType::CONSUMER},
                tt::tt_metal::DFBBinding{v_beta_dfb, "v_beta", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{t_inv_dfb, "t_inv", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{kd_dfb, "kd", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{q_decay_dfb, "q_decay", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{intra_dfb, "intra", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{k_decay_transposed_dfb, "k_decay_transposed", tt::tt_metal::DFBEndpointType::PRODUCER},
                tt::tt_metal::DFBBinding{final_decay_dfb, "final_decay", tt::tt_metal::DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"Ct", Ct},
             {"Kt", Kt},
             {"Vt", Vt},
             {"SCALE_BITS",
              [&] {
                  uint32_t bits;
                  const float value = 1.0F / std::sqrt(static_cast<float>(attrs.key_dim));
                  std::memcpy(&bits, &value, sizeof(bits));
                  return bits;
              }()},
             {"EPS_BITS", 0x358637BDU}},
        .runtime_arg_schema = {.runtime_arg_names = {"work_item_start", "work_item_count", "num_chunks"}},
        .hw_config = std::move(compute_hw),
    };

    tt::tt_metal::KernelRunArgs reader_run{.kernel = READER};
    tt::tt_metal::KernelRunArgs writer_run{.kernel = WRITER};
    tt::tt_metal::KernelRunArgs compute_run{.kernel = COMPUTE};
    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t work_item_start = distribution.wi_start[index];
        const uint32_t work_item_count = distribution.wi_count[index];
        tt::tt_metal::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"work_item_start", work_item_start},
             {"work_item_count", work_item_count},
             {"num_chunks", num_chunks},
             {"num_heads", attrs.num_heads}});
        tt::tt_metal::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"work_item_start", work_item_start}, {"work_item_count", work_item_count}, {"num_chunks", num_chunks}});
        tt::tt_metal::AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            core,
            {{"work_item_start", work_item_start}, {"work_item_count", work_item_count}, {"num_chunks", num_chunks}});
    }

    tt::tt_metal::ProgramSpec spec{
        .name = "prepare_chunk_recurrence",
        .dataflow_buffers = std::move(dfb_specs),
        .tensor_parameters =
            {
                tt::tt_metal::TensorParameter{.unique_id = Q_TENSOR, .spec = q.tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = K_TENSOR, .spec = k.tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = V_TENSOR, .spec = v.tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = G_TENSOR, .spec = g.tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = BETA_TENSOR, .spec = beta.tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = V_BETA_OUTPUT, .spec = outputs[0].mesh_tensor().tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = KD_OUTPUT, .spec = outputs[1].mesh_tensor().tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = Q_DECAY_OUTPUT, .spec = outputs[2].mesh_tensor().tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = INTRA_OUTPUT, .spec = outputs[3].mesh_tensor().tensor_spec()},
                tt::tt_metal::TensorParameter{
                    .unique_id = K_DECAY_TRANSPOSED_OUTPUT, .spec = outputs[4].mesh_tensor().tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = FINAL_DECAY_OUTPUT, .spec = outputs[5].mesh_tensor().tensor_spec()},
                tt::tt_metal::TensorParameter{.unique_id = T_INV_OUTPUT, .spec = outputs[6].mesh_tensor().tensor_spec()},
            },
        .work_units = {tt::tt_metal::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = cores}},
    };
    tt::tt_metal::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args = {
        {Q_TENSOR, q},
        {K_TENSOR, k},
        {V_TENSOR, v},
        {G_TENSOR, g},
        {BETA_TENSOR, beta},
        {V_BETA_OUTPUT, outputs[0].mesh_tensor()},
        {KD_OUTPUT, outputs[1].mesh_tensor()},
        {Q_DECAY_OUTPUT, outputs[2].mesh_tensor()},
        {INTRA_OUTPUT, outputs[3].mesh_tensor()},
        {K_DECAY_TRANSPOSED_OUTPUT, outputs[4].mesh_tensor()},
        {FINAL_DECAY_OUTPUT, outputs[5].mesh_tensor()},
        {T_INV_OUTPUT, outputs[6].mesh_tensor()},
    };
    if (attrs.precise_gate_factors) {
        reader.dfb_bindings.push_back(tt::tt_metal::DFBBinding{gate_center_dfb, "gate_center", tt::tt_metal::DFBEndpointType::PRODUCER});
        reader.dfb_bindings.push_back(
            tt::tt_metal::DFBBinding{gate_center_neg_dfb, "gate_center_neg", tt::tt_metal::DFBEndpointType::PRODUCER});
        reader.dfb_bindings.push_back(tt::tt_metal::DFBBinding{gate_half_dfb, "gate_half", tt::tt_metal::DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(tt::tt_metal::DFBBinding{gate_center_dfb, "gate_center", tt::tt_metal::DFBEndpointType::CONSUMER});
        compute.dfb_bindings.push_back(
            tt::tt_metal::DFBBinding{gate_center_neg_dfb, "gate_center_neg", tt::tt_metal::DFBEndpointType::CONSUMER});
        compute.dfb_bindings.push_back(tt::tt_metal::DFBBinding{gate_half_dfb, "gate_half", tt::tt_metal::DFBEndpointType::CONSUMER});
        reader.compiler_options.defines.insert({"PRECISE_GATE_FACTORS", "1"});
        compute.compiler_options.defines.insert({"PRECISE_GATE_FACTORS", "1"});
        // -O2: at -O3 the matmul init re-inlined into matmul_blocks and the pre-#60899 program overflowed the kernel
        // config buffer (73 KB of 70.6)
        compute.compiler_options.opt_level = KernelBuildOptLevel::O2;
    }
    reader.compile_time_args.insert({"has_actual_start", uint32_t(in.actual_start.has_value())});
    kda_factory_detail::bind_chronology(spec, run_args, in.actual_start, reader, compute);
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, reader);
    const tt::tt_metal::DFBSpecName writer_chronology{"chronology_writer"};
    spec.dataflow_buffers.push_back(
        {.unique_id = writer_chronology,
         .entry_size = 32,
         .num_entries = 1,
         .data_format_metadata = tt::DataFormat::UInt32});
    reader.dfb_bindings.push_back(tt::tt_metal::ProducerOf(writer_chronology, "chronology_writer"));
    writer.dfb_bindings.push_back(tt::tt_metal::ConsumerOf(writer_chronology, "chronology_writer"));
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    return kda_factory_detail::chronology_workload(
        {.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.num_chunks * TILE_HEIGHT,
        READER);
}

}  // namespace ttnn::experimental::prim::bringup
