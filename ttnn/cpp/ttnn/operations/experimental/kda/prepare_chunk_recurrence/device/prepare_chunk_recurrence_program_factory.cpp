// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/prepare_chunk_recurrence_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {
namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts PrepareChunkRecurrenceProgramFactory::create_program_artifacts(
    const PrepareChunkRecurrenceParams& attrs, const PrepareChunkRecurrenceInputs& in, std::vector<Tensor>& outputs) {
    const auto& q = in.q.mesh_tensor();
    const auto& k = in.k.mesh_tensor();
    const auto& v = in.v.mesh_tensor();
    const auto& g = in.g.mesh_tensor();
    const auto& beta = in.beta.mesh_tensor();
    const auto& device = q.device();
    const auto arch = device.arch();

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

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::DFBSpecName q_dfb{"q"};
    const m2::DFBSpecName k_dfb{"k"};
    const m2::DFBSpecName v_dfb{"v"};
    const m2::DFBSpecName g_dfb{"g"};
    const m2::DFBSpecName beta_dfb{"beta"};
    const m2::DFBSpecName eye_dfb{"eye"};
    const m2::DFBSpecName tril_dfb{"tril"};
    const m2::DFBSpecName ones_dfb{"ones"};
    const m2::DFBSpecName block_masks_dfb{"block_masks"};
    const m2::DFBSpecName workspace_0_dfb{"workspace_0"};
    const m2::DFBSpecName scan_decay_dfb{"scan_decay"};
    const m2::DFBSpecName centered_inverse_decay_dfb{"centered_inverse_decay"};
    const m2::DFBSpecName akk_dfb{"akk"};
    const m2::DFBSpecName t_inv_dfb{"t_inv"};
    const m2::DFBSpecName v_beta_dfb{"v_beta"};
    const m2::DFBSpecName kd_dfb{"kd"};
    const m2::DFBSpecName q_decay_dfb{"q_decay"};
    const m2::DFBSpecName intra_dfb{"intra"};
    const m2::DFBSpecName workspace_1_dfb{"workspace_1"};
    const m2::DFBSpecName final_decay_dfb{"final_decay"};
    const m2::DFBSpecName k_decay_transposed_dfb{"k_decay_transposed"};
    const m2::DFBSpecName anchor_decay_dfb{"anchor_decay"};
    const m2::DFBSpecName normalized_q_dfb{"normalized_q"};
    const m2::DFBSpecName normalized_k_dfb{"normalized_k"};
    const m2::DFBSpecName inverse_n3_dfb{"inverse_n3"};
    const m2::DFBSpecName workspace_3_dfb{"workspace_3"};
    const m2::DFBSpecName workspace_2_dfb{"workspace_2"};
    const m2::TensorParamName Q_TENSOR{"q"};
    const m2::TensorParamName K_TENSOR{"k"};
    const m2::TensorParamName V_TENSOR{"v"};
    const m2::TensorParamName G_TENSOR{"g"};
    const m2::TensorParamName BETA_TENSOR{"beta"};
    const m2::TensorParamName V_BETA_OUTPUT{"v_beta_output"};
    const m2::TensorParamName KD_OUTPUT{"kd_output"};
    const m2::TensorParamName Q_DECAY_OUTPUT{"q_decay_output"};
    const m2::TensorParamName INTRA_OUTPUT{"intra_output"};
    const m2::TensorParamName K_DECAY_TRANSPOSED_OUTPUT{"k_decay_transposed_output"};
    const m2::TensorParamName FINAL_DECAY_OUTPUT{"final_decay_output"};
    const m2::TensorParamName T_INV_OUTPUT{"t_inv_output"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto gate_format = datatype_to_dataformat_converter(in.g.dtype());
    std::vector<tt::DataFormat> output_formats;
    output_formats.reserve(outputs.size());
    for (const auto& output : outputs) {
        output_formats.push_back(datatype_to_dataformat_converter(output.dtype()));
    }
    auto make_dfb = [](const m2::DFBSpecName& name, uint32_t entries, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = entries,
            .data_format_metadata = format};
    };
    m2::Group<m2::DataflowBufferSpec> dfb_specs = {
        make_dfb(q_dfb, 2 * ck, bf16),
        make_dfb(k_dfb, 2 * ck, bf16),
        make_dfb(v_dfb, 2 * cv, bf16),
        make_dfb(g_dfb, 2 * ck, gate_format),
        make_dfb(beta_dfb, 2 * Ct, fp32),
        make_dfb(eye_dfb, cc, fp32),
        make_dfb(tril_dfb, cc, fp32),
        make_dfb(ones_dfb, cc, fp32),
        make_dfb(block_masks_dfb, 2, fp32),
        make_dfb(workspace_0_dfb, ck, fp32),
        make_dfb(scan_decay_dfb, ck, fp32),
        make_dfb(centered_inverse_decay_dfb, ck, fp32),
        make_dfb(akk_dfb, cc, fp32),
        make_dfb(t_inv_dfb, 2 * cc, output_formats[6]),
        make_dfb(v_beta_dfb, 2 * cv, output_formats[0]),
        make_dfb(kd_dfb, 2 * ck, output_formats[1]),
        make_dfb(q_decay_dfb, 2 * ck, output_formats[2]),
        make_dfb(intra_dfb, 2 * cc, output_formats[3]),
        make_dfb(workspace_1_dfb, kv * 2, fp32),
        make_dfb(final_decay_dfb, 2 * Kt, output_formats[5]),
        make_dfb(k_decay_transposed_dfb, 2 * kc, output_formats[4]),
        make_dfb(anchor_decay_dfb, kv, fp32),
        make_dfb(normalized_q_dfb, ck, fp32),
        make_dfb(normalized_k_dfb, std::max(ck, 2U), fp32),
        make_dfb(inverse_n3_dfb, 1, fp32),
        make_dfb(workspace_3_dfb, scratch, fp32),
        make_dfb(workspace_2_dfb, kv * 2, fp32),
    };
    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/"
            "reader_prepare_chunk_recurrence.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{q_dfb, "q", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{k_dfb, "k", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{v_dfb, "v", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{g_dfb, "g", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{beta_dfb, "beta", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{eye_dfb, "eye", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{tril_dfb, "tril", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{ones_dfb, "ones", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{block_masks_dfb, "block_masks", m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{Q_TENSOR, "q"},
                m2::TensorBinding{K_TENSOR, "k"},
                m2::TensorBinding{V_TENSOR, "v"},
                m2::TensorBinding{G_TENSOR, "g"},
                m2::TensorBinding{BETA_TENSOR, "beta"},
            },
        .compile_time_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"work_item_start", "work_item_count", "num_chunks", "num_heads"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/"
            "writer_prepare_chunk_recurrence.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{v_beta_dfb, "v_beta", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{t_inv_dfb, "t_inv", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{kd_dfb, "kd", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{q_decay_dfb, "q_decay", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{intra_dfb, "intra", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{k_decay_transposed_dfb, "k_decay_transposed", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{final_decay_dfb, "final_decay", m2::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{V_BETA_OUTPUT, "v_beta_output"},
                m2::TensorBinding{KD_OUTPUT, "kd_output"},
                m2::TensorBinding{Q_DECAY_OUTPUT, "q_decay_output"},
                m2::TensorBinding{INTRA_OUTPUT, "intra_output"},
                m2::TensorBinding{K_DECAY_TRANSPOSED_OUTPUT, "k_decay_transposed_output"},
                m2::TensorBinding{FINAL_DECAY_OUTPUT, "final_decay_output"},
                m2::TensorBinding{T_INV_OUTPUT, "t_inv_output"},
            },
        .compile_time_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"work_item_start", "work_item_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    unpack_modes[q_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[k_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[v_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[g_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[beta_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[eye_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[tril_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[ones_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[block_masks_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[workspace_0_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[scan_decay_dfb] = UnpackMode::UnpackToSrc;
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
    unpack_modes[inverse_n3_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[workspace_3_dfb] = UnpackMode::UnpackToSrc;
    unpack_modes[workspace_2_dfb] = UnpackMode::UnpackToSrc;
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/compute/"
            "prepare_chunk_recurrence.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{q_dfb, "q", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{k_dfb, "k", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{v_dfb, "v", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{g_dfb, "g", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{beta_dfb, "beta", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{eye_dfb, "eye", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{tril_dfb, "tril", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{ones_dfb, "ones", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{block_masks_dfb, "block_masks", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{workspace_0_dfb, "workspace_0", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{workspace_0_dfb, "workspace_0", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{scan_decay_dfb, "scan_decay", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{scan_decay_dfb, "scan_decay", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{centered_inverse_decay_dfb, "centered_inverse_decay", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{centered_inverse_decay_dfb, "centered_inverse_decay", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{akk_dfb, "akk", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{akk_dfb, "akk", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{workspace_1_dfb, "workspace_1", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{workspace_1_dfb, "workspace_1", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{anchor_decay_dfb, "anchor_decay", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{anchor_decay_dfb, "anchor_decay", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{normalized_q_dfb, "normalized_q", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{normalized_q_dfb, "normalized_q", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{normalized_k_dfb, "normalized_k", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{normalized_k_dfb, "normalized_k", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{inverse_n3_dfb, "inverse_n3", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{inverse_n3_dfb, "inverse_n3", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{workspace_3_dfb, "workspace_3", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{workspace_3_dfb, "workspace_3", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{workspace_2_dfb, "workspace_2", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{workspace_2_dfb, "workspace_2", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{v_beta_dfb, "v_beta", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{t_inv_dfb, "t_inv", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{kd_dfb, "kd", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{q_decay_dfb, "q_decay", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{intra_dfb, "intra", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{k_decay_transposed_dfb, "k_decay_transposed", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{final_decay_dfb, "final_decay", m2::DFBEndpointType::PRODUCER},
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
        .runtime_arg_schema = {.runtime_arg_names = {"work_item_count"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};
    m2::KernelRunArgs compute_run{.kernel = COMPUTE};
    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t work_item_start = distribution.wi_start[index];
        const uint32_t work_item_count = distribution.wi_count[index];
        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"work_item_start", work_item_start},
             {"work_item_count", work_item_count},
             {"num_chunks", num_chunks},
             {"num_heads", attrs.num_heads}});
        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"work_item_start", work_item_start}, {"work_item_count", work_item_count}});
        m2::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"work_item_count", work_item_count}});
    }

    m2::ProgramSpec spec{
        .name = "prepare_chunk_recurrence",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfb_specs),
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = Q_TENSOR, .spec = q.tensor_spec()},
                m2::TensorParameter{.unique_id = K_TENSOR, .spec = k.tensor_spec()},
                m2::TensorParameter{.unique_id = V_TENSOR, .spec = v.tensor_spec()},
                m2::TensorParameter{.unique_id = G_TENSOR, .spec = g.tensor_spec()},
                m2::TensorParameter{.unique_id = BETA_TENSOR, .spec = beta.tensor_spec()},
                m2::TensorParameter{.unique_id = V_BETA_OUTPUT, .spec = outputs[0].mesh_tensor().tensor_spec()},
                m2::TensorParameter{.unique_id = KD_OUTPUT, .spec = outputs[1].mesh_tensor().tensor_spec()},
                m2::TensorParameter{.unique_id = Q_DECAY_OUTPUT, .spec = outputs[2].mesh_tensor().tensor_spec()},
                m2::TensorParameter{.unique_id = INTRA_OUTPUT, .spec = outputs[3].mesh_tensor().tensor_spec()},
                m2::TensorParameter{
                    .unique_id = K_DECAY_TRANSPOSED_OUTPUT, .spec = outputs[4].mesh_tensor().tensor_spec()},
                m2::TensorParameter{.unique_id = FINAL_DECAY_OUTPUT, .spec = outputs[5].mesh_tensor().tensor_spec()},
                m2::TensorParameter{.unique_id = T_INV_OUTPUT, .spec = outputs[6].mesh_tensor().tensor_spec()},
            },
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = cores}},
    };
    m2::ProgramRunArgs run_args;
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
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
