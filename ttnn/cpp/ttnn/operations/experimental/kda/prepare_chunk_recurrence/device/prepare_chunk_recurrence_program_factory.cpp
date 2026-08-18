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

uint32_t prepare_chunk_recurrence_cb_size_bytes(
    uint32_t chunk_size, uint32_t key_dim, uint32_t value_dim, DataType gate_dtype, uint32_t output_bf16_mask) {
    const uint32_t Ct = chunk_size / TILE_HEIGHT;
    const uint32_t Kt = key_dim / TILE_WIDTH;
    const uint32_t Vt = value_dim / TILE_WIDTH;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scratch = std::max({cc, ck, cv, kv, kc});
    const auto format = [&](uint32_t index) {
        return (output_bf16_mask & (1U << index)) ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;
    };
    uint32_t bytes = 0;
    const auto add = [&](uint32_t tiles, uint32_t buffers = 1, tt::DataFormat data_format = tt::DataFormat::Float32) {
        bytes += tiles * buffers * tt::tile_size(data_format);
    };
    constexpr auto bf16 = tt::DataFormat::Float16_b;
    add(ck, 1, bf16);
    add(ck, 1, bf16);
    add(cv, 1, bf16);
    add(ck, 1, tt::tt_metal::datatype_to_dataformat_converter(gate_dtype));
    add(Ct);
    add(cc);
    add(cc);
    add(cc);
    add(kv, 2);
    add(ck);
    add(ck);
    add(ck);
    add(cc);
    add(cc, 1, format(6));
    add(cv, 1, format(0));
    add(ck);
    add(cv, 2, bf16);
    add(std::max(cv, 3U));
    add(ck, 1, format(1));
    add(ck, 1, format(2));
    add(cc, 1, format(3));
    add(kv, 2);
    add(std::max(cv, Kt), 1, format(5));
    add(std::max(cv, ck));
    add(kc, 1, format(4));
    add(kv);
    add(kv);
    add(kv);
    add(scratch);
    add(scratch);
    add(scratch);
    add(kv, 2);
    return bytes;
}

ttnn::device_operation::ProgramArtifacts PrepareChunkRecurrenceProgramFactory::create_program_artifacts(
    const PrepareChunkRecurrenceParams& attrs, const PrepareChunkRecurrenceInputs& in, std::vector<Tensor>& outputs) {
    const auto& q = in.q.mesh_tensor();
    const auto& k = in.k.mesh_tensor();
    const auto& v = in.v.mesh_tensor();
    const auto& g = in.g.mesh_tensor();
    const auto& beta = in.beta.mesh_tensor();
    const auto& eye = in.eye.mesh_tensor();
    const auto& tril = in.tril.mesh_tensor();
    const auto& ones = in.ones.mesh_tensor();
    const auto& masks = in.masks.mesh_tensor();
    const auto& device = q.device();
    const auto arch = device.arch();

    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    constexpr uint32_t chunk_size = TILE_HEIGHT;
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
        device.compute_with_storage_grid_size(), BH * NC, std::numeric_limits<uint32_t>::max());
    const auto& cores = distribution.core_set;

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::DFBSpecName Q{"q"};
    const m2::DFBSpecName K{"k"};
    const m2::DFBSpecName V{"v"};
    const m2::DFBSpecName GATE{"g"};
    const m2::DFBSpecName BETA{"beta"};
    const m2::DFBSpecName EYE{"eye"};
    const m2::DFBSpecName TRIL{"tril"};
    const m2::DFBSpecName ONES{"ones"};
    const m2::DFBSpecName STATE{"state"};
    const m2::DFBSpecName DECAY{"decay"};
    const m2::DFBSpecName DECAY_EXP{"decay_exp"};
    const m2::DFBSpecName DECAY_FACTOR{"decay_factor"};
    const m2::DFBSpecName LOWER_MASK{"lower_mask"};
    const m2::DFBSpecName T_INV{"t_inv"};
    const m2::DFBSpecName V_BETA{"v_beta"};
    const m2::DFBSpecName K_BETA{"k_beta"};
    const m2::DFBSpecName OUTPUT{"output"};
    const m2::DFBSpecName U{"u"};
    const m2::DFBSpecName W{"w"};
    const m2::DFBSpecName Q_DECAY{"q_decay"};
    const m2::DFBSpecName INTRA{"intra"};
    const m2::DFBSpecName STATE_TWO{"state_two"};
    const m2::DFBSpecName V_NEW{"v_new"};
    const m2::DFBSpecName OUTPUT_INTERMEDIATE{"output_intermediate"};
    const m2::DFBSpecName K_DECAY_TRANSPOSED{"k_decay_transposed"};
    const m2::DFBSpecName STATE_UPDATE{"state_update"};
    const m2::DFBSpecName STATE_TEMPORARY{"state_temporary"};
    const m2::DFBSpecName FINAL_STATE{"final_state"};
    const m2::DFBSpecName SCRATCH_ONE{"scratch_one"};
    const m2::DFBSpecName SCRATCH_TWO{"scratch_two"};
    const m2::DFBSpecName SCRATCH_THREE{"scratch_three"};
    const m2::DFBSpecName STATE_THREE{"state_three"};
    const m2::TensorParamName Q_TENSOR{"q"};
    const m2::TensorParamName K_TENSOR{"k"};
    const m2::TensorParamName V_TENSOR{"v"};
    const m2::TensorParamName G_TENSOR{"g"};
    const m2::TensorParamName BETA_TENSOR{"beta"};
    const m2::TensorParamName EYE_TENSOR{"eye"};
    const m2::TensorParamName TRIL_TENSOR{"tril"};
    const m2::TensorParamName ONES_TENSOR{"ones"};
    const m2::TensorParamName MASKS_TENSOR{"masks"};
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
        make_dfb(Q, ck, bf16),
        make_dfb(K, ck, bf16),
        make_dfb(V, cv, bf16),
        make_dfb(GATE, ck, gate_format),
        make_dfb(BETA, Ct, fp32),
        make_dfb(EYE, cc, fp32),
        make_dfb(TRIL, cc, fp32),
        make_dfb(ONES, cc, fp32),
        make_dfb(STATE, kv * 2, fp32),
        make_dfb(DECAY, ck, fp32),
        make_dfb(DECAY_EXP, ck, fp32),
        make_dfb(DECAY_FACTOR, ck, fp32),
        make_dfb(LOWER_MASK, cc, fp32),
        make_dfb(T_INV, cc, output_formats[6]),
        make_dfb(V_BETA, cv, output_formats[0]),
        make_dfb(K_BETA, ck, fp32),
        make_dfb(OUTPUT, cv * 2, bf16),
        make_dfb(U, std::max(cv, 3U), fp32),
        make_dfb(W, ck, output_formats[1]),
        make_dfb(Q_DECAY, ck, output_formats[2]),
        make_dfb(INTRA, cc, output_formats[3]),
        make_dfb(STATE_TWO, kv * 2, fp32),
        make_dfb(V_NEW, std::max(cv, Kt), output_formats[5]),
        make_dfb(OUTPUT_INTERMEDIATE, std::max(cv, ck), fp32),
        make_dfb(K_DECAY_TRANSPOSED, kc, output_formats[4]),
        make_dfb(STATE_UPDATE, kv, fp32),
        make_dfb(STATE_TEMPORARY, kv, fp32),
        make_dfb(FINAL_STATE, kv, fp32),
        make_dfb(SCRATCH_ONE, scratch, fp32),
        make_dfb(SCRATCH_TWO, scratch, fp32),
        make_dfb(SCRATCH_THREE, scratch, fp32),
        make_dfb(STATE_THREE, kv * 2, fp32),
    };
    TT_FATAL(
        prepare_chunk_recurrence_cb_size_bytes(
            chunk_size, attrs.key_dim, attrs.value_dim, in.g.dtype(), attrs.output_bf16_mask) ==
            [&] {
                uint32_t bytes = 0;
                for (const auto& spec : dfb_specs) {
                    bytes += spec.entry_size * spec.num_entries;
                }
                return bytes;
            }(),
        "KDA prep CB size estimator is out of sync with its program factory");

    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/"
            "reader_prepare_chunk_recurrence.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{Q, "q", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{K, "k", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{V, "v", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{GATE, "g", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{BETA, "beta", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{EYE, "eye", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{TRIL, "tril", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{ONES, "ones", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{U, "u", m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{Q_TENSOR, "q"},
                m2::TensorBinding{K_TENSOR, "k"},
                m2::TensorBinding{V_TENSOR, "v"},
                m2::TensorBinding{G_TENSOR, "g"},
                m2::TensorBinding{BETA_TENSOR, "beta"},
                m2::TensorBinding{EYE_TENSOR, "eye"},
                m2::TensorBinding{TRIL_TENSOR, "tril"},
                m2::TensorBinding{ONES_TENSOR, "ones"},
                m2::TensorBinding{MASKS_TENSOR, "masks"},
            },
        .compile_time_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count", "NC", "HV", "Hk"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/dataflow/"
            "writer_prepare_chunk_recurrence.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{V_BETA, "v_beta", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{T_INV, "t_inv", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{W, "w", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{Q_DECAY, "q_decay", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{INTRA, "intra", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{K_DECAY_TRANSPOSED, "k_decay_transposed", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{V_NEW, "v_new", m2::DFBEndpointType::CONSUMER},
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
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    unpack_modes[Q] = UnpackMode::UnpackToSrc;
    unpack_modes[K] = UnpackMode::UnpackToSrc;
    unpack_modes[V] = UnpackMode::UnpackToSrc;
    unpack_modes[GATE] = UnpackMode::UnpackToSrc;
    unpack_modes[BETA] = UnpackMode::UnpackToSrc;
    unpack_modes[EYE] = UnpackMode::UnpackToSrc;
    unpack_modes[TRIL] = UnpackMode::UnpackToSrc;
    unpack_modes[ONES] = UnpackMode::UnpackToSrc;
    unpack_modes[STATE] = UnpackMode::UnpackToSrc;
    unpack_modes[DECAY] = UnpackMode::UnpackToSrc;
    unpack_modes[DECAY_EXP] = UnpackMode::UnpackToSrc;
    unpack_modes[DECAY_FACTOR] = UnpackMode::UnpackToSrc;
    unpack_modes[LOWER_MASK] = UnpackMode::UnpackToSrc;
    unpack_modes[T_INV] = UnpackMode::UnpackToSrc;
    unpack_modes[V_BETA] = UnpackMode::UnpackToSrc;
    unpack_modes[K_BETA] = UnpackMode::UnpackToSrc;
    unpack_modes[OUTPUT] = UnpackMode::UnpackToSrc;
    unpack_modes[U] = UnpackMode::UnpackToSrc;
    unpack_modes[W] = UnpackMode::UnpackToSrc;
    unpack_modes[Q_DECAY] = UnpackMode::UnpackToSrc;
    unpack_modes[INTRA] = UnpackMode::UnpackToSrc;
    unpack_modes[STATE_TWO] = UnpackMode::UnpackToSrc;
    unpack_modes[V_NEW] = UnpackMode::UnpackToSrc;
    unpack_modes[OUTPUT_INTERMEDIATE] = UnpackMode::UnpackToSrc;
    unpack_modes[K_DECAY_TRANSPOSED] = UnpackMode::UnpackToSrc;
    unpack_modes[STATE_UPDATE] = UnpackMode::UnpackToSrc;
    unpack_modes[STATE_TEMPORARY] = UnpackMode::UnpackToSrc;
    unpack_modes[FINAL_STATE] = UnpackMode::UnpackToSrc;
    unpack_modes[SCRATCH_ONE] = UnpackMode::UnpackToSrc;
    unpack_modes[SCRATCH_TWO] = UnpackMode::UnpackToSrc;
    unpack_modes[SCRATCH_THREE] = UnpackMode::UnpackToSrc;
    unpack_modes[STATE_THREE] = UnpackMode::UnpackToSrc;
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/compute/"
            "prepare_chunk_recurrence.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{Q, "q", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{K, "k", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{V, "v", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{GATE, "g", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{BETA, "beta", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{EYE, "eye", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{TRIL, "tril", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{ONES, "ones", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{U, "u", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STATE, "state", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STATE, "state", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{DECAY, "decay", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{DECAY, "decay", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{DECAY_EXP, "decay_exp", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{DECAY_EXP, "decay_exp", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{DECAY_FACTOR, "decay_factor", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{DECAY_FACTOR, "decay_factor", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{LOWER_MASK, "lower_mask", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{LOWER_MASK, "lower_mask", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{K_BETA, "k_beta", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{K_BETA, "k_beta", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{OUTPUT, "output", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{OUTPUT, "output", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STATE_TWO, "state_two", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STATE_TWO, "state_two", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{OUTPUT_INTERMEDIATE, "output_intermediate", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{OUTPUT_INTERMEDIATE, "output_intermediate", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STATE_UPDATE, "state_update", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STATE_UPDATE, "state_update", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STATE_TEMPORARY, "state_temporary", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STATE_TEMPORARY, "state_temporary", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{FINAL_STATE, "final_state", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{FINAL_STATE, "final_state", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SCRATCH_ONE, "scratch_one", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SCRATCH_ONE, "scratch_one", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SCRATCH_TWO, "scratch_two", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SCRATCH_TWO, "scratch_two", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SCRATCH_THREE, "scratch_three", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SCRATCH_THREE, "scratch_three", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{STATE_THREE, "state_three", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{STATE_THREE, "state_three", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{V_BETA, "v_beta", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{T_INV, "t_inv", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{W, "w", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{Q_DECAY, "q_decay", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{INTRA, "intra", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{K_DECAY_TRANSPOSED, "k_decay_transposed", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{V_NEW, "v_new", m2::DFBEndpointType::PRODUCER},
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
        .runtime_arg_schema = {.runtime_arg_names = {"work_count"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};
    m2::KernelRunArgs compute_run{.kernel = COMPUTE};
    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t work_start = distribution.wi_start[index];
        const uint32_t work_count = distribution.wi_count[index];
        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"wi_start", work_start},
             {"wi_count", work_count},
             {"NC", NC},
             {"HV", attrs.num_heads},
             {"Hk", attrs.num_heads}});
        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"wi_start", work_start}, {"wi_count", work_count}});
        m2::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"work_count", work_count}});
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
                m2::TensorParameter{.unique_id = EYE_TENSOR, .spec = eye.tensor_spec()},
                m2::TensorParameter{.unique_id = TRIL_TENSOR, .spec = tril.tensor_spec()},
                m2::TensorParameter{.unique_id = ONES_TENSOR, .spec = ones.tensor_spec()},
                m2::TensorParameter{.unique_id = MASKS_TENSOR, .spec = masks.tensor_spec()},
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
        {EYE_TENSOR, eye},
        {TRIL_TENSOR, tril},
        {ONES_TENSOR, ones},
        {MASKS_TENSOR, masks},
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
