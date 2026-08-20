// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/recurrent_chunk_scan/device/recurrent_chunk_scan_program_factory.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {
namespace m2 = tt::tt_metal::experimental;
namespace {

struct ScanWorkDistribution {
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> head;
    std::vector<uint32_t> value_block;
    uint32_t value_tiles_per_core = 1;
    CoreRangeSet core_set;
};

ScanWorkDistribution distribute_scan(CoreCoord grid, uint32_t batch_heads, uint32_t value_tiles, bool summary) {
    const uint32_t num_cores = grid.x * grid.y;
    TT_FATAL(batch_heads <= num_cores, "KDA recurrent scan heads {} exceed compute cores {}", batch_heads, num_cores);
    uint32_t value_blocks = 1;
    if (!summary && batch_heads <= 8) {
        for (uint32_t candidate = value_tiles; candidate >= 1; --candidate) {
            if (value_tiles % candidate == 0 && batch_heads * candidate <= num_cores) {
                value_blocks = candidate;
                break;
            }
        }
    }
    ScanWorkDistribution result;
    result.value_tiles_per_core = value_tiles / value_blocks;
    std::set<CoreRange> ranges;
    for (uint32_t index = 0; index < batch_heads * value_blocks; ++index) {
        const CoreCoord core{index % grid.x, index / grid.x};
        result.cores.push_back(core);
        result.head.push_back(index / value_blocks);
        result.value_block.push_back(index % value_blocks);
        ranges.insert(CoreRange{core, core});
    }
    result.core_set = CoreRangeSet{ranges};
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
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt_full = attrs.value_dim / TILE_WIDTH;
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

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::DFBSpecName STATE{"state"};
    const m2::DFBSpecName T_INV{"t_inv"};
    const m2::DFBSpecName V_BETA{"v_beta"};
    const m2::DFBSpecName KD{"kd"};
    const m2::DFBSpecName Q_DECAY{"q_decay"};
    const m2::DFBSpecName INTRA{"intra"};
    const m2::DFBSpecName STATE_RING{"state_ring"};
    const m2::DFBSpecName VALUE_NEW{"value_new"};
    const m2::DFBSpecName FINAL_DECAY{"final_decay"};
    const m2::DFBSpecName OUTPUT{"output"};
    const m2::DFBSpecName OUTPUT_INTERMEDIATE{"output_intermediate"};
    const m2::DFBSpecName K_DECAY_TRANSPOSED{"k_decay_transposed"};
    const m2::DFBSpecName STATE_UPDATE{"state_update"};
    const m2::DFBSpecName STATE_TEMPORARY{"state_temporary"};
    const m2::DFBSpecName FINAL_STATE{"final_state"};
    const m2::DFBSpecName SCRATCH{"scratch"};
    const m2::DFBSpecName SUMMARY_RAW{"summary_raw"};
    const m2::DFBSpecName SUMMARY_SEED{"summary_seed"};
    const m2::DFBSpecName SUMMARY_RING{"summary_ring"};

    const m2::TensorParamName V_BETA_TENSOR{"v_beta"};
    const m2::TensorParamName KD_TENSOR{"kd"};
    const m2::TensorParamName Q_DECAY_TENSOR{"q_decay"};
    const m2::TensorParamName INTRA_TENSOR{"intra"};
    const m2::TensorParamName K_DECAY_TRANSPOSED_TENSOR{"k_decay_transposed"};
    const m2::TensorParamName FINAL_DECAY_TENSOR{"final_decay"};
    const m2::TensorParamName T_INV_TENSOR{"t_inv"};
    const m2::TensorParamName INITIAL_STATE_TENSOR{"initial_state"};
    const m2::TensorParamName OUTPUT_TENSOR{"output"};
    const m2::TensorParamName FINAL_STATE_TENSOR{"final_state"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto output_format = summary ? fp32 : tt::DataFormat::Float16_b;
    const auto input_format = [](const Tensor& tensor) { return datatype_to_dataformat_converter(tensor.dtype()); };
    const auto make_dfb = [](const m2::DFBSpecName& name, uint32_t entries, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = entries,
            .data_format_metadata = format};
    };
    m2::Group<m2::DataflowBufferSpec> dfb_specs = {
        make_dfb(STATE, kv, fp32),
        make_dfb(T_INV, cc, input_format(in.t_inv)),
        make_dfb(V_BETA, cv, input_format(in.v_beta)),
        make_dfb(KD, ck, input_format(in.kd)),
        make_dfb(Q_DECAY, summary ? 1 : ck, summary ? fp32 : input_format(in.q_decay)),
        make_dfb(INTRA, summary ? 1 : cc, summary ? fp32 : input_format(in.intra)),
        make_dfb(STATE_RING, 2 * kv, fp32),
        make_dfb(VALUE_NEW, cv, fp32),
        make_dfb(FINAL_DECAY, Kt, input_format(in.final_decay)),
        make_dfb(OUTPUT, summary ? kv : cv, output_format),
        make_dfb(OUTPUT_INTERMEDIATE, summary ? 1 : cv, fp32),
        make_dfb(K_DECAY_TRANSPOSED, kc, input_format(in.k_dec_t)),
        make_dfb(STATE_UPDATE, kv, fp32),
        make_dfb(STATE_TEMPORARY, kv, fp32),
        make_dfb(FINAL_STATE, kv, fp32),
        make_dfb(SCRATCH, scratch_entries, fp32),
        make_dfb(SUMMARY_RAW, kv, fp32),
        make_dfb(SUMMARY_SEED, kv, fp32),
        make_dfb(SUMMARY_RING, 2 * kv, fp32),
    };

    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/"
            "reader_recurrent_chunk_scan.cpp",
        .dfb_bindings =
            {
                m2::ProducerOf(STATE, "state"),
                m2::ProducerOf(T_INV, "t_inv"),
                m2::ProducerOf(V_BETA, "v_beta"),
                m2::ProducerOf(KD, "kd"),
                m2::ProducerOf(Q_DECAY, "q_decay"),
                m2::ProducerOf(INTRA, "intra"),
                m2::ProducerOf(SUMMARY_SEED, "summary_seed"),
                m2::ProducerOf(K_DECAY_TRANSPOSED, "k_decay_transposed"),
                m2::ProducerOf(FINAL_DECAY, "final_decay"),
            },
        .tensor_bindings =
            {
                m2::TensorBinding{V_BETA_TENSOR, "v_beta"},
                m2::TensorBinding{KD_TENSOR, "kd"},
                m2::TensorBinding{K_DECAY_TRANSPOSED_TENSOR, "k_decay_transposed"},
                m2::TensorBinding{FINAL_DECAY_TENSOR, "final_decay"},
                m2::TensorBinding{T_INV_TENSOR, "t_inv"},
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
        reader.tensor_bindings.push_back(m2::TensorBinding{Q_DECAY_TENSOR, "q_decay"});
        reader.tensor_bindings.push_back(m2::TensorBinding{INTRA_TENSOR, "intra"});
        reader.tensor_bindings.push_back(m2::TensorBinding{INITIAL_STATE_TENSOR, "initial_state"});
    } else {
        // The discarded recurrence branch is still parsed; aliases provide its binding names without extra parameters.
        reader.tensor_bindings.push_back(m2::TensorBinding{V_BETA_TENSOR, "q_decay"});
        reader.tensor_bindings.push_back(m2::TensorBinding{T_INV_TENSOR, "intra"});
        reader.tensor_bindings.push_back(m2::TensorBinding{V_BETA_TENSOR, "initial_state"});
    }

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/dataflow/"
            "writer_recurrent_chunk_scan.cpp",
        .dfb_bindings = {m2::ConsumerOf(OUTPUT, "output"), m2::ConsumerOf(FINAL_STATE, "final_state")},
        .tensor_bindings =
            {m2::TensorBinding{OUTPUT_TENSOR, "output"}, m2::TensorBinding{FINAL_STATE_TENSOR, "final_state"}},
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
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& name :
         {STATE,
          T_INV,
          V_BETA,
          KD,
          Q_DECAY,
          INTRA,
          STATE_RING,
          VALUE_NEW,
          FINAL_DECAY,
          OUTPUT,
          OUTPUT_INTERMEDIATE,
          K_DECAY_TRANSPOSED,
          STATE_UPDATE,
          STATE_TEMPORARY,
          FINAL_STATE,
          SCRATCH,
          SUMMARY_RAW,
          SUMMARY_SEED,
          SUMMARY_RING}) {
        unpack_modes[name] = UnpackMode::UnpackToSrc;
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan/device/kernels/compute/"
            "recurrent_chunk_scan.cpp",
        .dfb_bindings =
            {
                m2::ConsumerOf(STATE, "state"),
                m2::ConsumerOf(T_INV, "t_inv"),
                m2::ConsumerOf(V_BETA, "v_beta"),
                m2::ConsumerOf(KD, "kd"),
                m2::ConsumerOf(Q_DECAY, "q_decay"),
                m2::ConsumerOf(INTRA, "intra"),
                m2::ProducerOf(STATE_RING, "state_ring"),
                m2::ConsumerOf(STATE_RING, "state_ring"),
                m2::ProducerOf(VALUE_NEW, "value_new"),
                m2::ConsumerOf(VALUE_NEW, "value_new"),
                m2::ConsumerOf(FINAL_DECAY, "final_decay"),
                m2::ProducerOf(OUTPUT, "output"),
                m2::ProducerOf(OUTPUT_INTERMEDIATE, "output_intermediate"),
                m2::ConsumerOf(OUTPUT_INTERMEDIATE, "output_intermediate"),
                m2::ConsumerOf(K_DECAY_TRANSPOSED, "k_decay_transposed"),
                m2::ProducerOf(STATE_UPDATE, "state_update"),
                m2::ConsumerOf(STATE_UPDATE, "state_update"),
                m2::ProducerOf(STATE_TEMPORARY, "state_temporary"),
                m2::ConsumerOf(STATE_TEMPORARY, "state_temporary"),
                m2::ProducerOf(FINAL_STATE, "final_state"),
                m2::ProducerOf(SCRATCH, "scratch"),
                m2::ConsumerOf(SCRATCH, "scratch"),
                m2::ProducerOf(SUMMARY_RAW, "summary_raw"),
                m2::ConsumerOf(SUMMARY_RAW, "summary_raw"),
                m2::ConsumerOf(SUMMARY_SEED, "summary_seed"),
                m2::ProducerOf(SUMMARY_RING, "summary_ring"),
                m2::ConsumerOf(SUMMARY_RING, "summary_ring"),
            },
        .compile_time_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}, {"summary_pair", static_cast<uint32_t>(summary)}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_chunks"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};
    m2::KernelRunArgs compute_run{.kernel = COMPUTE};
    for (uint32_t index = 0; index < distribution.cores.size(); ++index) {
        const auto& core = distribution.cores[index];
        const uint32_t head = distribution.head[index];
        const uint32_t value_block = distribution.value_block[index];
        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"head", head}, {"value_block", value_block}, {"num_chunks", NC}});
        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"head", head}, {"value_block", value_block}, {"num_chunks", NC}});
        m2::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"num_chunks", NC}});
    }

    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = V_BETA_TENSOR, .spec = v_beta_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = KD_TENSOR, .spec = kd_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = K_DECAY_TRANSPOSED_TENSOR, .spec = k_dec_t_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = FINAL_DECAY_TENSOR, .spec = final_decay_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = T_INV_TENSOR, .spec = t_inv_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = outputs[0].mesh_tensor().tensor_spec()},
        m2::TensorParameter{.unique_id = FINAL_STATE_TENSOR, .spec = outputs[1].mesh_tensor().tensor_spec()},
    };
    if (!summary) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = Q_DECAY_TENSOR, .spec = q_decay_tensor.tensor_spec()});
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = INTRA_TENSOR, .spec = intra_tensor.tensor_spec()});
        tensor_parameters.push_back(m2::TensorParameter{
            .unique_id = INITIAL_STATE_TENSOR, .spec = in.initial_state->mesh_tensor().tensor_spec()});
    }
    m2::ProgramSpec spec{
        .name = summary ? "summarize_chunk_recurrence" : "recurrent_chunk_scan",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfb_specs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = cores}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args = {
        {V_BETA_TENSOR, v_beta_tensor},
        {KD_TENSOR, kd_tensor},
        {K_DECAY_TRANSPOSED_TENSOR, k_dec_t_tensor},
        {FINAL_DECAY_TENSOR, final_decay_tensor},
        {T_INV_TENSOR, t_inv_tensor},
        {OUTPUT_TENSOR, outputs[0].mesh_tensor()},
        {FINAL_STATE_TENSOR, outputs[1].mesh_tensor()},
    };
    if (!summary) {
        run_args.tensor_args.emplace(Q_DECAY_TENSOR, q_decay_tensor);
        run_args.tensor_args.emplace(INTRA_TENSOR, intra_tensor);
        run_args.tensor_args.emplace(INITIAL_STATE_TENSOR, in.initial_state->mesh_tensor());
    }
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
