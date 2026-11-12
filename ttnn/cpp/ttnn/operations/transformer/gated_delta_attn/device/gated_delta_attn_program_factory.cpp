// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the gated-delta-attention sequential scan kernel (Path A).
//
// Parallelism: one Tensix core per head.
// Each core: forward substitution (using pre-computed L_inv) + inter-chunk state scan.

#include "ttnn/operations/transformer/gated_delta_attn/device/gated_delta_attn_device_operation.hpp"

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts
GatedDeltaAttnSeqDeviceOperation::GatedDeltaAttnSeqProgramFactory::create_program_artifacts(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    const uint32_t BH = attrs.num_heads;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t C = attrs.chunk_size;
    const uint32_t Dk = attrs.key_dim;
    const uint32_t Dv = attrs.val_dim;

    const uint32_t Ct = C / TILE_HEIGHT;  // e.g. 4
    const uint32_t Kt = Dk / TILE_WIDTH;  // e.g. 4
    const uint32_t Vt = Dv / TILE_WIDTH;  // e.g. 4

    const uint32_t state_tiles = Kt * Vt;
    const uint32_t out_tiles = Ct * Vt;
    const uint32_t in_kv_tiles = Ct * Kt;
    const uint32_t attn_tiles = Ct * Ct;
    const uint32_t kdt_tiles = Kt * Ct;

    // Build per-head core coordinates (column-major to avoid harvested NOC columns).
    auto* device = in.L_unit.device();
    CoreCoord compute_grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = compute_grid.y;

    TT_FATAL(
        BH <= compute_grid.x * grid_y,
        "num_heads {} exceeds total compute cores {}×{}={}",
        BH,
        compute_grid.x,
        grid_y,
        compute_grid.x * grid_y);

    std::vector<CoreCoord> head_cores(BH);
    for (uint32_t h = 0; h < BH; h++) {
        head_cores[h] = CoreCoord{h / grid_y, h % grid_y};
    }

    std::set<CoreRange> core_set;
    for (auto& c : head_cores) {
        core_set.insert(CoreRange{c, c});
    }
    CoreRangeSet cores{core_set};

    tt::DataFormat df_f32 = tt::DataFormat::Float32;

    const bool has_initial_state = in.initial_state.has_value();

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // -----------------------------------------------------------------------
    // Dataflow buffers — all fp32
    // -----------------------------------------------------------------------
    // Each DFB's accessor name (the kernel-side dfb::<name>) is its spec name.
    std::vector<DataflowBufferSpec> dfbs;
    auto make_dfb = [&](const std::string& name, tt::DataFormat fmt, uint32_t n_tiles, uint32_t n_bufs = 1) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = DFBSpecName{name},
            .entry_size = tt::tile_size(fmt),
            .num_entries = n_tiles * n_bufs,
            .data_format_metadata = fmt,
        });
    };

    // Per-chunk inputs — single-buffered to stay within L1 budget (1.5 MB).
    make_dfb("L_unit", df_f32, attn_tiles, 1);     // L_unit [C,C]
    make_dfb("v_beta_sc", df_f32, out_tiles, 1);   // v_beta_sc [C,Dv]
    make_dfb("k_bd_sc", df_f32, in_kv_tiles, 1);   // k_bd_sc [C,Dk]
    make_dfb("intra_att", df_f32, attn_tiles, 1);  // intra_attn [C,C]
    make_dfb("q_decay", df_f32, in_kv_tiles, 1);   // q_decay [C,Dk]
    make_dfb("k_dt", df_f32, kdt_tiles, 1);        // k_decay_t [Dk,C]
    make_dfb("dl_exp", df_f32, 1, 1);              // dl_exp (fp32 scalar)
    make_dfb("S0", df_f32, state_tiles, 1);        // S0 — initial state, seeded by reader, consumed on chunk 0
    make_dfb("S", df_f32, state_tiles, 1);         // S — persistent state

    // Forward-substitution scratch
    // DFBs nm_P_a, nm_P_b, nm_R_a hold max(Vt, Kt) tiles for fwd_sub rows.
    const uint32_t scratch_Xt = std::max(Vt, Kt);
    make_dfb("nm_P_a", df_f32, scratch_Xt, 1);  // fwd_rhs
    make_dfb("nm_P_b", df_f32, scratch_Xt, 1);  // corr_mm
    make_dfb("nm_R_a", df_f32, scratch_Xt, 1);  // temp_rhs
    // Legacy indices 12, 13: unused (were Neumann-only)

    // Diagonal block inverses — 1 tile each, loaded per chunk by reader
    make_dfb("L_inv_0", df_f32, 1, 1);  // L_inv0
    make_dfb("L_inv_1", df_f32, 1, 1);  // L_inv1
    make_dfb("L_inv_2", df_f32, 1, 1);  // L_inv2
    make_dfb("L_inv_3", df_f32, 1, 1);  // L_inv3

    // Forward-substitution outputs
    make_dfb("v_cor", df_f32, out_tiles, 1);    // v_cor = L^{-1} @ v_beta_sc
    make_dfb("k_cum", df_f32, in_kv_tiles, 1);  // k_cum = L^{-1} @ k_bd_sc

    // 7-step scratch
    make_dfb("v_prime", df_f32, out_tiles, 1);        // v_prime
    make_dfb("v_new", df_f32, out_tiles, 1);          // v_new
    make_dfb("o_inter", df_f32, out_tiles, 1);        // o_inter
    make_dfb("intra_v", df_f32, out_tiles, 1);        // intra_v
    make_dfb("out", df_f32, out_tiles, 2);            // out (writer pops, double-buffered)
    make_dfb("s_upd", df_f32, state_tiles, 1);        // s_upd
    make_dfb("S_tmp", df_f32, state_tiles, 1);        // S_tmp
    make_dfb("final_state", df_f32, state_tiles, 1);  // final_state (writer reads once)

    // Endpoint roles. Reader fills the per-chunk inputs and seeds S0 (initial state, or zeros); compute
    // drains them; compute fills out and final_state for the writer. Compute-only buffers (scratch,
    // forward-substitution outputs, 7-step intermediates, and S, which compute recycles the updated
    // state into on every non-last chunk) are produced and consumed by compute itself.
    const std::vector<std::string> reader_to_compute = {
        "L_unit",
        "v_beta_sc",
        "k_bd_sc",
        "intra_att",
        "q_decay",
        "k_dt",
        "dl_exp",
        "S0",
        "L_inv_0",
        "L_inv_1",
        "L_inv_2",
        "L_inv_3",
    };
    const std::vector<std::string> compute_to_writer = {"out", "final_state"};
    const std::vector<std::string> compute_self_loop = {
        "S",
        "nm_P_a",
        "nm_P_b",
        "nm_R_a",
        "v_cor",
        "k_cum",
        "v_prime",
        "v_new",
        "o_inter",
        "intra_v",
        "s_upd",
        "S_tmp",
    };

    auto binding = [](const std::string& name, DFBEndpointType endpoint) {
        return DFBBinding{
            .dfb_spec_name = DFBSpecName{name},
            .accessor_name = name,
            .endpoint_type = endpoint,
        };
    };

    Group<DFBBinding> reader_dfb_bindings;
    Group<DFBBinding> writer_dfb_bindings;
    Group<DFBBinding> compute_dfb_bindings;
    for (const auto& name : reader_to_compute) {
        reader_dfb_bindings.push_back(binding(name, DFBEndpointType::PRODUCER));
        compute_dfb_bindings.push_back(binding(name, DFBEndpointType::CONSUMER));
    }
    for (const auto& name : compute_self_loop) {
        compute_dfb_bindings.push_back(binding(name, DFBEndpointType::PRODUCER));
        compute_dfb_bindings.push_back(binding(name, DFBEndpointType::CONSUMER));
    }
    for (const auto& name : compute_to_writer) {
        compute_dfb_bindings.push_back(binding(name, DFBEndpointType::PRODUCER));
        writer_dfb_bindings.push_back(binding(name, DFBEndpointType::CONSUMER));
    }

    // The legacy compute config left unpack_to_dest_mode at Default for every buffer. Every DFB is
    // Float32 and the compute kernel runs with a 32-bit Dest, so each one it consumes needs that
    // choice stated explicitly: UnpackToSrc.
    ComputeHardwareConfig::ComputeUnpackModes unpack_modes;
    for (const auto& b : compute_dfb_bindings) {
        if (b.endpoint_type == DFBEndpointType::CONSUMER) {
            unpack_modes.emplace(b.dfb_spec_name, UnpackMode::UnpackToSrc);
        }
    }

    // -----------------------------------------------------------------------
    // Tensors
    // -----------------------------------------------------------------------
    const TensorParamName L_UNIT{"L_unit"};
    const TensorParamName V_BETA_SC{"v_beta_sc"};
    const TensorParamName K_BD_SC{"k_bd_sc"};
    const TensorParamName INTRA_ATTN{"intra_attn"};
    const TensorParamName Q_DECAY{"q_decay"};
    const TensorParamName K_DECAY_T{"k_decay_t"};
    const TensorParamName DL_EXP{"dl_exp"};
    const TensorParamName L_INV{"L_inv"};
    const TensorParamName INITIAL_STATE{"initial_state"};
    const TensorParamName OUTPUT{"output"};
    const TensorParamName FINAL_STATE{"final_state"};

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once.
    std::vector<std::pair<TensorParamName, const MeshTensor*>> reader_tensors = {
        {L_UNIT, &in.L_unit.mesh_tensor()},
        {V_BETA_SC, &in.v_beta_sc.mesh_tensor()},
        {K_BD_SC, &in.k_bd_sc.mesh_tensor()},
        {INTRA_ATTN, &in.intra_attn.mesh_tensor()},
        {Q_DECAY, &in.q_decay.mesh_tensor()},
        {K_DECAY_T, &in.k_decay_t.mesh_tensor()},
        {DL_EXP, &in.dl_exp.mesh_tensor()},
        {L_INV, &in.L_inv.mesh_tensor()},
    };
    // initial_state is optional: it is bound only when present, and the reader zero-fills S0 otherwise.
    if (has_initial_state) {
        reader_tensors.emplace_back(INITIAL_STATE, &in.initial_state->mesh_tensor());
    }
    const std::vector<std::pair<TensorParamName, const MeshTensor*>> writer_tensors = {
        {OUTPUT, &outputs[0].mesh_tensor()},
        {FINAL_STATE, &outputs[1].mesh_tensor()},
    };

    Group<TensorBinding> reader_tensor_bindings;
    Group<TensorBinding> writer_tensor_bindings;
    Group<TensorParameter> tensor_parameters;
    ProgramRunArgs run_args;
    for (const auto& [name, tensor] : reader_tensors) {
        reader_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = name,
            .accessor_name = *name,
        });
        tensor_parameters.push_back(TensorParameter{.unique_id = name, .spec = tensor->tensor_spec()});
        run_args.tensor_args.emplace(name, *tensor);
    }
    for (const auto& [name, tensor] : writer_tensors) {
        writer_tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = name,
            .accessor_name = *name,
        });
        tensor_parameters.push_back(TensorParameter{.unique_id = name, .spec = tensor->tensor_spec()});
        run_args.tensor_args.emplace(name, *tensor);
    }

    // -----------------------------------------------------------------------
    // Kernels
    // -----------------------------------------------------------------------
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/gated_delta_attn/device/kernels/";

    const KernelSpec::CompileTimeArgs ct_args = {{"Ct", Ct}, {"Kt", Kt}, {"Vt", Vt}};

    KernelSpec::CompilerOptions::Defines reader_defines;
    if (has_initial_state) {
        reader_defines.emplace("HAS_INITIAL_STATE", "1");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = kdir + "dataflow/reader_gated_delta_attn.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = ct_args,
        .runtime_arg_schema = {.runtime_arg_names = {"head_idx", "num_chunks"}},
        // The legacy reader ran on RISCV_1 with NOC::RISCV_1_default (not the reader default NOC); kept verbatim.
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                    },
            },
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir + "dataflow/writer_gated_delta_attn.cpp",
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .compile_time_args = ct_args,
        .runtime_arg_schema = {.runtime_arg_names = {"head_idx", "num_chunks"}},
        // The legacy writer ran on RISCV_0 with NOC::RISCV_0_default (not the writer default NOC); kept verbatim.
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
                    },
            },
    };

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kdir + "compute/gated_delta_attn.cpp",
        // Legacy compute kernels default to O3; Metal 2.0 defaults every kernel to O2.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = compute_dfb_bindings,
        .compile_time_args = ct_args,
        .runtime_arg_schema = {.runtime_arg_names = {"num_chunks"}},
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = MathFidelity::HiFi2,
                .sfpu_precision_mode = Precision::Precise,
                .enable_32_bit_dest = true,
                .unpack_modes = unpack_modes,
            },
    };

    // -----------------------------------------------------------------------
    // Per-core runtime arguments
    // -----------------------------------------------------------------------
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t h = 0; h < BH; h++) {
        auto& core = head_cores[h];
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"head_idx", h}, {"num_chunks", NC}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"head_idx", h}, {"num_chunks", NC}});
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_chunks", NC}});
    }

    ProgramSpec spec{
        .name = "gated_delta_attn_seq",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "gated_delta_attn_seq",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = cores,
                },
            },
    };

    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
