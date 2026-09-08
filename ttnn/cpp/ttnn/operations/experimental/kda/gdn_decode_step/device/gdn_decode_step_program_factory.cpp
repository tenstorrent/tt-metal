// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/kda/gdn_decode_step/device/gdn_decode_step_program_factory.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {
uint32_t float_bits(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, sizeof(float));
    return b;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts GdnDecodeStepProgramFactory::create_program_artifacts(
    const GdnDecodeStepParams& a, const GdnDecodeStepInputs& in, Tensor& output_tensor) {
    const auto& qkv = in.qkv.mesh_tensor();
    const auto& beta = in.beta.mesh_tensor();
    const auto& g = in.g.mesh_tensor();
    const auto& state = in.state.mesh_tensor();
    const auto& weight = in.weight.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    const auto& device = qkv.device();
    const auto arch = device.arch();

    const uint32_t Kt = a.key_dim / TILE_WIDTH;
    const uint32_t Vt = a.value_dim / TILE_WIDTH;
    const uint32_t KV = Kt * Vt;
    const uint32_t Nv = a.num_value_heads;
    const uint32_t Nk = a.num_key_heads;

    const auto grid = device.compute_with_storage_grid_size();
    auto dist = kda_factory_detail::distribute_prep(grid, Nv, Nv);  // one head per core
    const auto& cores = dist.core_set;

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName QKV{"qkv"};
    const m2::TensorParamName BETA{"beta"};
    const m2::TensorParamName G{"g"};
    const m2::TensorParamName STATE{"state"};
    const m2::TensorParamName WEIGHT{"weight"};
    const m2::TensorParamName OUT{"out"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto in_fmt = datatype_to_dataformat_converter(in.qkv.dtype());
    const auto out_fmt = datatype_to_dataformat_converter(a.output_dtype);

    struct Dfb {
        const char* name;
        uint32_t tiles;
        tt::DataFormat fmt;
    };
    const uint32_t tmp_tiles = std::max(Kt, Vt);
    const std::vector<Dfb> dfb_list = {
        {"q_in", Kt, in_fmt}, {"k_in", Kt, in_fmt},   {"v_in", Vt, in_fmt}, {"beta_s", 1, fp32},
        {"g_s", 1, fp32},     {"state_in", KV, fp32}, {"w_in", Vt, bf16},   {"scaler", 1, fp32},
        {"eps_l2", 1, bf16},  {"eps_norm", 1, bf16},  {"mask", 1, bf16},    {"tmp", tmp_tiles, fp32},
        {"stats", 1, fp32},   {"scratch", 1, fp32},   {"inv", 1, fp32},     {"qn", Kt, fp32},
        {"kn", Kt, fp32},     {"vm", Vt, fp32},       {"dec", 1, fp32},     {"hd", KV, fp32},
        {"vread", Vt, fp32},  {"delta", Vt, fp32},    {"kt", Kt, fp32},     {"outer", KV, fp32},
        {"hn", KV, fp32},     {"hnew", KV, fp32},     {"o", Vt, fp32},      {"on", Vt, fp32},
        {"out", Vt, out_fmt},
    };
    m2::Group<m2::DataflowBufferSpec> dfbs;
    for (const auto& d : dfb_list) {
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{d.name},
            .entry_size = tt::tile_size(d.fmt),
            .num_entries = d.tiles,
            .data_format_metadata = d.fmt,
        });
    }
    auto bind = [](const char* name, m2::DFBEndpointType type) {
        return m2::DFBBinding{m2::DFBSpecName{name}, name, type};
    };
    using EP = m2::DFBEndpointType;

    const uint32_t beta_fp32 = in.beta.dtype() == DataType::FLOAT32 ? 1u : 0u;
    const uint32_t g_fp32 = in.g.dtype() == DataType::FLOAT32 ? 1u : 0u;

    m2::KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/"
            "reader_gdn_decode_step.cpp",
        .dfb_bindings =
            {bind("q_in", EP::PRODUCER),
             bind("k_in", EP::PRODUCER),
             bind("v_in", EP::PRODUCER),
             bind("beta_s", EP::PRODUCER),
             bind("g_s", EP::PRODUCER),
             bind("state_in", EP::PRODUCER),
             bind("w_in", EP::PRODUCER),
             bind("scaler", EP::PRODUCER),
             bind("eps_l2", EP::PRODUCER),
             bind("eps_norm", EP::PRODUCER),
             bind("mask", EP::PRODUCER)},
        .tensor_bindings =
            {m2::TensorBinding{QKV, "qkv"},
             m2::TensorBinding{BETA, "beta"},
             m2::TensorBinding{G, "g"},
             m2::TensorBinding{STATE, "state"},
             m2::TensorBinding{WEIGHT, "weight"}},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"Nk", Nk},
             {"Nv", Nv},
             {"beta_fp32", beta_fp32},
             {"g_fp32", g_fp32},
             {"l2_eps_bits", float_bits(a.l2_epsilon)},
             {"norm_eps_bits", float_bits(a.norm_epsilon)}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/"
            "writer_gdn_decode_step.cpp",
        .dfb_bindings = {bind("hnew", EP::CONSUMER), bind("out", EP::CONSUMER)},
        .tensor_bindings = {m2::TensorBinding{STATE, "state_out"}, m2::TensorBinding{OUT, "out"}},
        .compile_time_args = {{"Kt", Kt}, {"Vt", Vt}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, a.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& d : dfb_list) {
        if (d.fmt == fp32) {
            unpack_modes[m2::DFBSpecName{d.name}] = UnpackMode::UnpackToSrc;
        }
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_decode_step.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = {bind("q_in", EP::CONSUMER),     bind("k_in", EP::CONSUMER),    bind("v_in", EP::CONSUMER),
                         bind("beta_s", EP::CONSUMER),   bind("g_s", EP::CONSUMER),     bind("state_in", EP::CONSUMER),
                         bind("w_in", EP::CONSUMER),     bind("scaler", EP::CONSUMER),  bind("eps_l2", EP::CONSUMER),
                         bind("eps_norm", EP::CONSUMER), bind("mask", EP::CONSUMER),    bind("tmp", EP::PRODUCER),
                         bind("tmp", EP::CONSUMER),      bind("stats", EP::PRODUCER),   bind("stats", EP::CONSUMER),
                         bind("scratch", EP::PRODUCER),  bind("scratch", EP::CONSUMER), bind("inv", EP::PRODUCER),
                         bind("inv", EP::CONSUMER),      bind("qn", EP::PRODUCER),      bind("qn", EP::CONSUMER),
                         bind("kn", EP::PRODUCER),       bind("kn", EP::CONSUMER),      bind("vm", EP::PRODUCER),
                         bind("vm", EP::CONSUMER),       bind("dec", EP::PRODUCER),     bind("dec", EP::CONSUMER),
                         bind("hd", EP::PRODUCER),       bind("hd", EP::CONSUMER),      bind("vread", EP::PRODUCER),
                         bind("vread", EP::CONSUMER),    bind("delta", EP::PRODUCER),   bind("delta", EP::CONSUMER),
                         bind("kt", EP::PRODUCER),       bind("kt", EP::CONSUMER),      bind("outer", EP::PRODUCER),
                         bind("outer", EP::CONSUMER),    bind("hn", EP::PRODUCER),      bind("hn", EP::CONSUMER),
                         bind("hnew", EP::PRODUCER),     bind("o", EP::PRODUCER),       bind("o", EP::CONSUMER),
                         bind("on", EP::PRODUCER),       bind("on", EP::CONSUMER),      bind("out", EP::PRODUCER)},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"scale_bits", float_bits(a.scale)},
             {"inv_dv_bits", float_bits(1.0f / static_cast<float>(a.value_dim))}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_count"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_args{.kernel = READER};
    m2::KernelRunArgs writer_args{.kernel = WRITER};
    m2::KernelRunArgs compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"wi_count", dist.wi_count[i]}});
    }

    m2::ProgramSpec spec{
        .name = "gdn_decode_step",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {m2::TensorParameter{.unique_id = QKV, .spec = qkv.tensor_spec()},
             m2::TensorParameter{.unique_id = BETA, .spec = beta.tensor_spec()},
             m2::TensorParameter{.unique_id = G, .spec = g.tensor_spec()},
             m2::TensorParameter{.unique_id = STATE, .spec = state.tensor_spec()},
             m2::TensorParameter{.unique_id = WEIGHT, .spec = weight.tensor_spec()},
             m2::TensorParameter{.unique_id = OUT, .spec = output.tensor_spec()}},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = cores}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.kernel_run_args.push_back(std::move(compute_args));
    run_args.tensor_args = {{QKV, qkv}, {BETA, beta}, {G, g}, {STATE, state}, {WEIGHT, weight}, {OUT, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
