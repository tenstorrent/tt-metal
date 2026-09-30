// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/gdn_gates/device/gdn_gates_program_factory.hpp"

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
// bf16 bits of a float, round to nearest even (the packing binary_ng uses for a bf16 scalar).
uint32_t bf16_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(float));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return u >> 16;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts GdnGatesProgramFactory::create_program_artifacts(
    const GdnGatesParams& attrs, const GdnGatesInputs& in, std::vector<Tensor>& outputs) {
    const auto& gab = in.gab.mesh_tensor();
    const auto& dt_bias = in.dt_bias.mesh_tensor();
    const auto& a_neg = in.a_neg.mesh_tensor();
    const auto& beta_out = outputs[0].mesh_tensor();
    const auto& g_out = outputs[1].mesh_tensor();
    const auto& device = gab.device();
    const auto arch = device.arch();

    // beta_scale as binary_ng packs a bf16 scalar: rounded to bf16, then widened back to fp32 bits for fill_tile.
    const uint32_t scale_fp32_bits = bf16_bits(attrs.beta_scale) << 16;
    const uint32_t Mt = attrs.sequence / TILE_HEIGHT;
    const uint32_t gab_row_tiles = in.gab.padded_shape()[-1] / TILE_WIDTH;
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t core_cap = std::min<uint32_t>(Mt, grid.x * grid.y);
    auto dist = kda_factory_detail::distribute_prep(grid, Mt, core_cap);
    const auto& cores = dist.core_set;

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::DFBSpecName A_DFB{"a"};
    const m2::DFBSpecName B_DFB{"b"};
    const m2::DFBSpecName DT_DFB{"dt"};
    const m2::DFBSpecName ANEG_DFB{"aneg"};
    const m2::DFBSpecName DTF_DFB{"dtf"};
    const m2::DFBSpecName ANEGF_DFB{"anegf"};
    const m2::DFBSpecName BS_DFB{"bs"};
    const m2::DFBSpecName SP_DFB{"sp"};
    const m2::DFBSpecName BETA_DFB{"beta"};
    const m2::DFBSpecName G_DFB{"g"};

    const m2::TensorParamName GAB{"gab"};
    const m2::TensorParamName DT{"dt_bias"};
    const m2::TensorParamName ANEG{"a_neg"};
    const m2::TensorParamName BETA{"beta_out"};
    const m2::TensorParamName GOUT{"g_out"};

    auto make_dfb = [](const m2::DFBSpecName& name, uint32_t tiles, tt::DataFormat format) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::tile_size(format),
            .num_entries = tiles,
            .data_format_metadata = format,
        };
    };
    const auto out_format = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;

    m2::Group<m2::DataflowBufferSpec> dfbs = {
        make_dfb(A_DFB, 2, bf16),
        make_dfb(B_DFB, 2, bf16),
        make_dfb(DT_DFB, 1, bf16),
        make_dfb(ANEG_DFB, 1, bf16),
        make_dfb(DTF_DFB, 1, bf16),
        make_dfb(ANEGF_DFB, 1, bf16),
        make_dfb(BS_DFB, 1, bf16),
        make_dfb(SP_DFB, 1, bf16),
        make_dfb(BETA_DFB, 2, out_format),
        make_dfb(G_DFB, 2, out_format),
    };

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/gdn_gates/device/kernels/dataflow/reader_gdn_gates.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{A_DFB, "a", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{B_DFB, "b", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{DT_DFB, "dt", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{ANEG_DFB, "aneg", m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                m2::TensorBinding{GAB, "gab"},
                m2::TensorBinding{DT, "dt_bias"},
                m2::TensorBinding{ANEG, "a_neg"},
            },
        .compile_time_args =
            {{"gab_row_tiles", gab_row_tiles}, {"a_col_tile", attrs.a_col_tile}, {"b_col_tile", attrs.b_col_tile}},
        .runtime_arg_schema = {.runtime_arg_names = {"mt_start", "mt_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/gdn_gates/device/kernels/dataflow/writer_gdn_gates.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{BETA_DFB, "beta", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{G_DFB, "g", m2::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings = {m2::TensorBinding{BETA, "beta_out"}, m2::TensorBinding{GOUT, "g_out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"mt_start", "mt_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(arch, attrs.compute_kernel_config);

    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/kda/gdn_gates/device/kernels/compute/gdn_gates.cpp",
        .dfb_bindings =
            {
                m2::DFBBinding{A_DFB, "a", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{B_DFB, "b", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{DT_DFB, "dt", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{ANEG_DFB, "aneg", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{DTF_DFB, "dtf", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{DTF_DFB, "dtf", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{ANEGF_DFB, "anegf", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{ANEGF_DFB, "anegf", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{BS_DFB, "bs", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{BS_DFB, "bs", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{SP_DFB, "sp", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{SP_DFB, "sp", m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{BETA_DFB, "beta", m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{G_DFB, "g", m2::DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"left_faces_only", static_cast<uint32_t>(attrs.num_heads <= 16 ? 1 : 0)},
             {"scale_bits", scale_fp32_bits}},
        .runtime_arg_schema = {.runtime_arg_names = {"mt_count"}},
        .hw_config = std::move(compute_hw),
    };

    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); i++) {
        const auto& core = dist.cores[i];
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"mt_start", dist.wi_start[i]}, {"mt_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"mt_start", dist.wi_start[i]}, {"mt_count", dist.wi_count[i]}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"mt_count", dist.wi_count[i]}});
    }

    m2::ProgramSpec spec{
        .name = "gdn_gates",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = GAB, .spec = gab.tensor_spec()},
                m2::TensorParameter{.unique_id = DT, .spec = dt_bias.tensor_spec()},
                m2::TensorParameter{.unique_id = ANEG, .spec = a_neg.tensor_spec()},
                m2::TensorParameter{.unique_id = BETA, .spec = beta_out.tensor_spec()},
                m2::TensorParameter{.unique_id = GOUT, .spec = g_out.tensor_spec()},
            },
        .work_units =
            {
                m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = cores,
                },
            },
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {GAB, gab},
        {DT, dt_bias},
        {ANEG, a_neg},
        {BETA, beta_out},
        {GOUT, g_out},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
