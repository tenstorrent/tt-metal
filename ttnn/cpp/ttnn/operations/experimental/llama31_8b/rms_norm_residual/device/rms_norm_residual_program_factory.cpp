// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "rms_norm_residual_device_operation.hpp"
#include "rms_norm_residual_table.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::llama31_8b {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

namespace m2 = tt::tt_metal::experimental;

constexpr const char* KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/rms_norm_residual/device/kernels/";

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE{"compute"};

const m2::DFBSpecName IN{"in"};
const m2::DFBSpecName INB{"inb"};
const m2::DFBSpecName SCALER{"scaler"};
const m2::DFBSpecName EPS{"eps"};
const m2::DFBSpecName GAMMA{"gamma"};
const m2::DFBSpecName OUT{"out"};
const m2::DFBSpecName EX2{"ex2"};
const m2::DFBSpecName XMM2{"xmm2"};
const m2::DFBSpecName EX2PE{"ex2pe"};
const m2::DFBSpecName FUSION{"fusion"};
const m2::DFBSpecName XMM{"xmm"};
const m2::DFBSpecName HOUT{"hout"};

const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName RESIDUAL{"residual"};
const m2::TensorParamName WEIGHT{"weight"};
const m2::TensorParamName OUTPUT{"output"};
const m2::TensorParamName RESIDUAL_OUT{"residual_out"};

void bind_dfb(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name, m2::DFBEndpointType role) {
    kernel.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::move(accessor_name),
        .endpoint_type = role,
    });
}

void bind_self_loop(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name) {
    bind_dfb(kernel, dfb, accessor_name, m2::DFBEndpointType::PRODUCER);
    bind_dfb(kernel, dfb, std::move(accessor_name), m2::DFBEndpointType::CONSUMER);
}

void bind_tensor(m2::KernelSpec& kernel, const m2::TensorParamName& tensor, std::string accessor_name) {
    kernel.tensor_bindings.push_back(m2::TensorBinding{
        .tensor_parameter_name = tensor,
        .accessor_name = std::move(accessor_name),
    });
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts RmsNormResidualProgramFactory::create_program_artifacts(
    const RmsNormResidualOperation::operation_attributes_t& operation_attributes,
    const RmsNormResidualOperation::tensor_args_t& tensor_args,
    RmsNormResidualOperation::tensor_return_value_t& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual;
    const auto& gamma = tensor_args.weight;
    const auto& h = tensor_args.residual_out;
    auto* device = a.device();

    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const auto grid = device->compute_with_storage_grid_size();
    const auto* table = rms_norm_residual_table::find(grid.x, grid.y, a.padded_shape()[-2], a.padded_shape()[-1]);
    TT_FATAL(table != nullptr, "llama31_8b.rms_norm_residual: no entry for this shape and grid");
    const std::uint32_t W = a.logical_shape()[-1];
    const std::uint32_t Wt = table->row_tiles;
    const std::uint32_t block_size = table->block;

    const tt::DataFormat in_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat inb_format = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat out_format = datatype_to_dataformat_converter(output.dtype());
    const tt::DataFormat gamma_format = datatype_to_dataformat_converter(gamma.dtype());
    const tt::DataFormat interm_format = tt::DataFormat::Float32;
    const tt::DataFormat bf16_format = tt::DataFormat::Float16_b;

    const std::uint32_t in_tile = tt::tile_size(in_format);
    const std::uint32_t inb_tile = tt::tile_size(inb_format);
    const std::uint32_t out_tile = tt::tile_size(out_format);
    const std::uint32_t gamma_tile = tt::tile_size(gamma_format);
    const std::uint32_t interm_tile = tt::tile_size(interm_format);
    const std::uint32_t bf16_tile = tt::tile_size(bf16_format);
    std::vector<CoreRange> ranges;
    for (const auto& w : table->cores) {
        ranges.emplace_back(CoreCoord{w.x, w.y}, CoreCoord{w.x, w.y});
    }
    const CoreRangeSet all_cores(ranges);

    m2::ProgramSpec spec{.name = "llama31_8b_rms_norm_residual"};
    auto add_dfb = [&spec](
                       const m2::DFBSpecName& id, std::uint32_t entries, std::uint32_t size, tt::DataFormat format) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = id,
            .entry_size = size,
            .num_entries = entries,
            .data_format_metadata = format,
        });
    };
    add_dfb(IN, table->in_pages, in_tile, in_format);
    add_dfb(OUT, table->out_pages, out_tile, out_format);
    add_dfb(SCALER, 2, bf16_tile, bf16_format);
    add_dfb(EPS, 2, bf16_tile, bf16_format);
    add_dfb(EX2, 2, interm_tile, interm_format);
    add_dfb(XMM, Wt, in_tile, in_format);
    add_dfb(HOUT, table->hout_pages, in_tile, in_format);
    add_dfb(XMM2, Wt, interm_tile, interm_format);
    add_dfb(EX2PE, table->ex2pe_pages, interm_tile, interm_format);
    add_dfb(FUSION, table->fusion_pages, interm_tile, interm_format);
    add_dfb(GAMMA, Wt, gamma_tile, gamma_format);
    add_dfb(INB, table->in_pages, inb_tile, inb_format);
    std::uint64_t dfb_bytes = 0;
    for (const auto& dfb : spec.dataflow_buffers) {
        dfb_bytes += std::uint64_t{dfb.entry_size} * dfb.num_entries;
    }
    TT_FATAL(
        dfb_bytes == rms_norm_residual_table::cb_bytes(
                         *table, {in_tile, inb_tile, out_tile, gamma_tile, interm_tile, bf16_tile}),
        "llama31_8b.rms_norm_residual: buffer plan and L1 check disagree");

    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = RESIDUAL, .spec = b.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = WEIGHT, .spec = gamma.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = RESIDUAL_OUT, .spec = h.tensor_spec()});

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(KERNEL_DIR) + "reader.cpp",
        .compile_time_args = {{"block_size", block_size}, {"W", W}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt", "Wt", "reader_start", "eps"}},
        .hw_config = create_reader_datamovement_config(),
    };
    bind_dfb(reader, IN, "in", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, EPS, "eps", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, SCALER, "scaler", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, INB, "inb", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, GAMMA, "gamma", m2::DFBEndpointType::PRODUCER);
    bind_tensor(reader, INPUT, "src");
    bind_tensor(reader, RESIDUAL, "src_b");
    bind_tensor(reader, WEIGHT, "gamma");

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(KERNEL_DIR) + "writer.cpp",
        .compile_time_args = {{"block_size", block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"Wt", "num_tile_rows", "writer_start"}},
        .hw_config = create_writer_datamovement_config(),
    };
    bind_dfb(writer, OUT, "out", m2::DFBEndpointType::CONSUMER);
    bind_dfb(writer, HOUT, "hout", m2::DFBEndpointType::CONSUMER);
    bind_tensor(writer, OUTPUT, "dst");
    bind_tensor(writer, RESIDUAL_OUT, "srch");

    auto compute_hw = to_compute_hardware_config(operation_attributes.compute_kernel_config);
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(KERNEL_DIR) + "compute.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .compile_time_args =
            {{"Wt", Wt},
             {"block_size", block_size},
             {"do_gamma", 1u},
             {"do_beta", 0u},
             {"fp32_dest_acc_en", static_cast<std::uint32_t>(fp32_dest_acc_en)},
             {"W", W},
             {"tile_width", TILE_WIDTH},
             {"float32_reduction", static_cast<std::uint32_t>(fp32_dest_acc_en)}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
        .hw_config = compute_hw,
    };
    bind_dfb(compute, IN, "in", m2::DFBEndpointType::CONSUMER);
    bind_dfb(compute, OUT, "out", m2::DFBEndpointType::PRODUCER);
    bind_dfb(compute, EPS, "eps", m2::DFBEndpointType::CONSUMER);
    bind_self_loop(compute, EX2, "ex2");
    bind_self_loop(compute, EX2PE, "ex2pe");
    bind_dfb(compute, INB, "inb", m2::DFBEndpointType::CONSUMER);
    bind_dfb(compute, HOUT, "hout", m2::DFBEndpointType::PRODUCER);
    bind_dfb(compute, SCALER, "scaler", m2::DFBEndpointType::CONSUMER);
    bind_self_loop(compute, XMM2, "xmm2");
    bind_dfb(compute, GAMMA, "gamma", m2::DFBEndpointType::CONSUMER);
    bind_self_loop(compute, FUSION, "fusion");
    bind_self_loop(compute, XMM, "xmm");
    {
        auto& modes = std::get<m2::ComputeHardwareConfig>(compute.hw_config).unpack_modes;
        for (const auto& binding : compute.dfb_bindings) {
            if (binding.endpoint_type != m2::DFBEndpointType::CONSUMER) {
                continue;
            }
            for (const auto& dfb : spec.dataflow_buffers) {
                if (dfb.unique_id == binding.dfb_spec_name && dfb.data_format_metadata == tt::DataFormat::Float32) {
                    modes.emplace(binding.dfb_spec_name, UnpackMode::UnpackToSrc);
                }
            }
        }
    }

    m2::KernelRunArgs reader_run_args{.kernel = READER};
    m2::KernelRunArgs writer_run_args{.kernel = WRITER};
    m2::KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (const auto& w : table->cores) {
        const CoreCoord core{w.x, w.y};
        const std::uint32_t rows = w.tile_rows;
        const std::uint32_t tile_offset = w.tile_offset;
        m2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"NCHt", rows},
             {"Wt", Wt},
             {"reader_start", tile_offset},
             {"eps", std::bit_cast<std::uint32_t>(operation_attributes.eps)}});
        m2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"Wt", Wt}, {"num_tile_rows", rows}, {"writer_start", tile_offset}});
        m2::AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"NCHt", rows}});
    }

    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    }};

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args.emplace(INPUT, a.mesh_tensor());
    run_args.tensor_args.emplace(RESIDUAL, b.mesh_tensor());
    run_args.tensor_args.emplace(WEIGHT, gamma.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    run_args.tensor_args.emplace(RESIDUAL_OUT, h.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::llama31_8b
