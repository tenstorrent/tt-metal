// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// TransposeRmFactory implementation — see header for op semantics.
//
// Dispatch model: total work units = B * (A/32) * (C/32) where B is the
// product of leading dims.  We split units evenly across a pow-2-sized
// multi-core grid (same grid-picking logic as the FFT factories).  Each
// core processes `units_per_core` consecutive units.  Linear-to-3D
// decode is in the kernel.
//
// No twiddle, no compute kernel — only reader + writer.  Two CB slots
// double-buffer the 32×32 block staging area so the reader can stay one
// block ahead of the writer.

#include "transpose_rm_factory.hpp"

#include <cstdint>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "stockham_host.hpp"  // pick_batch_grid, max_cores_for_grid, batch_logical_core

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr uint32_t T_BLOCK = 32u;
const KernelSpecName TR_READER{"reader"};
const KernelSpecName TR_WRITER{"writer"};
const DFBSpecName TR_BLOCK{"block"};
const TensorParamName TR_INPUT{"input"};
const TensorParamName TR_OUTPUT{"output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeRmFactory::create_program_artifacts(
    const TransposeRmParams&, const TransposeRmTensorArgs& tensor_args, ttnn::Tensor& tensor_return_value) {
    const auto& x = tensor_args.input;
    const auto& y = tensor_return_value;
    const auto& s_x = x.padded_shape();

    const uint32_t A = static_cast<uint32_t>(s_x[-2]);
    const uint32_t C = static_cast<uint32_t>(s_x[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(s_x.size()) - 2; ++d) {
        B *= static_cast<uint32_t>(s_x[d]);
    }

    const uint32_t A_tiles = A / T_BLOCK;
    const uint32_t C_tiles = C / T_BLOCK;
    const uint32_t num_units = B * A_tiles * C_tiles;
    TT_FATAL(num_units > 0u, "transpose_rm: zero work units (B={}, A_tiles={}, C_tiles={}).", B, A_tiles, C_tiles);

    const DataType dtype = x.dtype();
    const bool is_bf16 = (dtype == DataType::BFLOAT16);
    const uint32_t elem_bytes = is_bf16 ? 2u : 4u;
    const uint32_t block_bytes = T_BLOCK * T_BLOCK * elem_bytes;

    TT_FATAL(x.buffer() && y.buffer(), "transpose_rm: input/output tensors must be on device.");

    auto* device_raw = x.device();

    // ── Pick a pow-2 core count that divides num_units cleanly ─────────
    const auto dev_grid = device_raw->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (num_units < max_cores) ? num_units : max_cores;
    while (num_cores > 1u && (num_units % num_cores) != 0u) {
        num_cores >>= 1;
    }
    TT_FATAL(
        num_cores >= 1u && (num_units % num_cores) == 0u,
        "transpose_rm: failed to pick num_cores for num_units={}.",
        num_units);
    const uint32_t units_per_core = num_units / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    DataflowBufferSpec block_dfb{
        .unique_id = TR_BLOCK,
        .entry_size = block_bytes,
        .num_entries = 2,
        .data_format_metadata = is_bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32,
    };

    const uint32_t is_bf16_flag = is_bf16 ? 1u : 0u;

    KernelSpec reader{
        .unique_id = TR_READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/transpose_rm_reader.cpp",
        .dfb_bindings =
            {DFBBinding{
                .dfb_spec_name = TR_BLOCK,
                .accessor_name = "block",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = TR_INPUT, .accessor_name = "src"}},
        .compile_time_args = {{"a_tiles", A_tiles}, {"c_tiles", C_tiles}, {"is_bf16", is_bf16_flag}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_unit", "num_units"}},
        .hw_config = ttnn::create_reader_datamovement_config(device_raw->arch()),
    };

    KernelSpec writer{
        .unique_id = TR_WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/transpose_rm_writer.cpp",
        .dfb_bindings =
            {DFBBinding{
                .dfb_spec_name = TR_BLOCK,
                .accessor_name = "block",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = TR_OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = {{"a_tiles", A_tiles}, {"c_tiles", C_tiles}, {"is_bf16", is_bf16_flag}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_unit", "num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(device_raw->arch()),
    };

    KernelRunArgs reader_run_args{.kernel = TR_READER};
    KernelRunArgs writer_run_args{.kernel = TR_WRITER};

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
        const uint32_t base = c * units_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            logical,
            {{"base_unit", base}, {"num_units", units_per_core}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            logical,
            {{"base_unit", base}, {"num_units", units_per_core}});
    }

    ProgramSpec spec{
        .name = "fft_transpose_rm",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(block_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = TR_INPUT, .spec = x.tensor_spec()},
             TensorParameter{.unique_id = TR_OUTPUT, .spec = y.tensor_spec()}},
        .work_units =
            {WorkUnitSpec{
                .name = "main",
                .kernels = {TR_READER, TR_WRITER},
                .target_nodes = crs,
            }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {TR_INPUT, TensorArgument{x.mesh_tensor()}},
        {TR_OUTPUT, TensorArgument{y.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
