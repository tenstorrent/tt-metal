// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// RebankRmMergeFactory — inverse page-size conversion for ROW_MAJOR tensors.
//
// Converts (B_total * N2, N1) with page_size = N1*elem_bytes
//       to (B_total, N1*N2)   with page_size = N1*N2*elem_bytes.
//
// No compute kernel — only reader + writer sharing one double-buffered CB.
// CB size = 2 * N1 * elem_bytes (one "slot" per double-buffer side, tiny).
//
// Work unit = one source row of N1 elements.
// num_units = B_total * N2.
// Reader: sequential full-page reads from source.
// Writer: writes at byte offset (unit % N2) * N1 * elem_bytes within dest page
//         (unit / N2).

#include "rebank_rm_merge_factory.hpp"

#include <cstdint>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "stockham_host.hpp"
#include "rebank_rm_device_operation_types.hpp"  // rebank_is_pow2

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

const KernelSpecName RM_READER{"reader"};
const KernelSpecName RM_WRITER{"writer"};
const DFBSpecName RM_BLOCK{"block"};
const TensorParamName RM_INPUT{"input"};
const TensorParamName RM_OUTPUT{"output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts RebankRmMergeFactory::create_program_artifacts(
    const RebankRmMergeParams& operation_attributes,
    const RebankRmMergeTensorArgs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const auto& x = tensor_args.input;
    const auto& s_x = x.padded_shape();
    const uint32_t chunks_per_merge = operation_attributes.chunks_per_merge;

    TT_FATAL(s_x.size() == 2u, "rebank_rm_merge: input must be 2D (got {}D).", s_x.size());

    const uint32_t N1 = static_cast<uint32_t>(s_x[-1]);         // source last-dim
    const uint32_t B_total_in = static_cast<uint32_t>(s_x[0]);  // B * chunks_per_merge
    TT_FATAL(
        B_total_in % chunks_per_merge == 0u,
        "rebank_rm_merge: input rows {} not divisible by chunks_per_merge {}.",
        B_total_in,
        chunks_per_merge);

    const uint32_t num_units = B_total_in;  // one unit per source row

    const DataType dtype = x.dtype();
    const bool is_bf16 = (dtype == DataType::BFLOAT16);
    const uint32_t elem_bytes = is_bf16 ? 2u : 4u;
    const uint32_t chunk_bytes = N1 * elem_bytes;

    const auto& y = tensor_return_value;
    TT_FATAL(x.buffer() && y.buffer(), "rebank_rm_merge: input/output tensors must be on device.");

    auto* device_raw = x.device();

    // ── Core grid ─────────────────────────────────────────────────────
    const auto dev_grid = device_raw->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(dev_grid.x, dev_grid.y);
    uint32_t num_cores = (num_units < max_cores) ? num_units : max_cores;
    while (num_cores > 1u && (num_units % num_cores) != 0u) {
        num_cores >>= 1u;
    }
    TT_FATAL(
        num_cores >= 1u && (num_units % num_cores) == 0u,
        "rebank_rm_merge: failed to pick num_cores for num_units={}.",
        num_units);

    // Validate that the resulting core grid fits within the physical device.
    {
        auto [gc, gr] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);
        while (num_cores > 1u && gr > dev_grid.y) {
            --num_cores;
            while (num_cores > 1u && (num_units % num_cores) != 0u) {
                --num_cores;
            }
            std::tie(gc, gr) = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);
        }
    }

    const uint32_t units_per_core = num_units / num_cores;
    auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, dev_grid.x);

    const CoreCoord first{0, 0};
    const CoreCoord last{grid_cols - 1u, grid_rows - 1u};
    const CoreRange cr(first, last);
    const CoreRangeSet crs({cr});

    DataflowBufferSpec block_dfb{
        .unique_id = RM_BLOCK,
        .entry_size = chunk_bytes,
        .num_entries = 2,
        .data_format_metadata = is_bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Float32,
    };

    const uint32_t is_bf16_flag = is_bf16 ? 1u : 0u;

    KernelSpec reader{
        .unique_id = RM_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/rebank_rm_merge_reader.cpp",
        .dfb_bindings =
            {DFBBinding{
                .dfb_spec_name = RM_BLOCK,
                .accessor_name = "block",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = RM_INPUT, .accessor_name = "src"}},
        .compile_time_args = {{"chunk", N1}, {"is_bf16", is_bf16_flag}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_unit", "num_units"}},
        .hw_config = ttnn::create_reader_datamovement_config(device_raw->arch()),
    };

    KernelSpec writer{
        .unique_id = RM_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/rebank_rm_merge_writer.cpp",
        .dfb_bindings =
            {DFBBinding{
                .dfb_spec_name = RM_BLOCK,
                .accessor_name = "block",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = RM_OUTPUT, .accessor_name = "dst"}},
        .compile_time_args =
            {{"chunk", N1}, {"chunks_per_merge", chunks_per_merge}, {"is_bf16", is_bf16_flag}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_unit", "num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(device_raw->arch()),
    };

    KernelRunArgs reader_run_args{.kernel = RM_READER};
    KernelRunArgs writer_run_args{.kernel = RM_WRITER};

    for (uint32_t c = 0u; c < num_cores; ++c) {
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
        .name = "fft_rebank_rm_merge",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(block_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = RM_INPUT, .spec = x.tensor_spec()},
             TensorParameter{.unique_id = RM_OUTPUT, .spec = y.tensor_spec()}},
        .work_units =
            {WorkUnitSpec{
                .name = "main",
                .kernels = {RM_READER, RM_WRITER},
                .target_nodes = crs,
            }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {RM_INPUT, TensorArgument{x.mesh_tensor()}},
        {RM_OUTPUT, TensorArgument{y.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
