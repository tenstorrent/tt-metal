// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "batched_stockham_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "stockham_host.hpp"
#include "stockham_program_spec.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
constexpr uint32_t log2u_bs(uint32_t n) {
    uint32_t result = 0;
    while ((1u << result) < n) {
        ++result;
    }
    return result;
}
constexpr bool is_pow2_bs(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }
}  // namespace

ttnn::device_operation::ProgramArtifacts BatchedStockhamFactory::create_program_artifacts(
    const FFTParams&,
    const FFTTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    namespace shared = stockham_spec;
    const auto& in_r = tensor_args.input_real;
    const auto& in_i = tensor_args.input_imag;
    const auto& tw_r = tensor_args.tw_real;
    const auto& tw_i = tensor_args.tw_imag;
    const auto& out_r = std::get<0>(tensor_return_value);
    const auto& out_i = std::get<1>(tensor_return_value);
    const auto& shape = in_r.padded_shape();
    const uint32_t N = static_cast<uint32_t>(shape[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(shape[d]);
    }
    TT_FATAL(is_pow2_bs(N) && N >= 2u && N <= shared::kTileElems, "BatchedStockhamFactory: invalid N={}", N);
    TT_FATAL(is_pow2_bs(B), "BatchedStockhamFactory: batch must be pow-2 (got {})", B);
    const bool is_bf16 = in_r.dtype() == DataType::BFLOAT16;
    TT_FATAL(
        is_bf16 || in_r.dtype() == DataType::FLOAT32,
        "BatchedStockhamFactory: only fp32 / bf16 supported");
    TT_FATAL(
        in_r.buffer() && in_i.buffer() && tw_r.buffer() && tw_i.buffer() && out_r.buffer() && out_i.buffer(),
        "BatchedStockhamFactory: all tensors must be on device");

    auto* device = in_r.device();
    auto md = device->get_mesh_device();
    const auto grid = md->compute_with_storage_grid_size();
    const uint32_t max_cores = fft_stockham::max_cores_for_grid(grid.x, grid.y);
    const uint32_t num_cores = std::min(B, max_cores);
    TT_FATAL(B % num_cores == 0u, "BatchedStockhamFactory: batch/core split must be exact");
    const uint32_t batch_per_core = B / num_cores;
    const auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, grid.x);
    const CoreRangeSet cores({CoreRange({0, 0}, {grid_cols - 1u, grid_rows - 1u})});

    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (is_bf16) {
        reader_defines["INPUT_BF16"] = "1";
        writer_defines["OUTPUT_BF16"] = "1";
    }
    KernelSpec reader{
        .unique_id = shared::READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_reader.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = shared::reader_bindings(is_bf16, false),
        .tensor_bindings = {
            {.tensor_parameter_name = shared::IN_R, .accessor_name = "in_r"},
            {.tensor_parameter_name = shared::IN_I, .accessor_name = "in_i"},
            {.tensor_parameter_name = shared::TW_R, .accessor_name = "tw_r"},
            {.tensor_parameter_name = shared::TW_I, .accessor_name = "tw_i"}},
        .compile_time_args = {{"sub_n", N}, {"log2_sub_n", log2u_bs(N)}, {"bit_reverse_on_load", 1u}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_tile_idx", "batch_per_core", "noc_x", "noc_y"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch())};
    KernelSpec writer{
        .unique_id = shared::WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/batch_fft_writer.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = shared::writer_bindings(is_bf16, false),
        .tensor_bindings = {
            {.tensor_parameter_name = shared::OUT_R, .accessor_name = "out_r"},
            {.tensor_parameter_name = shared::OUT_I, .accessor_name = "out_i"}},
        .compile_time_args = {{"sub_n", N}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_tile_idx", "batch_per_core"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch())};
    KernelSpec compute = shared::make_compute(log2u_bs(N));

    KernelRunArgs reader_args{.kernel = shared::READER};
    KernelRunArgs writer_args{.kernel = shared::WRITER};
    KernelRunArgs compute_args{.kernel = shared::COMPUTE};
    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, grid_cols);
        const CoreCoord physical = md->worker_core_from_logical_core(logical);
        const uint32_t base = c * batch_per_core;
        AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            logical,
            {{"base_tile_idx", base},
             {"batch_per_core", batch_per_core},
             {"noc_x", physical.x},
             {"noc_y", physical.y}});
        AddRuntimeArgsForNode(
            writer_args.runtime_arg_values,
            logical,
            {{"base_tile_idx", base}, {"batch_per_core", batch_per_core}});
        AddRuntimeArgsForNode(
            compute_args.runtime_arg_values, logical, {{"batch_per_core", batch_per_core}});
    }

    ProgramSpec spec{
        .name = "fft_batched_stockham",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = shared::make_dataflow_buffers(is_bf16, false),
        .tensor_parameters = {
            {.unique_id = shared::IN_R, .spec = in_r.tensor_spec()},
            {.unique_id = shared::IN_I, .spec = in_i.tensor_spec()},
            {.unique_id = shared::TW_R, .spec = tw_r.tensor_spec()},
            {.unique_id = shared::TW_I, .spec = tw_i.tensor_spec()},
            {.unique_id = shared::OUT_R, .spec = out_r.tensor_spec()},
            {.unique_id = shared::OUT_I, .spec = out_i.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {shared::READER, shared::WRITER, shared::COMPUTE}, .target_nodes = cores}}};
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    run_args.tensor_args = {
        {shared::IN_R, in_r.mesh_tensor()},
        {shared::IN_I, in_i.mesh_tensor()},
        {shared::TW_R, tw_r.mesh_tensor()},
        {shared::TW_I, tw_i.mesh_tensor()},
        {shared::OUT_R, out_r.mesh_tensor()},
        {shared::OUT_I, out_i.mesh_tensor()}};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
