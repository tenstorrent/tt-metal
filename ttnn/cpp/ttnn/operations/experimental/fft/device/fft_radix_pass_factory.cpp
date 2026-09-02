// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fft_radix_pass_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
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
constexpr uint32_t log2u_rp(uint32_t n) {
    uint32_t result = 0;
    while ((1u << result) < n) {
        ++result;
    }
    return result;
}
constexpr bool is_pow2_rp(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }
uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
struct WorkGeometry {
    uint32_t batch;
    uint32_t num_cores;
    uint32_t batch_per_core;
    uint32_t grid_cols;
    uint32_t grid_rows;
};
WorkGeometry work_geometry(const Tensor& input) {
    const auto& shape = input.padded_shape();
    uint32_t batch = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        batch *= static_cast<uint32_t>(shape[d]);
    }
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min(batch, fft_stockham::max_cores_for_grid(grid.x, grid.y));
    const auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, grid.x);
    return {batch, num_cores, batch / num_cores, grid_cols, grid_rows};
}
}  // namespace

ttnn::device_operation::ProgramArtifacts FftRadixPassFactory::create_program_artifacts(
    const FftRadixPassParams& attrs,
    const FftRadixPassTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value) {
    namespace shared = stockham_spec;
    const auto& in_r = tensor_args.input_real;
    const auto& in_i = tensor_args.input_imag;
    const auto& tw_r = tensor_args.tw_real;
    const auto& tw_i = tensor_args.tw_imag;
    const auto& out_r = std::get<0>(tensor_return_value);
    const auto& out_i = std::get<1>(tensor_return_value);
    const uint32_t N = static_cast<uint32_t>(in_r.padded_shape()[-1]);
    const auto work = work_geometry(in_r);
    const bool apply_pt = attrs.twiddle_N2 != 0u;
    const bool apply_scale = attrs.output_scale != 1.0f;
    const bool is_bf16 = in_r.dtype() == DataType::BFLOAT16;
    TT_FATAL(N == attrs.P && is_pow2_rp(N) && N >= 2u && N <= 1024u, "FftRadixPassFactory: invalid P");
    TT_FATAL(is_pow2_rp(work.batch) && work.batch % work.num_cores == 0u, "FftRadixPassFactory: invalid batch split");
    TT_FATAL(is_bf16 || in_r.dtype() == DataType::FLOAT32, "FftRadixPassFactory: only fp32 / bf16 supported");
    if (apply_pt) {
        TT_FATAL(
            is_pow2_rp(attrs.twiddle_N2) && attrs.twiddle_N2 <= 1024u &&
                is_pow2_rp(attrs.stride) && attrs.stride <= work.batch &&
                (work.batch % attrs.stride) == 0u &&
                ((work.batch / attrs.stride) % attrs.twiddle_N2) == 0u,
            "FftRadixPassFactory: invalid post-twiddle geometry");
    }

    auto* device = in_r.device();
    auto md = device->get_mesh_device();
    const CoreRangeSet cores(
        {CoreRange({0, 0}, {work.grid_cols - 1u, work.grid_rows - 1u})});

    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (is_bf16) {
        reader_defines["INPUT_BF16"] = "1";
        writer_defines["OUTPUT_BF16"] = "1";
    }
    if (apply_pt) {
        reader_defines["APPLY_POST_TWIDDLE"] = "1";
        writer_defines["APPLY_POST_TWIDDLE"] = "1";
    }
    if (apply_scale) {
        writer_defines["APPLY_SCALE"] = "1";
    }

    Group<TensorBinding> reader_tensor_bindings = {
        {.tensor_parameter_name = shared::IN_R, .accessor_name = "in_r"},
        {.tensor_parameter_name = shared::IN_I, .accessor_name = "in_i"},
        {.tensor_parameter_name = shared::TW_R, .accessor_name = "tw_r"},
        {.tensor_parameter_name = shared::TW_I, .accessor_name = "tw_i"}};
    if (apply_pt) {
        reader_tensor_bindings.push_back(
            {.tensor_parameter_name = shared::PT_R, .accessor_name = "post_tw_r"});
        reader_tensor_bindings.push_back(
            {.tensor_parameter_name = shared::PT_I, .accessor_name = "post_tw_i"});
    }
    KernelSpec reader{
        .unique_id = shared::READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/radix_pass_reader.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = shared::reader_bindings(is_bf16, apply_pt),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args = {{"sub_n", N}, {"log2_sub_n", log2u_rp(N)}, {"bit_reverse_on_load", 1u}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"base_tile_idx", "batch_per_core", "noc_x", "noc_y", "pt_modulus", "pt_stride"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch())};
    KernelSpec writer{
        .unique_id = shared::WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/radix_pass_writer.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = shared::writer_bindings(is_bf16, apply_pt),
        .tensor_bindings = {
            {.tensor_parameter_name = shared::OUT_R, .accessor_name = "out_r"},
            {.tensor_parameter_name = shared::OUT_I, .accessor_name = "out_i"}},
        .compile_time_args = {{"sub_n", N}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_tile_idx", "batch_per_core", "output_scale_bits"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch())};
    KernelSpec compute = shared::make_compute(log2u_rp(N));

    KernelRunArgs reader_args{.kernel = shared::READER};
    KernelRunArgs writer_args{.kernel = shared::WRITER};
    KernelRunArgs compute_args{.kernel = shared::COMPUTE};
    for (uint32_t c = 0; c < work.num_cores; ++c) {
        const CoreCoord logical = fft_stockham::batch_logical_core(c, work.grid_cols);
        const CoreCoord physical = md->worker_core_from_logical_core(logical);
        const uint32_t base = c * work.batch_per_core;
        AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            logical,
            {{"base_tile_idx", base},
             {"batch_per_core", work.batch_per_core},
             {"noc_x", physical.x},
             {"noc_y", physical.y},
             {"pt_modulus", apply_pt ? attrs.twiddle_N2 : 0u},
             {"pt_stride", apply_pt ? attrs.stride : 1u}});
        AddRuntimeArgsForNode(
            writer_args.runtime_arg_values,
            logical,
            {{"base_tile_idx", base},
             {"batch_per_core", work.batch_per_core},
             {"output_scale_bits", float_bits(attrs.output_scale)}});
        AddRuntimeArgsForNode(
            compute_args.runtime_arg_values, logical, {{"batch_per_core", work.batch_per_core}});
    }

    Group<TensorParameter> tensor_parameters = {
        {.unique_id = shared::IN_R, .spec = in_r.tensor_spec()},
        {.unique_id = shared::IN_I, .spec = in_i.tensor_spec()},
        {.unique_id = shared::TW_R, .spec = tw_r.tensor_spec()},
        {.unique_id = shared::TW_I, .spec = tw_i.tensor_spec()},
        {.unique_id = shared::OUT_R, .spec = out_r.tensor_spec()},
        {.unique_id = shared::OUT_I, .spec = out_i.tensor_spec()}};
    if (apply_pt) {
        tensor_parameters.push_back(
            {.unique_id = shared::PT_R, .spec = tensor_args.post_tw_real.tensor_spec()});
        tensor_parameters.push_back(
            {.unique_id = shared::PT_I, .spec = tensor_args.post_tw_imag.tensor_spec()});
    }
    ProgramSpec spec{
        .name = "fft_radix_pass",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = shared::make_dataflow_buffers(is_bf16, apply_pt),
        .tensor_parameters = std::move(tensor_parameters),
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
    if (apply_pt) {
        run_args.tensor_args.emplace(shared::PT_R, tensor_args.post_tw_real.mesh_tensor());
        run_args.tensor_args.emplace(shared::PT_I, tensor_args.post_tw_imag.mesh_tensor());
    }
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

ProgramRunArgs FftRadixPassFactory::override_runtime_arguments(
    const FftRadixPassParams& attrs,
    const FftRadixPassTensorArgs& tensor_args,
    std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>&) {
    namespace shared = stockham_spec;
    const auto work = work_geometry(tensor_args.input_real);
    KernelRunArgs writer_args{.kernel = shared::WRITER};
    for (uint32_t c = 0; c < work.num_cores; ++c) {
        AddRuntimeArgsForNode(
            writer_args.runtime_arg_values,
            fft_stockham::batch_logical_core(c, work.grid_cols),
            {{"output_scale_bits", float_bits(attrs.output_scale)}});
    }
    ProgramRunArgs result;
    result.kernel_run_args = {std::move(writer_args)};
    result.tensor_args = {
        {shared::IN_R, tensor_args.input_real.mesh_tensor()},
        {shared::IN_I, tensor_args.input_imag.mesh_tensor()},
        {shared::TW_R, tensor_args.tw_real.mesh_tensor()},
        {shared::TW_I, tensor_args.tw_imag.mesh_tensor()},
        {shared::OUT_R, std::get<0>(tensor_return_value).mesh_tensor()},
        {shared::OUT_I, std::get<1>(tensor_return_value).mesh_tensor()}};
    if (attrs.twiddle_N2 != 0u) {
        result.tensor_args.emplace(shared::PT_R, tensor_args.post_tw_real.mesh_tensor());
        result.tensor_args.emplace(shared::PT_I, tensor_args.post_tw_imag.mesh_tensor());
    }
    return result;
}

}  // namespace ttnn::experimental::prim
