// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bluestein_chirp_device_operation.hpp"

#include <algorithm>
#include <cstdint>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "apply_twiddles_shared.hpp"
#include "kernels/dataflow/bluestein_streaming_common.h"
#include "stockham_host.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

BluesteinChirpDeviceOperation::program_factory_t BluesteinChirpDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return BluesteinChirpFactory{};
}

void BluesteinChirpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    for (const auto* tensor : {&args.input_real, &args.input_imag, &args.chirp_real, &args.chirp_imag}) {
        TT_FATAL(tensor->storage_type() == StorageType::DEVICE, "bluestein_chirp: inputs must be on device.");
        TT_FATAL(tensor->buffer() != nullptr, "bluestein_chirp: inputs must have allocated device buffers.");
        TT_FATAL(tensor->dtype() == DataType::FLOAT32, "bluestein_chirp: inputs must be Float32.");
        TT_FATAL(tensor->layout() == Layout::ROW_MAJOR, "bluestein_chirp: inputs must be ROW_MAJOR.");
        TT_FATAL(
            tensor->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                tensor->memory_config().buffer_type() == BufferType::DRAM,
            "bluestein_chirp: inputs must use interleaved DRAM storage.");
        TT_FATAL(tensor->device() == args.input_real.device(), "bluestein_chirp: inputs must share a device.");
        const auto& shape = tensor->logical_shape();
        TT_FATAL(
            shape.size() == 2 && shape[0] == 1 && shape[1] >= 1 && shape == tensor->padded_shape(),
            "bluestein_chirp: inputs must have an unpadded logical shape (1, N), got {}.",
            shape);
    }
    TT_FATAL(
        args.input_real.device()->arch() == tt::ARCH::WORMHOLE_B0,
        "bluestein_chirp: this streaming path supports Wormhole B0.");
    TT_FATAL(
        args.input_real.logical_shape() == args.input_imag.logical_shape(),
        "bluestein_chirp: real and imaginary input shapes must match.");
    TT_FATAL(
        args.chirp_real.logical_shape() == args.chirp_imag.logical_shape(),
        "bluestein_chirp: real and imaginary chirp shapes must match.");
    const uint32_t chirp_size = args.chirp_real.logical_shape()[1];
    TT_FATAL(
        args.input_real.logical_shape()[1] >= chirp_size && attrs.output_size >= chirp_size &&
            attrs.output_size <= (1u << 20),
        "bluestein_chirp: input and output lengths must cover the chirp, and output length must not exceed 2^20. "
        "Got input={}, chirp={}, output={}.",
        args.input_real.logical_shape()[1],
        chirp_size,
        attrs.output_size);
}

void BluesteinChirpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}

BluesteinChirpDeviceOperation::spec_return_value_t BluesteinChirpDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const TensorSpec spec(
        ttnn::Shape{ttnn::SmallVector<uint32_t>{1u, attrs.output_size}},
        TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), args.input_real.memory_config()));
    return {spec, spec};
}

BluesteinChirpDeviceOperation::tensor_return_value_t BluesteinChirpDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    const auto [real_spec, imag_spec] = compute_output_specs(attrs, args);
    return {
        create_device_tensor(real_spec, args.input_real.device()),
        create_device_tensor(imag_spec, args.input_real.device())};
}

tt::stl::hash::hash_t BluesteinChirpDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return tt::tt_metal::operation::hash_operation<BluesteinChirpDeviceOperation>(
        attrs.output_size,
        attrs.input_imag_provided,
        args.input_real.tensor_spec(),
        args.input_imag.tensor_spec(),
        args.chirp_real.tensor_spec(),
        args.chirp_imag.tensor_spec());
}

ttnn::device_operation::ProgramArtifacts BluesteinChirpFactory::create_program_artifacts(
    const BluesteinChirpParams& attrs, const BluesteinChirpTensorArgs& args, std::tuple<Tensor, Tensor>& outputs) {
    namespace shared = apply_tw_shared;
    const KernelSpecName reader_id{"reader"};
    const TensorParamName input_real_id{"input_real"};
    const TensorParamName input_imag_id{"input_imag"};
    const TensorParamName chirp_real_id{"chirp_real"};
    const TensorParamName chirp_imag_id{"chirp_imag"};
    auto* device = args.input_real.device();
    const uint32_t chunks = bluestein_streaming::chunk_count(attrs.output_size);
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t cap = std::min(chunks, fft_stockham::max_cores_for_grid(grid.x, grid.y));
    uint32_t num_cores = 1;
    while (num_cores * 2 <= cap) {
        num_cores *= 2;
    }
    const auto [grid_cols, grid_rows] = fft_stockham::pick_batch_grid(num_cores, grid.x);
    const CoreRangeSet cores({CoreRange({0, 0}, {grid_cols - 1u, grid_rows - 1u})});

    KernelSpec reader{
        .unique_id = reader_id,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/bluestein_chirp_reader.cpp",
        .dfb_bindings = shared::reader_dfb_bindings(false),
        .tensor_bindings =
            {{.tensor_parameter_name = input_real_id, .accessor_name = "input_real"},
             {.tensor_parameter_name = input_imag_id, .accessor_name = "input_imag"},
             {.tensor_parameter_name = chirp_real_id, .accessor_name = "chirp_real"},
             {.tensor_parameter_name = chirp_imag_id, .accessor_name = "chirp_imag"}},
        .compile_time_args =
            {{"chirp_size", args.chirp_real.logical_shape()[1]}, {"has_imag", attrs.input_imag_provided ? 1u : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_chunk", "num_chunks"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch())};
    KernelSpec writer{
        .unique_id = shared::WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/fft/device/kernels/dataflow/bluestein_chirp_writer.cpp",
        .dfb_bindings =
            {{.dfb_spec_name = shared::B_R, .accessor_name = "b_r", .endpoint_type = DFBEndpointType::CONSUMER},
             {.dfb_spec_name = shared::B_I, .accessor_name = "b_i", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {{.tensor_parameter_name = shared::OUT_R, .accessor_name = "out_r"},
             {.tensor_parameter_name = shared::OUT_I, .accessor_name = "out_i"}},
        .compile_time_args = {{"output_size", attrs.output_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"base_chunk", "num_chunks"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch())};
    KernelSpec compute = shared::make_compute();

    KernelRunArgs reader_args{.kernel = reader_id};
    KernelRunArgs writer_args{.kernel = shared::WRITER};
    KernelRunArgs compute_args{.kernel = shared::COMPUTE};
    for (uint32_t core = 0; core < num_cores; ++core) {
        const CoreCoord logical = fft_stockham::batch_logical_core(core, grid_cols);
        const uint32_t begin = chunks * core / num_cores;
        const uint32_t end = chunks * (core + 1u) / num_cores;
        AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, logical, {{"base_chunk", begin}, {"num_chunks", end - begin}});
        AddRuntimeArgsForNode(
            writer_args.runtime_arg_values, logical, {{"base_chunk", begin}, {"num_chunks", end - begin}});
        AddRuntimeArgsForNode(compute_args.runtime_arg_values, logical, {{"num_tiles", end - begin}});
    }

    const auto& [out_real, out_imag] = outputs;
    ProgramSpec spec{
        .name = "fft_bluestein_chirp",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        // Four double-buffered inputs, two double-buffered outputs, and two
        // scratch entries: 14 * 4096 = 57344 bytes per core, independent of N/M.
        .dataflow_buffers = shared::make_dataflow_buffers(false),
        .tensor_parameters =
            {{.unique_id = input_real_id, .spec = args.input_real.tensor_spec()},
             {.unique_id = input_imag_id, .spec = args.input_imag.tensor_spec()},
             {.unique_id = chirp_real_id, .spec = args.chirp_real.tensor_spec()},
             {.unique_id = chirp_imag_id, .spec = args.chirp_imag.tensor_spec()},
             {.unique_id = shared::OUT_R, .spec = out_real.tensor_spec()},
             {.unique_id = shared::OUT_I, .spec = out_imag.tensor_spec()}},
        .work_units = {
            {.name = "main", .kernels = {reader_id, shared::WRITER, shared::COMPUTE}, .target_nodes = cores}}};
    auto run_args = override_runtime_arguments(attrs, args, outputs);
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args), std::move(compute_args)};
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

ProgramRunArgs BluesteinChirpFactory::override_runtime_arguments(
    const BluesteinChirpParams&,
    const BluesteinChirpTensorArgs& args,
    std::tuple<Tensor, Tensor>& outputs,
    const std::optional<ttnn::MeshCoordinate>&) {
    ProgramRunArgs result;
    // Bind by semantic role on every call, including a cache miss that used
    // the same tensor for both complex input halves. Tensor pointer identity
    // on that call must not determine which role is rebound on later hits.
    result.tensor_args = {
        {TensorParamName{"input_real"}, args.input_real.mesh_tensor()},
        {TensorParamName{"input_imag"}, args.input_imag.mesh_tensor()},
        {TensorParamName{"chirp_real"}, args.chirp_real.mesh_tensor()},
        {TensorParamName{"chirp_imag"}, args.chirp_imag.mesh_tensor()},
        {apply_tw_shared::OUT_R, std::get<0>(outputs).mesh_tensor()},
        {apply_tw_shared::OUT_I, std::get<1>(outputs).mesh_tensor()}};
    return result;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> bluestein_chirp(
    const Tensor& input_real,
    const std::optional<Tensor>& input_imag,
    const Tensor& chirp_real,
    const Tensor& chirp_imag,
    uint32_t output_size) {
    using Op = ttnn::experimental::prim::BluesteinChirpDeviceOperation;
    const Op::operation_attributes_t attrs{.output_size = output_size, .input_imag_provided = input_imag.has_value()};
    // Keep the reflected tensor aggregate populated when the input is real.
    // The reader synthesizes zero imaginary lanes and never reads this alias.
    const Op::tensor_args_t args{
        .input_real = input_real,
        .input_imag = input_imag.value_or(input_real),
        .chirp_real = chirp_real,
        .chirp_imag = chirp_imag};
    return ttnn::device_operation::launch<Op>(attrs, args);
}

}  // namespace ttnn::prim
