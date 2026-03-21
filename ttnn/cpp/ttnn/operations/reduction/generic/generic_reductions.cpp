// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;

namespace ttnn::operations::reduction {

std::map<string, string> get_defines(ReduceOpMath reduce_op, ReduceOpDim reduce_dim) {
    std::map<string, string> defines;
    switch (reduce_op) {
        case ReduceOpMath::SUM: defines["REDUCE_OP"] = "PoolType::SUM"; break;
        case ReduceOpMath::MEAN: defines["REDUCE_OP"] = "PoolType::AVG"; break;
        case ReduceOpMath::MIN: defines["REDUCE_OP"] = "PoolType::MIN"; break;
        case ReduceOpMath::MAX: defines["REDUCE_OP"] = "PoolType::MAX"; break;
        default: TT_FATAL("Unsupported reduce op");
    }

    switch (reduce_dim) {
        case ReduceOpDim::H: defines["REDUCE_DIM"] = "PoolType::REDUCE_ROW"; break;
        case ReduceOpDim::W: defines["REDUCE_DIM"] = "PoolType::REDUCE_COL"; break;
        case ReduceOpDim::HW: defines["REDUCE_DIM"] = "PoolType::REDUCE_SCALAR"; break;
        default: TT_FATAL("Unsupported reduce dim");
    }

    return defines;
}

operation::ProgramWithCallbacks reduce_single_core_sharded(
    const Tensor &input_tensor,
    const Tensor &output_tensor,
    ReduceOpMath reduce_op,
    ReduceOpDim reduce_dim,
    float scaler,
    DeviceComputeKernelConfig compute_kernel_config) {

    tt_metal::Program program{};

    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    uint32_t num_output_tiles = output_tensor.volume() / TILE_HW;

    tt_metal::Device *device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    uint32_t input_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    uint32_t output_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange all_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Create circular buffers
    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * input_tile_size, {{src0_cb_index, input_cb_data_format}})
        .set_page_size(src0_cb_index, input_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t output_cb_index = CB::c_out0;
    uint32_t num_output_tiles_cb = 2;
    tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles_cb * output_tile_size, {{output_cb_index, output_cb_data_format}})
        .set_page_size(output_cb_index, output_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // Create compute kernel
    std::string compute_kernel_name;
    std::map<string, string> reduce_defines = get_defines(reduce_op, reduce_dim);

    switch (reduce_dim) {
        case ReduceOpDim::H:
            compute_kernel_name = "ttnn/cpp/ttnn/operations/reduction/generic/kernels/reduce_h.cpp";
            break;
        case ReduceOpDim::W:
            compute_kernel_name = "ttnn/cpp/ttnn/operations/reduction/generic/kernels/reduce_w.cpp";
            break;
        case ReduceOpDim::HW:
            compute_kernel_name = "ttnn/cpp/ttnn/operations/reduction/generic/kernels/reduce_hw.cpp";
            break;
    }

    auto reduce_compute_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel_name,
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = compute_kernel_config.math_fidelity,
            .fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en,
            .math_approx_mode = compute_kernel_config.math_approx_mode,
            .compile_args = {},
            .defines = reduce_defines
        }
    );

    // Get padding info
    auto input_padding = input_tensor.get_padding();
    auto padded_shape = input_tensor.get_padded_shape();
    
    // Calculate padding parameters
    uint32_t padded_row_size_bytes = padded_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    
    bool has_padding = (input_padding[0].back > 0) || (input_padding[1].back > 0);

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) num_tiles,
        (std::uint32_t) has_padding,
        (std::uint32_t) padded_row_size_bytes,
        (std::uint32_t) unpadded_row_size_bytes,
        (std::uint32_t) input_padding[0].back, // height padding
        (std::uint32_t) input_padding[1].back  // width padding
    };

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/kernels/reader_reduce.cpp",
        all_cores,
        tt_metal::ReaderConfig{.compile_args = reader_compile_time_args});

    // Create writer kernel  
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) num_output_tiles
    };

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/kernels/writer_reduce.cpp", 
        all_cores,
        tt_metal::WriterConfig{.compile_args = writer_compile_time_args});

    // Set runtime args
    CoreCoord core = {0, 0};
    
    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {input_buffer->address()}
    );

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {output_buffer->address()}
    );

    tt_metal::SetRuntimeArgs(
        program,
        reduce_compute_kernel_id,
        core,
        {num_tiles, static_cast<uint32_t>(std::bit_cast<uint32_t>(scaler))}
    );

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, reduce_compute_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks reduce_multi_core_sharded(
    const Tensor &input_tensor,
    const Tensor &output_tensor,
    ReduceOpMath reduce_op,
    ReduceOpDim reduce_dim,
    float scaler,
    DeviceComputeKernelConfig compute_kernel_config) {

    // For multi-core, delegate to single core for now
    // TODO: Implement proper multi-core sharding
    return reduce_single_core_sharded(input_tensor, output_tensor, reduce_op, reduce_dim, scaler, compute_kernel_config);
}

void GenericReduceDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffer on device");
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::SINGLE_BANK, "Input tensor must be in single bank memory layout");
}

std::vector<tt::tt_metal::Shape> GenericReduceDeviceOperation::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    auto input_shape = input.get_legacy_shape();
    
    auto output_shape = input_shape;
    switch (this->dim) {
        case ReduceOpDim::H:
            output_shape[-2] = 32; // TILE_HEIGHT
            break;
        case ReduceOpDim::W:
            output_shape[-1] = 32; // TILE_WIDTH
            break;
        case ReduceOpDim::HW:
            output_shape[-2] = 32; // TILE_HEIGHT
            output_shape[-1] = 32; // TILE_WIDTH
            break;
    }
    
    return {output_shape};
}

std::vector<Tensor> GenericReduceDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    auto output_shapes = this->compute_output_shapes(input_tensors);
    
    return {create_device_tensor(
        output_shapes.at(0),
        this->output_dtype.value_or(input.get_dtype()),
        Layout::TILE,
        input.device(),
        this->memory_config)};
}

operation::ProgramWithCallbacks GenericReduceDeviceOperation::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& output = output_tensors.at(0);

    return reduce_single_core_sharded(input, output, this->math_op, this->dim, this->scaler, this->compute_kernel_config);
}

Tensor reduce(
    const Tensor &input_tensor,
    ReduceOpMath math_op,
    ReduceOpDim dim,
    float scaler,
    const std::optional<DataType> &output_dtype,
    const std::optional<MemoryConfig> &memory_config,
    const std::optional<DeviceComputeKernelConfig> &compute_kernel_config) {

    return operation::run(GenericReduceDeviceOperation{
        math_op,
        dim, 
        scaler,
        output_dtype,
        memory_config.value_or(input_tensor.memory_config()),
        compute_kernel_config.value_or(init_device_compute_kernel_config(input_tensor.device()->arch()))
    }, {input_tensor}).at(0);
}

}  // namespace ttnn::operations::reduction