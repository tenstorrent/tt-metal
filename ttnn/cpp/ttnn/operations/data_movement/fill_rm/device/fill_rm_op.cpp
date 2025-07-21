// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/fill_rm/device/fill_rm_op.hpp"
#include <tt-metalium/tilize_utils.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/util.hpp>

using namespace tt::tt_metal;

using uint32_t = uint32_t;

namespace ttnn::operations::data_movement {

operation::ProgramWithCallbacks fill_rm_single_core(
    const Tensor& any,
    Tensor& output,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    float val_hi,
    float val_lo) {
    tt::tt_metal::IDevice* device = any.device();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(any.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_cb_tiles = 16;
    TT_ASSERT(W < 1024 * num_cb_tiles);  // Limitation for simplifying the kernel

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{0, cb_data_format}})
            .set_page_size(0, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_cb_tiles * single_tile_size, {{1, cb_data_format}})
            .set_page_size(1, single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle binary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {dst_buffer->address(),
         uint32_t(N * C),
         uint32_t(H),
         uint32_t(W),
         uint32_t(hFill),
         uint32_t(wFill),
         uint32_t(bfloat16(val_hi).to_uint16()),
         uint32_t(bfloat16(val_lo).to_uint16())});

    auto override_runtime_args_callback = [kernel_id = binary_reader_kernel_id](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto dst_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = GetRuntimeArgs(program, kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void FillRM::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL((this->N > 0 && this->C > 0 && this->H > 0 && this->W > 0), "Error");
    TT_FATAL((this->hFill <= this->H && this->wFill <= this->W), "Error");
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16, "Error");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "FillRM does not currently support sharding");
    TT_FATAL(
        this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "FillRM does not currently support sharding");
}

std::vector<ttnn::TensorSpec> FillRM::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    ttnn::Shape shape({this->N, this->C, this->H, this->W});
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::ROW_MAJOR), output_mem_config))};
}

operation::ProgramWithCallbacks FillRM::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return fill_rm_single_core(
        input_tensor,
        output_tensor,
        this->N,
        this->C,
        this->H,
        this->W,
        this->hFill,
        this->wFill,
        this->val_hi,
        this->val_lo);
}

}  // namespace ttnn::operations::data_movement
