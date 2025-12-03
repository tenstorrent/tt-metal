
#include "pad_program_factory.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::data_movement::pad::program {
PadProgramFactory::cached_program_t PadProgramFactory::create(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    const float pad_value) {
    tt::tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device

    auto output_shape = output_padded_shape;

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "pad_tile");
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);
    log_debug(tt::LogOp, "input_tensor_start: {}", input_tensor_start);
    log_debug(tt::LogOp, "pad_value: {}", pad_value);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;  // For pad buffer
    uint32_t num_pad_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_pad_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    uint32_t num_unpadded_Xt = a.padded_shape()[3] / TILE_WIDTH;
    uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = a.padded_shape()[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = a.padded_shape()[1];
    uint32_t num_total_Z = output_shape[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = a.padded_shape()[0];
    uint32_t num_total_W = output_shape[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    uint32_t num_unpadded_tiles = a.physical_volume() / TILE_HW;

    const std::array reader_kernel_args = {
        src0_buffer->address(),
        num_unpadded_tiles,
        std::uint32_t{0},
    };
    const std::array writer_kernel_args = {
        dst_buffer->address(),
        num_unpadded_W,
        num_padded_Wt,
        num_unpadded_Z,
        num_padded_Zt,
        num_unpadded_Yt,
        num_padded_Yt,
        num_unpadded_Xt,
        num_padded_Xt,
        packed_pad_value,
    };

    // Reader compile-time args
    // Data is 32 byte aligned
    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index, src1_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto* src_dram_buffer = input_tensors.at(0).buffer();

        auto* dst_dram_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::pad::program
