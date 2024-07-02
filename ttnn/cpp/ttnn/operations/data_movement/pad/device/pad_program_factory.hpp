// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/host_buffer/functions.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"

namespace ttnn::operations::data_movement::detail {


operation::ProgramWithCallbacks pad_rm_reader_writer(const Tensor &a,
                                                     Tensor &output,
                                                     const tt::tt_metal::Shape &output_tensor_shape,
                                                     const tt::tt_metal::Shape &input_tensor_start,
                                                     const float pad_value) {
    Program program{};

    auto output_shape = output_tensor_shape;

    uint32_t unpadded_row_size_nbytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();   // Assuming output is same datatype as input
    TT_ASSERT(unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    // construct const buffer with the pad_value
    Device *device = a.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer = tt::tt_metal::owned_buffer::create(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    const Tensor pad_value_const_tensor = Tensor(OwnedStorage{pad_value_const_buffer},
                                                 Shape({1, 1, 1, pad_value_const_buffer_size}),
                                                 DataType::BFLOAT16, Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    CoreRange cores({0, 0}, {0, 0});
    uint32_t cb_id = tt::CB::c_in0;
    uint32_t cb_npages = 16; // multibuffering
    uint32_t cb_pagesize = tt::round_up(padded_row_size_nbytes, tt::constants::TILE_WIDTH);
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(cb_npages * cb_pagesize, {{cb_id, in_df}})
		.set_page_size(cb_id, cb_pagesize);
    auto cb = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);


    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(unpadded_row_size_nbytes);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(unpadded_row_size_nbytes) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(padded_row_size_nbytes);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(padded_row_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) src0_is_dram,
                                            (std::uint32_t) dst_is_dram,
                                            (std::uint32_t) src_stick_size_is_power_of_two,
                                            (std::uint32_t) src_log2_stick_size,
                                            (std::uint32_t) dst_stick_size_is_power_of_two,
                                            (std::uint32_t) dst_log2_stick_size};
    std::vector<uint32_t> writer_ct_args = reader_ct_args;

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    bfloat16 bfloat_zero = bfloat16(0.0f);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_zero, bfloat_pad_value});

    KernelHandle reader_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
                                                        cores,
                                                        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
                                                        cores,
                                                        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;

    #if 0
    {
        log_debug("src0_buffer_addr: {}", src0_buffer->address());
        log_debug("dst_buffer_addr: {}", dst_buffer->address());
        log_debug("a.shape[0]: {}", a.get_legacy_shape()[0]);
        log_debug("out.shape[0]: {}", output_shape[0]);
        log_debug("a.shape[1]: {}", a.get_legacy_shape()[1]);
        log_debug("out.shape[1]: {}", output_shape[1]);
        log_debug("a.shape[2]: {}", a.get_legacy_shape()[2]);
        log_debug("out.shape[2]: {}", output_shape[2]);
        log_debug("s.shape[3]: {}", a.get_legacy_shape()[3]);
        log_debug("out.shape[3]: {}", output_shape[3]);
        log_debug("unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug("padded_row_size_nbytes: {}", padded_row_size_nbytes);
        log_debug("padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug("pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        log_debug("pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug("packed_pad_value: {}", packed_pad_value);
    }
    #endif

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                       dst_buffer->address(),
                                       a.get_legacy_shape()[0],
                                       output_shape[0],
                                       a.get_legacy_shape()[1],
                                       output_shape[1],
                                       a.get_legacy_shape()[2],
                                       output_shape[2],
                                       a.get_legacy_shape()[3],
                                       output_shape[3],
                                       unpadded_row_size_nbytes,
                                       padded_row_size_nbytes,
                                       padded_row_diff_size_nbytes,
                                       pad_value_const_tensor_addr,
                                       pad_value_const_buffer_nbytes,
                                       packed_pad_value,
                                       start_src_stick_id,
                                       start_dst_stick_id,
                                       0,
                                       0,
                                       0,
                                       output_shape[2],
                                       a.get_legacy_shape()[2],
                                       unpadded_row_size_nbytes,
                                       padded_row_size_nbytes,
                                       0,
                                       output.get_legacy_shape()[0]
                                       };
    vector<uint32_t> writer_rt_args = reader_rt_args;
    tt::tt_metal::SetRuntimeArgs(program,
                   reader_kernel_id,
                   cores,
                   reader_rt_args);
    tt::tt_metal::SetRuntimeArgs(program,
                   writer_kernel_id,
                   cores,
                   writer_rt_args);

    auto override_runtime_args_callback =
        [reader_kernel_id=reader_kernel_id, writer_kernel_id=writer_kernel_id](
            const Program &program,
            const std::vector<Buffer*>& input_buffers,
            const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
        {
            auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_rm_opt(const Tensor &a,
                                           Tensor &output,
                                           const Shape &output_tensor_shape,
                                           const Shape &input_tensor_start,
                                           const float pad_value) {
    Program program{};

    auto output_shape = output_tensor_shape;

    uint32_t unpadded_row_size_nbytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();   // Assuming output is same datatype as input
    TT_ASSERT(unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    Device *device = a.device();
    auto dst_buffer_l1 = Buffer(device, padded_row_size_nbytes, padded_row_size_nbytes, BufferType::L1);

    // construct const buffer with the pad_value
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer = owned_buffer::create(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    const Tensor pad_value_const_tensor = Tensor(OwnedStorage{pad_value_const_buffer},
                                                 Shape({1, 1, 1, pad_value_const_buffer_size}),
                                                 DataType::BFLOAT16, Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(unpadded_row_size_nbytes);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(unpadded_row_size_nbytes) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(padded_row_size_nbytes);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(padded_row_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) src0_is_dram,
                                            (std::uint32_t) dst_is_dram,
                                            (std::uint32_t) src_stick_size_is_power_of_two,
                                            (std::uint32_t) src_log2_stick_size,
                                            (std::uint32_t) dst_stick_size_is_power_of_two,
                                            (std::uint32_t) dst_log2_stick_size};

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    bfloat16 bfloat_zero = bfloat16(0.0f);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_zero, bfloat_pad_value});

    CoreRange core({0, 0}, {0, 0});
    KernelHandle reader_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/pad_dims_rm_interleaved_opt.cpp",
                                                        core,
                                                        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;

    #if 0
    {
        tt::log_debug("src0_buffer_addr: {}", src0_buffer->address());
        tt::log_debug("dst_buffer_addr: {}", dst_buffer->address());
        tt::log_debug("a.shape[0]: {}", a.get_legacy_shape()[0]);
        tt::log_debug("out.shape[0]: {}", output_shape[0]);
        tt::log_debug("a.shape[1]: {}", a.get_legacy_shape()[1]);
        tt::log_debug("out.shape[1]: {}", output_shape[1]);
        tt::log_debug("a.shape[2]: {}", a.get_legacy_shape()[2]);
        tt::log_debug("out.shape[2]: {}", output_shape[2]);
        tt::log_debug("s.shape[3]: {}", a.get_legacy_shape()[3]);
        tt::log_debug("out.shape[3]: {}", output_shape[3]);
        tt::log_debug("unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        tt::log_debug("padded_row_size_nbytes: {}", padded_row_size_nbytes);
        tt::log_debug("padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        tt::log_debug("pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        tt::log_debug("pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        tt::log_debug("packed_pad_value: {}", packed_pad_value);
        tt::log_debug("dst_buffer_l1_addr: {}", dst_buffer_l1.address());
    }
    #endif

    vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                       dst_buffer->address(),
                                       a.get_legacy_shape()[0],
                                       output_shape[0],
                                       a.get_legacy_shape()[1],
                                       output_shape[1],
                                       a.get_legacy_shape()[2],
                                       output_shape[2],
                                       a.get_legacy_shape()[3],
                                       output_shape[3],
                                       unpadded_row_size_nbytes,
                                       padded_row_size_nbytes,
                                       padded_row_diff_size_nbytes,
                                       pad_value_const_tensor_addr,
                                       pad_value_const_buffer_nbytes,
                                       packed_pad_value,
                                       dst_buffer_l1.address()};
    tt::tt_metal::SetRuntimeArgs(program,
                   reader_kernel_id,
                   core,
                   reader_rt_args);

    auto override_runtime_args_callback = [kernel_id=reader_kernel_id](const Program &program,
                                                                             const std::vector<Buffer*>& input_buffers,
                                                                             const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_rm(const Tensor &a, Tensor &output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value) {

    tt::tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    uint32_t unpadded_row_size_bytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_bytes = output_shape[3] * a.element_size();

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src_stick_size = unpadded_row_size_bytes;
    uint32_t dst_stick_size = padded_row_size_bytes;

    uint32_t dst_buffer_size = dst_stick_size;

    tt::tt_metal::InterleavedBufferConfig buff_config{
                    .device= device,
                    .size = dst_buffer_size,
                    .page_size = dst_buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };

    auto dst_buffer_l1 = tt::tt_metal::CreateBuffer(buff_config);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        dst_buffer->address(),
        a.get_legacy_shape()[0],
        output_shape[0],
        a.get_legacy_shape()[1],
        output_shape[1],
        a.get_legacy_shape()[2],
        output_shape[2],
        a.get_legacy_shape()[3],
        output_shape[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        padded_row_size_bytes - unpadded_row_size_bytes,
        packed_pad_value,
        dst_buffer_l1->address()
    };
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(src_stick_size);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(src_stick_size) : 0;
    bool dst_stick_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(dst_stick_size);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(dst_stick_size) : 0;
    std::vector<uint32_t> compile_time_args_vec = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) src_stick_size_is_power_of_two,
        (std::uint32_t) src_log2_stick_size,
        (std::uint32_t) dst_stick_size_is_power_of_two,
        (std::uint32_t) dst_log2_stick_size,

    };

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/pad_dims_rm_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args_vec));

    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    auto override_runtime_args_callback = [kernel_id=unary_reader_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_tile(const Tensor &a, Tensor& output, const tt::tt_metal::Shape &output_tensor_shape, const tt::tt_metal::Shape &input_tensor_start, const float pad_value) {

    tt::tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;

    tt::tt_metal::Buffer *src0_buffer = a.buffer();

    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    tt::log_debug("pad_tile");
    tt::log_debug("cb_data_format: {}", cb_data_format);
    tt::log_debug("single_tile_size: {}", single_tile_size);
    tt::log_debug("output_tensor_shape: {}", output_tensor_shape);
    tt::log_debug("input_tensor_start: {}", input_tensor_start);
    tt::log_debug("pad_value: {}", pad_value);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1; // For pad buffer
    uint32_t num_pad_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config = tt::tt_metal::CircularBufferConfig(num_pad_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    uint32_t num_unpadded_Xt = a.get_legacy_shape()[3] / TILE_WIDTH;
    uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = a.get_legacy_shape()[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = a.get_legacy_shape()[1];
    uint32_t num_total_Z = output_shape[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = a.get_legacy_shape()[0];
    uint32_t num_total_W = output_shape[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    uint32_t num_unpadded_tiles = a.volume() / TILE_HW;

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        num_unpadded_tiles, 0
    };
    vector<uint32_t> writer_kernel_args = {
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
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_is_dram
    };
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src1_cb_index,
        (std::uint32_t) dst_is_dram
    };
    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

   tt::tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


inline void log_rt_args(const CoreCoord& core,  vector<uint32_t>& args) {
    for (auto v : args) {
        tt::log_debug(tt::LogOp, "{},{} :: {}", core.x, core.y, v);
    }
}

// This is currently mostly hardcoded for resnet shapes
inline std::tuple<uint32_t, uint32_t, uint32_t, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t, uint32_t, uint32_t>
    split_across_cores(CoreCoord grid_size, uint32_t nbatch, uint32_t nchannel, uint32_t ntiles_h, uint32_t ntiles_w) {

    uint32_t ncores, ncores_h, ncores_w, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h;

    ncores_h = 1;

    // each batch needs to be padded independently
    switch (nbatch) {
        case 1:
            ncores_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2: ncores_h = 2; ntiles_per_core_h = 1; break;
                case 4: ncores_h = 4; ntiles_per_core_h = 1; break;
                case 8: ncores_h = 8; ntiles_per_core_h = 1; break;
                case 64: ncores_h = 8; ntiles_per_core_h = 8; break;
            }
            ncores_per_batch_h = ncores_h;
            break;

        case 2:
            ncores_h = 1;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2: ncores_per_batch_h = 2; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 1; break;
                case 4: ncores_per_batch_h = 4; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 1; break;
                case 8: ncores_per_batch_h = 4; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 2; break;
                case 64: ncores_per_batch_h = 4; ncores_h = ncores_per_batch_h * nbatch; ntiles_per_core_h = 16; break;
            }
            break;

        case 8:
            ncores_h = 8;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = ntiles_h;
            break;

        default:
            TT_ASSERT(false, "unhandled nbatch. TODO");

            // generic case -- TODO

            // one of the following will be 0 when grid_size.y != nbatch
            uint32_t nbatch_per_core_h = nbatch / grid_size.y;  // floor
            uint32_t ncores_per_batch_h = grid_size.y / nbatch; // floor
            if (nbatch == grid_size.y) {
                nbatch_per_core_h = 1;
                ncores_per_batch_h = 1;
            }

            // currently uses hardcoded values for resnet50
            // TT_ASSERT(ntiles_h == 1 || ntiles_h == 2 || ntiles_h == 4 || ntiles_h == 16, "Only Resnet50 shapes are supported in multicore version for now.");
            // TT_ASSERT(ntiles_w == 64, "Only Resnet50 shapes are supported in multicore version for now.");

            TT_ASSERT(nbatch <= grid_size.y, "Unsupported case with nbatch > grid_size.y!");

            uint32_t ncores_h = 1;
            uint32_t ntiles_per_core_h = ntiles_h / ncores_h;
            if (nbatch_per_core_h == 0) {
                // there are multiple cores along h per batch
                nbatch_per_core_h = 1;
                ncores_h = ncores_per_batch_h * nbatch;
                ntiles_per_core_h = ntiles_h / ncores_per_batch_h;
            } else if (ncores_per_batch_h == 0) {
                // unsupported case. TODO.
                TT_ASSERT(false);
                // there are multiple batch per core along h
                // ncores_per_batch_h = 1;
                // ncores_h = (uint32_t) ceil((float) nbatch / nbatch_per_core_h);
                // ntiles_per_core_h = nbatch_per_core_h * ntiles_h;
            } else {
                TT_ASSERT("Something went terribly wrong in splitting acrtoss cores");
            }
            break;
    }

    ncores_w = 1;
    switch (ntiles_w) {
        case 2: ncores_w = 2; break;
        case 4: ncores_w = 4; break;
        case 8: ncores_w = 8; break;
        case 64: ncores_w = 8; break;
    }
    ncores = ncores_h * ncores_w;
    ntiles_per_core_w = ntiles_w / ncores_w;
    std::set<CoreRange> all_cores;
    std::set<CoreRange> core_range;

    all_cores.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));

    return std::make_tuple(ncores, ncores_h, ncores_w, all_cores, core_range, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h);
}

operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core(const Tensor &a,
                                                                Tensor &output,
                                                                const tt::tt_metal::Shape &output_tensor_shape,
                                                                const tt::tt_metal::Shape &input_tensor_start,
                                                                const float pad_value) {
    Program program{};

    auto output_shape = output_tensor_shape;

    uint32_t unpadded_row_size_nbytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();   // Assuming output is same datatype as input
    TT_ASSERT(unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    Device *device = a.device();

    // construct const buffer with the pad_value
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer = owned_buffer::create(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    // NOTE: The const buffer is always in L1
    // TODO: make a local buffer for each core?
    const Tensor pad_value_const_tensor = Tensor(OwnedStorage{pad_value_const_buffer},
                                                 Shape({1, 1, 1, pad_value_const_buffer_size}),
                                                 DataType::BFLOAT16, Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    // uint32_t ntiles_h = output_tensor_shape[0] * output_tensor_shape[1] * output_tensor_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_h = output_tensor_shape[2] / TILE_HEIGHT;
    uint32_t ntiles_w = output_tensor_shape[3] / TILE_WIDTH;

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t nbatch = output_tensor_shape[0];
    uint32_t nchannel = output_tensor_shape[1];
    // first the batch dim is distributed along H, and within each batch then the tiles are distributed.
    auto [ncores, ncores_h, ncores_w, all_cores, core_range, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h] = split_across_cores(grid_size, nbatch, nchannel, ntiles_h, ntiles_w);

    int32_t src_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * a.element_size();
    int32_t dst_nbytes_per_core_w = ntiles_per_core_w * TILE_WIDTH * output.element_size();

    uint32_t cb_id = tt::CB::c_in0;
    uint32_t cb_npages = 16; // multibuffering for perf
    // uint32_t cb_npages = 1; // multibuffering for perf
    uint32_t cb_pagesize = (uint32_t) ceil((float) dst_nbytes_per_core_w / tt::constants::TILE_WIDTH) * tt::constants::TILE_WIDTH;
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(cb_npages * cb_pagesize, {{cb_id, in_df}})
		.set_page_size(cb_id, cb_pagesize);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(unpadded_row_size_nbytes);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(unpadded_row_size_nbytes) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(padded_row_size_nbytes);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(padded_row_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) src0_is_dram,
                                            (std::uint32_t) dst_is_dram,
                                            (std::uint32_t) src_stick_size_is_power_of_two,
                                            (std::uint32_t) src_log2_stick_size,
                                            (std::uint32_t) dst_stick_size_is_power_of_two,
                                            (std::uint32_t) dst_log2_stick_size};
    std::vector<uint32_t> writer_ct_args = reader_ct_args;

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    bfloat16 bfloat_zero = bfloat16(0.0f);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_zero, bfloat_pad_value});

    KernelHandle reader_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
                                                        all_cores,
                                                        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
                                                        all_cores,
                                                        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    // int32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;
    log_rt_args(CoreCoord{0, 0}, reader_ct_args);

    #if 1
    {
        tt::log_debug("ncores: {}", ncores);
        tt::log_debug("ncores_h: {}", ncores_h);
        tt::log_debug("ncores_w: {}", ncores_w);
        tt::log_debug("ntiles_per_core_h: {}", ntiles_per_core_h);
        tt::log_debug("ntiles_per_core_w: {}", ntiles_per_core_w);
        tt::log_debug("src0_buffer_addr: {}", src0_buffer->address());
        tt::log_debug("dst_buffer_addr: {}", dst_buffer->address());
        tt::log_debug("a.shape[0]: {}", a.get_legacy_shape()[0]);
        tt::log_debug("out.shape[0]: {}", output_shape[0]);
        tt::log_debug("a.shape[1]: {}", a.get_legacy_shape()[1]);
        tt::log_debug("out.shape[1]: {}", output_shape[1]);
        tt::log_debug("a.shape[2]: {}", a.get_legacy_shape()[2]);
        tt::log_debug("out.shape[2]: {}", output_shape[2]);
        tt::log_debug("s.shape[3]: {}", a.get_legacy_shape()[3]);
        tt::log_debug("out.shape[3]: {}", output_shape[3]);
        tt::log_debug("unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        tt::log_debug("padded_row_size_nbytes: {}", padded_row_size_nbytes);
        // tt::log_debug("padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        tt::log_debug("pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        tt::log_debug("pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        tt::log_debug("packed_pad_value: {}", packed_pad_value);
        tt::log_debug("src_nbytes_per_core_w: {}", src_nbytes_per_core_w);
        tt::log_debug("dst_nbytes_per_core_w: {}", dst_nbytes_per_core_w);
        tt::log_debug("nbatch_per_core_h: {}", nbatch_per_core_h);
        tt::log_debug("ncores_per_batch_h: {}", ncores_per_batch_h);
    }
    #endif

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    uint32_t start_src_stick_wi = 0; // start of stick segment for 2d decomp
    uint32_t start_dst_stick_wi = 0;
    int32_t local_nsticks = ntiles_per_core_h * TILE_HEIGHT;
    int32_t rem_nbatch = nbatch;    // per core h, there are ncores_per_batch_h cores, ie each batch ncores_h = ncores_per_batch_h
    for (int32_t b = 0; b < nbatch; ++ b) {
        int32_t rem_src_nsticks = a.get_legacy_shape()[2];
        for (uint32_t j = 0; j < ncores_per_batch_h; ++ j) {
            uint32_t num_local_unpadded_nsticks = local_nsticks;
            if (rem_src_nsticks - local_nsticks >= 0) {
                // not reached padding sticks yet
                rem_src_nsticks -= local_nsticks;
            } else {
                num_local_unpadded_nsticks = rem_src_nsticks;
                rem_src_nsticks = 0;
            }
            start_src_stick_wi = 0;
            start_dst_stick_wi = 0;
            int32_t rem_src_stick_size_nbytes = unpadded_row_size_nbytes;
            for (uint32_t i = 0; i < ncores_w; ++ i) {
                CoreCoord core = {i, b * ncores_per_batch_h + j};
                uint32_t curr_stick_size_nbytes = 0;
                int32_t curr_stick_diff_nbytes = 0;
                if (rem_src_stick_size_nbytes - dst_nbytes_per_core_w >= 0) {
                    // no padding on this core
                    curr_stick_size_nbytes = dst_nbytes_per_core_w;
                    rem_src_stick_size_nbytes -= dst_nbytes_per_core_w;
                } else {
                    // this core has padding
                    curr_stick_size_nbytes = rem_src_stick_size_nbytes;
                    curr_stick_diff_nbytes = dst_nbytes_per_core_w - curr_stick_size_nbytes;
                    rem_src_stick_size_nbytes = 0;
                }
                vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                                    dst_buffer->address(),
                                                    a.get_legacy_shape()[0],
                                                    output_shape[0],
                                                    a.get_legacy_shape()[1],
                                                    output_shape[1],
                                                    a.get_legacy_shape()[2],
                                                    output_shape[2],
                                                    a.get_legacy_shape()[3],
                                                    output_shape[3],
                                                    curr_stick_size_nbytes,
                                                    (uint32_t) dst_nbytes_per_core_w,
                                                    (uint32_t) curr_stick_diff_nbytes,
                                                    pad_value_const_tensor_addr,
                                                    pad_value_const_buffer_nbytes,
                                                    packed_pad_value,
                                                    start_src_stick_id,
                                                    start_dst_stick_id,
                                                    start_src_stick_wi,
                                                    start_dst_stick_wi,
                                                    start_src_stick_wi * a.element_size(),
                                                    (uint32_t) local_nsticks,
                                                    num_local_unpadded_nsticks,
                                                    unpadded_row_size_nbytes,
                                                    padded_row_size_nbytes,
                                                    start_dst_stick_wi * output.element_size(),
                                                    nbatch_per_core_h
                                                    };
                // if (core.x == 0) log_rt_args(core, reader_rt_args);
                // if (core.x == 0) {
                //     log_debug("{} :: start_src_stick_id: {}", core.y, start_src_stick_id);
                //     log_debug("{} :: start_dst_stick_id: {}", core.y, start_dst_stick_id);
                //     log_debug("{} :: local_nsticks: {}", core.y, local_nsticks);
                //     log_debug("{} :: num_local_unpadded_nsticks: {}", core.y, num_local_unpadded_nsticks);
                //     log_debug("{} :: nbatch_per_core_h: {}", core.y, nbatch_per_core_h);
                //     log_debug("{} :: ncores_per_batch_h: {}", core.y, ncores_per_batch_h);
                // }
                vector<uint32_t> writer_rt_args = reader_rt_args;
                tt::tt_metal::SetRuntimeArgs(program,
                                reader_kernel_id,
                                core,
                                reader_rt_args);
                tt::tt_metal::SetRuntimeArgs(program,
                                writer_kernel_id,
                                core,
                                writer_rt_args);
                start_src_stick_wi += ntiles_per_core_w * TILE_WIDTH;
                start_dst_stick_wi += ntiles_per_core_w * TILE_WIDTH;
            } // for ncores_w
            start_src_stick_id += num_local_unpadded_nsticks;
            start_dst_stick_id += local_nsticks;
        } // for ncores_h
    }

    auto override_runtime_args_callback = [reader_kernel_id = reader_kernel_id,
                                           writer_kernel_id = writer_kernel_id,
                                           ncores_h = ncores_h,
                                           ncores_w = ncores_w](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t j = 0; j < ncores_h; ++ j) {
            for (uint32_t i = 0; i < ncores_w; ++ i) {
                CoreCoord core = {i, j};
                {
                    auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
                {
                    auto &runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = src_buffer->address();
                    runtime_args[1] = dst_buffer->address();
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}



} // namespace ttnn::operations::reduction::detail
