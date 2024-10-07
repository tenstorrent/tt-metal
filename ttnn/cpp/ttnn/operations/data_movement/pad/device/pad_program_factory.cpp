// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/math.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"
using namespace tt::constants;

namespace ttnn::operations::data_movement::detail {


operation::ProgramWithCallbacks pad_rm_reader_writer(const Tensor &a,
                                                     Tensor &output,
                                                     const tt::tt_metal::LegacyShape &output_tensor_shape,
                                                     const tt::tt_metal::LegacyShape &input_tensor_start,
                                                     const float pad_value) {
    auto program = tt::tt_metal::CreateProgram();

    auto output_shape = output_tensor_shape;

    uint32_t unpadded_row_size_nbytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();   // Assuming output is same datatype as input
    TT_ASSERT(unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    // construct const buffer with the pad_value
    Device *device = a.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer = tt::tt_metal::owned_buffer::create(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    const Tensor pad_value_const_tensor =
        Tensor(
            OwnedStorage{pad_value_const_buffer},
            Shape(std::array<uint32_t, 4>{1, 1, 1, pad_value_const_buffer_size}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR)
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
                                                        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
                                                        cores,
                                                        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
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
            const ProgramHandle program,
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

    return {program, override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_rm_opt(const Tensor &a,
                                           Tensor &output,
                                           const Shape &output_tensor_shape,
                                           const Shape &input_tensor_start,
                                           const float pad_value) {
    auto program = tt::tt_metal::CreateProgram();

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
    const Tensor pad_value_const_tensor =
        Tensor(
            OwnedStorage{pad_value_const_buffer},
            Shape(std::array<uint32_t, 4>{1, 1, 1, pad_value_const_buffer_size}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR)
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
                                                        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/pad_dims_rm_interleaved_opt.cpp",
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

    auto override_runtime_args_callback = [kernel_id=reader_kernel_id](const ProgramHandle program,
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

    return {program, override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_rm(const Tensor &a, Tensor &output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value) {

    auto program = tt::tt_metal::CreateProgram();

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
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/pad_dims_rm_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args_vec));

    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    auto override_runtime_args_callback = [kernel_id=unary_reader_kernel_id](
        const ProgramHandle program,
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

    return {program, override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_tile(const Tensor &a, Tensor& output, const tt::tt_metal::LegacyShape &output_tensor_shape, const tt::tt_metal::LegacyShape &input_tensor_start, const float pad_value) {

    auto program = tt::tt_metal::CreateProgram();

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
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp",
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
        const ProgramHandle program,
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

    return {program, override_runtime_args_callback};
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
                TT_THROW("Something went terribly wrong in splitting acrtoss cores");
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
                                                                const tt::tt_metal::LegacyShape &output_tensor_shape,
                                                                const tt::tt_metal::LegacyShape &input_tensor_start,
                                                                const float pad_value) {
    auto program = tt::tt_metal::CreateProgram();

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
    const Tensor pad_value_const_tensor =
        Tensor(
            OwnedStorage{pad_value_const_buffer},
            Shape(std::array<uint32_t, 4>{1, 1, 1, pad_value_const_buffer_size}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR)
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
                                                        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
                                                        all_cores,
                                                        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
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
                                              const ProgramHandle program,
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

    return {program, override_runtime_args_callback};
}


std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > get_runtime_args_rm(const Tensor &input_tensor,
                                                                                        Tensor &output_tensor,
                                                                                        const tt::tt_metal::LegacyShape &input_tensor_start,
                                                                                        uint32_t num_cores_total,
                                                                                        uint32_t num_cores,
                                                                                        uint32_t num_cores_y,
                                                                                        CoreRangeSet core_group_1,
                                                                                        uint32_t num_w_sticks_per_core_group_1,
                                                                                        CoreRangeSet core_group_2,
                                                                                        uint32_t num_w_sticks_per_core_group_2
                                                                                        ){

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t W_padded = output_shape[3], H_padded = output_shape[2], C_padded = output_shape[1], N_padded = output_shape[0];
    uint32_t W_padded_bytes = W_padded * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> start_dim_offset(num_dims, 0);


    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val(num_cores_total);


    uint32_t max_read_size = 2048;
    auto front_pad = ttnn::Shape(input_tensor_start);
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for(uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};


        uint32_t num_sticks_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_2;
        } else {
            //no-op
            num_sticks_per_core = 0;
        }


        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, W_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
        }

        // reader
        std::vector<uint32_t> reader_runtime_args = {
            input_buffer->address(),
            num_sticks_per_core_read,
            num_read_per_barrier,
            curr_sticks_read,
            front_pad[-4],
            front_pad[-3],
            front_pad[-2],
        };
        reader_runtime_args.insert(reader_runtime_args.end(), start_dim_offset.begin(), start_dim_offset.end());

        // writer
        std::vector<uint32_t> writer_runtime_args = {
            output_buffer->address(),
            num_sticks_per_core_read,
            num_read_per_barrier,
            curr_sticks_write
        };

        ret_val[i] = {reader_runtime_args, writer_runtime_args};

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t i = 0; i < num_sticks_per_core; ++i) {

            if ((curr_h >= front_pad[-2] and curr_h < (H + front_pad[-2])) and (curr_c >= front_pad[-3] and curr_c < (C + front_pad[-3])) and (curr_n >= front_pad[-4] and curr_n < (N + front_pad[-4]))) {
                curr_sticks_read++;
            }

            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }

        start_dim_offset = {0, curr_h, curr_c, curr_n};

    }

    return ret_val;
}

operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core_v2(const Tensor &a,
                                                                Tensor &output,
                                                                const tt::tt_metal::LegacyShape &output_tensor_shape,
                                                                const tt::tt_metal::LegacyShape &input_tensor_start,
                                                                const float pad_value) {
    auto program = tt::tt_metal::CreateProgram();

    auto output_shape = output_tensor_shape;
    uint32_t W = a.shape()[3], H = a.shape()[2], C = a.shape()[1], N = a.shape()[0];
    uint32_t NCH = H * C * N;
    uint32_t W_padded = output_tensor_shape[3], H_padded = output_tensor_shape[2], C_padded = output_tensor_shape[1], N_padded = output_tensor_shape[0];
    uint32_t NCH_padded = H_padded * C_padded * N_padded;

    auto front_pad = ttnn::Shape(input_tensor_start);

    auto stick_size = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    auto stick_size_padded_front = front_pad[-1] * a.element_size();
    auto stick_size_padded_end = stick_size_padded - stick_size - stick_size_padded_front;
    uint32_t row_major_min_bytes = 16;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x-1, num_cores_y-1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_padded_per_core_group_1, num_sticks_padded_per_core_group_2] = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, NCH_padded);

    uint32_t src0_cb_index = 0;
    auto num_sticks = num_sticks_padded_per_core_group_1 > num_sticks_padded_per_core_group_2 ? num_sticks_padded_per_core_group_1 : num_sticks_padded_per_core_group_2;

    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(num_sticks * stick_size_padded, {{src0_cb_index, cb_data_format}})
        .set_page_size(src0_cb_index, stick_size_padded);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;
    if (not_pad_by_zero) {
        uint32_t src1_cb_index = 1;
        tt::tt_metal::CircularBufferConfig cb_src1_config = tt::tt_metal::CircularBufferConfig(row_major_min_bytes, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, row_major_min_bytes);
        auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);
    }

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(stick_size) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size_padded);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(stick_size_padded) : 0;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) src0_is_dram,
                                            (std::uint32_t) N + front_pad[-4],
                                            (std::uint32_t) H + front_pad[-2],
                                            (std::uint32_t) C + front_pad[-3],
                                            (std::uint32_t) stick_size,
                                            (std::uint32_t) N_padded,
                                            (std::uint32_t) H_padded,
                                            (std::uint32_t) C_padded,
                                            (std::uint32_t) stick_size_padded,
                                            (std::uint32_t) stick_size_padded_front,
                                            (std::uint32_t) stick_size_padded_end,
                                            (std::uint32_t) tt::div_up(stick_size_padded, 512), // max zero size is 512B
                                            (std::uint32_t) (stick_size_padded % 512 == 0 ? 512 : stick_size_padded % 512),
                                            (std::uint32_t) not_pad_by_zero,
                                            (std::uint32_t) packed_pad_value,
                                            (std::uint32_t) row_major_min_bytes,
                                            (std::uint32_t) (stick_size_padded_front / row_major_min_bytes),
                                            (std::uint32_t) (stick_size_padded_end / row_major_min_bytes),
                                            (std::uint32_t) (stick_size_padded / row_major_min_bytes),
                                            (std::uint32_t) src_stick_size_is_power_of_two,
                                            (std::uint32_t) src_stick_size_is_power_of_two ? src_log2_stick_size : stick_size};
    std::vector<uint32_t> writer_ct_args = {(std::uint32_t) src0_cb_index,
                                            (std::uint32_t) dst_is_dram,
                                            (std::uint32_t) stick_size_padded,
                                            (std::uint32_t) dst_stick_size_is_power_of_two,
                                            (std::uint32_t) dst_stick_size_is_power_of_two ? dst_log2_stick_size : stick_size_padded};

    KernelHandle reader_kernel_id = CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved_v2.cpp",
                                                        total_cores,
                                                        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved_v2.cpp",
                                                        total_cores,
                                                        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto all_runtime_args = get_runtime_args_rm(a, output, input_tensor_start, num_cores_total, num_cores, num_cores_y, core_group_1, num_sticks_padded_per_core_group_1, core_group_2, num_sticks_padded_per_core_group_2);

    for(uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            all_runtime_args[i].first
        );

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            all_runtime_args[i].second

        );
    }

    auto override_runtime_args_callback = [
            reader_kernel_id,
            writer_kernel_id,
            compute_with_storage_grid_size,
            input_tensor_start
        ]
    (
        const void* operation,
        const ProgramHandle program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;

        auto output_tensor_shape = dst_tensor.shape();
        uint32_t W_padded = output_tensor_shape[3], H_padded = output_tensor_shape[2], C_padded = output_tensor_shape[1], N_padded = output_tensor_shape[0];
        uint32_t NCH_padded = H_padded * C_padded * N_padded;

        auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_padded_per_core_group_1, num_sticks_padded_per_core_group_2] = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, NCH_padded);
        auto all_runtime_args = get_runtime_args_rm(src_tensor, dst_tensor, input_tensor_start, num_cores_total, num_cores, num_cores_y, core_group_1, num_sticks_padded_per_core_group_1, core_group_2, num_sticks_padded_per_core_group_2);

        for(uint32_t i = 0; i < num_cores_total; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
            }

            {
                SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

inline std::vector<std::vector<uint32_t>> group_contiguous_and_repeated_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) return chunks;

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.push_back(values[0]);

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1] + 1 or values[i] == values[i - 1]) {
            current_chunk.push_back(values[i]);
        } else {
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_chunk.push_back(values[i]);
        }
    }
    // Add the last chunk
    chunks.push_back(current_chunk);
    return chunks;
}

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_pad_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const tt::tt_metal::LegacyShape &input_tensor_start,
    uint32_t num_cores_padded,
    bool row_major,
    uint32_t num_cores_x_padded,
    uint32_t num_cores_y_padded,
    uint32_t shard_height_padded,
    uint32_t shard_height_unpadded,
    uint32_t num_cores_x_unpadded,
    uint32_t num_cores_y_unpadded) {

    tt::tt_metal::Device* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t W_bytes = W * input_tensor.element_size();

    uint32_t W_padded = output_shape[3], H_padded = output_shape[2], C_padded = output_shape[1], N_padded = output_shape[0];
    uint32_t W_padded_bytes = W_padded * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> start_dim_offset(num_dims, 0);

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val(num_cores_padded);

    auto front_pad = ttnn::Shape(input_tensor_start);
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for(uint32_t i = 0, curr_sticks_read = 0; i < num_cores_padded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_padded, i / num_cores_x_padded};
        } else {
            core = {i / num_cores_y_padded, i % num_cores_y_padded};
        }
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // writer rt args, set on top here as interleaved version.
        std::vector<uint32_t> writer_kernel_args = {
            num_sticks_per_core_padded,
            curr_sticks_read,
            front_pad[-4],
            front_pad[-3],
            front_pad[-2],
        };
        writer_kernel_args.insert(writer_kernel_args.end(), start_dim_offset.begin(), start_dim_offset.end());

        // figure out the start read stick id for each core, and the start id for each dim
        std::vector<int> stick_ids_per_core;
        int front_pad_stick_id = -2;
        int pad_stick_id = -1;
        for (uint32_t i = 0; i < num_sticks_per_core_padded; ++i) {
            if ((curr_h >= front_pad[-2] and curr_h < (H + front_pad[-2])) and (curr_c >= front_pad[-3] and curr_c < (C + front_pad[-3])) and (curr_n >= front_pad[-4] and curr_n < (N + front_pad[-4]))) {
                stick_ids_per_core.push_back(curr_sticks_read);
                curr_sticks_read++;
            } else {
                if (curr_h < front_pad[-2] or curr_c < front_pad[-3] or curr_n < front_pad[-4]) {
                    stick_ids_per_core.push_back(front_pad_stick_id);
                } else {
                    stick_ids_per_core.push_back(pad_stick_id);
                }
            }

            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }

        start_dim_offset = {0, curr_h, curr_c, curr_n};

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        auto first_core = device->worker_core_from_logical_core(CoreCoord{0, 0});
        std::pair<uint32_t, uint32_t> prev_xy_pair = std::make_pair(first_core.x, first_core.y);
        for (uint32_t j = 0; j < num_sticks_per_core_padded; ++j) {
            int stick_id = stick_ids_per_core[j];

            // if it is pad stick, we need to leave a gap between the previous non-pad stick and next non-pad stick.
            if (stick_id == -2) { // front padding
                core_stick_map[prev_xy_pair].push_back(stick_id);
            } else if (stick_id == -1) { // end padding
                core_stick_map[prev_xy_pair].push_back(stick_id);
            } else {
                uint32_t shard_id = stick_id / num_sticks_per_core_unpadded;
                uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_unpadded);

                uint32_t shard_grid_inner_dim = row_major ? num_cores_x_unpadded : num_cores_y_unpadded;
                uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
                uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

                uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
                uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

                if (worker_x_logical < num_cores_x_unpadded and worker_y_logical < num_cores_y_unpadded) {
                    auto core_physical = device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});
                    // save stick id in a shard, and core coord into a map
                    std::pair<uint32_t, uint32_t> xy_pair = row_major ? std::make_pair(core_physical.y, core_physical.x)
                                                                        : std::make_pair(core_physical.x, core_physical.y);
                    core_stick_map[xy_pair].push_back(stick_id_in_shard);
                    prev_xy_pair = xy_pair;
                }
            }
        }

        // reader rt args
        vector<uint32_t> reader_kernel_args;
        reader_kernel_args.push_back(core_stick_map.size()); // num_cores

        tt::log_debug("num_cores: {}", core_stick_map.size());

        for (auto core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_kernel_args.push_back((std::uint32_t) xy_pair.second); // noc x
                reader_kernel_args.push_back((std::uint32_t) xy_pair.first); // noc y
            } else {
                reader_kernel_args.push_back((std::uint32_t) xy_pair.first); // noc x
                reader_kernel_args.push_back((std::uint32_t) xy_pair.second); // noc y
            }

            tt::log_debug("xy_pair.first: {}", xy_pair.first);
            tt::log_debug("xy_pair.second: {}", xy_pair.second);
        }

        // coalesce the sticks into chunks
        vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_and_repeated_values(core_stick_pair.second);
            stick_chunks_per_core.push_back(stick_chunks);

            reader_kernel_args.push_back(stick_chunks.size()); // num_chunks for current core
            tt::log_debug("chunk_size: {}", stick_chunks.size());
        }
        for (auto stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]); // start id of a chunk
                tt::log_debug("chunk_start_id: {}", chunk[0]);

                reader_kernel_args.push_back(chunk.size()); // length of a chunk
                tt::log_debug("chunk_length: {}", chunk.size());
            }
        }

        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks pad_rm_sharded(const Tensor &a,
                                                                Tensor &output,
                                                                const tt::tt_metal::LegacyShape &output_tensor_shape,
                                                                const tt::tt_metal::LegacyShape &input_tensor_start,
                                                                const float pad_value) {
    auto program = tt::tt_metal::CreateProgram();

    auto output_shape = output_tensor_shape;
    uint32_t W = a.shape()[3], H = a.shape()[2], C = a.shape()[1], N = a.shape()[0];
    uint32_t num_unpadded_sticks = H * C * N;
    uint32_t W_padded = output_tensor_shape[3], H_padded = output_tensor_shape[2], C_padded = output_tensor_shape[1], N_padded = output_tensor_shape[0];
    uint32_t num_padded_sticks = H_padded * C_padded * N_padded;

    auto front_pad = ttnn::Shape(input_tensor_start);

    tt::log_debug("H_padded: {}", H_padded);
    tt::log_debug("front_pad: {}", front_pad);

    // stick sizes
    auto stick_size_unpadded = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    auto rem_stick_size_padded = stick_size_padded - stick_size_unpadded;
    uint32_t row_major_min_bytes = 16;

    uint32_t zero_pad_stick_size = tt::tt_metal::find_max_divisor(stick_size_padded, 512);
    uint32_t num_zero_pad_sticks_read = stick_size_padded / zero_pad_stick_size;

    tt::log_debug("zero_pad_stick_size: {}", zero_pad_stick_size);
    tt::log_debug("num_zero_pad_sticks_read: {}", num_zero_pad_sticks_read);

    // TODO: add a general case, where we can pad on any dim.
    TT_FATAL(stick_size_unpadded == stick_size_padded, "sharded pad does not support pad on last dim currently as that will cause perf degradation");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    Device *device = a.device();

    // input shard spec
    auto shard_spec_unpadded = a.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    uint32_t shard_width_unpadded = shard_spec_unpadded.shape[1];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = shard_spec_unpadded.grid.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y+1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    tt::log_debug("num_unpadded_sticks: {}", num_unpadded_sticks);
    tt::log_debug("shard_height_unpadded: {}", shard_height_unpadded);
    tt::log_debug("all_cores_unpadded: {}", all_cores_unpadded);
    tt::log_debug("num_cores_unpadded: {}", num_cores_unpadded);


    // output shard spec
    auto shard_spec_padded = output.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];
    uint32_t shard_width_padded = shard_spec_padded.shape[1];

    auto& all_cores_padded = shard_spec_padded.grid;
    uint32_t num_cores_padded = shard_spec_padded.num_cores();
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y+1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    tt::log_debug("num_unpadded_sticks: {}", num_unpadded_sticks);
    tt::log_debug("shard_height_unpadded: {}", shard_height_unpadded);
    tt::log_debug("all_cores_unpadded: {}", all_cores_unpadded);
    tt::log_debug("num_cores_unpadded: {}", num_cores_unpadded);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});


    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(shard_height_unpadded * stick_size_unpadded, {{src0_cb_index, cb_data_format}})
        .set_page_size(src0_cb_index, stick_size_unpadded).set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CB::c_out0; // output operands start at index 16
    tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(shard_height_padded * stick_size_padded, {{output_cb_index, dst_cb_data_format}})
        .set_page_size(output_cb_index, stick_size_padded).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;
    uint32_t src1_cb_index = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config = tt::tt_metal::CircularBufferConfig(stick_size_padded, {{src1_cb_index, cb_data_format}})
        .set_page_size(src1_cb_index, stick_size_padded);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);


    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) stick_size_padded,
                                            (std::uint32_t) shard_height_padded};

    std::vector<uint32_t> writer_ct_args = {(std::uint32_t) N + front_pad[-4],
                                            (std::uint32_t) H + front_pad[-2],
                                            (std::uint32_t) C + front_pad[-3],
                                            (std::uint32_t) stick_size_padded,
                                            (std::uint32_t) N_padded,
                                            (std::uint32_t) H_padded,
                                            (std::uint32_t) C_padded,
                                            (std::uint32_t) num_zero_pad_sticks_read,
                                            (std::uint32_t) zero_pad_stick_size,
                                            (std::uint32_t) not_pad_by_zero,
                                            (std::uint32_t) packed_pad_value,
                                            (std::uint32_t) row_major_min_bytes,
                                            (std::uint32_t) (stick_size_padded / row_major_min_bytes)};

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded.cpp",
        all_cores_padded,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp",
        all_cores_padded,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto all_runtime_args = get_pad_runtime_args_rm_sharded(
        a,
        output,
        input_tensor_start,
        num_cores_padded,
        row_major,
        num_cores_x_padded,
        num_cores_y_padded,
        shard_height_padded,
        shard_height_unpadded,
        num_cores_x_unpadded,
        num_cores_y_unpadded
        );

    for (uint32_t i = 0; i < num_cores_padded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_padded, i / num_cores_x_padded};
        } else {
            core = {i / num_cores_y_padded, i % num_cores_y_padded};
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
    }

    auto override_runtime_args_callback = [
            cb_src0,
            cb_output
        ]
    (
        const void* operation,
        ProgramHandle program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}



} // namespace ttnn::operations::reduction::detail
