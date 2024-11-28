// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

#define MASK_64      0xFFFFFFFFFFFFFFC0
#define OFFSET_64    0x000000000000003F
#define MASK_16      0xFFFFFFFFFFFFFFF0
#define OFFSET_16    0x000000000000000F

namespace ttnn::operations::data_movement::rm_reshape{

operation::ProgramWithCallbacks rm_reshape_preparer(const Tensor& input, const Tensor& output)
{
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    //get datum size
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const uint32_t data_size = input.element_size();
    CoreRange core({0, 0}, {0, 0});

    tt::tt_metal::Device *device = input.device();
    ttnn::Shape input_log_shape = ttnn::Shape(input.get_logical_shape().view());
    ttnn::Shape output_log_shape = ttnn::Shape(output.get_logical_shape().view());
    tt::log_debug("row major reshape");
    tt::log_debug("input shape: {}", input_log_shape);
    tt::log_debug("output shape: {}", output_log_shape);
    tt::log_debug("data size: {}", data_size);
    uint32_t source_page_size_bytes = input_log_shape[-1] * data_size;
    uint32_t dest_page_size_bytes = output_log_shape[-1] * data_size;
    uint32_t source_read_size_bytes = ((source_page_size_bytes-1) & MASK_64) + 128;
    uint32_t read_start_page = 0;
    uint32_t read_end_page = input_log_shape[-2];
    uint32_t write_start_page = 0;
    tt::tt_metal::Buffer *src_buffer = input.buffer();
    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t cb_size0 = source_read_size_bytes;
    const uint32_t cb_size1 = ((dest_page_size_bytes-1)&MASK_64) + 80;

    uint32_t src0_cb_index = 0;
    uint32_t src1_cb_index = 1;
    tt::tt_metal::CircularBufferConfig cb_src0_config = tt::tt_metal::CircularBufferConfig(cb_size0*2, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, cb_size0);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    tt::tt_metal::CircularBufferConfig cb_src1_config = tt::tt_metal::CircularBufferConfig(cb_size1, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, cb_size1);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
    //set the runtime args
    //set the compile time args
    uint32_t src0_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) (source_page_size_bytes%64==0) ? 1 : 0,
        (std::uint32_t) (source_page_size_bytes%16==0) ? 1 : 0,
        (std::uint32_t) (dest_page_size_bytes%16==0) ? 1 : 0,
    };

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args));
    std::vector<uint32_t> reader_runtime_args = {
        src_buffer->address(),
        dst_buffer->address(),
        source_page_size_bytes,
        dest_page_size_bytes,
        source_read_size_bytes,
        read_start_page,
        read_end_page,
        write_start_page,
        src0_cb_index,
        src1_cb_index
    };
    tt::tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        reader_runtime_args
    );
    return {.program=std::move(program)};
}
}; // namespace ttnn::operations::data_movement::rm_reshape
