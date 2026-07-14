// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

void emit_runtime_args_hc_tiled(
    KernelDescriptor& reader_desc,
    KernelDescriptor& writer_desc,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW_bytes = C * HW * input_tensor.element_size();

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    reader_desc.runtime_args.reserve(num_cores_total);
    writer_desc.runtime_args.reserve(num_cores_total);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;
        }

        uint32_t h = num_tiles_read / CtWt % H;
        uint32_t ct = num_tiles_read / Wt % Ct;

        reader_desc.emplace_runtime_args(
            core,
            {input_buffer,
             Wt,
             H,
             Ct,
             HW_bytes,
             CHW_bytes,
             num_tiles_read,
             num_tiles_per_core,
             num_tiles_read / CtHWt * CHW_bytes,
             h,
             h / TILE_HEIGHT * Wt,
             ct,
             ct * TILE_HEIGHT * HW_bytes,
             num_tiles_read % Wt});

        writer_desc.emplace_runtime_args(core, {output_buffer, num_tiles_per_core, num_tiles_read});

        num_tiles_read += num_tiles_per_core;
    }
}

}  // namespace

tt::tt_metal::ProgramDescriptor TransposeHCTiledProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    uint32_t sub_tile_line_bytes = 16 * input_tensor.element_size();
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;

    ProgramDescriptor desc;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "transpose_hc_tiled");
    log_debug(tt::LogOp, "sub_tile_line_bytes: {}", sub_tile_line_bytes);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    // check if we need to allocate a scratch buffer
    // The kernel reads several 16 element face lines (32B for BFLOAT16) from different input tiles to form a single
    // output tile, one output tile at a time Each face line is 32 bytes, so if our minimum read alignment is greater
    // than that (64B for Blackhole) then we will have reads from unaligned face-lines into differently aligned
    // destination face-lines
    // TODO: noc_async_write only require 16B alignment for both DRAM and L1 for Blackhole, so instead of reading in
    // face-lines from C tiles to form a single tile, we can load a single tile and then write out its face-lines to C
    // tiles
    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;

    uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // need some scratch memory here - if we need data from a misaligned address then we need to read from the
    // nearest aligned address and then copy the data to the correct location
    if (misaligned) {
        uint32_t src1_cb_index = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = alignment,
            .core_ranges = total_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src1_cb_index),
                .data_format = cb_data_format,
                .page_size = alignment,
            }}},
        });
    }

    Buffer* src0_buffer = input_tensor.buffer();
    std::vector<uint32_t> reader_compile_time_args;
    reader_compile_time_args.push_back(sub_tile_line_bytes);
    reader_compile_time_args.push_back(cb_data_format == tt::DataFormat::Float32 ? 1 : 0);
    reader_compile_time_args.push_back(alignment);
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_interleaved_partitioned.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = total_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    emit_runtime_args_hc_tiled(
        reader_desc,
        writer_desc,
        input_tensor,
        output_tensor,
        num_cores_total,
        num_cores_y,
        core_group_1,
        num_tiles_per_core_group_1,
        core_group_2,
        num_tiles_per_core_group_2);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
