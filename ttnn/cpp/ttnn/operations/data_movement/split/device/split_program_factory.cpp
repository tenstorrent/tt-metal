// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/split/device/split_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

void setup_runtime(
    KernelDescriptor& reader_desc,
    KernelDescriptor& writer_desc,
    const uint32_t& num_cores_c,
    const uint32_t& z,
    const uint32_t& num_cores_x,
    const uint32_t& per_core_tiles_y,
    const uint32_t& per_core_tiles_x,
    const uint32_t& num_tiles_per_z,
    Buffer* in0_buffer,
    Buffer* out0_buffer,
    Buffer* out1_buffer) {
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    if (num_cores_c > 1) {
        TT_FATAL(num_cores_c % 2 == 0, "Must be even number of cores");
    }
    uint32_t idc_outer_limit = 1;
    uint32_t idc_inner_limit = num_cores_c;

    for (int id_r_outer = 0; id_r_outer < z; id_r_outer++) {
        for (int id_r_inner = 0; id_r_inner < num_cores_x; id_r_inner++) {
            uint32_t id_r = (id_r_outer * num_cores_x) + id_r_inner;

            uint32_t id_r_reader =
                (id_r_outer * num_tiles_per_z) + (id_r_inner * per_core_tiles_y * num_cores_c * per_core_tiles_x);
            uint32_t id_r_writer = id_r_reader / 2;
            if (num_cores_c > 1) {
                idc_outer_limit = 2;
                idc_inner_limit = num_cores_c / 2;
            }
            for (int id_c_outer = 0; id_c_outer < idc_outer_limit; id_c_outer++) {
                for (int id_c_inner = 0; id_c_inner < idc_inner_limit; id_c_inner++) {
                    uint32_t id_c = (id_c_outer * idc_inner_limit) + id_c_inner;
                    CoreCoord core = {(std::size_t)start_core_x + id_r, (std::size_t)start_core_y + id_c};

                    uint32_t reader_core_id = id_c * per_core_tiles_y;
                    reader_core_id += id_r_reader;

                    bool out0_only = false;
                    bool out1_only = false;
                    if (num_cores_c > 1) {
                        out0_only = (id_c_outer == 0);
                        out1_only = (id_c_outer == 1);
                    }

                    uint32_t writer_core_id = (id_c_inner * per_core_tiles_y) + (id_r_writer);

                    // Buffer* entries register binding slots so the framework patches their
                    // addresses on cache hits without rebuilding the descriptor.
                    reader_desc.emplace_runtime_args(
                        core, {(std::uint32_t)reader_core_id, in0_buffer, (std::uint32_t)0});  // split on last dim

                    writer_desc.emplace_runtime_args(
                        core,
                        {writer_core_id,
                         out0_buffer,  // first base addr
                         out1_buffer,  // second base addr
                         (std::uint32_t)out0_only,
                         (std::uint32_t)out1_only});
                }
            }
        }
    }
}

}  // namespace

ProgramDescriptor SplitProgramFactory::create_descriptor(
    const SplitParams& operation_attributes, const SplitInputs& tensor_args, std::vector<Tensor>& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    const uint32_t num_chunks = operation_attributes.num_splits;

    auto input_shape = input_tensor.padded_shape();

    IDevice* device = input_tensor.device();
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    Buffer* in0_buffer = input_tensor.buffer();

    // Output buffers
    TT_FATAL(
        tensor_return_value.size() == num_chunks,
        "Number of output tensors ({}) must equal number of chunks ({})",
        tensor_return_value.size(),
        num_chunks);
    Tensor& out0 = tensor_return_value[0];
    Tensor& out1 = tensor_return_value[1];

    Buffer* out0_buffer = out0.buffer();
    TT_FATAL(out0_buffer != nullptr, "Output 0 buffer should be allocated on device!");
    Buffer* out1_buffer = out1.buffer();
    TT_FATAL(out1_buffer != nullptr, "Output 1 buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t z = input_shape[1];
    uint32_t num_tiles_dim_2 = input_shape[2] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_dim_3 = input_shape[3] / tt::constants::TILE_WIDTH;
    uint32_t num_cores_x_limit = device->compute_with_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_with_storage_grid_size().y;

    // parallelize z
    auto num_cores_z = z;

    // parallelize y
    auto [num_cores_y, per_core_tiles_y] = get_max_cores_divisible_by_tiles_per_core_tiles(
        num_tiles_dim_3, num_cores_y_limit, /*request_even=*/(num_tiles_dim_3 > 1));

    // parallelize x
    auto [num_cores_x, per_core_tiles_x] =
        get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_2, num_cores_x_limit / num_cores_z);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_cores_c = num_cores_y;
    uint32_t num_cores_r = num_cores_x * num_cores_z;

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_r - 1, (std::size_t)start_core_y + num_cores_c - 1});

    TT_FATAL(out0_buffer->buffer_type() == out1_buffer->buffer_type(), "Output buffers should be the same type");

    uint32_t num_tiles_per_z = (per_core_tiles_x * num_cores_x) * (per_core_tiles_y * num_cores_y);
    uint32_t z_stride_read = num_tiles_per_z;
    uint32_t y_stride_read = per_core_tiles_y * num_cores_y;

    std::vector<uint32_t> reader_compile_time_args = {// READER COMPILE TIME ARGS
                                                      (std::uint32_t)(z / num_cores_z),
                                                      (std::uint32_t)per_core_tiles_x,  // out_num_tiles_per_tensor
                                                      (std::uint32_t)per_core_tiles_y,  // out_num_tiles_per_tensor
                                                      (std::uint32_t)z_stride_read,
                                                      (std::uint32_t)y_stride_read};
    TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);

    uint32_t z_stride_write = num_tiles_per_z / num_chunks;
    uint32_t y_stride_write = per_core_tiles_y * (num_cores_c / num_chunks);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)per_core_tiles_x,  // out_num_tiles_per_tensor
        (std::uint32_t)per_core_tiles_y,  // out_num_tiles_per_tensor

        (std::uint32_t)(z / num_cores_z),
        (std::uint32_t)z_stride_write,
        (std::uint32_t)y_stride_write

    };
    TensorAccessorArgs(*out0_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*out1_buffer).append_to(writer_compile_time_args);

    ProgramDescriptor desc;

    // Circular buffer for input tile staging.
    constexpr uint32_t src0_cb_index = 0;
    constexpr uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = CoreRangeSet{all_cores},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/"
        "reader_tm_tile_layout_split_two_chunks.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet{all_cores};
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/"
        "writer_tm_tile_layout_split_two_chunks.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = CoreRangeSet{all_cores};
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    setup_runtime(
        reader_desc,
        writer_desc,
        num_cores_c,
        num_cores_z,
        num_cores_x,
        per_core_tiles_y,
        per_core_tiles_x,
        num_tiles_per_z,
        in0_buffer,
        out0_buffer,
        out1_buffer);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
