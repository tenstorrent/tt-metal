// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_device_operation.hpp"

#include <array>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/types.hpp"

namespace {

using namespace tt::tt_metal;
using namespace ttsl;

enum class IntImgCB : uint32_t {
    START,
    INPUT,
    ACC,
    CUMSUM_STAGE_0,
    CUMSUM_STAGE_1,
    CUMSUM_STAGE_2,
    OUTPUT,
    AXIS_2_BUFFER,  // memoizing last tile (for the "deeper" block) for propagation along axis 2
    AXIS_3_BUFFER,  // memoizing upper 32 tiles for propagation along axis 3
};

CBDescriptor make_cb(
    const DataType& dtype, const IntImgCB& intimg_cb, const CoreRangeSet& core_range_set, const uint32_t& num_tiles) {
    const uint32_t cb_id{static_cast<uint32_t>(intimg_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{tt::tile_size(cb_data_format)};
    return CBDescriptor{
        .total_size = num_tiles * single_tile_size,
        .core_ranges = core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    };
}

constexpr std::array<const char*, 3> KERNEL_PATHS{
    "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/"
    "intimg_reader.cpp",
    "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/intimg_compute.cpp",
    "ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/"
    "intimg_writer.cpp"};

}  // namespace

namespace ttnn::experimental::prim {

// it is expected that this operator is used primarily on BOS' custom chips, which are 4 rows and 5 columns, however the
// expected parallelisation of the maximal input shape is calculated to be 4 rows and 2 columns
constexpr uint32_t CORES_X = 2;
constexpr uint32_t CORES_Y = 4;

tt::tt_metal::ProgramDescriptor IntImgDeviceOperation::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    constexpr uint32_t BLOCK_DEPTH = 48;

    auto* src_buffer{input_tensor.buffer()};
    auto* dst_buffer{output_tensor.buffer()};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(input_tensor.dtype())};
    const bool fp32_dest_acc_en{
        (dst_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Int32) ||
        (dst_cb_data_format == DataFormat::UInt32)};

    const auto tile_spec = input_tensor.tensor_spec().tile();

    const uint32_t tiles_num_per_full_block_depth_cb = BLOCK_DEPTH;
    const uint32_t tiles_num_per_small_cb = 2;
    const auto core_range_set = CoreRangeSet{{{0, 0}, {CORES_X - 1, CORES_Y - 1}}};

    ProgramDescriptor desc;

    desc.cbs.push_back(make_cb(input_tensor.dtype(), IntImgCB::START, core_range_set, tiles_num_per_small_cb));
    desc.cbs.push_back(
        make_cb(input_tensor.dtype(), IntImgCB::INPUT, core_range_set, tiles_num_per_full_block_depth_cb));
    desc.cbs.push_back(make_cb(input_tensor.dtype(), IntImgCB::ACC, core_range_set, tiles_num_per_small_cb));
    desc.cbs.push_back(
        make_cb(input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_0, core_range_set, tiles_num_per_full_block_depth_cb));
    desc.cbs.push_back(
        make_cb(input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_1, core_range_set, tiles_num_per_full_block_depth_cb));
    desc.cbs.push_back(
        make_cb(input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_2, core_range_set, tiles_num_per_full_block_depth_cb));
    desc.cbs.push_back(
        make_cb(input_tensor.dtype(), IntImgCB::OUTPUT, core_range_set, tiles_num_per_full_block_depth_cb));
    desc.cbs.push_back(make_cb(input_tensor.dtype(), IntImgCB::AXIS_2_BUFFER, core_range_set, tiles_num_per_small_cb));
    desc.cbs.push_back(
        make_cb(input_tensor.dtype(), IntImgCB::AXIS_3_BUFFER, core_range_set, tiles_num_per_full_block_depth_cb));
    // create_cb(program, input_tensor.dtype(), IntImgCB::AXIS_3_BUFFER_1, core_range_set, tiles_num_per_cb);

    std::vector<uint32_t> compute_compile_time_args{
        static_cast<uint32_t>(IntImgCB::START),
        static_cast<uint32_t>(IntImgCB::INPUT),
        static_cast<uint32_t>(IntImgCB::ACC),
        static_cast<uint32_t>(IntImgCB::CUMSUM_STAGE_0),
        static_cast<uint32_t>(IntImgCB::CUMSUM_STAGE_1),
        static_cast<uint32_t>(IntImgCB::CUMSUM_STAGE_2),
        static_cast<uint32_t>(IntImgCB::OUTPUT),
        static_cast<uint32_t>(IntImgCB::AXIS_2_BUFFER),
        static_cast<uint32_t>(IntImgCB::AXIS_3_BUFFER),
        tile_spec.get_height(),
        tile_spec.get_width(),
        BLOCK_DEPTH,
        input_shape[3],
        input_shape[2],
        input_shape[1],
        input_shape[0],
        CORES_X,
        CORES_Y};
    auto dataflow_compile_time_args = compute_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(dataflow_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(dataflow_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = KERNEL_PATHS[0];
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range_set;
    reader_desc.compile_time_args = dataflow_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};
    // Replicate the same runtime args (input buffer address) across every core in the grid.
    for (uint32_t x = 0; x < CORES_X; ++x) {
        for (uint32_t y = 0; y < CORES_Y; ++y) {
            reader_desc.emplace_runtime_args(CoreCoord{x, y}, {src_buffer});
        }
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = KERNEL_PATHS[1];
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_range_set;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = false,
    };

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = KERNEL_PATHS[2];
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range_set;
    writer_desc.compile_time_args = dataflow_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};
    // Replicate the same runtime args (output buffer address) across every core in the grid.
    for (uint32_t x = 0; x < CORES_X; ++x) {
        for (uint32_t y = 0; y < CORES_Y; ++y) {
            writer_desc.emplace_runtime_args(CoreCoord{x, y}, {dst_buffer});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
