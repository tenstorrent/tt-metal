// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_device_operation.hpp"
#include "intimg_program_factory.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace {

using namespace tt::tt_metal;
using namespace tt::stl;

enum class IntImgCB : uint32_t {
    START,
    INPUT,
    ACC,
    CUMSUM_STAGE_0,
    CUMSUM_STAGE_1,
    CUMSUM_STAGE_2,
    CUMSUM_STAGE_3,
    OUTPUT,
    AXIS_2_BUFFER,  // memoizing last tile (for the "deeper" block) for propagation along axis 2
    AXIS_3_BUFFER,  // memoizing upper 32 tiles for propagation along axis 3
};

CBHandle create_cb(
    Program& program,
    const DataType& dtype,
    const IntImgCB& intimg_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    const uint32_t cb_id{static_cast<uint32_t>(intimg_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{tt::tile_size(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args = {}) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

}  // namespace

namespace ttnn::experimental::prim {

// it is expected that this operator is used primarily on BOS' custom chips, which are 4 rows and 5 columns, however the
// expected parallelisation of the maximal input shape is calculated to be 4 rows and 2 columns
constexpr uint32_t CORES_X = 2;
constexpr uint32_t CORES_Y = 4;

IntImgProgramFactory::cached_program_t IntImgProgramFactory::create(
    const IntImgParams& /*operation_attributes*/, const Tensor& tensor_args, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    constexpr uint32_t BLOCK_DEPTH = 48;

    Program program{};

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
    create_cb(program, input_tensor.dtype(), IntImgCB::START, core_range_set, tiles_num_per_small_cb);
    create_cb(program, input_tensor.dtype(), IntImgCB::INPUT, core_range_set, tiles_num_per_full_block_depth_cb);
    create_cb(program, input_tensor.dtype(), IntImgCB::ACC, core_range_set, tiles_num_per_small_cb);
    create_cb(
        program, input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_0, core_range_set, tiles_num_per_full_block_depth_cb);
    create_cb(
        program, input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_1, core_range_set, tiles_num_per_full_block_depth_cb);
    create_cb(
        program, input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_2, core_range_set, tiles_num_per_full_block_depth_cb);
    create_cb(program, input_tensor.dtype(), IntImgCB::OUTPUT, core_range_set, tiles_num_per_full_block_depth_cb);
    create_cb(program, input_tensor.dtype(), IntImgCB::AXIS_2_BUFFER, core_range_set, tiles_num_per_small_cb);
    create_cb(
        program, input_tensor.dtype(), IntImgCB::AXIS_3_BUFFER, core_range_set, tiles_num_per_full_block_depth_cb);
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
    const ReaderDataMovementConfig reader_config{dataflow_compile_time_args};
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = false,
        .compile_args = compute_compile_time_args,
        .defines = {}};
    const WriterDataMovementConfig writer_config{dataflow_compile_time_args};

    auto reader_kernel_id{create_kernel(program, KERNEL_PATHS[0], core_range_set, reader_config)};
    auto compute_kernel_id{create_kernel(program, KERNEL_PATHS[1], core_range_set, compute_config)};
    auto writer_kernel_id{create_kernel(program, KERNEL_PATHS[2], core_range_set, writer_config)};

    SetRuntimeArgs(program, reader_kernel_id, core_range_set, {src_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core_range_set, {dst_buffer->address()});

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id}};
}

void IntImgProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const IntImgParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& tensor_return_value) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    auto input_buffer_address = tensor_args.buffer()->address();
    auto output_buffer_address = tensor_return_value.buffer()->address();
    for (uint32_t x = 0; x < CORES_X; ++x) {
        for (uint32_t y = 0; y < CORES_Y; ++y) {
            const auto core = CoreCoord{x, y};
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

            reader_runtime_args[0] = input_buffer_address;
            writer_runtime_args[0] = output_buffer_address;
        }
    }
}

}  // namespace ttnn::experimental::prim
