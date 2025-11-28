// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "image_rotate_device_operation.hpp"

#include <cmath>
#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/operations/pool/grid_sample/device/grid_sample_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::image_rotate {

using namespace tt;
using namespace tt::tt_metal;

// Constants matching grid_sample
constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
constexpr uint32_t BUFFERING_FACTOR = 2;
constexpr uint32_t REDUCTION_SIZE = 4;           // Bilinear uses 4 corners
constexpr uint32_t MAX_ROWS_FOR_REDUCTION = 16;  // Height of one face
constexpr bool ONE_SCALAR_PER_CORE = false;
constexpr uint32_t DUMMY_CB_ID = 32;

// Helper to convert float to bfloat16 representation
static uint16_t float_to_bfloat16(float value) { return static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16); }

ImageRotateDeviceOperation::ProgramFactory::cached_program_t ImageRotateDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = output;

    tt::tt_metal::Program program{};

    // Data formats
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::IDevice* const device = output_tensor.device();

    // Shape and dimensions (NHWC format)
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
    const uint32_t input_batch = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t input_channels = input_shape[3];

    // Calculate rotation parameters
    const float angle_rad = -operation_attributes.angle * M_PI / 180.0f;  // Negative for inverse rotation
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    // Center point - default to image center if not specified
    float center_x, center_y;
    if (operation_attributes.center.has_value()) {
        center_x = std::get<0>(operation_attributes.center.value());
        center_y = std::get<1>(operation_attributes.center.value());
    } else {
        center_x = (static_cast<float>(input_width) - 1.0f) / 2.0f;
        center_y = (static_cast<float>(input_height) - 1.0f) / 2.0f;
    }

    // Fill value as bfloat16
    const uint16_t fill_value_bf16 = float_to_bfloat16(operation_attributes.fill);

    // Work distribution
    // Total work units = N * H * W (one output pixel per work unit)
    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    // Calculate cores needed
    const auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);

    const uint32_t num_cores_y = compute_grid_size.y;

    // Get logical cores for setting runtime args
    std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, num_cores, true);

    // Calculate stick sizes (aligned to 32 bytes)
    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;

    // CB indices - matching grid_sample pattern but simpler (no grid CB needed)
    uint32_t cb_idx = tt::CBIndex::c_0;

    // CB_0: Not used (grid_sample uses this for grid tensor)
    // We skip c_0 to keep CB indices aligned with grid_sample for potential code reuse
    cb_idx++;

    // CB_1: Input data CB - holds 4 corner sticks per work unit, double-buffered
    const uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_channels / tt::constants::TILE_WIDTH);
    const uint32_t input_cb_page_size = in_ntiles_c * tt::constants::TILE_HW * element_size;
    const auto [input_cb_index, input_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, input_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);

    // CB_2: Not used (grid_sample uses this for split reader input)
    cb_idx++;

    // CB_3: Scalar CB for bilinear interpolation weights
    const uint32_t scalar_cb_page_size = tt::tile_size(input_cb_data_format);
    const auto [scalar_cb_index, scalar_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, scalar_cb_page_size, BUFFERING_FACTOR, input_cb_data_format);

    // CB_4: Not used (grid_sample uses this for split reader scalars)
    cb_idx++;

    // CB_5: Output CB for computed output sticks
    const uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[-1] / tt::constants::FACE_WIDTH);
    const uint32_t output_cb_page_size = tt::constants::FACE_WIDTH * element_size;
    const uint32_t output_cb_pages = out_ntiles_c * BUFFERING_FACTOR;
    const auto [output_cb_index, output_cb_handle] = tt::tt_metal::create_cb(
        cb_idx++, program, all_cores, output_cb_page_size, output_cb_pages, output_cb_data_format);

    // Reader compile-time arguments
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,      // ct_arg[0]: input_cb_index
        scalar_cb_index,     // ct_arg[1]: scalar_cb_index
        input_stick_nbytes,  // ct_arg[2]: input_stick_nbytes
        input_batch,         // ct_arg[3]: input_batch
        input_height,        // ct_arg[4]: input_height
        input_width,         // ct_arg[5]: input_width
    };

    // Append cos/sin/center as reinterpreted uint32_t values
    reader_compile_time_args.push_back(std::bit_cast<uint32_t>(cos_angle));  // ct_arg[6]: cos_angle (as uint32)
    reader_compile_time_args.push_back(std::bit_cast<uint32_t>(sin_angle));  // ct_arg[7]: sin_angle (as uint32)
    reader_compile_time_args.push_back(std::bit_cast<uint32_t>(center_x));   // ct_arg[8]: center_x (as uint32)
    reader_compile_time_args.push_back(std::bit_cast<uint32_t>(center_y));   // ct_arg[9]: center_y (as uint32)
    reader_compile_time_args.push_back(fill_value_bf16);                     // ct_arg[10]: fill_value_bf16

    // Append tensor accessor args for input tensor
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    // Create reader kernel
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/image_rotate/device/kernels/dataflow/reader_image_rotate_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Compute kernel compile-time arguments - match grid_sample's compute_pool_2d.cpp interface
    const uint32_t in_nblocks_c = (uint32_t)std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);

    auto create_compute_kernel = [&](tt::tt_metal::CoreRangeSet cores, uint32_t total_interpolations) {
        std::vector<uint32_t> compute_compile_time_args = {
            in_ntiles_c,                    // ct_arg[0]: in_ntiles_c
            REDUCTION_SIZE,                 // ct_arg[1]: window_size_hw = 4 for bilinear
            0,                              // ct_arg[2]: split_reader = false
            total_interpolations,           // ct_arg[3]: nsticks_per_core_by_nblocks
            input_channels,                 // ct_arg[4]: in_c
            in_nblocks_c,                   // ct_arg[5]: in_nblocks_c
            MAX_ROWS_FOR_REDUCTION,         // ct_arg[6]: max_rows_for_reduction
            input_cb_index,                 // ct_arg[7]: in_cb_id_0
            DUMMY_CB_ID,                    // ct_arg[8]: in_cb_id_1 (unused, no split reader)
            scalar_cb_index,                // ct_arg[9]: in_scalar_cb_id_0
            DUMMY_CB_ID,                    // ct_arg[10]: in_scalar_cb_id_1 (unused)
            DUMMY_CB_ID,                    // ct_arg[11]: in_idx_cb_id (unused)
            DUMMY_CB_ID,                    // ct_arg[12]: pack_tmp_cb_id (unused)
            DUMMY_CB_ID,                    // ct_arg[13]: pack_idx_tmp_cb_id (unused)
            DUMMY_CB_ID,                    // ct_arg[14]: right_inc_cb_id (unused)
            DUMMY_CB_ID,                    // ct_arg[15]: down_left_wrap_inc_cb_id (unused)
            DUMMY_CB_ID,                    // ct_arg[16]: up_left_wrap_inc_cb_id (unused)
            output_cb_index,                // ct_arg[17]: out_cb_id
            DUMMY_CB_ID,                    // ct_arg[18]: out_idx_cb_id (unused)
            ONE_SCALAR_PER_CORE ? 1U : 0U,  // ct_arg[19]: one_scalar_per_core
            DUMMY_CB_ID,                    // ct_arg[20]: pre_tilize_cb_id (unused for row-major output)
            0U,                             // ct_arg[21]: is_output_tiled = false (ROW_MAJOR)
            0U,                             // ct_arg[22]: is_output_block_format = false
            0U,                             // ct_arg[23]: return_indices = false
            1U,                             // ct_arg[24]: stride_h (unused)
            1U,                             // ct_arg[25]: stride_w (unused)
            1U,                             // ct_arg[26]: in_h_padded (unused)
            1U,                             // ct_arg[27]: in_w_padded (unused)
            1U,                             // ct_arg[28]: eff_kernel_h (unused)
            1U,                             // ct_arg[29]: eff_kernel_w (unused)
            0U,                             // ct_arg[30]: pad_l (unused)
        };

        return tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp",
            cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_compile_time_args,
                .defines = pool::get_defines(pool::Pool2DType::AVG_POOL2D)});
    };

    // Create compute kernels for each core group
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    if (core_group_1.num_cores() > 0) {
        compute_kernel_id = create_compute_kernel(core_group_1, num_sticks_per_core_group_1);
    }
    if (core_group_2.num_cores() > 0) {
        create_compute_kernel(core_group_2, num_sticks_per_core_group_2);
    }

    // Writer compile-time arguments
    const uint32_t output_stick_size = input_channels * element_size;
    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,    // ct_arg[0]: output_cb_index
        output_stick_size,  // ct_arg[1]: output_stick_size
        out_ntiles_c        // ct_arg[2]: out_ntiles_c
    };
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    // Create writer kernel - reuse grid_sample's writer
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime arguments for each core
    uint32_t sticks_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const uint32_t num_sticks =
            core_group_1.contains(core) ? num_sticks_per_core_group_1 : num_sticks_per_core_group_2;

        // Reader runtime args
        std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(),  // rt_arg[0]: input_buffer_address
            num_sticks,                        // rt_arg[1]: num_sticks
            sticks_processed                   // rt_arg[2]: start_stick_id
        };

        // Writer runtime args
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),  // rt_arg[0]: output_buffer_address
            num_sticks,                         // rt_arg[1]: output_sticks
            sticks_processed                    // rt_arg[2]: output_processed
        };

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        sticks_processed += num_sticks;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void ImageRotateDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::image_rotate
