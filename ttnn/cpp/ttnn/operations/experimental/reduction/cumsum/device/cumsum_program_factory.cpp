// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <numeric>
#include <random>
#include "cumsum_device_operation.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/command_queue.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/data_types.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::reduction {

CumSumDeviceOperation::SingleCore::cached_program_t CumSumDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;

    // Device setup
    Program program;
    IDevice* device = output_tensor.device();

    // Parameters setup
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_dtype = input_tensor.dtype();
    const auto& output_dtype = output_tensor.dtype();
    const auto& tensor_shape = input_tensor.padded_shape();
    const uint32_t tensor_rank = tensor_shape.rank();
    int32_t dim = operation_attributes.dim;

    if (dim < 0) {  // Handle cases where dim is negative
        dim += tensor_rank;
    }

    constexpr CoreCoord core{0, 0};

    TT_FATAL(
        input_dtype == output_dtype,
        "In-device type conversion not supported yet: received {} input dtype and {} output dtype",
        input_dtype,
        output_dtype);

    TT_FATAL(
        output_dtype == DataType::FLOAT32 || output_dtype == DataType::INT32 || output_dtype == DataType::UINT32 ||
            output_dtype == DataType::BFLOAT16,
        "Only float32, bfloat16, uint32 and int32 data type supported for now: received {}",
        output_dtype);

    TT_FATAL(
        output_tensor.layout() == Layout::TILE,
        "Only supported tensor layout is TILE: received {}",
        output_tensor.layout());

    TT_FATAL(
        tensor_rank >= 3, "Device operation only support 3D tensor and above: received tensor of rank {}", tensor_rank);

    TT_FATAL(
        input_tensor.buffer()->size() == input_tensor.physical_volume() * input_tensor.element_size(),
        "Input tensor size ({}) does not match expected volume ({})",
        input_tensor.buffer()->size(),
        input_tensor.physical_volume() * input_tensor.element_size());

    TT_FATAL(input_tensor.logical_volume() > 0, "Input must not be empty");

    TT_ASSERT(dim >= 0, "dim argument must be positive: received {}", dim);

    TT_FATAL(
        dim + 2 < tensor_rank, "cumsum on x and y axes not supported: received dim = {}, rank = {}", dim, tensor_rank);

    // Buffer setup
    const uint32_t single_tile_size = output_tensor.element_size() * tt::constants::TILE_HW;

    constexpr uint32_t cb_in_index = CBIndex::c_0;
    constexpr uint32_t cb_out_index = CBIndex::c_1;
    constexpr uint32_t cb_zero_index = CBIndex::c_2;
    constexpr uint32_t cb_intermed_index = CBIndex::c_3;

    // Device operation does not handle on-the-fly type conversion yet and we ensured that input_dtype == ouptut_dtype
    DataFormat in_df = datatype_to_dataformat_converter(output_dtype);
    DataFormat out_df = in_df;

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(single_tile_size, {{cb_in_index, in_df}}).set_page_size(cb_in_index, single_tile_size);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(single_tile_size, {{cb_out_index, out_df}}).set_page_size(cb_out_index, single_tile_size);

    CircularBufferConfig cb_zero_config = CircularBufferConfig(single_tile_size, {{cb_zero_index, out_df}})
                                              .set_page_size(cb_zero_index, single_tile_size);

    CircularBufferConfig cb_intermed_config = CircularBufferConfig(single_tile_size, {{cb_intermed_index, out_df}})
                                                  .set_page_size(cb_intermed_index, single_tile_size);

    CreateCircularBuffer(program, core, cb_in_config);
    CreateCircularBuffer(program, core, cb_out_config);
    CreateCircularBuffer(program, core, cb_zero_config);
    CreateCircularBuffer(program, core, cb_intermed_config);

    // Create kernels
    KernelHandle cumsum_reader_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/cumsum/device/kernels/dataflow/cumsum_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle cumsum_writer_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/cumsum/device/kernels/dataflow/cumsum_writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {};
    std::map<std::string, std::string> defines_kernel_args = {};

    if (is_integer_format(out_df)) {
        // Used to switch to add_tile_int32() instead of add_tiles()
        defines_kernel_args["CUMSUM_USE_INT32"] = "1";
    }

    KernelHandle cumsum_compute_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/cumsum/device/kernels/compute/cumsum_compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = defines_kernel_args});

    // Parameters setup
    uint32_t num_tiles = output_tensor.physical_volume() / tt::constants::TILE_HW;
    const uint32_t xy_volume = tensor_shape[tensor_rank - 1] * tensor_shape[tensor_rank - 2];  // W * H
    const uint32_t num_tiles_per_row = tensor_shape[dim];     // each row contains N independent tiles
    const uint32_t num_rows = num_tiles / num_tiles_per_row;  // total number of rows in tensor
    const uint32_t HtWt = xy_volume / tt::constants::TILE_HW;  // padded shape => xy_volume is multiple of tile_size

    // Depending on tensor rank and dim parameter, we may have to iterative on several tensor axis, with varying offset
    // To solve this problem (and generalize the approach), we can compute two offsets: for dimensions > dim and for
    // dimensions < dim We thus two parameters product_high_dims (product High) and product_low_dims (product Low):
    // product_high_dims is the number of iterations on 'high dims', it is the product of all axes length for dimensions
    // > `dim` (excluding x and y axes) product_low_dims is the number of iterations on 'low dims', it is the product of
    // all axes length for dimensions < `dim`
    uint32_t product_high_dims = 1;
    uint32_t product_low_dims = 1;

    for (int i = dim + 1; i + 2 < tensor_rank; i++) {
        product_high_dims *= tensor_shape[i];
    }
    for (int i = 0; i < dim; i++) {
        product_low_dims *= tensor_shape[i];
    }

    SetRuntimeArgs(
        program,
        cumsum_reader_handle_id,
        core,
        {
            input_tensor.buffer()->address(),
            num_tiles_per_row,
            product_high_dims,
            product_low_dims,
            HtWt,
        });

    SetRuntimeArgs(
        program,
        cumsum_writer_handle_id,
        core,
        {
            output_tensor.buffer()->address(),
            num_tiles_per_row,
            product_high_dims,
            product_low_dims,
            HtWt,
        });

    SetRuntimeArgs(
        program,
        cumsum_compute_handle_id,
        core,
        {
            product_high_dims * product_low_dims * HtWt,
            num_tiles_per_row,
        });

    return {std::move(program), {}};
}

void CumSumDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dtype = operation_attributes.dtype;

    const auto& input_dtype = input_tensor.dtype();

    // Support for override_runtime_arguments() will be added in resolution of issue #21097
    TT_THROW("override_runtime_arguments() not yet supported");
}

}  // namespace ttnn::operations::experimental::reduction
