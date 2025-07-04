// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/data_types.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt-metalium/work_split.hpp"

namespace ttnn::operations::reduction {

CumSumDeviceOperation::ProgramFactory::cached_program_t CumSumDeviceOperation::ProgramFactory::create(
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
    const bool flip = operation_attributes.flip;

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

    // Parameters setup
    const auto& tile = input_tensor.tensor_spec().tile();
    uint32_t num_tiles = output_tensor.physical_volume() / tile.get_tile_hw();

    const uint32_t xy_volume = tensor_shape[-1] * tensor_shape[-2];  // W * H
    const uint32_t num_tiles_per_row = tensor_shape[dim];            // each row contains N independent tiles
    const uint32_t num_rows = num_tiles / num_tiles_per_row;         // total number of rows in tensor
    const uint32_t HtWt = xy_volume / tile.get_tile_hw();  // padded shape => xy_volume is multiple of tile_size

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

    // Buffer setup
    const uint32_t single_tile_size = output_tensor.element_size() * tile.get_tile_hw();

    constexpr uint32_t cb_in_index = CBIndex::c_0;
    constexpr uint32_t cb_out_index = CBIndex::c_1;
    constexpr uint32_t cb_zero_index = CBIndex::c_2;
    constexpr uint32_t cb_intermed_index = CBIndex::c_3;

    auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt_metal::split_work_to_cores(grid, num_rows);

    // Device operation does not handle on-the-fly type conversion yet and we ensured that input_dtype == ouptut_dtype
    DataFormat in_df = datatype_to_dataformat_converter(output_dtype);
    DataFormat out_df = in_df;

    constexpr uint32_t TILES_PER_CB = 4;
    uint32_t total_cb_size = TILES_PER_CB * single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(total_cb_size, {{cb_in_index, in_df}}).set_page_size(cb_in_index, single_tile_size);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(total_cb_size, {{cb_out_index, out_df}}).set_page_size(cb_out_index, single_tile_size);

    CircularBufferConfig cb_zero_config =
        CircularBufferConfig(total_cb_size, {{cb_zero_index, out_df}}).set_page_size(cb_zero_index, single_tile_size);

    CircularBufferConfig cb_intermed_config = CircularBufferConfig(total_cb_size, {{cb_intermed_index, out_df}})
                                                  .set_page_size(cb_intermed_index, single_tile_size);

    CreateCircularBuffer(
        program,
        all_cores,
        in_df,
        {{tt::CBIndex::c_0, 1}, {tt::CBIndex::c_1, 1}, {tt::CBIndex::c_2, 1}, {tt::CBIndex::c_3, 1}});

    std::vector<uint32_t> reader_kernel_compile_args = {
        num_tiles_per_row,
        HtWt,
        product_high_dims,
        product_low_dims,
        flip,
        input_tensor.memory_config().buffer_type() == BufferType::DRAM};
    std::vector<uint32_t> writer_kernel_compile_args = {
        num_tiles_per_row,
        HtWt,
        product_high_dims,
        product_low_dims,
        flip,
        output_tensor.memory_config().buffer_type() == BufferType::DRAM};

    ////////////////////////////////////////////////////////////////////////////
    //                      Data Movement Kernel Setup
    ////////////////////////////////////////////////////////////////////////////
    KernelHandle cumsum_reader_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/cumsum/device/kernels/dataflow/cumsum_reader.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_kernel_compile_args});

    KernelHandle cumsum_writer_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/cumsum/device/kernels/dataflow/cumsum_writer.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_kernel_compile_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> compute_kernel_args = {};
    std::map<std::string, std::string> defines_kernel_args = {};

    if (is_integer_format(out_df)) {
        // Used to switch to add_tile_int32() instead of add_tiles()
        defines_kernel_args["CUMSUM_USE_INT32"] = "1";
    }

    KernelHandle cumsum_compute_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/cumsum/device/kernels/compute/cumsum_compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = defines_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    uint32_t start_row = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / grid.y, i % grid.y};

        uint32_t rows_per_core = 0;
        if (core_group_1.contains(core)) {
            rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core outside specified core ranges");
        }

        uint32_t start_low_tile_index = start_row / (product_high_dims * HtWt);
        uint32_t start_high_tile_index = start_row % (product_high_dims * HtWt);

        SetRuntimeArgs(
            program,
            cumsum_reader_handle_id,
            core,
            {input_tensor.buffer()->address(), start_row, rows_per_core, start_high_tile_index, start_low_tile_index});

        SetRuntimeArgs(
            program,
            cumsum_writer_handle_id,
            core,
            {output_tensor.buffer()->address(), start_row, rows_per_core, start_high_tile_index, start_low_tile_index});

        SetRuntimeArgs(
            program,
            cumsum_compute_handle_id,
            core,
            {
                rows_per_core,
                num_tiles_per_row,
            });

        start_row += rows_per_core;
    }
    return {std::move(program), {cumsum_reader_handle_id, cumsum_writer_handle_id, num_cores, num_cores_y}};
}

void CumSumDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dtype = operation_attributes.dtype;

    const auto& input_dtype = input_tensor.dtype();

    auto& cumsum_reader_kernel_id = cached_program.shared_variables.cumsum_reader_kernel_id;
    auto& cumsum_writer_kernel_id = cached_program.shared_variables.cumsum_writer_kernel_id;

    auto num_cores = cached_program.shared_variables.num_cores;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    auto& program = cached_program.program;

    auto input_buffer_addr = input_tensor.buffer()->address();
    auto output_buffer_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, cumsum_reader_kernel_id, core);
            runtime_args[0] = input_buffer_addr;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, cumsum_writer_kernel_id, core);
            runtime_args[0] = output_buffer_addr;
        }
    }
}

}  // namespace ttnn::operations::reduction
