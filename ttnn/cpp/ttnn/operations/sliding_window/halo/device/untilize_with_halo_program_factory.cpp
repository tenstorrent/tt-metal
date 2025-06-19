// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/common/constants.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

// In order to make circular buffer indicies sequential, we use variable to keep track of the next available index.
// Circular buffer indices should be assigned right before their creation.
struct CBIndices {
    // Invalid value for cb id is 32, number greater than the maximum number of index circular buffer can have.
    // Not assigning get_next_cb_index() value before creating cb will throw exception in circular_buffer_config.cpp
    // which can be used as a reminder.
    uint32_t src_cb_id = 32;
    uint32_t pad_cb_id = 32;
    uint32_t out_cb_id = 32;

    // Additional CBs for sharded data kernel configs
    uint32_t padding_config = 32;
    uint32_t gather_config0 = 32;
    uint32_t gather_config1 = 32;
    uint32_t untilize_out_cb_id0 = 32;
    uint32_t untilize_out_cb_id1 = 32;
    uint32_t get_next_cb_id() { return next_cb_id++; }

private:
    uint32_t next_cb_id = tt::CBIndex::c_0;
};

static inline CBHandle create_circular_buffer(
    Program& program,
    const CoreRangeSet& cores,
    uint32_t cb_id,
    tt::DataFormat df,
    uint32_t npages,
    uint32_t pagesize,
    Buffer* buffer = nullptr) {
    return std::get<1>(tt::tt_metal::create_cb(cb_id, program, cores, pagesize, npages, df, buffer));
}

constexpr bool ENABLE_UNTILIZE_DOUBLE_BUFFERING = true;

operation::ProgramWithCallbacks untilize_with_halo_multi_core(
    Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config,
    const Tensor& gather_config0,
    const Tensor& gather_config1,
    const std::vector<uint16_t>& number_of_blocks_per_core,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const int block_size,
    const bool capture_buffers) {
    IDevice* device = input_tensor.device();
    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const bool skip_untilize = input_tensor.layout() == Layout::ROW_MAJOR;

    const auto input_shape = input_tensor.padded_shape();
    const auto output_shape = output_tensor.padded_shape();

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t out_nbytes = datum_size(out_df);

    const CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;

    const ShardOrientation shard_orientation = output_tensor.shard_spec().value().orientation;
    const auto input_shard_shape = output_tensor.shard_spec().value().shape;
    const auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1], "Expected input and output shard shapes to match");

    const uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);

    uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;

    uint32_t in_page_size = tt::tt_metal::detail::TileSize(in_df);
    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = input_shard_shape[1] * in_nbytes;
        input_npages = remapped_input_shard_shape_for_output_grid;
    }

    const uint32_t out_stick_nbytes = output_shard_shape[1] * out_nbytes;
    const uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);

    CBIndices cb_indices = CBIndices();

    // The input CB can either be tiled or row-major
    cb_indices.src_cb_id = cb_indices.get_next_cb_id();
    auto src_cb =
        create_circular_buffer(program, all_cores, cb_indices.src_cb_id, in_df, input_npages, in_page_size, src_buffer);

    // We need to clamp in the case that the block size is larger than the nhw input size
    TT_FATAL(block_size % TILE_HEIGHT == 0, "Block size must be a multiple of tile height (was {})", block_size);
    const uint32_t clamped_block_size_height =
        std::min(static_cast<uint32_t>(block_size), input_nblocks_per_core * TILE_HEIGHT);
    TT_FATAL(
        clamped_block_size_height % TILE_HEIGHT == 0,
        "Block size must be a multiple of tile height (was {})",
        clamped_block_size_height);

    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    cb_indices.out_cb_id = cb_indices.get_next_cb_id();
    auto out_cb = create_circular_buffer(
        program, all_cores, cb_indices.out_cb_id, out_df, out_cb_npages, out_cb_pagesize, dst_buffer);

    // Used for storing padding immediate values (TODO: use zeroed memory region instead)
    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    cb_indices.pad_cb_id = cb_indices.get_next_cb_id();
    auto pad_cb =
        create_circular_buffer(program, all_cores, cb_indices.pad_cb_id, out_df, pad_cb_npages, pad_cb_pagesize);

    tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // NOTE: UInt16 is not supported for CB types
    uint32_t pagesize = 0;

    uint32_t input_to_writer_cb_id0 = cb_indices.src_cb_id;
    uint32_t input_to_writer_cb_id1 = cb_indices.src_cb_id;
    if (!skip_untilize) {
        cb_indices.untilize_out_cb_id0 = cb_indices.get_next_cb_id();
        cb_indices.untilize_out_cb_id1 = cb_indices.get_next_cb_id();
        input_to_writer_cb_id0 = cb_indices.untilize_out_cb_id0;
        input_to_writer_cb_id1 = cb_indices.untilize_out_cb_id1;
        const uint32_t output_ntiles = (clamped_block_size_height / TILE_HEIGHT) * ntiles_per_block;
        const uint32_t untilize_out_cb_num_pages = ENABLE_UNTILIZE_DOUBLE_BUFFERING ? 2 * output_ntiles : output_ntiles;
        auto untilize_out_cb0 = create_circular_buffer(
            program, all_cores, cb_indices.untilize_out_cb_id0, out_df, untilize_out_cb_num_pages, out_tile_size);
        auto untilize_out_cb1 = create_circular_buffer(
            program, all_cores, cb_indices.untilize_out_cb_id1, out_df, untilize_out_cb_num_pages, out_tile_size);

        const std::string compute_kernel_name =
            "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp";
        const std::vector<uint32_t> compute_ct_args = {
            cb_indices.src_cb_id,
            input_to_writer_cb_id0,
            input_to_writer_cb_id1,
            ntiles_per_block,                        // number of tiles in the width dimension (channels)
            clamped_block_size_height / TILE_HEIGHT  // number of tiles in height dimension that make up a block

        };
        KernelHandle untilize_kernel_id =
            CreateKernel(program, compute_kernel_name, all_cores, ComputeConfig{.compile_args = compute_ct_args});

        const bool is_rm_orientation = shard_orientation == ShardOrientation::ROW_MAJOR;
        const auto cores = corerange_to_cores(all_cores, std::nullopt, is_rm_orientation);
        for (int core_id = 0; core_id < cores.size(); core_id++) {
            SetRuntimeArgs(program, untilize_kernel_id, cores[core_id], {number_of_blocks_per_core[core_id]});
        }
    }

    TT_ASSERT(padding_config.dtype() == DataType::UINT16);
    TT_ASSERT(gather_config0.dtype() == DataType::UINT16);
    TT_ASSERT(gather_config1.dtype() == DataType::UINT16);

    const uint32_t num_cores = all_cores.num_cores();

    auto padding_config_storage = padding_config.device_storage();
    auto padding_config_buffer = padding_config_storage.get_buffer();
    cb_indices.padding_config = cb_indices.get_next_cb_id();
    auto padding_config_cb = create_circular_buffer(
        program,
        all_cores,
        cb_indices.padding_config,
        kernel_config_df,
        1,
        padding_config_buffer->size() / num_cores,
        padding_config_buffer);

    auto gather_config_storage0 = gather_config0.device_storage();
    auto gather_config_buffer0 = gather_config_storage0.get_buffer();
    cb_indices.gather_config0 = cb_indices.get_next_cb_id();
    auto gather_config_cb0 = create_circular_buffer(
        program,
        all_cores,
        cb_indices.gather_config0,
        kernel_config_df,
        1,
        gather_config_buffer0->size() / num_cores,
        gather_config_buffer0);

    auto gather_config_storage1 = gather_config1.device_storage();
    auto gather_config_buffer1 = gather_config_storage1.get_buffer();
    cb_indices.gather_config1 = cb_indices.get_next_cb_id();
    auto gather_config_cb1 = create_circular_buffer(
        program,
        all_cores,
        cb_indices.gather_config1,
        kernel_config_df,
        1,
        gather_config_buffer1->size() / num_cores,
        gather_config_buffer1);

    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    auto aligned_input_nstick_nbytes = out_stick_nbytes;
    if (out_stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_input_nstick_nbytes = tt::round_up(out_stick_nbytes, input_tensor.buffer()->alignment());
    }

    const uint32_t block_stride = 2;  // Skip every 2nd block because of split reader
    const std::string reader_kernel_name =
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp";
    std::vector<uint32_t> reader_ct_args = {
        0,  // padding_config_cb_id
        0,  // gather_config_cb_id
        cb_indices.src_cb_id,
        input_to_writer_cb_id0,
        cb_indices.out_cb_id,
        cb_indices.pad_cb_id,
        pad_val,
        input_npages,
        out_stick_nbytes,
        is_block_sharded,
        remote_read,
        (uint32_t)(transpose_mcast ? 1 : 0),
        is_width_sharded,
        aligned_input_nstick_nbytes,
        skip_untilize,
        clamped_block_size_height,  // Block size in sticks
        ntiles_per_block,
        0,            // Block start offset
        block_stride  // Block stride
    };

    reader_ct_args[0] = 0;
    reader_ct_args[1] = cb_indices.gather_config0;
    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        reader_kernel_name,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args});

    reader_ct_args[0] = cb_indices.padding_config;
    reader_ct_args[1] = cb_indices.gather_config1;
    reader_ct_args[3] = input_to_writer_cb_id1;
    reader_ct_args[17] = 1;  // Block start offset
    KernelHandle reader_kernel_id1 = CreateKernel(
        program,
        reader_kernel_name,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Capture padding_config_buffer, local_config_buffer, remote_config_buffer to cache this with the program
    if (!capture_buffers) {
        padding_config_storage = {};
        gather_config_storage0 = {};
        gather_config_storage1 = {};
    }
    auto override_runtime_arguments_callback = [src_cb,
                                                out_cb,
                                                padding_config_cb,
                                                gather_config_cb0,
                                                gather_config_cb1,
                                                padding_config_storage,
                                                gather_config_storage0,
                                                gather_config_storage1](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

struct InplaceCBIndices {
    uint32_t src_cb_id = 32;
    uint32_t pad_cb_id = 32;
    uint32_t out_cb_id = 32;
    uint32_t padding_config_cb_id = 32;
    uint32_t local_config_cb_id = 32;
    uint32_t remote_config_cb_id = 32;
    uint32_t untilize_out_cb_id = 32;
    uint32_t get_next_cb_id() { return next_cb_id++; }

private:
    uint32_t next_cb_id = tt::CBIndex::c_0;
};

operation::ProgramWithCallbacks inplace_untilize_with_halo_multi_core(
    Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const bool padding_exists,
    const uint32_t ncores_nhw,
    const uint32_t ncores_c,
    const uint32_t max_out_nsticks_per_core,
    const uint32_t max_ref_size,
    const Tensor& padding_config,
    const Tensor& local_config,
    const Tensor& remote_config,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const bool capture_buffers) {
    IDevice* device = input_tensor.device();
    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t out_nbytes = datum_size(out_df);

    CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;
    auto input_shard_shape = output_tensor.shard_spec().value().shape;
    auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1]);
    uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);
    uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;

    uint32_t out_stick_nbytes = output_shard_shape[1] * out_nbytes;

    uint32_t in_page_size = tt::tt_metal::detail::TileSize(in_df);
    uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);

    const bool skip_untilize = input_tensor.layout() == Layout::ROW_MAJOR;
    bool wide_tensor = ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH;
    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = input_shard_shape[1] * in_nbytes;
        input_npages = remapped_input_shard_shape_for_output_grid;
    }

    // Construct CBs
    InplaceCBIndices cb_indices = InplaceCBIndices();
    cb_indices.src_cb_id = cb_indices.get_next_cb_id();
    auto src_cb =
        create_circular_buffer(program, all_cores, cb_indices.src_cb_id, in_df, input_npages, in_page_size, src_buffer);

    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    cb_indices.out_cb_id = cb_indices.get_next_cb_id();
    auto out_cb = create_circular_buffer(
        program, all_cores, cb_indices.out_cb_id, out_df, out_cb_npages, out_cb_pagesize, dst_buffer);

    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    cb_indices.pad_cb_id = cb_indices.get_next_cb_id();
    auto pad_cb =
        create_circular_buffer(program, all_cores, cb_indices.pad_cb_id, out_df, pad_cb_npages, pad_cb_pagesize);

    tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // NOTE: UInt16 is not supported for CB types
    uint32_t config_nbytes =
        tt::datum_size(kernel_config_df) * 2;  // each config is a pair "start, size", so double the size
    uint32_t pagesize = 0;

    uint32_t temp_cb_id = 0;
    uint32_t input_to_writer_cb_id = cb_indices.src_cb_id;
    if (!skip_untilize) {
        cb_indices.untilize_out_cb_id = cb_indices.get_next_cb_id();
        input_to_writer_cb_id = cb_indices.untilize_out_cb_id;
        // output of untilize from compute kernel goes into this CB
        uint32_t output_ntiles = ntiles_per_block * input_nblocks_per_core;
        auto untilize_out_cb_config =
            CircularBufferConfig(output_ntiles * out_tile_size, {{cb_indices.untilize_out_cb_id, out_df}})
                .set_page_size(cb_indices.untilize_out_cb_id, out_tile_size)
                .set_globally_allocated_address(*dst_buffer);  // untilize into the dst buffer for in place untilize
        auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);
        log_debug(
            tt::LogOp,
            "CB {} :: npages = {}, pagesize = {}",
            cb_indices.untilize_out_cb_id,
            output_ntiles,
            out_tile_size);

        // compute kernel
        std::string compute_kernel;
        std::vector<uint32_t> compute_ct_args;
        if (wide_tensor) {
            // wide tensors use a different compute kernel which requires use of a temp buffer for the intermediate
            // untilized results
            temp_cb_id = cb_indices.get_next_cb_id();
            auto temp_cb =
                create_circular_buffer(program, all_cores, temp_cb_id, out_df, ntiles_per_block, out_tile_size);
            log_debug(
                tt::LogOp,
                "Falling back to slow untilize since ntiles_per_block {} > MAX_PACK_UNTILIZE_WIDTH {}",
                ntiles_per_block,
                MAX_PACK_UNTILIZE_WIDTH);
            compute_ct_args = {input_nblocks_per_core, ntiles_per_block, cb_indices.src_cb_id, temp_cb_id};
            compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
        } else {
            compute_ct_args = {input_nblocks_per_core, ntiles_per_block, cb_indices.src_cb_id, input_to_writer_cb_id};
            compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp";
        }
        KernelHandle untilize_kernel_id =
            CreateKernel(program, compute_kernel, all_cores, ComputeConfig{.compile_args = compute_ct_args});
    }

    TT_ASSERT(padding_config.dtype() == DataType::UINT16);
    TT_ASSERT(local_config.dtype() == DataType::UINT16);
    TT_ASSERT(remote_config.dtype() == DataType::UINT16);

    const uint32_t num_cores = all_cores.num_cores();

    auto padding_config_storage = padding_config.device_storage();
    auto padding_config_buffer = padding_config_storage.get_buffer();
    cb_indices.padding_config_cb_id = cb_indices.get_next_cb_id();
    auto padding_config_cb = create_circular_buffer(
        program,
        all_cores,
        cb_indices.padding_config_cb_id,
        kernel_config_df,
        1,
        padding_config_buffer->size() / num_cores,
        padding_config_buffer);

    auto local_config_storage = local_config.device_storage();
    auto local_config_buffer = local_config_storage.get_buffer();
    cb_indices.local_config_cb_id = cb_indices.get_next_cb_id();
    auto local_config_cb = create_circular_buffer(
        program,
        all_cores,
        cb_indices.local_config_cb_id,
        kernel_config_df,
        1,
        local_config_buffer->size() / num_cores,
        local_config_buffer);

    auto remote_config_storage = remote_config.device_storage();
    auto remote_config_buffer = remote_config_storage.get_buffer();
    cb_indices.remote_config_cb_id = cb_indices.get_next_cb_id();
    auto remote_config_cb = create_circular_buffer(
        program,
        all_cores,
        cb_indices.remote_config_cb_id,
        kernel_config_df,
        1,
        remote_config_buffer->size() / num_cores,
        remote_config_buffer);

    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    int32_t in_out_buffer_start_delta = max_out_nsticks_per_core - input_npages;
    if (!skip_untilize) {
        in_out_buffer_start_delta = 0;
    }
    const auto delta = output_tensor.buffer()->aligned_size_per_bank() - input_tensor.buffer()->aligned_size_per_bank();
    TT_ASSERT(
        src_buffer->sharded_page_address(0, 0) == dst_buffer->sharded_page_address(0, 0) + delta,
        "In-place halo requires input and output buffers to be sharded at the same address");
    TT_ASSERT(!remote_read, "remote_read is not supported for in place operation");

    // create the remote temp CB
    uint32_t remote_temp_cb_id = 0;
    if (max_ref_size > 0) {
        remote_temp_cb_id = cb_indices.get_next_cb_id();
        auto remote_temp_cb_config =
            CircularBufferConfig(
                max_ref_size * output_shard_shape[1] * out_nbytes, {{remote_temp_cb_id, kernel_config_df}})
                .set_page_size(remote_temp_cb_id, output_shard_shape[1] * out_nbytes);
        CBHandle remote_temp_cb = CreateCircularBuffer(program, all_cores, remote_temp_cb_config);
    }

    // noc conversion function
    auto core_id_to_noc_coords = [is_block_sharded, transpose_mcast, device](uint32_t core_id) -> CoreCoord {
        auto num_cores_x = device->compute_with_storage_grid_size().x;
        auto core_coord = CoreCoord(core_id % num_cores_x, core_id / num_cores_x);
        return device->worker_core_from_logical_core(core_coord);
    };

    // find the noc coordinate of core 0,0
    CoreCoord noc_TL = core_id_to_noc_coords(0);

    // compute the number of noop cores
    const bool is_rm_orientation = input_tensor.shard_spec()->orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, is_rm_orientation);
    int32_t num_cores_x = device->compute_with_storage_grid_size().x;
    int32_t num_cores_y = device->compute_with_storage_grid_size().y;
    int32_t num_active_cores = cores.size();
    int32_t num_cores_rectangular = is_block_sharded ? num_active_cores : tt::round_up(num_active_cores, num_cores_x);
    int32_t num_noop_cores = is_block_sharded ? 0 : num_cores_rectangular - num_active_cores;
    TT_FATAL(
        !is_block_sharded || all_cores.ranges().size() == 1,
        "for block sharding the implementation depends on the assumption that there is only 1 core range");
    CoreCoord last_active_coord = is_block_sharded
                                      ? device->worker_core_from_logical_core(all_cores.ranges()[0].end_coord)
                                      : core_id_to_noc_coords(num_active_cores - 1);
    uint32_t last_active_x = last_active_coord.x;

    // create the rectangular core range
    uint32_t rectangular_x = is_block_sharded ? all_cores.ranges()[0].end_coord.x + 1 : num_cores_x;
    uint32_t rectangular_y =
        is_block_sharded ? all_cores.ranges()[0].end_coord.y + 1
                         : (num_noop_cores ? num_active_cores / num_cores_x + 1 : num_active_cores / num_cores_x);
    std::set<CoreRange> rectangular_cores_set;
    if (is_block_sharded) {
        rectangular_cores_set.insert(all_cores.ranges()[0]);
    } else {
        rectangular_cores_set.insert(CoreRange(CoreCoord(0, 0), CoreCoord(rectangular_x - 1, rectangular_y - 1)));
    }
    CoreRangeSet rectangular_cores(rectangular_cores_set);
    CoreCoord noc_BR = is_block_sharded ? last_active_coord : core_id_to_noc_coords(rectangular_x * rectangular_y - 1);

    // create semaphore
    uint32_t semaphore_id = tt::tt_metal::CreateSemaphore(program, rectangular_cores, 0);

    auto aligned_input_nstick_nbytes = out_stick_nbytes;
    log_debug(tt::LogOp, "out_stick_nbytes = {}", out_stick_nbytes);
    log_debug(tt::LogOp, "input_tensor.buffer()->alignment() = {}", input_tensor.buffer()->alignment());

    if (out_stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_input_nstick_nbytes = tt::round_up(out_stick_nbytes, input_tensor.buffer()->alignment());
    }

    // create the NC/BR sync CBs
    int32_t sync_cb_id1 = cb_indices.get_next_cb_id();
    auto sync_cb1 = create_circular_buffer(program, all_cores, sync_cb_id1, tt::DataFormat::UInt16, 1, 2);
    int32_t sync_cb_id2 = cb_indices.get_next_cb_id();
    auto sync_cb2 = create_circular_buffer(program, all_cores, sync_cb_id2, tt::DataFormat::UInt16, 1, 2);

    // reader kernel
    std::vector<uint32_t> reader_ct_args = {
        true,  // main thread
        padding_exists,
        cb_indices.padding_config_cb_id,
        cb_indices.local_config_cb_id,
        cb_indices.remote_config_cb_id,
        remote_temp_cb_id,
        cb_indices.src_cb_id,
        input_to_writer_cb_id,
        cb_indices.out_cb_id,
        cb_indices.pad_cb_id,
        pad_val,
        input_npages,
        out_stick_nbytes,
        is_block_sharded,
        (uint32_t)(transpose_mcast ? 1 : 0),
        is_width_sharded,
        aligned_input_nstick_nbytes,
        remote_read,
        num_active_cores,
        noc_TL.x,
        noc_TL.y,
        noc_BR.x,
        noc_BR.y,
        rectangular_x,
        rectangular_y,
        last_active_x,
        semaphore_id,
        in_out_buffer_start_delta,
        temp_cb_id,
        ntiles_per_block,
        input_nblocks_per_core,
        sync_cb_id1,
        sync_cb_id2};

    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather_in_place.cpp",
        rectangular_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args});

    reader_ct_args[0] = false;  // secondary thread
    KernelHandle reader_kernel_id1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather_in_place.cpp",
        rectangular_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    for (uint32_t core_i = 0; core_i < num_cores_rectangular; core_i++) {
        uint32_t core_x_i = core_i % rectangular_x;
        uint32_t core_y_i = core_i / rectangular_x;
        CoreRange core(CoreCoord(core_x_i, core_y_i), CoreCoord(core_x_i, core_y_i));
        bool noop_core = core_i >= num_active_cores;
        bool cast_core = core_i == 0;  // first core controls the multicasting

        std::vector<uint32_t> reader_rt_args0 = {(uint32_t)noop_core, (uint32_t)cast_core};
        std::vector<uint32_t> reader_rt_args1 = {(uint32_t)noop_core, (uint32_t)false};
        SetRuntimeArgs(program, reader_kernel_id0, core, reader_rt_args0);
        SetRuntimeArgs(program, reader_kernel_id1, core, reader_rt_args1);
    }

    if (!capture_buffers) {
        padding_config_storage = {};
        local_config_storage = {};
        remote_config_storage = {};
    }
    // Capture padding_config_storage, local_config_storage, remote_config_storage to cache this with the program
    auto override_runtime_arguments_callback = [src_cb,
                                                out_cb,
                                                padding_config_cb,
                                                local_config_cb,
                                                remote_config_cb,
                                                padding_config_storage,
                                                local_config_storage,
                                                remote_config_storage](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail
