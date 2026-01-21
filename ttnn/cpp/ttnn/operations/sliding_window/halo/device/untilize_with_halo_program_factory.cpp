// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/sliding_window/halo/device/untilize_with_halo_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/types.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

// TODO: Look into increasing this to tradeoff some L1 for performance (#19980)
constexpr int UNTILIZE_BLOCK_SIZE = 32;

// In order to make circular buffer indices sequential, we use variable to keep track of the next available index.
// Circular buffer indices should be assigned right before their creation.
struct CBIndices {
    // Invalid value for cb id is 32, number greater than the maximum number of index circular buffer can have.
    // Not assigning get_next_cb_index() value before creating cb will throw exception in circular_buffer_config.cpp
    // which can be used as a reminder.
    uint32_t src_cb_id = 32;
    uint32_t pad_cb_id0 = 32;
    uint32_t pad_cb_id1 = 32;
    uint32_t out_cb_id = 32;

    // Additional CBs for sharded data kernel configs
    uint32_t padding_config0 = 32;
    uint32_t padding_config1 = 32;
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

UntilizeWithHaloProgramFactory::cached_program_t UntilizeWithHaloProgramFactory::create(
    const HaloParams& operation_attributes, const Tensor& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args;
    const auto& pad_val = operation_attributes.pad_val;
    const int block_size = UNTILIZE_BLOCK_SIZE;
    const uint32_t ncores_nhw = operation_attributes.config.num_cores_nhw;
    const uint32_t max_out_nsticks_per_core = operation_attributes.max_out_nsticks_per_core;
    const bool config_tensors_in_dram = operation_attributes.config_tensors_in_dram;
    const bool remote_read = operation_attributes.remote_read;
    const bool transpose_mcast = operation_attributes.transpose_mcast;

    auto *device = input_tensor.device();

    bool is_in_tiled = input_tensor.layout() == Layout::TILE;
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    using namespace ttnn::operations;
    auto pad_metadata = sliding_window::generate_pad_metadata(operation_attributes.config);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(operation_attributes.config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(operation_attributes.config);
    const uint32_t input_shard_height = input_tensor.memory_config().shard_spec()->shape[0];
    auto tensor_metadata =
        sliding_window::generate_tensor_metadata(pad_metadata, operation_attributes.config, input_shard_height);

    uint32_t num_cores_x = input_tensor.memory_config().shard_spec()->grid.bounding_box().grid_size().x;

    auto kernel_config = sliding_window::generate_halo_kernel_config_tensors(
        tensor_metadata,
        shard_boundaries,
        is_block_sharded,
        transpose_mcast,
        remote_read,
        device,
        num_cores_x,
        is_in_tiled,
        UNTILIZE_BLOCK_SIZE);

    const auto& pad_config0 = kernel_config.pad_config0;
    const auto& pad_config1 = kernel_config.pad_config1;
    const auto& gather_config0 = kernel_config.gather_config0;
    const auto& gather_config1 = kernel_config.gather_config1;

    const auto pad_config_tensor0 = sliding_window::construct_on_host_config_tensor(
        pad_config0, operation_attributes.parallel_config, operation_attributes.config_tensors_in_dram);
    const auto pad_config_tensor1 = sliding_window::construct_on_host_config_tensor(
        pad_config1, operation_attributes.parallel_config, operation_attributes.config_tensors_in_dram);
    const auto gather_config_tensor0 = sliding_window::construct_on_host_config_tensor(
        gather_config0, operation_attributes.parallel_config, operation_attributes.config_tensors_in_dram);
    const auto gather_config_tensor1 = sliding_window::construct_on_host_config_tensor(
        gather_config1, operation_attributes.parallel_config, operation_attributes.config_tensors_in_dram);

    auto pad_config_device_tensor0 = sliding_window::move_config_tensor_to_device(
        pad_config_tensor0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);
    auto pad_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
        pad_config_tensor1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);
    auto gather_config_device_tensor0 = sliding_window::move_config_tensor_to_device(
        gather_config_tensor0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);
    auto gather_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
        gather_config_tensor1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);

    const auto number_of_blocks_per_core = sliding_window::remap_nhw_scalar_argument_across_full_grid(
        kernel_config.number_of_blocks_per_core, operation_attributes.parallel_config);

    Program program = CreateProgram();

    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const bool skip_untilize = input_tensor.layout() == Layout::ROW_MAJOR;

    const auto& input_shape = input_tensor.padded_shape();

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t out_nbytes = datum_size(out_df);

    const CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;

    const ShardOrientation shard_orientation = output_tensor.shard_spec().value().orientation;
    const auto input_shard_shape = input_tensor.shard_spec().value().shape;
    const auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1], "Expected input and output shard widths to match");

    const uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);

    uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;

    uint32_t in_page_size = tt::tile_size(in_df);
    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = input_shard_shape[1] * in_nbytes;
        input_npages = remapped_input_shard_shape_for_output_grid;
    }

    // Calculate aligned stick size - used for both input and output since channels don't change
    const uint32_t stick_nbytes = output_shard_shape[1] * out_nbytes;
    uint32_t aligned_stick_nbytes = stick_nbytes;
    if (stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_stick_nbytes = tt::round_up(stick_nbytes, input_tensor.buffer()->alignment());
    }
    const uint32_t out_tile_size = tt::tile_size(out_df);

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

    uint32_t out_cb_pagesize = aligned_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    cb_indices.out_cb_id = cb_indices.get_next_cb_id();
    auto out_cb = create_circular_buffer(
        program, all_cores, cb_indices.out_cb_id, out_df, out_cb_npages, out_cb_pagesize, dst_buffer);

    // Used for storing padding immediate values (only used if not zero padding)
    uint32_t pad_cb_pagesize = aligned_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    cb_indices.pad_cb_id0 = cb_indices.get_next_cb_id();
    create_circular_buffer(program, all_cores, cb_indices.pad_cb_id0, out_df, pad_cb_npages, pad_cb_pagesize);
    cb_indices.pad_cb_id1 = cb_indices.get_next_cb_id();
    create_circular_buffer(program, all_cores, cb_indices.pad_cb_id1, out_df, pad_cb_npages, pad_cb_pagesize);

    tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // NOTE: UInt16 is not supported for CB types

    uint32_t input_to_writer_cb_id0 = cb_indices.src_cb_id;
    uint32_t input_to_writer_cb_id1 = cb_indices.src_cb_id;
    const bool is_rm_orientation = shard_orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, is_rm_orientation);

    if (!skip_untilize) {
        cb_indices.untilize_out_cb_id0 = cb_indices.get_next_cb_id();
        cb_indices.untilize_out_cb_id1 = cb_indices.get_next_cb_id();
        input_to_writer_cb_id0 = cb_indices.untilize_out_cb_id0;
        input_to_writer_cb_id1 = cb_indices.untilize_out_cb_id1;
        const uint32_t output_ntiles = (clamped_block_size_height / TILE_HEIGHT) * ntiles_per_block;
        const uint32_t untilize_out_cb_num_pages = ENABLE_UNTILIZE_DOUBLE_BUFFERING ? 2 * output_ntiles : output_ntiles;
        create_circular_buffer(
            program, all_cores, cb_indices.untilize_out_cb_id0, out_df, untilize_out_cb_num_pages, out_tile_size);
        create_circular_buffer(
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

        for (int core_id = 0; core_id < cores.size(); core_id++) {
            SetRuntimeArgs(program, untilize_kernel_id, cores[core_id], {number_of_blocks_per_core[core_id]});
        }
    }

    TT_ASSERT(pad_config_device_tensor0.dtype() == DataType::UINT16);
    TT_ASSERT(pad_config_device_tensor1.dtype() == DataType::UINT16);
    TT_ASSERT(gather_config_device_tensor0.dtype() == DataType::UINT16);
    TT_ASSERT(gather_config_device_tensor1.dtype() == DataType::UINT16);

    const auto& padding_config_storage0 = pad_config_device_tensor0.device_storage();
    auto* padding_config_buffer0 = padding_config_storage0.get_buffer();
    cb_indices.padding_config0 = cb_indices.get_next_cb_id();
    auto padding_config_cb0 = create_circular_buffer(
        program,
        all_cores,
        cb_indices.padding_config0,
        kernel_config_df,
        1,
        padding_config_buffer0->page_size(),
        config_tensors_in_dram ? nullptr : padding_config_buffer0);

    const auto& padding_config_storage1 = pad_config_device_tensor1.device_storage();
    auto* padding_config_buffer1 = padding_config_storage1.get_buffer();
    cb_indices.padding_config1 = cb_indices.get_next_cb_id();
    auto padding_config_cb1 = create_circular_buffer(
        program,
        all_cores,
        cb_indices.padding_config1,
        kernel_config_df,
        1,
        padding_config_buffer1->page_size(),
        config_tensors_in_dram ? nullptr : padding_config_buffer1);

    const auto& gather_config_storage0 = gather_config_device_tensor0.device_storage();
    auto* gather_config_buffer0 = gather_config_storage0.get_buffer();
    cb_indices.gather_config0 = cb_indices.get_next_cb_id();
    auto gather_config_cb0 = create_circular_buffer(
        program,
        all_cores,
        cb_indices.gather_config0,
        kernel_config_df,
        1,
        gather_config_buffer0->page_size(),
        config_tensors_in_dram ? nullptr : gather_config_buffer0);

    const auto& gather_config_storage1 = gather_config_device_tensor1.device_storage();
    auto* gather_config_buffer1 = gather_config_storage1.get_buffer();
    cb_indices.gather_config1 = cb_indices.get_next_cb_id();
    auto gather_config_cb1 = create_circular_buffer(
        program,
        all_cores,
        cb_indices.gather_config1,
        kernel_config_df,
        1,
        gather_config_buffer1->page_size(),
        config_tensors_in_dram ? nullptr : gather_config_buffer1);

    const bool is_height_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    const uint32_t block_stride = 2;  // Skip every 2nd block because of split reader
    const std::string reader_kernel_name =
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp";
    std::vector<uint32_t> common_reader_ct_args = {
        0,  // padding config cb
        0,  // gather config cb
        cb_indices.src_cb_id,
        input_to_writer_cb_id0,
        cb_indices.out_cb_id,
        0,  // padding value cb
        pad_val,
        input_npages,
        aligned_stick_nbytes,
        is_block_sharded,
        remote_read,
        (uint32_t)(transpose_mcast ? 1 : 0),
        is_width_sharded,
        skip_untilize,
        clamped_block_size_height,  // Block size in sticks
        ntiles_per_block,
        0,            // Block start offset
        block_stride  // Block stride
    };
    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> core_0_reader_ct_args = common_reader_ct_args;
    std::vector<uint32_t> core_1_reader_ct_args = common_reader_ct_args;

    if (config_tensors_in_dram) {
        reader_defines["CONFIG_TENSOR_IN_DRAM"] = "1";
        core_0_reader_ct_args.push_back(padding_config_storage0.get_buffer()->address());
        core_0_reader_ct_args.push_back(padding_config_storage0.get_buffer()->page_size());

        core_0_reader_ct_args.push_back(gather_config_storage0.get_buffer()->address());
        core_0_reader_ct_args.push_back(gather_config_storage0.get_buffer()->page_size());

        core_1_reader_ct_args.push_back(padding_config_storage1.get_buffer()->address());
        core_1_reader_ct_args.push_back(padding_config_storage1.get_buffer()->page_size());

        core_1_reader_ct_args.push_back(gather_config_storage1.get_buffer()->address());
        core_1_reader_ct_args.push_back(gather_config_storage1.get_buffer()->page_size());

        tt::tt_metal::TensorAccessorArgs(padding_config_storage0.get_buffer()).append_to(core_0_reader_ct_args);
        tt::tt_metal::TensorAccessorArgs(gather_config_storage0.get_buffer()).append_to(core_0_reader_ct_args);

        tt::tt_metal::TensorAccessorArgs(padding_config_storage1.get_buffer()).append_to(core_1_reader_ct_args);
        tt::tt_metal::TensorAccessorArgs(gather_config_storage1.get_buffer()).append_to(core_1_reader_ct_args);
    }
    const uint32_t EMPTY_PADDING_CONFIG_BUFFER_SIZE = 4;
    const bool enable_padding = config_tensors_in_dram ||
                                padding_config_buffer0->page_size() != EMPTY_PADDING_CONFIG_BUFFER_SIZE ||
                                padding_config_buffer1->page_size() != EMPTY_PADDING_CONFIG_BUFFER_SIZE;

    core_0_reader_ct_args[0] = enable_padding ? cb_indices.padding_config0 : 0;
    core_0_reader_ct_args[1] = cb_indices.gather_config0;
    core_0_reader_ct_args[5] = cb_indices.pad_cb_id0;

    core_1_reader_ct_args[0] = enable_padding ? cb_indices.padding_config1 : 0;
    core_1_reader_ct_args[1] = cb_indices.gather_config1;
    core_1_reader_ct_args[3] = input_to_writer_cb_id1;
    core_1_reader_ct_args[5] = cb_indices.pad_cb_id1;
    core_1_reader_ct_args[16] = 1;  // Block start offset

    auto reader_0_kernel_id = CreateKernel(
        program,
        reader_kernel_name,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = core_0_reader_ct_args,
            .defines = reader_defines});

    auto reader_1_kernel_id = CreateKernel(
        program,
        reader_kernel_name,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = core_1_reader_ct_args,
            .defines = reader_defines});

    if (config_tensors_in_dram) {
        uint32_t core_index = 0;
        for (auto core : cores) {
            if (is_height_sharded) {
                SetRuntimeArgs(program, reader_0_kernel_id, core, {core_index});
                SetRuntimeArgs(program, reader_1_kernel_id, core, {core_index});
            } else if (is_width_sharded) {
                SetRuntimeArgs(program, reader_0_kernel_id, core, {0});
                SetRuntimeArgs(program, reader_1_kernel_id, core, {0});
            } else if (is_block_sharded) {
                auto nhw_index = is_rm_orientation ? core.y : core.x;
                SetRuntimeArgs(program, reader_0_kernel_id, core, {nhw_index});
                SetRuntimeArgs(program, reader_1_kernel_id, core, {nhw_index});
            }
            core_index++;
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .src_cb = src_cb,
            .out_cb = out_cb,
            .padding_config_cb0 = padding_config_cb0,
            .padding_config_cb1 = padding_config_cb1,
            .gather_config_cb0 = gather_config_cb0,
            .gather_config_cb1 = gather_config_cb1,
            .padding_config_storage0 = padding_config_storage0,
            .padding_config_storage1 = padding_config_storage1,
            .gather_config_storage0 = gather_config_storage0,
            .gather_config_storage1 = gather_config_storage1,
        }};
}

void UntilizeWithHaloProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const HaloParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;

    auto* src_buffer = tensor_args.buffer();
    auto *dst_buffer = output_tensor.buffer();
    auto& src_cb = cached_program.shared_variables.src_cb;
    auto& out_cb = cached_program.shared_variables.out_cb;

    UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
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

}  // namespace ttnn::prim
