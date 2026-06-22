// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/halo/device/untilize_with_halo_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/types.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

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

constexpr bool ENABLE_UNTILIZE_DOUBLE_BUFFERING = true;

namespace {

// Append a CBDescriptor for a single-format circular buffer.  When `buffer`
// is non-null the CB is globally-allocated against that buffer so address
// patching on cache hit works via the framework's dynamic CB update path.
void add_cb(
    ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t cb_id,
    tt::DataFormat df,
    uint32_t npages,
    uint32_t pagesize,
    Buffer* buffer = nullptr) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = npages * pagesize,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = df,
            .page_size = pagesize,
        }}},
        .buffer = buffer,
    });
}

// Build the per-coord halo ProgramDescriptor.  All workload-scoped
// intermediate buffers (the four halo config tensors) are passed in by
// pointer — their owning Tensors live on the WorkloadDescriptor.
ProgramDescriptor build_halo_program(
    const HaloParams& operation_attributes,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    Buffer* padding_config_buffer0,
    Buffer* padding_config_buffer1,
    Buffer* gather_config_buffer0,
    Buffer* gather_config_buffer1,
    const std::vector<uint16_t>& number_of_blocks_per_core) {
    const auto& pad_val = operation_attributes.pad_val;
    const int block_size = UNTILIZE_BLOCK_SIZE;
    const uint32_t ncores_nhw = operation_attributes.config.num_cores_nhw;
    const uint32_t max_out_nsticks_per_core = operation_attributes.max_out_nsticks_per_core;
    const bool config_tensors_in_dram = operation_attributes.config_tensors_in_dram;
    const bool remote_read = operation_attributes.remote_read;
    const bool transpose_mcast = operation_attributes.transpose_mcast;

    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

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

    // Calculate aligned stick size - used for both input and output since channels don't change
    const uint32_t stick_nbytes = output_shard_shape[1] * out_nbytes;
    uint32_t aligned_stick_nbytes = stick_nbytes;
    if (stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_stick_nbytes = tt::round_up(stick_nbytes, input_tensor.buffer()->alignment());
    }
    const uint32_t out_tile_size = tt::tile_size(out_df);

    // For ROW_MAJOR input the kernel reads with aligned_stick_nbytes stride
    // across the full input shard, so the CB must match the actual buffer layout:
    // page size = aligned stick width, npages = actual shard height.
    if (skip_untilize) {
        in_page_size = aligned_stick_nbytes;
        input_npages = input_shard_shape[0];
    }

    ProgramDescriptor desc;

    CBIndices cb_indices = CBIndices();

    // The input CB can either be tiled or row-major
    cb_indices.src_cb_id = cb_indices.get_next_cb_id();
    add_cb(desc, all_cores, cb_indices.src_cb_id, in_df, input_npages, in_page_size, src_buffer);

    // We need to clamp in the case that the block size is larger than the nhw input size
    TT_FATAL(block_size % TILE_HEIGHT == 0, "Block size must be a multiple of tile height (was {})", block_size);
    const uint32_t clamped_block_size_height =
        std::min(static_cast<uint32_t>(block_size), input_nblocks_per_core * TILE_HEIGHT);
    TT_FATAL(
        clamped_block_size_height % TILE_HEIGHT == 0,
        "Block size must be a multiple of tile height (was {})",
        clamped_block_size_height);

    const uint32_t out_cb_pagesize = aligned_stick_nbytes;
    const uint32_t out_cb_npages = max_out_nsticks_per_core;
    cb_indices.out_cb_id = cb_indices.get_next_cb_id();
    add_cb(desc, all_cores, cb_indices.out_cb_id, out_df, out_cb_npages, out_cb_pagesize, dst_buffer);

    // Used for storing padding immediate values (only used if not zero padding)
    const uint32_t pad_cb_pagesize = aligned_stick_nbytes;
    const uint32_t pad_cb_npages = 1;
    cb_indices.pad_cb_id0 = cb_indices.get_next_cb_id();
    add_cb(desc, all_cores, cb_indices.pad_cb_id0, out_df, pad_cb_npages, pad_cb_pagesize);
    cb_indices.pad_cb_id1 = cb_indices.get_next_cb_id();
    add_cb(desc, all_cores, cb_indices.pad_cb_id1, out_df, pad_cb_npages, pad_cb_pagesize);

    const tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // NOTE: UInt16 is not supported for CB types

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
        add_cb(desc, all_cores, cb_indices.untilize_out_cb_id0, out_df, untilize_out_cb_num_pages, out_tile_size);
        add_cb(desc, all_cores, cb_indices.untilize_out_cb_id1, out_df, untilize_out_cb_num_pages, out_tile_size);

        const std::vector<uint32_t> compute_ct_args = {
            cb_indices.src_cb_id,
            input_to_writer_cb_id0,
            input_to_writer_cb_id1,
            ntiles_per_block,                        // number of tiles in the width dimension (channels)
            clamped_block_size_height / TILE_HEIGHT  // number of tiles in height dimension that make up a block

        };
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(input_tensor.device()->arch(), operation_attributes.compute_kernel_config);

        KernelDescriptor compute_desc;
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/halo/device/kernels/compute/pack_untilize.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = compute_ct_args;
        compute_desc.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };

        for (size_t core_id = 0; core_id < cores.size(); core_id++) {
            compute_desc.runtime_args.emplace_back(
                cores[core_id], std::vector<uint32_t>{number_of_blocks_per_core[core_id]});
        }

        desc.kernels.push_back(std::move(compute_desc));
    }

    TT_ASSERT(padding_config_buffer0 != nullptr);
    TT_ASSERT(padding_config_buffer1 != nullptr);
    TT_ASSERT(gather_config_buffer0 != nullptr);
    TT_ASSERT(gather_config_buffer1 != nullptr);

    cb_indices.padding_config0 = cb_indices.get_next_cb_id();
    add_cb(
        desc,
        all_cores,
        cb_indices.padding_config0,
        kernel_config_df,
        1,
        padding_config_buffer0->page_size(),
        config_tensors_in_dram ? nullptr : padding_config_buffer0);

    cb_indices.padding_config1 = cb_indices.get_next_cb_id();
    add_cb(
        desc,
        all_cores,
        cb_indices.padding_config1,
        kernel_config_df,
        1,
        padding_config_buffer1->page_size(),
        config_tensors_in_dram ? nullptr : padding_config_buffer1);

    cb_indices.gather_config0 = cb_indices.get_next_cb_id();
    add_cb(
        desc,
        all_cores,
        cb_indices.gather_config0,
        kernel_config_df,
        1,
        gather_config_buffer0->page_size(),
        config_tensors_in_dram ? nullptr : gather_config_buffer0);

    cb_indices.gather_config1 = cb_indices.get_next_cb_id();
    add_cb(
        desc,
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
        "ttnn/cpp/ttnn/operations/experimental/quasar/halo/device/kernels/dataflow/halo_gather.cpp";
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
    KernelDescriptor::Defines reader_defines;
    std::vector<uint32_t> core_0_reader_ct_args = common_reader_ct_args;
    std::vector<uint32_t> core_1_reader_ct_args = common_reader_ct_args;

    if (config_tensors_in_dram) {
        reader_defines.emplace_back("CONFIG_TENSOR_IN_DRAM", "1");
        // NOTE: the config-buffer addresses are baked into compile-time args
        // here.  The buffers themselves are parked on the WorkloadDescriptor,
        // so their device allocations stay valid for the cached workload's
        // lifetime — the address embedded below remains correct on cache hit
        // because the buffers are not re-allocated.
        core_0_reader_ct_args.push_back(padding_config_buffer0->address());
        core_0_reader_ct_args.push_back(padding_config_buffer0->page_size());

        core_0_reader_ct_args.push_back(gather_config_buffer0->address());
        core_0_reader_ct_args.push_back(gather_config_buffer0->page_size());

        core_1_reader_ct_args.push_back(padding_config_buffer1->address());
        core_1_reader_ct_args.push_back(padding_config_buffer1->page_size());

        core_1_reader_ct_args.push_back(gather_config_buffer1->address());
        core_1_reader_ct_args.push_back(gather_config_buffer1->page_size());

        tt::tt_metal::TensorAccessorArgs(padding_config_buffer0).append_to(core_0_reader_ct_args);
        tt::tt_metal::TensorAccessorArgs(gather_config_buffer0).append_to(core_0_reader_ct_args);

        tt::tt_metal::TensorAccessorArgs(padding_config_buffer1).append_to(core_1_reader_ct_args);
        tt::tt_metal::TensorAccessorArgs(gather_config_buffer1).append_to(core_1_reader_ct_args);
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

    KernelDescriptor reader_0_desc;
    reader_0_desc.kernel_source = reader_kernel_name;
    reader_0_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_0_desc.core_ranges = all_cores;
    reader_0_desc.compile_time_args = std::move(core_0_reader_ct_args);
    reader_0_desc.defines = reader_defines;
    reader_0_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };

    KernelDescriptor reader_1_desc;
    reader_1_desc.kernel_source = reader_kernel_name;
    reader_1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_1_desc.core_ranges = all_cores;
    reader_1_desc.compile_time_args = std::move(core_1_reader_ct_args);
    reader_1_desc.defines = std::move(reader_defines);
    reader_1_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
    };

    if (config_tensors_in_dram) {
        uint32_t core_index = 0;
        for (const auto& core : cores) {
            if (is_height_sharded) {
                reader_0_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{core_index});
                reader_1_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{core_index});
            } else if (is_width_sharded) {
                reader_0_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{0});
                reader_1_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{0});
            } else if (is_block_sharded) {
                const auto nhw_index = is_rm_orientation ? core.y : core.x;
                reader_0_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{nhw_index});
                reader_1_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{nhw_index});
            }
            core_index++;
        }
    }

    desc.kernels.push_back(std::move(reader_0_desc));
    desc.kernels.push_back(std::move(reader_1_desc));

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor UntilizeWithHaloProgramFactory::create_workload_descriptor(
    const HaloParams& operation_attributes,
    const Tensor& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& input_tensor = tensor_args;
    auto* device = input_tensor.device();

    const bool is_in_tiled = input_tensor.layout() == Layout::TILE;
    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool remote_read = operation_attributes.remote_read;
    const bool transpose_mcast = operation_attributes.transpose_mcast;

    using namespace ttnn::operations;
    auto pad_metadata = sliding_window::generate_pad_metadata(operation_attributes.config);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(operation_attributes.config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(operation_attributes.config);
    const uint32_t input_shard_height = input_tensor.memory_config().shard_spec()->shape[0];
    auto tensor_metadata =
        sliding_window::generate_tensor_metadata(pad_metadata, operation_attributes.config, input_shard_height);

    const uint32_t num_cores_x = input_tensor.memory_config().shard_spec()->grid.bounding_box().grid_size().x;

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

    Tensor pad_config_device_tensor0 = sliding_window::move_config_tensor_to_device(
        pad_config_tensor0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);
    Tensor pad_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
        pad_config_tensor1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);
    Tensor gather_config_device_tensor0 = sliding_window::move_config_tensor_to_device(
        gather_config_tensor0,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);
    Tensor gather_config_device_tensor1 = sliding_window::move_config_tensor_to_device(
        gather_config_tensor1,
        operation_attributes.parallel_config,
        is_block_sharded,
        device,
        operation_attributes.config_tensors_in_dram);

    TT_ASSERT(pad_config_device_tensor0.dtype() == DataType::UINT16);
    TT_ASSERT(pad_config_device_tensor1.dtype() == DataType::UINT16);
    TT_ASSERT(gather_config_device_tensor0.dtype() == DataType::UINT16);
    TT_ASSERT(gather_config_device_tensor1.dtype() == DataType::UINT16);

    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    // Park each intermediate halo-config Tensor on the WorkloadDescriptor.
    // The Tensor itself (not just shared_ptr<MeshBuffer>) must outlive the
    // cached workload: ~Tensor force-deallocates the underlying device memory
    // via DeviceStorage::deallocate regardless of other shared owners.  See
    // pool_multi_core_program_factory.cpp for the lifetime rationale.
    auto pad0_owner = std::make_shared<Tensor>(std::move(pad_config_device_tensor0));
    Buffer* pad0_buf = pad0_owner->buffer();
    workload_descriptor.buffers.push_back({pad0_owner, pad0_buf});

    auto pad1_owner = std::make_shared<Tensor>(std::move(pad_config_device_tensor1));
    Buffer* pad1_buf = pad1_owner->buffer();
    workload_descriptor.buffers.push_back({pad1_owner, pad1_buf});

    auto gather0_owner = std::make_shared<Tensor>(std::move(gather_config_device_tensor0));
    Buffer* gather0_buf = gather0_owner->buffer();
    workload_descriptor.buffers.push_back({gather0_owner, gather0_buf});

    auto gather1_owner = std::make_shared<Tensor>(std::move(gather_config_device_tensor1));
    Buffer* gather1_buf = gather1_owner->buffer();
    workload_descriptor.buffers.push_back({gather1_owner, gather1_buf});

    const auto number_of_blocks_per_core = sliding_window::remap_nhw_scalar_argument_across_full_grid(
        kernel_config.number_of_blocks_per_core, operation_attributes.parallel_config);

    // Single-device op: the kernel program is structurally identical for every
    // coord in `tensor_coords` (halo doesn't depend on cluster position).
    // Build the per-coord ProgramDescriptor ONCE and copy it into each
    // coord-range entry to avoid redundant work on multi-coord workloads.
    auto desc = build_halo_program(
        operation_attributes,
        input_tensor,
        output_tensor,
        pad0_buf,
        pad1_buf,
        gather0_buf,
        gather1_buf,
        number_of_blocks_per_core);

    auto ranges = tensor_coords.ranges();
    workload_descriptor.programs.reserve(ranges.size());
    for (size_t i = 0; i + 1 < ranges.size(); ++i) {
        workload_descriptor.programs.push_back({ranges[i], desc});
    }
    if (!ranges.empty()) {
        workload_descriptor.programs.push_back({ranges.back(), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::prim::qsr
