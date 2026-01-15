// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/hal.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

#include "gather.hpp"

#include <algorithm>

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

namespace {

// Per-core tiling and addressing parameters used by the writers and compute kernels
struct BlockTilingParams {
    uint32_t total_tiles_per_block;
    uint32_t total_tiles_per_core;
    uint32_t tiles_per_block_writer0;
    uint32_t tiles_per_block_writer1;
    uint32_t output_addr_stride;
    uint32_t block_size_bytes;
};

struct GroupingResult {
    std::vector<std::vector<convert_to_hwc::detail::BlockedTransferGroup>> per_core_groups;
    uint32_t num_blocks;
};

inline BlockTilingParams compute_block_tiling_params(
    const ConvertToHwcConfig& config, uint32_t block_size_width, uint32_t num_blocks) {
    const uint32_t total_tiles_per_block = tt::div_up(block_size_width, TILE_HEIGHT);
    const uint32_t total_tiles_per_core = total_tiles_per_block * num_blocks;
    const uint32_t tiles_per_block_writer0 = tt::div_up(total_tiles_per_block, 2);
    const uint32_t tiles_per_block_writer1 = total_tiles_per_block - tiles_per_block_writer0;
    const uint32_t output_stride_sticks = TILE_WIDTH;
    // Inter-writer L1 output stride (bytes) between consecutive tiles written by a single writer.
    // If a block is only one tile tall, only one writer is active, so no stride is needed.
    const uint32_t output_addr_stride =
        (block_size_width != TILE_HEIGHT) ? output_stride_sticks * config.output_shard_width * config.element_size_bytes
                                          : 0;
    const uint32_t block_size_bytes =
        config.gather_l1_output_shard_height * block_size_width * config.element_size_bytes;
    return {
        total_tiles_per_block,
        total_tiles_per_core,
        tiles_per_block_writer0,
        tiles_per_block_writer1,
        output_addr_stride,
        block_size_bytes};
}

inline std::vector<uint32_t> make_writer_compile_args(
    bool is_reader,
    uint32_t cb_in_transpose_index,
    const ConvertToHwcConfig& config,
    const BlockTilingParams& tiling,
    uint32_t tiles_per_block_for_writer,
    uint32_t initial_write_stick_offset,
    uint32_t num_blocks) {
    return {
        CBIndex::CB_IN,
        CBIndex::CB_IN_BATCH,
        cb_in_transpose_index,
        CBIndex::CB_OUT,
        config.output_shard_width,
        tiles_per_block_for_writer,
        initial_write_stick_offset,
        config.element_size_bytes,
        static_cast<uint32_t>(config.is_input_in_dram),
        static_cast<uint32_t>(is_reader),
        is_reader ? config.gather_l1_output_shard_height : 0u,
        num_blocks,
        tiling.output_addr_stride,
        tiling.block_size_bytes};
}

inline std::vector<uint32_t> make_compute_compile_args(
    uint32_t total_tiles_per_block, uint32_t total_sticks_per_block, uint32_t num_blocks) {
    return {
        CBIndex::CB_IN_BATCH,
        CBIndex::CB_IN_TILED,
        CBIndex::CB_IN_TRANSPOSE_0,
        CBIndex::CB_IN_TRANSPOSE_1,
        total_tiles_per_block,
        total_sticks_per_block,
        num_blocks};
}

// Select an appropriate block size that evenly divides gather_l1_output_shard_width
// Tries to find a block size >= 1024 that is a multiple of 32 and divides evenly
inline uint32_t select_block_size(uint32_t gather_l1_output_shard_width) {
    const uint32_t min_block_size_width = 1024;
    uint32_t block_size_width = gather_l1_output_shard_width;
    for (uint32_t candidate = min_block_size_width; candidate <= gather_l1_output_shard_width; candidate += 32) {
        if (gather_l1_output_shard_width % candidate == 0) {
            block_size_width = candidate;
            break;
        }
    }
    return block_size_width;
}

// Generate gather transfers, group them into output column blocks, and coalesce contiguous copies.
GroupingResult group_and_coalesce_transfers(
    const ConvertToHwcConfig& config,
    const std::vector<CoreCoord>& in_cores,
    uint32_t effective_hw_for_gather,
    uint32_t block_size_width) {
    // Use the actual output shard width for transfer generation (determines which output core)
    // block_size_width is only used for grouping transfers into blocks
    const auto gather_transfers = convert_to_hwc::detail::precompute_gather_transfers(
        config.batch_size,
        config.input_channels,
        effective_hw_for_gather,
        in_cores,
        config.output_cores,
        config.gather_l1_output_shard_width);

    const auto blocked_result = convert_to_hwc::detail::group_transfers_by_output_column_blocks(
        gather_transfers,
        config.batch_size,
        config.input_channels,
        effective_hw_for_gather,
        in_cores,
        config.output_cores.size(),
        /*element_size_bytes=*/config.element_size_bytes,
        /*block_size=*/block_size_width,
        /*output_shard_width=*/config.gather_l1_output_shard_width);

    auto blocked_gather_transfers = blocked_result.blocked_transfers;
    auto per_core_blocked_gather_transfers =
        convert_to_hwc::detail::split_by_destination_core(blocked_gather_transfers, config.output_cores.size());

    // Verify all cores have the same number of blocks
    // This is critical because the compute kernel expects total_num_blocks blocks from each core
    const uint32_t expected_blocks_per_core = blocked_result.num_logical_blocks;
    for (size_t core_idx = 0; core_idx < per_core_blocked_gather_transfers.size(); core_idx++) {
        uint32_t core_blocks = static_cast<uint32_t>(per_core_blocked_gather_transfers[core_idx].size());
        TT_FATAL(
            core_blocks == expected_blocks_per_core,
            "Core {} has {} blocks but expected {} blocks per core. "
            "All cores must have the same number of blocks for the compute kernel to work correctly.",
            core_idx,
            core_blocks,
            expected_blocks_per_core);
    }

    // Coalesce contiguous transfers for each core
    for (auto& core_transfers : per_core_blocked_gather_transfers) {
        core_transfers = convert_to_hwc::detail::coalesce_contiguous_transfers(core_transfers);
    }
    return {std::move(per_core_blocked_gather_transfers), blocked_result.num_logical_blocks};
}

// Serialize grouped transfers per destination core with the provided source-address mapping.
inline std::vector<std::vector<uint32_t>> serialize_transfers_per_core(
    const std::vector<std::vector<convert_to_hwc::detail::BlockedTransferGroup>>& per_core_groups,
    const std::vector<CoreCoord>& in_cores,
    const std::function<CoreCoord(const CoreCoord&)>& logical_to_addr_id) {
    std::vector<std::vector<uint32_t>> per_core_serialized;
    per_core_serialized.resize(per_core_groups.size());
    for (size_t core_idx = 0; core_idx < per_core_groups.size(); core_idx++) {
        per_core_serialized[core_idx] = convert_to_hwc::detail::serialize_blocked_transfer_groups(
            per_core_groups[core_idx], in_cores, logical_to_addr_id);
    }
    return per_core_serialized;
}

}  // namespace

// Effective HW used by gather: always the padded capacity per input core
uint32_t calculate_effective_hw_for_sharding(
    uint32_t /*hw_total*/, uint32_t /*batch_size*/, uint32_t padded_shard_width, uint32_t num_cores) {
    // Covers both even sharding (exact fit) and uneven sharding (B=1; padded)
    return num_cores * padded_shard_width;
}

struct CircularBufferHandles {
    tt::tt_metal::CBHandle cb_in;
    tt::tt_metal::CBHandle cb_out;
};

// Helper function to create a circular buffer
tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_grid,
    uint32_t index,
    uint32_t total_size,
    uint32_t page_size,
    const tt::DataFormat& format,
    tt::tt_metal::Buffer* buffer = nullptr);

// Setup all circular buffers for the convert_to_hwc operation
CircularBufferHandles setup_circular_buffers(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_grid,
    const ConvertToHwcConfig& config,
    const Tensor& input,
    const Tensor& output,
    uint32_t block_size_width);

ConvertToHwcConfig ConvertToHwcConfig::create_from_tensors(const Tensor& input, const Tensor& output) {
    ConvertToHwcConfig config;

    // Input tensor properties
    config.batch_size = input.logical_shape()[1];
    config.input_channels = input.logical_shape()[2];
    config.hw_total = input.logical_shape()[3];
    config.input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    config.element_size_bytes = tt::datum_size(config.input_format);

    // DRAM/L1 configuration
    config.is_input_in_dram = input.buffer()->core_type() == tt::CoreType::DRAM;
    config.remote_address = input.buffer()->address();
    config.remote_buffer_type = input.buffer()->buffer_type();
    config.remote_core_type = input.buffer()->core_type();

    // Shard specifications
    config.output_shard_height = output.shard_spec()->shape[0];
    config.output_shard_width = output.shard_spec()->shape[1];
    config.l1_input_shard_height = config.is_input_in_dram ? input.logical_shape()[-2] : input.shard_spec()->shape[0];
    // Use input's padded sharded width (WIDTH_SHARDED) for both DRAM and L1 inputs
    config.l1_input_shard_width = input.shard_spec()->shape[1];

    // Core information
    // Kernels run on output cores; data sources are the input cores
    config.output_core_grid = output.shard_spec()->grid;
    config.output_cores = corerange_to_cores(
        config.output_core_grid,
        std::nullopt,
        output.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    // Always derive input core locations from the input tensor's shard grid
    config.l1_input_core_grid = input.shard_spec()->grid;
    config.l1_input_cores = corerange_to_cores(
        config.l1_input_core_grid,
        std::nullopt,
        input.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    config.dram_input_cores = corerange_to_cores(
        input.shard_spec()->grid,
        std::nullopt,
        input.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    // Gather output shard specifications (for the intermediate gather result)
    // The gather operation transforms from [B, C, HW] to [C, B, HW] layout
    // So the gather output has height=C and width=B*HW_effective/num_output_cores
    config.gather_l1_output_shard_height = config.input_channels;

    // Set per-destination-core gather width to the padded B*HW per output core (output shard height)
    config.gather_l1_output_shard_width = config.output_shard_height;

    // Alignment requirements
    config.alignment_elements = compute_alignment_requirement_in_elements(output);

    log_debug(
        tt::LogType::LogOp,
        "convert_to_hwc config: B={}, C={}, HW={}, input_in_dram={}, in_cores={}, out_cores={}, out_shard=[{}x{}], "
        "gather_width={}",
        config.batch_size,
        config.input_channels,
        config.hw_total,
        config.is_input_in_dram,
        config.l1_input_cores.size(),
        config.output_cores.size(),
        config.output_shard_height,
        config.output_shard_width,
        config.gather_l1_output_shard_width);

    return config;
}

void ConvertToHwcConfig::validate() const {
    TT_FATAL(alignment_elements != 0, "Number of alignment elements cannot be 0");
    TT_FATAL(
        output_shard_width % alignment_elements == 0,
        "Output shard width {} must be multiple of {} to satisfy alignment constraints",
        output_shard_width,
        alignment_elements);
    TT_FATAL(output_shard_height % 32 == 0, "Shard height {} must be multiple of tile width (32)", output_shard_height);
    TT_FATAL(!output_cores.empty(), "No output cores available for processing");

    // Check for uneven sharding and validate B=1 requirement
    uint32_t input_num_cores = l1_input_cores.size();
    // Uneven sharding occurs when the last core has fewer logical elements than the shard width
    uint32_t total_padded_elements = input_num_cores * l1_input_shard_width;
    bool is_uneven_sharding = hw_total < total_padded_elements;
    if (is_uneven_sharding) {
        TT_FATAL(
            batch_size == 1,
            "Uneven sharding (HW={} < total_padded_capacity={}) is only supported for batch_size=1, got batch_size={}",
            hw_total,
            total_padded_elements,
            batch_size);
    }
}

tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_grid,
    uint32_t index,
    uint32_t total_size,
    uint32_t page_size,
    const tt::DataFormat& format,
    tt::tt_metal::Buffer* buffer) {
    auto config = tt::tt_metal::CircularBufferConfig(total_size, {{index, format}}).set_page_size(index, page_size);
    if (buffer != nullptr) {
        config = config.set_globally_allocated_address(*buffer);
    }
    return tt::tt_metal::CreateCircularBuffer(program, core_grid, config);
}

CircularBufferHandles setup_circular_buffers(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_grid,
    const ConvertToHwcConfig& config,
    const Tensor& input,
    const Tensor& output,
    uint32_t block_size_width) {
    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    // CB_IN: full input shard per core
    const uint32_t cb_in_page_size = config.l1_input_shard_width * config.element_size_bytes;
    const uint32_t cb_in_total_size = config.l1_input_shard_height * cb_in_page_size;
    auto cb_in = create_circular_buffer(
        program,
        core_grid,
        CBIndex::CB_IN,
        cb_in_total_size,
        cb_in_page_size,
        config.input_format,
        config.is_input_in_dram ? nullptr : input.buffer());

    // CB_IN_BATCH: [C x block_size_width] staging for gathered sticks
    const uint32_t cb_in_batch_page_size = block_size_width * config.element_size_bytes;
    const uint32_t cb_in_batch_total_size = config.gather_l1_output_shard_height * cb_in_batch_page_size;
    create_circular_buffer(
        program, core_grid, CBIndex::CB_IN_BATCH, cb_in_batch_total_size, cb_in_batch_page_size, config.input_format);

    // CB_IN_TILED: intermediate tiles
    const uint32_t cb_in_tiled_page_size = intermediary_tile_size;
    const uint32_t cb_in_tiled_total_size = tt::div_up(block_size_width, TILE_WIDTH) * intermediary_tile_size;
    create_circular_buffer(
        program, core_grid, CBIndex::CB_IN_TILED, cb_in_tiled_total_size, cb_in_tiled_page_size, intermediary_format);

    // CB_IN_TRANSPOSE_[0/1]
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;
    const uint32_t cb_in_transpose_total_size = tt::div_up(block_size_width, TILE_WIDTH) * intermediary_tile_size;
    create_circular_buffer(
        program,
        core_grid,
        CBIndex::CB_IN_TRANSPOSE_0,
        cb_in_transpose_total_size,
        cb_in_transpose_page_size,
        intermediary_format);
    create_circular_buffer(
        program,
        core_grid,
        CBIndex::CB_IN_TRANSPOSE_1,
        cb_in_transpose_total_size,
        cb_in_transpose_page_size,
        intermediary_format);

    // CB_OUT: output shard per core
    const uint32_t cb_out_page_size = config.output_shard_width * config.element_size_bytes;
    const uint32_t cb_out_total_size = cb_out_page_size * config.output_shard_height;  // same size as input
    auto cb_out = create_circular_buffer(
        program, core_grid, CBIndex::CB_OUT, cb_out_total_size, cb_out_page_size, config.input_format, output.buffer());

    return {cb_in, cb_out};
}

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor) {
    const uint32_t element_size_bytes = input_tensor.element_size();
    const uint32_t l1_alignment_bytes = tt::tt_metal::hal::get_l1_alignment();
    return l1_alignment_bytes / element_size_bytes;
}

}  // namespace ttnn::operations::experimental::cnn::detail

namespace ttnn::operations::experimental::cnn::program {

namespace {

void set_runtime_arguments(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    bool is_input_in_dram,
    const std::vector<tt::tt_metal::CoreCoord>& output_cores,
    const std::vector<std::vector<uint32_t>>& per_core_serialized_transfers,
    tt::tt_metal::KernelHandle writer_kernel_id0,
    tt::tt_metal::KernelHandle writer_kernel_id1,
    uint32_t remote_address,
    tt::tt_metal::CBHandle cb_in,
    tt::tt_metal::CBHandle cb_out) {
    // Set per-core runtime arguments for writer kernels
    for (uint32_t core_idx = 0; core_idx < output_cores.size(); core_idx++) {
        std::vector<uint32_t> runtime_args_0 = {remote_address};
        std::vector<uint32_t> runtime_args_1 = {remote_address};
        const auto& transfer_args = per_core_serialized_transfers.at(core_idx);
        runtime_args_0.insert(runtime_args_0.end(), transfer_args.begin(), transfer_args.end());
        runtime_args_1.insert(runtime_args_1.end(), transfer_args.begin(), transfer_args.end());
        SetRuntimeArgs(program, writer_kernel_id0, output_cores[core_idx], runtime_args_0);
        SetRuntimeArgs(program, writer_kernel_id1, output_cores[core_idx], runtime_args_1);
    }
    // Only update input CB address for L1 input (DRAM input doesn't need CB update)
    if (!is_input_in_dram) {
        UpdateDynamicCircularBufferAddress(program, cb_in, *input_tensor.buffer());
    }
    UpdateDynamicCircularBufferAddress(program, cb_out, *output_tensor.buffer());
}

}  // namespace

ConvertToHWCProgramFactory::cached_program_t ConvertToHWCProgramFactory::create(
    const CnnParams& /*operation_attributes*/, const CnnInputs& tensor_args, Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& a = tensor_args.input;
    auto& output = tensor_return_value;

    // Create configuration from input tensors
    auto config = detail::ConvertToHwcConfig::create_from_tensors(a, output);
    config.validate();

    // Select input cores based on source memory (DRAM vs L1)
    const auto& in_cores = config.is_input_in_dram ? config.dram_input_cores : config.l1_input_cores;

    // Effective HW for gather transfers (padded capacity per input core)
    uint32_t effective_hw_for_gather = detail::calculate_effective_hw_for_sharding(
        config.hw_total, config.batch_size, config.l1_input_shard_width, static_cast<uint32_t>(in_cores.size()));

    // Use smaller block size to reduce L1 consumption
    // Find a block size that evenly divides gather_l1_output_shard_width
    // This reduces the CB_IN_BATCH buffer size significantly
    const auto block_width = detail::select_block_size(config.gather_l1_output_shard_width);

    // Setup circular buffers on the output cores (where the kernels execute)
    auto cb_handles = detail::setup_circular_buffers(program, config.output_core_grid, config, a, output, block_width);
    auto grouping = detail::group_and_coalesce_transfers(config, in_cores, effective_hw_for_gather, block_width);
    const uint32_t num_blocks = grouping.num_blocks;

    // Source-address mapping for serialization:
    // - L1 input: logical core -> worker core (x,y)
    // - DRAM input: x := bank_id, y := 0
    std::function<CoreCoord(const CoreCoord&)> logical_to_addr_id;
    if (config.is_input_in_dram) {
        std::map<std::pair<int, int>, uint32_t> bank_id_by_core;
        for (const auto& c : config.dram_input_cores) {
            auto bank_ids = a.device()->allocator()->get_bank_ids_from_logical_core(config.remote_buffer_type, c);
            uint32_t bank_id = bank_ids.empty() ? 0 : bank_ids[0];
            bank_id_by_core[{c.x, c.y}] = bank_id;
        }
        logical_to_addr_id = [bank_id_by_core = std::move(bank_id_by_core)](const CoreCoord& logical_core) {
            auto it = bank_id_by_core.find({logical_core.x, logical_core.y});
            uint32_t bank_id = (it == bank_id_by_core.end()) ? 0 : it->second;
            return CoreCoord(bank_id, 0);
        };
    } else {
        logical_to_addr_id = [&a](const CoreCoord& logical_core) {
            return a.device()->worker_core_from_logical_core(logical_core);
        };
    }

    // Serialize blocked transfer groups for each core
    auto per_core_serialized_transfers =
        detail::serialize_transfers_per_core(grouping.per_core_groups, in_cores, logical_to_addr_id);

    // Compute per-core tiling/state based on the chosen block width
    const detail::BlockTilingParams tiling = detail::compute_block_tiling_params(config, block_width, num_blocks);

    // Split tiles within each block between the two writers
    const uint32_t tiles_per_block_writer0 = tiling.tiles_per_block_writer0;
    const uint32_t tiles_per_block_writer1 = tiling.tiles_per_block_writer1;

    // Ensure tiles divide evenly across blocks
    TT_FATAL(
        tiling.total_tiles_per_core % num_blocks == 0,
        "total_tiles_per_core={} must be divisible by num_blocks={}",
        tiling.total_tiles_per_core,
        num_blocks);
    auto writer_compile_time_args0 = detail::make_writer_compile_args(
        /*is_reader=*/true,
        detail::CBIndex::CB_IN_TRANSPOSE_0,
        config,
        tiling,
        tiles_per_block_writer0,
        /*initial_write_stick_offset=*/0,
        num_blocks);

    auto writer_compile_time_args1 = detail::make_writer_compile_args(
        /*is_reader=*/false,
        detail::CBIndex::CB_IN_TRANSPOSE_1,
        config,
        tiling,
        tiles_per_block_writer1,
        /*initial_write_stick_offset=*/tt::constants::TILE_WIDTH,
        num_blocks);

    auto compute_compile_time_args = detail::make_compute_compile_args(
        tiling.total_tiles_per_block, config.gather_l1_output_shard_height, num_blocks);

    auto writer_kernel_id0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        config.output_core_grid,
        tt::tt_metal::ReaderDataMovementConfig(writer_compile_time_args0));

    auto writer_kernel_id1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        config.output_core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args1));

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/convert_to_hwc.cpp",
        config.output_core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    // Set runtime arguments during program creation (required to prevent hangs)
    set_runtime_arguments(
        program,
        a,
        output,
        config.is_input_in_dram,
        config.output_cores,
        per_core_serialized_transfers,
        writer_kernel_id0,
        writer_kernel_id1,
        config.remote_address,
        cb_handles.cb_in,
        cb_handles.cb_out);

    // Store shared variables for override
    shared_variables_t shared_variables{
        .cb_in = cb_handles.cb_in,
        .cb_out = cb_handles.cb_out,
        .is_input_in_dram = config.is_input_in_dram,
        .output_cores = config.output_cores,
        .per_core_serialized_transfers = per_core_serialized_transfers,
        .writer_kernel_id0 = writer_kernel_id0,
        .writer_kernel_id1 = writer_kernel_id1,
        .remote_address = config.remote_address};

    return {std::move(program), std::move(shared_variables)};
}

void ConvertToHWCProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const CnnParams& /*operation_attributes*/,
    const CnnInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;
    const auto& a = tensor_args.input;
    auto& output = tensor_return_value;

    set_runtime_arguments(
        program,
        a,
        output,
        shared_vars.is_input_in_dram,
        shared_vars.output_cores,
        shared_vars.per_core_serialized_transfers,
        shared_vars.writer_kernel_id0,
        shared_vars.writer_kernel_id1,
        shared_vars.remote_address,
        shared_vars.cb_in,
        shared_vars.cb_out);
}

}  // namespace ttnn::operations::experimental::cnn::program
