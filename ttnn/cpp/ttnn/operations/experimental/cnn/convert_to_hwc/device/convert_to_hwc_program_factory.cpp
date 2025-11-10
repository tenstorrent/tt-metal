// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/hal.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

#include "gather.hpp"

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

// Helper function to calculate effective HW dimension for sharding calculations
uint32_t calculate_effective_hw_for_sharding(
    uint32_t hw_total, uint32_t batch_size, uint32_t padded_shard_width, uint32_t num_cores) {
    // Always use the full padded capacity for all batch sizes
    // For even sharding: hw_total == num_cores * padded_shard_width (same result)
    // For uneven sharding (B=1 only): hw_total < num_cores * padded_shard_width (handles padding correctly)
    return num_cores * padded_shard_width;
}

// Helper struct to hold circular buffer handles
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

    log_info(tt::LogType::LogAlways, "=== ConvertToHwcConfig::create_from_tensors ===");
    log_info(
        tt::LogType::LogAlways,
        "Input tensor logical shape: [{}, {}, {}, {}]",
        input.logical_shape()[0],
        input.logical_shape()[1],
        input.logical_shape()[2],
        input.logical_shape()[3]);
    log_info(
        tt::LogType::LogAlways,
        "Output tensor logical shape: [{}, {}, {}, {}]",
        output.logical_shape()[0],
        output.logical_shape()[1],
        output.logical_shape()[2],
        output.logical_shape()[3]);
    log_info(
        tt::LogType::LogAlways,
        "Parsed: batch_size={}, input_channels={}, hw_total={}, element_size_bytes={}",
        config.batch_size,
        config.input_channels,
        config.hw_total,
        config.element_size_bytes);

    // DRAM/L1 configuration
    config.is_input_in_dram = input.buffer()->core_type() == tt::CoreType::DRAM;
    config.remote_address = input.buffer()->address();
    config.remote_buffer_type = input.buffer()->buffer_type();
    config.remote_core_type = input.buffer()->core_type();

    log_info(
        tt::LogType::LogAlways,
        "Input buffer: is_input_in_dram={}, remote_address={}, core_type={}",
        config.is_input_in_dram,
        config.remote_address,
        (uint32_t)config.remote_core_type);

    // Shard specifications
    config.output_shard_height = output.shard_spec()->shape[0];
    config.output_shard_width = output.shard_spec()->shape[1];
    config.l1_input_shard_height = config.is_input_in_dram ? input.logical_shape()[-2] : input.shard_spec()->shape[0];
    config.l1_input_shard_width = config.is_input_in_dram ? config.output_shard_height : input.shard_spec()->shape[1];

    log_info(
        tt::LogType::LogAlways,
        "Shard specs: output_shard=[{}x{}], l1_input_shard=[{}x{}]",
        config.output_shard_height,
        config.output_shard_width,
        config.l1_input_shard_height,
        config.l1_input_shard_width);

    // Core information
    config.l1_input_core_grid = config.is_input_in_dram ? output.shard_spec()->grid : input.shard_spec()->grid;
    config.l1_input_cores = corerange_to_cores(
        config.l1_input_core_grid,
        std::nullopt,
        input.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    config.dram_input_cores = corerange_to_cores(
        input.shard_spec()->grid,
        std::nullopt,
        input.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    log_info(
        tt::LogType::LogAlways,
        "Core info: l1_input_cores.size()={}, dram_input_cores.size()={}",
        config.l1_input_cores.size(),
        config.dram_input_cores.size());
    log_info(tt::LogType::LogAlways, "L1 input cores:");
    for (size_t i = 0; i < config.l1_input_cores.size(); i++) {
        log_info(tt::LogType::LogAlways, "  [{}]: ({}, {})", i, config.l1_input_cores[i].x, config.l1_input_cores[i].y);
    }

    // Gather output shard specifications (for the intermediate gather result)
    // The gather operation transforms from [B, C, HW] to [C, B, HW] layout
    // So the gather output has height=C and width=B*HW_effective/num_output_cores
    config.gather_l1_output_shard_height = config.input_channels;

    // For uneven sharding with B=1, use padded capacity instead of logical HW
    uint32_t effective_hw = calculate_effective_hw_for_sharding(
        config.hw_total, config.batch_size, config.l1_input_shard_width, config.l1_input_cores.size());
    config.gather_l1_output_shard_width = config.batch_size * effective_hw / config.l1_input_cores.size();

    log_info(
        tt::LogType::LogAlways,
        "Gather shard specs: gather_l1_output_shard=[{}x{}] (C={}, B*HW_effective/num_cores={}*{}/{})",
        config.gather_l1_output_shard_height,
        config.gather_l1_output_shard_width,
        config.input_channels,
        config.batch_size,
        effective_hw,
        config.l1_input_cores.size());

    // Alignment requirements
    config.alignment_elements = compute_alignment_requirement_in_elements(output);

    log_info(tt::LogType::LogAlways, "Alignment: alignment_elements={}", config.alignment_elements);
    log_info(tt::LogType::LogAlways, "=== End ConvertToHwcConfig ===");

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
    TT_FATAL(!l1_input_cores.empty(), "No input cores available for processing");

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

    log_info(tt::LogType::LogAlways, "=== setup_circular_buffers ===");
    log_info(
        tt::LogType::LogAlways,
        "intermediary_format={}, intermediary_tile_size={}",
        (uint32_t)intermediary_format,
        intermediary_tile_size);

    // CB in (full input)
    const uint32_t cb_in_page_size = config.l1_input_shard_width * config.element_size_bytes;
    const uint32_t cb_in_total_size = config.l1_input_shard_height * cb_in_page_size;
    log_info(
        tt::LogType::LogAlways,
        "CB_IN: page_size={}, total_size={}, buffer={}",
        cb_in_page_size,
        cb_in_total_size,
        (void*)(config.is_input_in_dram ? nullptr : input.buffer()));
    auto cb_in = create_circular_buffer(
        program,
        core_grid,
        CBIndex::CB_IN,
        cb_in_total_size,
        cb_in_page_size,
        config.input_format,
        config.is_input_in_dram ? nullptr : input.buffer());

    // CB in batch - using block_size_width to ensure alignment with transfer blocks
    const uint32_t cb_in_batch_page_size = block_size_width * config.element_size_bytes;
    const uint32_t cb_in_batch_total_size = config.gather_l1_output_shard_height * cb_in_batch_page_size;
    log_info(
        tt::LogType::LogAlways,
        "CB_IN_BATCH: page_size={}, total_size={} (using block_size_width={}, height={})",
        cb_in_batch_page_size,
        cb_in_batch_total_size,
        block_size_width,
        config.gather_l1_output_shard_height);
    create_circular_buffer(
        program, core_grid, CBIndex::CB_IN_BATCH, cb_in_batch_total_size, cb_in_batch_page_size, config.input_format);

    // CB in tiled
    const uint32_t cb_in_tiled_page_size = intermediary_tile_size;
    const uint32_t cb_in_tiled_total_size = tt::div_up(block_size_width, TILE_WIDTH) * intermediary_tile_size;
    log_info(
        tt::LogType::LogAlways,
        "CB_IN_TILED: page_size={}, total_size={}",
        cb_in_tiled_page_size,
        cb_in_tiled_total_size);
    create_circular_buffer(
        program, core_grid, CBIndex::CB_IN_TILED, cb_in_tiled_total_size, cb_in_tiled_page_size, intermediary_format);

    // CB in transpose buffers
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;
    const uint32_t cb_in_transpose_total_size = tt::div_up(block_size_width, TILE_WIDTH) * intermediary_tile_size;
    log_info(
        tt::LogType::LogAlways,
        "CB_IN_TRANSPOSE_0/1: page_size={}, total_size={}",
        cb_in_transpose_page_size,
        cb_in_transpose_total_size);
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

    // CB out
    const uint32_t cb_out_page_size = config.output_shard_width * config.element_size_bytes;
    const uint32_t cb_out_total_size = cb_out_page_size * config.output_shard_height;  // same size as input
    log_info(
        tt::LogType::LogAlways,
        "CB_OUT: page_size={}, total_size={}, buffer={}",
        cb_out_page_size,
        cb_out_total_size,
        (void*)output.buffer());
    auto cb_out = create_circular_buffer(
        program, core_grid, CBIndex::CB_OUT, cb_out_total_size, cb_out_page_size, config.input_format, output.buffer());

    log_info(tt::LogType::LogAlways, "=== End setup_circular_buffers ===");

    return {cb_in, cb_out};
}

// Generate individual transfers for a single destination core
std::map<uint32_t, std::vector<TransferData>> generate_transfers_for_output_core(
    uint32_t dst_core,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    uint32_t input_num_cores,
    uint32_t output_num_cores,
    uint32_t element_size_bytes,
    uint32_t padded_shard_width) {
    std::map<uint32_t, std::vector<TransferData>> transfers_by_src;

    // Helper function to get actual HW count for a given core (first cores get full shard, last may be partial)
    auto get_hw_count_for_core = [&](uint32_t core_idx) -> uint32_t {
        uint32_t remaining_hw = hw_total - (core_idx * padded_shard_width);
        return std::min(padded_shard_width, remaining_hw);
    };

    // Check if we have uneven sharding (total logical HW < total padded capacity)
    uint32_t total_padded_capacity = input_num_cores * padded_shard_width;
    bool is_uneven_sharding = hw_total < total_padded_capacity;

    uint32_t dst_bhw_start, dst_bhw_end;

    if (is_uneven_sharding && batch_size == 1) {
        // For uneven sharding with B=1, we need to process the full padded shard width
        // to include padding for alignment purposes
        dst_bhw_start = dst_core * padded_shard_width;
        dst_bhw_end = (dst_core + 1) * padded_shard_width;
    } else {
        // For even sharding or B>1, use uniform distribution
        uint32_t bhw_total = batch_size * hw_total;
        uint32_t bhw_per_output_core = bhw_total / output_num_cores;
        dst_bhw_start = dst_core * bhw_per_output_core;
        dst_bhw_end = std::min(dst_bhw_start + bhw_per_output_core, bhw_total);
    }

    for (uint32_t bhw_idx = dst_bhw_start; bhw_idx < dst_bhw_end; bhw_idx++) {
        uint32_t src_core, src_hw_offset, batch_id, hw_idx;

        if (is_uneven_sharding && batch_size == 1) {
            // For uneven sharding with B=1, map directly to padded shard layout
            batch_id = 0;                             // B=1
            hw_idx = bhw_idx;                         // Direct mapping since we're processing padded shard width
            src_core = dst_core;                      // Same core as destination
            src_hw_offset = bhw_idx - dst_bhw_start;  // Offset within the padded shard
        } else {
            // Convert BHW index back to (batch_id, hw_idx)
            batch_id = bhw_idx / hw_total;
            hw_idx = bhw_idx % hw_total;

            // Find which input core has this hw_idx data
            src_core = hw_idx / padded_shard_width;
            src_hw_offset = hw_idx % padded_shard_width;

            // Skip if this hw_idx exceeds the logical HW range for this core
            uint32_t src_core_hw_count = get_hw_count_for_core(src_core);
            if (src_hw_offset >= src_core_hw_count) {
                continue;  // Skip invalid logical indices
            }
        }

        // Generate one transfer per C and B (stick-by-stick)
        for (uint32_t c_idx = 0; c_idx < channels; c_idx++) {
            // Calculate source offset: [B*C, padded_shard_width] layout (interleaved batch-channel)
            // Each core physically has padded_shard_width elements, but logically may have fewer
            uint32_t src_offset =
                (batch_id * channels * padded_shard_width + src_hw_offset * channels + c_idx) * element_size_bytes;

            // Calculate destination offset: [1, C, BHW_per_output_core] layout
            uint32_t dst_bhw_offset = bhw_idx - dst_bhw_start;
            uint32_t dst_offset = (dst_bhw_offset * channels + c_idx) * element_size_bytes;

            // Group by source core
            if (transfers_by_src.find(src_core) == transfers_by_src.end()) {
                transfers_by_src[src_core] = std::vector<TransferData>();
            }

            // Create one transfer per element (C and B combination)
            transfers_by_src[src_core].emplace_back(src_offset, dst_offset, element_size_bytes);
        }
    }

    // Log transfers before optimization
    log_info(tt::LogType::LogAlways, "BEFORE OPTIMIZATION - dst_core={}, total transfers by src:", dst_core);
    for (const auto& [src_core, transfer_list] : transfers_by_src) {
        log_info(tt::LogType::LogAlways, "  src_core={}, num_transfers={}", src_core, transfer_list.size());
        for (size_t i = 0; i < transfer_list.size() && i < 10; i++) {  // Limit to first 10 for readability
            const auto& t = transfer_list[i];
            log_info(
                tt::LogType::LogAlways,
                "    [{}]: src_offset={}, dst_offset={}, size={}",
                i,
                t.src_offset,
                t.dst_offset,
                t.size);
        }
        if (transfer_list.size() > 10) {
            log_info(tt::LogType::LogAlways, "    ... ({} more transfers)", transfer_list.size() - 10);
        }
    }

    return transfers_by_src;
}

// Optimize transfers using batch-aware grouping
std::vector<BatchTransferInstruction> optimize_transfers(
    const std::map<uint32_t, std::vector<TransferData>>& transfers_by_src,
    uint32_t dst_core,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    uint32_t input_num_cores,
    uint32_t element_size_bytes,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t padded_shard_width) {
    std::vector<BatchTransferInstruction> instructions;

    for (const auto& [src_core, transfer_list] : transfers_by_src) {
        // Sort by source offset
        std::vector<TransferData> sorted_transfers = transfer_list;
        std::sort(sorted_transfers.begin(), sorted_transfers.end(), [](const TransferData& a, const TransferData& b) {
            return a.src_offset < b.src_offset;
        });

        if (sorted_transfers.empty()) {
            continue;
        }

        // Group transfers by batch to ensure at least one transfer per batch
        std::map<uint32_t, std::vector<TransferData>> transfers_by_batch;
        uint32_t batch_size_bytes = channels * padded_shard_width * element_size_bytes;

        for (const TransferData& transfer : sorted_transfers) {
            // Determine which batch this transfer belongs to
            // The src_offset was calculated as: batch_id * channels * padded_shard_width + src_hw_offset * channels
            uint32_t batch_id = transfer.src_offset / batch_size_bytes;
            transfers_by_batch[batch_id].push_back(transfer);
        }

        // Process each batch separately to ensure at least one transfer per batch
        for (auto& [batch_id, batch_transfers] : transfers_by_batch) {
            // Group consecutive transfers within this batch
            TransferData current_transfer = batch_transfers[0];

            for (size_t i = 1; i < batch_transfers.size(); i++) {
                const TransferData& next_transfer = batch_transfers[i];

                // Check if we can combine transfers (within same batch)
                if (next_transfer.src_offset == current_transfer.src_offset + current_transfer.size &&
                    next_transfer.dst_offset == current_transfer.dst_offset + current_transfer.size) {
                    current_transfer.size += next_transfer.size;
                } else {
                    // Emit current transfer and start new one
                    instructions.emplace_back(
                        src_core,
                        dst_core,
                        input_cores[src_core],
                        output_cores[dst_core],
                        current_transfer.src_offset,
                        current_transfer.dst_offset,
                        current_transfer.size,
                        0);  // bank_id = 0 for L1 transfers
                    current_transfer = next_transfer;
                }
            }

            // Emit final transfer for this batch
            instructions.emplace_back(
                src_core,
                dst_core,
                input_cores[src_core],
                output_cores[dst_core],
                current_transfer.src_offset,
                current_transfer.dst_offset,
                current_transfer.size,
                0);  // bank_id = 0 for L1 transfers
        }
    }

    // Log transfers after optimization
    log_info(
        tt::LogType::LogAlways,
        "AFTER OPTIMIZATION - dst_core={}, final instructions={}",
        dst_core,
        instructions.size());
    for (size_t i = 0; i < instructions.size() && i < 20; i++) {  // Limit to first 20 for readability
        const auto& instr = instructions[i];
        log_info(
            tt::LogType::LogAlways,
            "  [{}]: src_core={}, src_offset={}, dst_offset={}, size={}",
            i,
            instr.src_core_idx,
            instr.src_offset,
            instr.dst_offset,
            instr.transfer_size);
    }
    if (instructions.size() > 20) {
        log_info(tt::LogType::LogAlways, "  ... ({} more instructions)", instructions.size() - 20);
    }

    return instructions;
}

void populate_dram_bank_ids(
    std::vector<BatchTransferInstruction>& transfers,
    const std::vector<CoreCoord>& dram_cores,
    const tt::tt_metal::BufferType& dram_buffer_type,
    tt::tt_metal::IDevice* device) {
    for (auto& transfer : transfers) {
        // Map source core index to DRAM core and get bank ID
        TT_FATAL(
            transfer.src_core_idx < dram_cores.size(),
            "Source core index {} exceeds available DRAM cores {}",
            transfer.src_core_idx,
            dram_cores.size());

        transfer.src_core_coord = dram_cores[transfer.src_core_idx];

        // Get bank ID for the DRAM core
        transfer.bank_id =
            device->allocator()->get_bank_ids_from_logical_core(dram_buffer_type, dram_cores[transfer.src_core_idx])[0];
    }
}

// Log transfer generation parameters and results
void log_transfer_generation_info(
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    uint32_t input_num_cores,
    uint32_t output_num_cores,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    const std::vector<BatchTransferInstruction>& instructions) {
    uint32_t hw_per_input_core = hw_total / input_num_cores;
    uint32_t bhw_total = batch_size * hw_total;
    uint32_t bhw_per_output_core = bhw_total / output_num_cores;

    log_info(
        tt::LogType::LogAlways,
        "generate_batch_redistribution_transfers: B={}, C={}, HW={}, input_cores={}, output_cores={}",
        batch_size,
        channels,
        hw_total,
        input_num_cores,
        output_num_cores);
    log_info(
        tt::LogType::LogAlways,
        "  hw_per_input_core={}, bhw_per_output_core={}",
        hw_per_input_core,
        bhw_per_output_core);

    // Log core coordinates for debugging
    log_info(tt::LogType::LogAlways, "Input cores:");
    for (size_t i = 0; i < input_cores.size(); i++) {
        log_info(tt::LogType::LogAlways, "  [{}]: ({}, {})", i, input_cores[i].x, input_cores[i].y);
    }
    log_info(tt::LogType::LogAlways, "Output cores:");
    for (size_t i = 0; i < output_cores.size(); i++) {
        log_info(tt::LogType::LogAlways, "  [{}]: ({}, {})", i, output_cores[i].x, output_cores[i].y);
    }

    log_info(tt::LogType::LogAlways, "Generated {} batch transfer instructions", instructions.size());
    for (size_t i = 0; i < instructions.size(); i++) {
        const auto& instr = instructions[i];
        log_info(
            tt::LogType::LogAlways,
            "  {}: src_core={}({},{}), dst_core={}({},{}), src_offset={}, dst_offset={}, size={}",
            i,
            instr.src_core_idx,
            instr.src_core_coord.x,
            instr.src_core_coord.y,
            instr.dst_core_idx,
            instr.dst_core_coord.x,
            instr.dst_core_coord.y,
            instr.src_offset,
            instr.dst_offset,
            instr.transfer_size);
    }
}

std::vector<BatchTransferInstruction> generate_batch_redistribution_transfers(
    uint32_t batch_size,
    uint32_t channels,
    uint32_t hw_total,
    const std::vector<CoreCoord>& input_cores,
    const std::vector<CoreCoord>& output_cores,
    uint32_t element_size_bytes,
    uint32_t padded_shard_width) {
    std::vector<BatchTransferInstruction> instructions;

    uint32_t input_num_cores = input_cores.size();
    uint32_t output_num_cores = output_cores.size();

    // For each output core, generate and optimize transfers
    for (uint32_t dst_core = 0; dst_core < output_num_cores; dst_core++) {
        // Generate individual transfers for this destination core
        auto transfers_by_src = generate_transfers_for_output_core(
            dst_core,
            batch_size,
            channels,
            hw_total,
            input_num_cores,
            output_num_cores,
            element_size_bytes,
            padded_shard_width);

        // Optimize transfers using batch-aware grouping
        auto core_instructions = optimize_transfers(
            transfers_by_src,
            dst_core,
            batch_size,
            channels,
            hw_total,
            input_num_cores,
            element_size_bytes,
            input_cores,
            output_cores,
            padded_shard_width);

        // Add to overall instruction list
        instructions.insert(instructions.end(), core_instructions.begin(), core_instructions.end());
    }

    // Log transfer generation information
    log_transfer_generation_info(
        batch_size, channels, hw_total, input_num_cores, output_num_cores, input_cores, output_cores, instructions);

    return instructions;
}

template <typename T>
std::vector<std::vector<T>> group_by_destination_core(const std::vector<T>& transfers, int num_output_cores) {
    std::vector<std::vector<T>> output(num_output_cores);
    for (const auto& transfer : transfers) {
        output[transfer.dst_core_idx].push_back(transfer);
    }
    // Ensure transfers for each destination core are ordered by destination offset
    // This guarantees segments are consumed in BHW order across source cores.
    for (auto& per_core_transfers : output) {
        std::sort(per_core_transfers.begin(), per_core_transfers.end(), [](const T& a, const T& b) {
            if (a.dst_offset == b.dst_offset) {
                // Stable tie-breaker to keep deterministic order across sources
                if (a.src_core_idx == b.src_core_idx) {
                    return a.src_offset < b.src_offset;
                }
                return a.src_core_idx < b.src_core_idx;
            }
            return a.dst_offset < b.dst_offset;
        });
    }
    return output;
}

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor) {
    const uint32_t element_size_bytes = input_tensor.element_size();
    const uint32_t l1_alignment_bytes = tt::tt_metal::hal::get_l1_alignment();
    return l1_alignment_bytes / element_size_bytes;
}

tt::tt_metal::operation::ProgramWithCallbacks multi_core_convert_to_hwc(const Tensor& a, Tensor& output) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // Create configuration from input tensors
    auto config = ConvertToHwcConfig::create_from_tensors(a, output);
    config.validate();

    // Use effective HW for gather transfers in uneven sharding cases
    uint32_t effective_hw_for_gather = calculate_effective_hw_for_sharding(
        config.hw_total, config.batch_size, config.l1_input_shard_width, config.l1_input_cores.size());
    const auto gather_transfers = convert_to_hwc::detail::precompute_gather_transfers(
        config.batch_size,
        config.input_channels,
        effective_hw_for_gather,
        config.l1_input_cores,
        config.l1_input_cores);  // TODO: L1 input grid can be different if resharding
    const uint32_t block_size_width = config.l1_input_shard_width * config.batch_size;  // Back to working configuration

    // Setup circular buffers after block_size_width is calculated
    auto cb_handles = setup_circular_buffers(program, config.l1_input_core_grid, config, a, output, block_size_width);
    const auto blocked_result = convert_to_hwc::detail::group_transfers_by_output_column_blocks_with_count(
        gather_transfers,
        config.batch_size,
        config.input_channels,
        effective_hw_for_gather,
        config.l1_input_cores,
        config.l1_input_cores.size(),
        a.element_size(),
        block_size_width);

    auto blocked_gather_transfers = std::move(blocked_result.blocked_transfers);
    const uint32_t num_blocks = blocked_result.num_logical_blocks;
    log_info(
        tt::LogType::LogAlways,
        "num_blocks={} (logical blocks), block_size_width={}, transfer_groups={}",
        num_blocks,
        block_size_width,
        blocked_gather_transfers.size());

    // Split transfers by destination core first, then apply coalescing per-core
    auto per_core_blocked_gather_transfers =
        convert_to_hwc::detail::split_by_destination_core(blocked_gather_transfers, config.l1_input_cores.size());

    // Apply transfer coalescing optimization to reduce NOC operations per-core
    for (auto& core_transfers : per_core_blocked_gather_transfers) {
        core_transfers = convert_to_hwc::detail::coalesce_contiguous_transfers(core_transfers);
    }

    // Create lambda for logical to worker core conversion
    auto logical_to_worker_core = [&a](const CoreCoord& logical_core) {
        return a.device()->worker_core_from_logical_core(logical_core);
    };

    // Serialize blocked transfer groups for each core
    std::vector<std::vector<uint32_t>> per_core_serialized_transfers(config.l1_input_cores.size());

    for (int core_idx = 0; core_idx < config.l1_input_cores.size(); core_idx++) {
        log_info(tt::LogType::LogAlways, "--- CORE {} ---", core_idx);
        const auto& core_transfers = per_core_blocked_gather_transfers[core_idx];

        for (const auto& blocked_gather_transfer : core_transfers) {
            log_info(tt::LogType::LogAlways, "Blocked groups: {}:", blocked_gather_transfer);
            for (const auto& transfer : blocked_gather_transfer.transfers) {
                log_info(tt::LogType::LogAlways, " - {}", transfer);
            }
        }

        // Serialize this core's blocked transfer groups
        per_core_serialized_transfers[core_idx] = convert_to_hwc::detail::serialize_blocked_transfer_groups(
            core_transfers, config.l1_input_cores, logical_to_worker_core);

        log_info(
            tt::LogType::LogAlways,
            "Core {} serialized {} uint32_t values: {}",
            core_idx,
            per_core_serialized_transfers[core_idx].size(),
            per_core_serialized_transfers[core_idx]);
    }

    // Generate transfer instructions using effective HW for uneven sharding
    uint32_t effective_hw_for_transfers = calculate_effective_hw_for_sharding(
        config.hw_total, config.batch_size, config.l1_input_shard_width, config.l1_input_cores.size());
    auto transfers = generate_batch_redistribution_transfers(
        config.batch_size,
        config.input_channels,
        effective_hw_for_transfers,
        config.l1_input_cores,
        config.l1_input_cores,
        config.element_size_bytes,
        config.l1_input_shard_width);

    // Calculate tiles based on block width (which becomes input to compute pipeline)
    const uint32_t total_tiles_per_block = tt::div_up(block_size_width, TILE_HEIGHT);  // assumes C < 32
    const uint32_t total_tiles_per_core = total_tiles_per_block * num_blocks;
    const uint32_t total_tiles_writer0 =
        tt::div_up(total_tiles_per_core, 2);  // each writer should process half of the output tiles
    const uint32_t total_tiles_writer1 = total_tiles_per_core - total_tiles_writer0;

    // Validation: ensure tiles divide evenly across blocks
    TT_FATAL(
        total_tiles_per_core % num_blocks == 0,
        "total_tiles_per_core={} must be divisible by num_blocks={}",
        total_tiles_per_core,
        num_blocks);
    uint32_t output_stride_sticks = TILE_WIDTH;

    // Update transfers with DRAM bank IDs if input is in DRAM
    if (config.is_input_in_dram) {
        populate_dram_bank_ids(transfers, config.dram_input_cores, config.remote_buffer_type, a.device());
    }

    const auto grouped_transfers = group_by_destination_core(transfers, config.l1_input_cores.size());

    // If there is only one HW tile we shouldn't stride the output copies because only one writer is working
    const uint32_t output_addr_stride =
        block_size_width != TILE_HEIGHT ? output_stride_sticks * config.output_shard_width * config.element_size_bytes
                                        : 0;

    // Writer kernel processes gather output blocks - height is the number of sticks per block
    const uint32_t num_sticks_block_size_kernel_0 = config.gather_l1_output_shard_height;
    // const uint32_t channel_size = config.output_shard_width * config.element_size_bytes;

    // block_size_bytes should be exactly the same as CB_IN_BATCH total size
    // Using the block-aligned dimensions for consistency
    const uint32_t block_size_bytes =
        config.gather_l1_output_shard_height * block_size_width * config.element_size_bytes;

    log_info(tt::LogType::LogAlways, "=== KERNEL COMPILE TIME ARGS ===");
    log_info(
        tt::LogType::LogAlways,
        "Total tiles per core: {} (block_width={} / TILE_HEIGHT={})",
        total_tiles_per_core,
        block_size_width,
        TILE_HEIGHT);
    log_info(
        tt::LogType::LogAlways,
        "Num blocks: {}, Tiles per block: {} (total_tiles_per_core / num_blocks)",
        num_blocks,
        total_tiles_per_block);
    log_info(tt::LogType::LogAlways, "Writer0 tiles: {}, Writer1 tiles: {}", total_tiles_writer0, total_tiles_writer1);
    log_info(
        tt::LogType::LogAlways,
        "Writer num_output_channels_padded arg: {} (output_shard_width - padded to min 8)",
        config.output_shard_width);
    log_info(
        tt::LogType::LogAlways,
        "Writer input_block_size_sticks_per_core: {} (gather_l1_output_shard_height)",
        num_sticks_block_size_kernel_0);
    log_info(
        tt::LogType::LogAlways, "Compute total_tiles_per_block: {} (tiles processed per block)", total_tiles_per_block);
    log_info(
        tt::LogType::LogAlways,
        "Compute total_sticks_per_block: {} (gather_l1_output_shard_height)",
        config.gather_l1_output_shard_height);
    log_info(
        tt::LogType::LogAlways,
        "Setting block_size_bytes={} (same as CB_IN_BATCH total size for alignment)",
        block_size_bytes);

    std::vector<uint32_t> writer_compile_time_args0 = {
        CBIndex::CB_IN,
        CBIndex::CB_IN_BATCH,
        CBIndex::CB_IN_TRANSPOSE_0,
        CBIndex::CB_OUT,
        config.output_shard_width,  // output channels (padded to minimum 8)
        total_tiles_writer0,
        output_stride_sticks,
        0,
        config.element_size_bytes,
        config.is_input_in_dram,
        true,  // is_reader - this writer kernel acts as the reader
        num_sticks_block_size_kernel_0,
        num_blocks,
        output_addr_stride,
        block_size_bytes};

    std::vector<uint32_t> writer_compile_time_args1 = {
        CBIndex::CB_IN,
        CBIndex::CB_IN_BATCH,
        CBIndex::CB_IN_TRANSPOSE_1,
        CBIndex::CB_OUT,
        config.output_shard_width,  // output channels (padded to minimum 8)
        total_tiles_writer1,
        output_stride_sticks,
        output_stride_sticks,
        config.element_size_bytes,
        config.is_input_in_dram,
        false,  // is_reader - this writer kernel does not read input
        0,      // num_sticks_block_size_kernel_1 - unused
        num_blocks,
        output_addr_stride,
        block_size_bytes};

    std::vector<uint32_t> compute_compile_time_args = {
        CBIndex::CB_IN_BATCH,
        CBIndex::CB_IN_TILED,
        CBIndex::CB_IN_TRANSPOSE_0,
        CBIndex::CB_IN_TRANSPOSE_1,
        total_tiles_per_block,                 // tiles per block, not total tiles
        config.gather_l1_output_shard_height,  // total_sticks_per_block - height of gather output
        num_blocks};

    auto writer_kernel_id0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        config.l1_input_core_grid,
        tt::tt_metal::ReaderDataMovementConfig(writer_compile_time_args0));

    auto writer_kernel_id1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        config.l1_input_core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args1));

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/convert_to_hwc.cpp",
        config.l1_input_core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    auto set_runtime_args =
        [&cb_handles, &config, grouped_transfers, per_core_serialized_transfers, writer_kernel_id0, writer_kernel_id1](
            tt::tt_metal::Program& program, const Tensor& a, const Tensor& output) {
            for (uint32_t core_idx = 0; core_idx < config.l1_input_cores.size(); core_idx++) {
                std::vector<uint32_t> runtime_args_0 = {config.remote_address};
                std::vector<uint32_t> runtime_args_1 = {config.remote_address};

                const auto& transfer_args = per_core_serialized_transfers.at(core_idx);
                runtime_args_0.insert(runtime_args_0.end(), transfer_args.begin(), transfer_args.end());
                runtime_args_1.insert(runtime_args_1.end(), transfer_args.begin(), transfer_args.end());

                /*
                const auto& args_for_all_segments = grouped_transfers.at(core_idx);
                runtime_args_0.push_back(args_for_all_segments.size());
                runtime_args_1.push_back(args_for_all_segments.size());

                for (const auto& args : args_for_all_segments) {
                    auto core = a.device()->worker_core_from_logical_core(args.src_core_coord);
                    // Always pass bank_id (0 for L1 transfers, actual bank_id for DRAM transfers)
                    const std::vector<uint32_t> segment_args = {
                        core.x, core.y, args.src_offset, args.dst_offset, args.transfer_size, args.bank_id};
                    runtime_args_0.insert(runtime_args_0.end(), segment_args.begin(), segment_args.end());
                    runtime_args_1.insert(runtime_args_1.end(), segment_args.begin(), segment_args.end());
                }
                */
                SetRuntimeArgs(program, writer_kernel_id0, config.l1_input_cores[core_idx], runtime_args_0);
                SetRuntimeArgs(program, writer_kernel_id1, config.l1_input_cores[core_idx], runtime_args_1);
            }
            // Only update input CB address for L1 input (DRAM input doesn't need CB update)
            if (!config.is_input_in_dram) {
                UpdateDynamicCircularBufferAddress(program, cb_handles.cb_in, *a.buffer());
            }
            UpdateDynamicCircularBufferAddress(program, cb_handles.cb_out, *output.buffer());
        };
    set_runtime_args(program, a, output);

    auto override_runtime_arguments_callback = [set_runtime_args](
                                                   const void* operation,
                                                   tt::tt_metal::Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& output_tensor = output_tensors.at(0);
        set_runtime_args(program, input_tensors.at(0), output_tensor);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::cnn::detail
