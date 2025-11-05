// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/hal.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

namespace ttnn::operations::experimental::cnn::detail {

using namespace tt::constants;

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
    const Tensor& output);

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
    config.l1_input_shard_width = config.is_input_in_dram ? config.output_shard_height : input.shard_spec()->shape[1];

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

    // Alignment requirements
    config.alignment_elements = compute_alignment_requirement_in_elements(output);

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
    const Tensor& output) {
    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    // CB in (full input)
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

    // CB in batch
    const uint32_t cb_in_batch_page_size = config.l1_input_shard_width * config.element_size_bytes;
    const uint32_t cb_in_batch_total_size = (config.l1_input_shard_height / config.batch_size) * cb_in_batch_page_size;
    create_circular_buffer(
        program, core_grid, CBIndex::CB_IN_BATCH, cb_in_batch_total_size, cb_in_batch_page_size, config.input_format);

    // CB in tiled
    const uint32_t cb_in_tiled_total_size =
        tt::div_up(config.l1_input_shard_width, TILE_WIDTH) * intermediary_tile_size;
    const uint32_t cb_in_tiled_page_size = intermediary_tile_size;
    create_circular_buffer(
        program, core_grid, CBIndex::CB_IN_TILED, cb_in_tiled_total_size, cb_in_tiled_page_size, intermediary_format);

    // CB in transpose buffers
    const uint32_t cb_in_transpose_total_size =
        tt::div_up(config.l1_input_shard_width, TILE_WIDTH) * intermediary_tile_size;
    const uint32_t cb_in_transpose_page_size = intermediary_tile_size;
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
    const uint32_t cb_out_total_size = cb_in_total_size;  // same size as input
    const uint32_t cb_out_page_size = config.l1_input_shard_height * config.element_size_bytes;
    auto cb_out = create_circular_buffer(
        program, core_grid, CBIndex::CB_OUT, cb_out_total_size, cb_out_page_size, config.input_format, output.buffer());

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
    uint32_t element_size_bytes) {
    std::map<uint32_t, std::vector<TransferData>> transfers_by_src;

    uint32_t hw_per_input_core = hw_total / input_num_cores;
    uint32_t bhw_total = batch_size * hw_total;
    uint32_t bhw_per_output_core = bhw_total / output_num_cores;

    // Calculate BHW range this output core handles
    uint32_t dst_bhw_start = dst_core * bhw_per_output_core;
    uint32_t dst_bhw_end = std::min(dst_bhw_start + bhw_per_output_core, bhw_total);

    for (uint32_t bhw_idx = dst_bhw_start; bhw_idx < dst_bhw_end; bhw_idx++) {
        // Convert BHW index back to (batch_id, hw_idx)
        uint32_t batch_id = bhw_idx / hw_total;
        uint32_t hw_idx = bhw_idx % hw_total;

        // Find which input core has this hw_idx data
        uint32_t src_core = hw_idx / hw_per_input_core;
        uint32_t src_hw_offset = hw_idx % hw_per_input_core;

        // Calculate source offset: [B*C, HW_per_input_core] layout (interleaved batch-channel)
        // For batch_id and hw_offset, we need to find the position in the flattened [B*C, HW] layout
        // Each batch occupies 'channels' consecutive rows starting at batch_id * channels
        uint32_t src_offset = (batch_id * channels * hw_per_input_core + src_hw_offset * channels) * element_size_bytes;

        // Calculate destination offset: [1, C, BHW_per_output_core] layout
        uint32_t dst_bhw_offset = bhw_idx - dst_bhw_start;
        uint32_t dst_offset = dst_bhw_offset * channels * element_size_bytes;

        // Group by source core
        if (transfers_by_src.find(src_core) == transfers_by_src.end()) {
            transfers_by_src[src_core] = std::vector<TransferData>();
        }

        transfers_by_src[src_core].emplace_back(src_offset, dst_offset, channels * element_size_bytes);
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
    const std::vector<CoreCoord>& output_cores) {
    std::vector<BatchTransferInstruction> instructions;
    uint32_t hw_per_input_core = hw_total / input_num_cores;

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
        uint32_t batch_size_bytes = channels * hw_per_input_core * element_size_bytes;

        for (const TransferData& transfer : sorted_transfers) {
            // Determine which batch this transfer belongs to
            // The src_offset was calculated as: batch_id * channels * hw_per_input_core + src_hw_offset * channels
            // Let's use the original calculation that worked before for now
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
    uint32_t element_size_bytes) {
    std::vector<BatchTransferInstruction> instructions;

    uint32_t input_num_cores = input_cores.size();
    uint32_t output_num_cores = output_cores.size();

    // For each output core, generate and optimize transfers
    for (uint32_t dst_core = 0; dst_core < output_num_cores; dst_core++) {
        // Generate individual transfers for this destination core
        auto transfers_by_src = generate_transfers_for_output_core(
            dst_core, batch_size, channels, hw_total, input_num_cores, output_num_cores, element_size_bytes);

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
            output_cores);

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

    // Setup circular buffers
    auto cb_handles = setup_circular_buffers(program, config.l1_input_core_grid, config, a, output);

    // Generate transfer instructions
    auto transfers = generate_batch_redistribution_transfers(
        config.batch_size,
        config.input_channels,
        config.hw_total,
        config.l1_input_cores,
        config.l1_input_cores,
        config.element_size_bytes);

    const uint32_t total_tiles_per_core = tt::div_up(config.l1_input_shard_width, TILE_HEIGHT);
    const uint32_t total_tiles_writer0 = tt::div_up(total_tiles_per_core, 2);
    const uint32_t total_tiles_writer1 = total_tiles_per_core - total_tiles_writer0;
    uint32_t output_stride_sticks = TILE_WIDTH;

    // Update transfers with DRAM bank IDs if input is in DRAM
    if (config.is_input_in_dram) {
        populate_dram_bank_ids(transfers, config.dram_input_cores, config.remote_buffer_type, a.device());
    }

    const auto grouped_transfers = group_by_destination_core(transfers, config.l1_input_cores.size());

    // If there is only one HW tile we shouldn't stride the output copies because only one writer is working
    const uint32_t output_addr_stride =
        config.l1_input_shard_width != TILE_HEIGHT
            ? output_stride_sticks * config.output_shard_width * config.element_size_bytes
            : 0;

    const uint32_t num_sticks_block_size_kernel_0 = (config.l1_input_shard_height / config.batch_size);

    std::vector<uint32_t> writer_compile_time_args0 = {
        CBIndex::CB_IN,
        CBIndex::CB_IN_BATCH,
        CBIndex::CB_IN_TRANSPOSE_0,
        CBIndex::CB_OUT,
        config.output_shard_width,  // output channels
        total_tiles_writer0,
        output_stride_sticks,
        0,
        config.element_size_bytes,
        config.is_input_in_dram,
        true,  // is_reader - this writer kernel acts as the reader
        num_sticks_block_size_kernel_0,
        config.batch_size,
        output_addr_stride};

    std::vector<uint32_t> writer_compile_time_args1 = {
        CBIndex::CB_IN,
        CBIndex::CB_IN_BATCH,
        CBIndex::CB_IN_TRANSPOSE_1,
        CBIndex::CB_OUT,
        config.output_shard_width,  // output channels
        total_tiles_writer1,
        output_stride_sticks,
        output_stride_sticks,
        config.element_size_bytes,
        config.is_input_in_dram,
        false,  // is_reader - this writer kernel does not read input
        0,      // num_sticks_block_size_kernel_1 - unused
        config.batch_size,
        output_addr_stride};

    std::vector<uint32_t> compute_compile_time_args = {
        CBIndex::CB_IN_BATCH,
        CBIndex::CB_IN_TILED,
        CBIndex::CB_IN_TRANSPOSE_0,
        CBIndex::CB_IN_TRANSPOSE_1,
        total_tiles_per_core,
        config.l1_input_shard_height / config.batch_size,
        config.batch_size};

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

    auto set_runtime_args = [&cb_handles, &config, grouped_transfers, writer_kernel_id0, writer_kernel_id1](
                                tt::tt_metal::Program& program, const Tensor& a, const Tensor& output) {
        for (uint32_t core_idx = 0; core_idx < config.l1_input_cores.size(); core_idx++) {
            std::vector<uint32_t> runtime_args_0 = {config.remote_address};
            std::vector<uint32_t> runtime_args_1 = {config.remote_address};

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
