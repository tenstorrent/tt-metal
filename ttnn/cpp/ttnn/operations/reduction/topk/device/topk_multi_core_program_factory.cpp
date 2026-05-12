// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt_stl/assert.hpp"
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

#include <cmath>
#include <map>
#include <string>

using namespace tt::tt_metal;

namespace ttnn::prim {

/**
 * Core Configuration Utility
 *
 * Determines optimal work distribution across available cores while respecting hardware
 * constraints and memory limitations. Each core must process at least min_dim elements
 * (64 minimum for LLK efficiency), and total memory usage must fit within L1 constraints.
 *
 * @param width Input tensor width dimension
 * @param min_dim Minimum elements per core (64 for LLK compatibility)
 * @param max_dim Maximum elements per core (width/2 for load balancing)
 * @param k TopK value
 * @param core_range Available core grid
 * @param l1_size L1 memory size per core
 * @param value_tile_size Memory size of value tiles
 * @param index_tile_size Memory size of index tiles
 * @return Tuple of (num_cores, split_size, remainder, final_input_size, selected_x, selected_y)
 */
static inline std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t> cores_utilized(
    uint32_t width,
    uint32_t min_dim,
    uint32_t max_dim,
    uint32_t k,
    const CoreRange core_range,
    const uint32_t l1_size,
    const uint32_t value_tile_size,
    const uint32_t index_tile_size,
    uint32_t tile_width = 32) {
    const auto config_opt = find_topk_core_config(
        width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size, tile_width);
    if (config_opt.has_value()) {
        const auto& config = config_opt.value();
        return {
            config.num_cores + 1,
            config.split_size,
            config.rem,
            config.final_input_size,
            config.selected_x,
            config.selected_y};
    }
    const auto max_cores =
        (core_range.end_coord.y - core_range.start_coord.y - 1) * (core_range.end_coord.x - core_range.start_coord.x);
    return {max_cores + 1, width, 0, width * k, 0, 0};
}

tt::tt_metal::ProgramDescriptor TopKDeviceOperation::TopKMultiCoreProgramFactory::create_descriptor(
    const TopkParams& operation_attributes,
    const TopkInputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;
    // Tensor references
    const auto& input_tensor = tensor_args.input;
    const auto& input_indices_tensor = tensor_args.indices;
    const auto& value_tensor = std::get<0>(output_tensors);
    const auto& index_tensor = std::get<1>(output_tensors);

    ProgramDescriptor desc;

    // Data format configuration for all circular buffers
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    const tt::DataFormat index_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());
    const bool is32_bit_data = index_cb_data_format == tt::DataFormat::UInt32;

    // Use bf16 for compute intermediate buffers to avoid precision loss from bfp8/bfp4
    // shared-exponent grouping during sort (e.g. a single inf in a block makes all other
    // elements in that block encode to 0, corrupting the sort result).
    const tt::DataFormat compute_cb_data_format =
        (input_cb_data_format == tt::DataFormat::Bfp8_b || input_cb_data_format == tt::DataFormat::Bfp4_b)
            ? tt::DataFormat::Float16_b
            : input_cb_data_format;

    // Core grid and tile size calculations
    const auto first_core_range = args.sub_core_grids.ranges().at(0);
    const auto first_core_range_set = CoreRangeSet(first_core_range);

    const uint32_t input_tile_size = tile_size(input_cb_data_format);
    const uint32_t value_tile_size = tile_size(value_cb_data_format);
    const uint32_t index_tile_size = tile_size(index_cb_data_format);
    const uint32_t compute_tile_size = tile_size(compute_cb_data_format);

    // DRAM buffer pointers for kernel runtime arguments
    auto* const input_buffer = input_tensor.buffer();
    auto* const values_buffer = value_tensor.buffer();
    auto* const index_buffer = index_tensor.buffer();
    auto* const input_indices_buffer = input_indices_tensor.has_value() ? input_indices_tensor->buffer() : nullptr;

    const auto* device = input_tensor.device();

    const auto input_shape = input_tensor.padded_shape();
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;

    // Determine optimal core configuration based on input dimensions, K value, and memory constraints
    const auto& [num_cores, local_topk_input_size, rem, final_topk_input_size, selected_x, selected_y] = cores_utilized(
        input_shape[args.dim],       // Total width dimension
        64,                          // Minimum elements per core (LLK requirement)
        input_shape[args.dim] / 2,   // Maximum elements per core (load balancing)
        args.k,                      // TopK value
        first_core_range,            // Available core grid
        device->l1_size_per_core(),  // L1 memory per core
        value_tile_size,             // Value tile memory footprint
        index_tile_size,             // Index tile memory footprint
        tile_width);

    constexpr bool select_cores_row_wise = false;

    // Configure local processing cores (handle width chunks)
    auto local_cores_range =
        select_contiguous_range_from_corerangeset(first_core_range_set, selected_x - 1, selected_y - 1);
    TT_FATAL(local_cores_range.has_value(), "Failed to select local cores range");

    auto local_cores_range_set = CoreRangeSet(local_cores_range.value());
    auto local_cores =
        corerange_to_cores(local_cores_range_set, local_cores_range_set.num_cores(), select_cores_row_wise);

    // Configure final aggregation core (handles global TopK computation)
    auto final_cores_range_set =
        select_from_corerangeset(first_core_range_set, selected_y, selected_y, select_cores_row_wise);
    auto final_core = corerange_to_cores(final_cores_range_set, 1u, select_cores_row_wise).at(0);

    // Combined core set for shared circular buffer allocation
    auto all_cores_range_set = local_cores_range_set;
    all_cores_range_set = all_cores_range_set.merge(final_cores_range_set);

    // Calculate processing dimensions in tile units
    const uint32_t Wt_local = local_topk_input_size / tile_width;  // Width tiles per local core
    const uint32_t Wt_final = final_topk_input_size / tile_width;  // Total width tiles for final core
    const uint32_t Kt = args.k % tile_width == 0 ? args.k / tile_width : (args.k / tile_width) + 1;  // TopK in tiles

    const uint32_t num_cb_unit = 2;                // Base buffering unit
    const uint32_t cb_in_units = 2 * num_cb_unit;  // 4 units total for double-buffered input

    // ==================================================================================
    // CIRCULAR BUFFER ALLOCATION
    //
    // Allocation order matters: allocate_circular_buffers() processes CBs sequentially
    // and assigns each CB the MAX address across its core ranges. To avoid L1 gaps,
    // shared CBs (all_cores) must be allocated BEFORE core-specific CBs.
    //
    // Layout after allocation:
    //   Local cores:  CB0 → CB1 → CB4 → CB5 → CB8 → CB9 → CB2 → CB3
    //   Final core:   CB0 → CB1 → CB4 → CB5 → CB8 → CB9 → CB6 → CB7
    // ==================================================================================
    // Input values (double-buffered for continuous DRAM streaming)
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * value_tile_size,
        .core_ranges = all_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_tile_size,
        }}},
    });

    // Input indices (double-buffered, generated on-demand or read from DRAM)
    constexpr uint32_t index_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_in_units * index_tile_size,
        .core_ranges = all_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Gathered values (aggregation buffer for final core).
    // Uses compute_cb_data_format (bf16 when input is bfp8/bfp4): the local cores write
    // tiles in transposed layout where each tile row mixes values from different H positions
    // (e.g. [normal_H0, INF_H1, ..., INF_H31]). If stored as bfp8 the shared-exponent
    // block is dominated by INF, zeroing out H=0's value. Keeping bf16 here and in
    // values_cb_index (local) avoids that precision loss for the inter-core transfer.
    constexpr uint32_t gathered_values_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt_final * compute_tile_size,
        .core_ranges = all_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(gathered_values_cb_index),
            .data_format = compute_cb_data_format,
            .page_size = compute_tile_size,
        }}},
    });

    // Gathered indices (aggregation buffer for final core)
    constexpr uint32_t gathered_indices_cb_index = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt_final * index_tile_size,
        .core_ranges = all_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(gathered_indices_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Local TopK indices output (local cores → writer → final core)
    constexpr uint32_t output_ind_cb_index = tt::CBIndex::c_9;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * index_tile_size,
        .core_ranges = all_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_ind_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Transposed values (single-buffered for in-place bitonic operations)
    // Holds all Wt_local tiles for complete width chunk processing.
    // Uses bf16 when input is bfp8/bfp4 to avoid precision loss from shared-exponent
    // grouping during the sort (inf in one slot zeroes out its block-mates).
    constexpr uint32_t input_transposed_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt_local * compute_tile_size,
        .core_ranges = local_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_transposed_cb_index),
            .data_format = compute_cb_data_format,
            .page_size = compute_tile_size,
        }}},
    });

    // Transposed indices (single-buffered for in-place bitonic operations)
    constexpr uint32_t index_transposed_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt_local * index_tile_size,
        .core_ranges = local_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_transposed_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Local TopK values output — split format between local and final cores.
    //
    // Local cores (bf16): the sorted Kt tile in transposed layout has rows of the form
    // [normal_H0, INF_H1, ..., INF_H31]. Packing such a row to bfp8 makes the
    // shared-exponent INF-dominated, reducing H=0's value to 0. Using bf16 here
    // preserves all values through the NoC transfer to the final core's c_4 buffer
    // (also bf16, same tile size, so the raw-byte NoC copy is format-consistent).
    //
    // Final core (bfp8): after the final merge and transpose-back, the tile rows are
    // per-H-row (H=0 row has only normal values, no INF mixing), so bfp8 quantisation
    // is safe. bfp8 also matches the output tensor dtype for the DRAM write.
    constexpr uint32_t values_cb_index = tt::CBIndex::c_8;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * compute_tile_size,
        .core_ranges = local_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(values_cb_index),
            .data_format = compute_cb_data_format,
            .page_size = compute_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cb_unit * value_tile_size,
        .core_ranges = final_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(values_cb_index),
            .data_format = value_cb_data_format,
            .page_size = value_tile_size,
        }}},
    });

    // Final values (staging buffer for final compute output).
    // Uses bf16 when input is bfp8/bfp4 so that the final bitonic merge operates at
    // higher precision (same rationale as input_transposed_cb_index above).
    constexpr uint32_t final_values_cb_index = tt::CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt_final * compute_tile_size,
        .core_ranges = final_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(final_values_cb_index),
            .data_format = compute_cb_data_format,
            .page_size = compute_tile_size,
        }}},
    });

    // Final indices (staging buffer for final compute output)
    constexpr uint32_t final_indices_cb_index = tt::CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Wt_final * index_tile_size,
        .core_ranges = final_cores_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(final_indices_cb_index),
            .data_format = index_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Semaphore-based flow control for coordinating data transfer between local and final cores
    const uint32_t sender_semaphore_id = 0;    // Tracks data transmission completion
    const uint32_t receiver_semaphore_id = 1;  // Signals readiness to receive data
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sender_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores_range_set,
        .initial_value = INVALID,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = receiver_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_cores_range_set,
        .initial_value = INVALID,
    });

    // Local reader - Data Input and Index Generation/Reading
    // Responsibility: Stream input tensor data from DRAM to local cores
    // Two variants: generate indices on-demand or read pre-existing index tensor
    std::vector<uint32_t> reader_local_compile_time_args = {
        input_cb_index,                // CB0: Input values destination
        index_cb_index,                // CB1: Input indices destination
        Ht,                            // Height tiles in tensor
        Wt_local,                      // Width tiles per local core
        input_shape[-1] / tile_width,  // Total width tiles (Wt)
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_local_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_indices_buffer).append_to(reader_local_compile_time_args);
    const std::map<std::string, std::string> reader_specialization_defines_map = {
        {"GENERATE_INDICES", tensor_args.indices.has_value() ? "0" : "1"},
    };
    KernelDescriptor::Defines reader_local_defines(
        reader_specialization_defines_map.begin(), reader_specialization_defines_map.end());

    KernelDescriptor reader_local_desc;
    reader_local_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk.cpp";
    reader_local_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_local_desc.core_ranges = local_cores_range_set;  // Runs on all local processing cores
    reader_local_desc.compile_time_args = std::move(reader_local_compile_time_args);
    reader_local_desc.defines = std::move(reader_local_defines);
    reader_local_desc.config = ReaderConfigDescriptor{};

    // Final reader - Local TopK Results Aggregation Coordinator
    // Responsibility: Coordinate reception of TopK results from all local cores
    // Uses semaphore protocol to synchronize with multiple sender cores
    CoreCoord local_cores_physical_start = device->worker_core_from_logical_core(local_cores.at(0));
    CoreCoord local_cores_physical_end = device->worker_core_from_logical_core(local_cores.at(num_cores - 2u));
    const std::vector<uint32_t> reader_final_compile_time_args = {
        static_cast<std::uint32_t>(receiver_semaphore_id),         // Semaphore for coordinating data reception
        static_cast<std::uint32_t>(sender_semaphore_id),           // Semaphore for tracking transmission completion
        static_cast<std::uint32_t>(local_cores_physical_start.x),  // NoC coordinates of local core range
        static_cast<std::uint32_t>(local_cores_physical_start.y),
        static_cast<std::uint32_t>(local_cores_physical_end.x),
        static_cast<std::uint32_t>(local_cores_physical_end.y),
        static_cast<std::uint32_t>(Ht),             // Height tiles to process
        static_cast<std::uint32_t>(Wt_final),       // Total aggregated width tiles
        static_cast<std::uint32_t>(num_cores - 1),  // Number of local cores sending data
        gathered_values_cb_index,                   // Final TopK values destination
        gathered_indices_cb_index                   // Final TopK indices destination
    };

    KernelDescriptor reader_final_desc;
    reader_final_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp";
    reader_final_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_final_desc.core_ranges = final_cores_range_set;  // Runs only on final aggregation core
    reader_final_desc.compile_time_args = reader_final_compile_time_args;
    reader_final_desc.config = ReaderConfigDescriptor{};

    // Local writer - Local TopK Results Transmission
    // Responsibility: Send local TopK results from each core to final aggregation core
    // Implements sender side of semaphore-based synchronization protocol
    const CoreCoord final_cores_physical = device->worker_core_from_logical_core(final_core);
    const std::vector<uint32_t> writer_local_compile_time_args = {
        static_cast<std::uint32_t>(receiver_semaphore_id),   // Semaphore to check final core readiness
        static_cast<std::uint32_t>(sender_semaphore_id),     // Semaphore to signal transmission completion
        static_cast<std::uint32_t>(final_cores_physical.x),  // Target final core NoC coordinates
        static_cast<std::uint32_t>(final_cores_physical.y),
        Ht,                          // Height tiles to send
        args.k,                      // TopK value
        Kt,                          // TopK in tile units
        values_cb_index,             // Local TopK values source
        output_ind_cb_index,         // Local TopK indices source
        gathered_values_cb_index,    // Final TopK values destination
        gathered_indices_cb_index};  // Final TopK indices destination

    KernelDescriptor writer_local_desc;
    writer_local_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp";
    writer_local_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_local_desc.core_ranges = local_cores_range_set;  // Runs on all local processing cores
    writer_local_desc.compile_time_args = writer_local_compile_time_args;
    writer_local_desc.config = WriterConfigDescriptor{};

    // Final writer - Global TopK Results Output to DRAM
    // Responsibility: Write final globally optimal TopK results to output tensors
    // Handles proper interleaved tensor formatting for host consumption
    std::vector<uint32_t> writer_final_compile_time_args = {
        values_cb_index,      // Final TopK values source
        output_ind_cb_index,  // Final TopK indices source
        Ht,                   // Height tiles to write
        Kt                    // TopK tiles per height row
    };
    tt::tt_metal::TensorAccessorArgs(values_buffer).append_to(writer_final_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(index_buffer).append_to(writer_final_compile_time_args);

    KernelDescriptor writer_final_desc;
    writer_final_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_final_topk.cpp";
    writer_final_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_final_desc.core_ranges = final_cores_range_set;  // Runs only on final aggregation core
    writer_final_desc.compile_time_args = std::move(writer_final_compile_time_args);
    writer_final_desc.config = WriterConfigDescriptor{};

    // Local compute - Local Width Chunk Bitonic Sorting
    // Responsibility: Perform bitonic sort on assigned width chunk to extract local TopK
    // Uses iterative divide-and-conquer with log(Wt_local) merge phases
    const std::vector<uint32_t> compute_args = {
        input_cb_index,                                   // CB0: Input values stream
        index_cb_index,                                   // CB1: Input indices stream
        input_transposed_cb_index,                        // CB24: Transposed values workspace
        index_transposed_cb_index,                        // CB25: Transposed indices workspace
        values_cb_index,                                  // CB16: Local TopK values output
        output_ind_cb_index,                              // CB17: Local TopK indices output
        Ht,                                               // Height tiles to process
        Wt_local,                                         // Width tiles per local core
        args.k,                                           // TopK value
        Kt,                                               // TopK in tile units
        static_cast<std::uint32_t>(std::log2(args.k)),    // log2(K) for bitonic network depth
        static_cast<std::uint32_t>(std::log2(Wt_local)),  // log2(width) for merge iterations
        static_cast<std::uint32_t>(args.largest),         // Sort direction (largest=1, smallest=0)
        static_cast<std::uint32_t>(args.sorted),          // Output sorting requirement
    };

    KernelDescriptor compute_local_desc;
    compute_local_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_local.cpp";
    compute_local_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_local_desc.core_ranges = local_cores_range_set;  // Runs on all local processing cores
    compute_local_desc.compile_time_args = compute_args;
    compute_local_desc.config = ComputeConfigDescriptor{
        .dst_full_sync_en = false,
    };

    // Final compute - Global TopK Bitonic Merge
    // Responsibility: Perform final bitonic merge of all local TopK results
    // Produces globally optimal TopK from aggregated local results
    const std::vector<uint32_t> compute_args_final = {
        gathered_values_cb_index,                         // CB26: Aggregated local TopK values
        gathered_indices_cb_index,                        // CB27: Aggregated local TopK indices
        final_values_cb_index,                            // CB28: Final processing workspace values
        final_indices_cb_index,                           // CB29: Final processing workspace indices
        values_cb_index,                                  // CB16: Global TopK values output
        output_ind_cb_index,                              // CB17: Global TopK indices output
        Ht,                                               // Height tiles to process
        Wt_final,                                         // Total aggregated width tiles
        args.k,                                           // TopK value
        Kt,                                               // TopK in tile units
        static_cast<std::uint32_t>(std::log2(args.k)),    // log2(K) for bitonic network depth
        static_cast<std::uint32_t>(std::log2(Wt_final)),  // log2(final_width) for merge iterations
        static_cast<std::uint32_t>(args.largest),         // Sort direction (largest=1, smallest=0)
        static_cast<std::uint32_t>(args.sorted),          // Output sorting requirement
    };

    KernelDescriptor compute_final_desc;
    compute_final_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp";
    compute_final_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_final_desc.core_ranges = final_cores_range_set;  // Runs only on final aggregation core
    compute_final_desc.compile_time_args = compute_args_final;
    compute_final_desc.config = ComputeConfigDescriptor{
        .dst_full_sync_en = false,
    };

    uint32_t core_id = 0;            // Width offset counter for core assignment
    bool ascending = !args.largest;  // Initial sort direction for bitonic properties

    // Configure runtime arguments for each local processing core
    for (auto core : local_cores) {
        // Local reader
        reader_local_desc.emplace_runtime_args(
            core,
            {
                input_buffer,                          // DRAM address of input values tensor
                0u,                                    // Height offset (no height parallelism currently)
                core_id * Wt_local,                    // Width offset for this core's chunk
                static_cast<uint32_t>(is32_bit_data),  // Flag indicating if data is 32-bit
                input_indices_tensor.has_value() ? input_indices_buffer->address()
                                                 : 0u,  // DRAM address of input indices tensor (if provided)
            });

        // Local writer
        writer_local_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                core_id,  // Width position for placement in final aggregation buffer
            });

        // Local compute
        compute_local_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                static_cast<uint32_t>(ascending),  // Sort direction for bitonic properties
            });

        core_id++;               // Advance to next width chunk
        ascending = !ascending;  // Alternate sort direction for bitonic sequence
    }

    // Final writer
    writer_final_desc.emplace_runtime_args(
        final_core,
        {
            values_buffer,  // DRAM address for TopK values output tensor
            index_buffer,   // DRAM address for TopK indices output tensor
        });

    desc.kernels.push_back(std::move(reader_local_desc));
    desc.kernels.push_back(std::move(reader_final_desc));
    desc.kernels.push_back(std::move(writer_local_desc));
    desc.kernels.push_back(std::move(writer_final_desc));
    desc.kernels.push_back(std::move(compute_local_desc));
    desc.kernels.push_back(std::move(compute_final_desc));

    return desc;
}

}  // namespace ttnn::prim
