// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

#include <cmath>
#include <iostream>

using namespace tt::tt_metal;
using namespace std;

namespace ttnn::prim {

/**
 * ================================================================================================
 * TOPK MULTICORE PROGRAM FACTORY - DIVIDE-AND-CONQUER ARCHITECTURE
 * ================================================================================================
 *
 * This factory implements a high-performance multicore TopK algorithm that leverages width-based
 * parallelization with bitonic sorting for optimal hardware utilization on Tenstorrent devices.
 *
 * ALGORITHM OVERVIEW:
 * The multicore TopK uses a two-stage divide-and-conquer approach:
 * 1. LOCAL PROCESSING: Multiple cores process disjoint width chunks independently
 * 2. GLOBAL AGGREGATION: Single core merges all local results to find global TopK
 *
 * CORE ARCHITECTURE:
 * - Local Cores (N-1): Each processes Wt_local consecutive width tiles using bitonic sort
 * - Final Core (1): Aggregates N-1 local TopK results and computes global TopK
 * - Communication: Semaphore-synchronized NoC transfers between local and final cores
 *
 * KERNEL PIPELINE OVERVIEW:
 *
 * ┌─────────────────────────────────────────────────────────────────────────────────────┐
 * │                              LOCAL PROCESSING CORES                                 │
 * │                                                                                     │
 * │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                   │
 * │  │ Reader (Local)  │  │ Compute (Local) │  │ Writer (Local)  │                   │
 * │  │                 │  │                 │  │                 │                   │
 * │  │ • Read width    │  │ • Bitonic sort  │  │ • Send TopK to  │                   │
 * │  │   chunk from    │→ │   of Wt_local   │→ │   final core    │ ──┐               │
 * │  │   DRAM          │  │   tiles         │  │ • Semaphore     │   │               │
 * │  │ • Generate      │  │ • Extract       │  │   coordination  │   │               │
 * │  │   indices       │  │   local TopK    │  │                 │   │               │
 * │  └─────────────────┘  └─────────────────┘  └─────────────────┘   │               │
 * └─────────────────────────────────────────────────────────────────┼───────────────┘
 *                                                                     │
 *                                                                     │ NoC
 *                                                                     │ Transfer
 * ┌─────────────────────────────────────────────────────────────────┼───────────────┐
 * │                              FINAL AGGREGATION CORE              ↓               │
 * │                                                                                   │
 * │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
 * │  │ Reader (Final)  │  │ Compute (Final) │  │ Writer (Final)  │                 │
 * │  │                 │  │                 │  │                 │                 │
 * │  │ • Collect all   │  │ • Bitonic merge │  │ • Write final   │                 │
 * │  │   local TopK    │→ │   across all    │→ │   TopK to DRAM  │                 │
 * │  │ • Semaphore     │  │   cores' results│  │ • Interleaved   │                 │
 * │  │   coordination  │  │ • Global TopK   │  │   format        │                 │
 * │  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
 * └─────────────────────────────────────────────────────────────────────────────────┘
 *
 * DETAILED KERNEL DESCRIPTIONS:
 *
 * 1. READER KERNELS:
 *    - reader_create_index_local_topk.cpp: Streams input data + generates position indices
 *    - reader_read_index_local_topk.cpp: Streams input data + pre-existing indices
 *    - reader_final_topk.cpp: Coordinates reception of local TopK results
 *
 * 2. COMPUTE KERNELS:
 *    - topk_local.cpp: Bitonic sort on local width chunks, outputs local TopK
 *    - topk_final.cpp: Bitonic merge of all local TopK results, outputs global TopK
 *
 * 3. WRITER KERNELS:
 *    - writer_local_topk.cpp: Transmits local TopK results to final core via NoC
 *    - writer_final_topk.cpp: Writes final global TopK results to DRAM
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Scalability: O(width/cores + log(cores)) - near-linear scaling with core count
 * - Memory: O(K + width_chunk_size) per core - efficient memory utilization
 * - Bandwidth: Optimized DRAM access patterns and minimal inter-core communication
 *
 * CONFIGURATION STRATEGY:
 * The factory automatically determines optimal core allocation based on:
 * - K value and memory constraints
 * - Available core grid geometry
 * - Hardware limits (minimum 64 elements per core for LLK efficiency)
 */

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
    const uint32_t index_tile_size) {
    auto config_opt =
        find_topk_core_config(width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size);
    if (config_opt.has_value()) {
        auto config = config_opt.value();
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

/**
 * TopK Multicore Program Factory - Main Implementation
 *
 * This function orchestrates the complete multicore TopK pipeline by:
 * 1. Analyzing input dimensions and determining optimal core allocation
 * 2. Setting up circular buffer network for data flow management
 * 3. Creating and configuring all reader, compute, and writer kernels
 * 4. Establishing semaphore-based synchronization between cores
 * 5. Managing runtime arguments for dynamic work assignment
 *
 * CIRCULAR BUFFER ARCHITECTURE:
 *
 * Local Cores Buffer Layout:
 * ┌─────────────────────────────────────────────────────────────┐
 * │ CB0,1: Input (Double-buffered)     │ CB24,25: Transposed     │
 * │ CB16,17: Local TopK Output         │ CB26,27: Aggregation    │
 * │ CB28,29: Final Results             │ (Final core only)       │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Data Flow:
 * DRAM → CB0,1 → Compute → CB24,25 → CB16,17 → NoC → CB26,27 → CB28,29 → DRAM
 *
 * SYNCHRONIZATION PROTOCOL:
 * - Semaphores coordinate between local cores and final aggregation core
 * - Flow control prevents buffer overflow during inter-core transfers
 * - Multicast operations enable efficient coordination across multiple senders
 *
 * CORE WORK ASSIGNMENT:
 * - Dynamic load balancing based on input dimensions and available cores
 * - Each local core processes Wt_local consecutive width tiles
 * - Final core processes aggregated results from all local cores
 */
TopKMultiCoreProgramFactory::cached_program_t TopKMultiCoreProgramFactory::create(
    const TopkParams& args, const TopkInputs& tensor_args, std::tuple<Tensor, Tensor>& output_tensors) {
    using namespace tt::constants;

    // ===============================================================================
    // STEP 1: TENSOR AND DEVICE CONFIGURATION
    // ===============================================================================

    const auto& input_tensor = tensor_args.input;
    const auto& input_indices_tensor = tensor_args.indices;
    const auto& value_tensor = std::get<0>(output_tensors);
    const auto& index_tensor = std::get<1>(output_tensors);

    tt::tt_metal::Program program{};

    // Data format configuration for all circular buffers
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    tt::DataFormat index_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    // Core grid and tile size calculations
    auto first_core_range = args.sub_core_grids.ranges().at(0);
    auto first_core_range_set = CoreRangeSet(first_core_range);

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    // DRAM buffer pointers for kernel runtime arguments
    auto* input_buffer = input_tensor.buffer();
    auto* values_buffer = value_tensor.buffer();
    auto* index_buffer = index_tensor.buffer();
    auto* input_indices_buffer = input_indices_tensor.has_value() ? input_indices_tensor->buffer() : nullptr;

    auto* device = input_tensor.device();

    // ===============================================================================
    // STEP 2: OPTIMAL CORE ALLOCATION AND WORK DISTRIBUTION
    // ===============================================================================

    auto input_shape = input_tensor.padded_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;

    // Determine optimal core configuration based on input dimensions, K value, and memory constraints
    const auto& [num_cores, local_topk_input_size, rem, final_topk_input_size, selected_x, selected_y] = cores_utilized(
        input_shape[args.dim],       // Total width dimension
        64,                          // Minimum elements per core (LLK requirement)
        input_shape[args.dim] / 2,   // Maximum elements per core (load balancing)
        args.k,                      // TopK value
        first_core_range,            // Available core grid
        device->l1_size_per_core(),  // L1 memory per core
        value_tile_size,             // Value tile memory footprint
        index_tile_size);            // Index tile memory footprint

    constexpr bool select_cores_row_wise = false;

    // Configure local processing cores (handle width chunks)
    auto local_cores_range =
        select_contiguous_range_from_corerangeset(first_core_range_set, selected_x - 1, selected_y - 1);
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
    uint32_t Wt_local = local_topk_input_size / TILE_WIDTH;  // Width tiles per local core
    uint32_t Wt_final = final_topk_input_size / TILE_WIDTH;  // Total width tiles for final core
    uint32_t Kt = args.k % TILE_WIDTH == 0 ? args.k / TILE_WIDTH : (args.k / TILE_WIDTH) + 1;  // TopK in tiles

    // ===============================================================================
    // STEP 3: CIRCULAR BUFFER NETWORK SETUP
    // ===============================================================================
    //
    // The circular buffer architecture implements a multi-stage pipeline:
    // Input Buffers → Compute Buffers → Aggregation Buffers → Output Buffers
    //
    // Buffer Sizing Strategy:
    // - Double buffering for input streams to hide DRAM latency
    // - Single buffering for intermediate results with in-place operations
    // - Sized to minimize memory footprint while avoiding pipeline stalls

    uint32_t num_cb_unit = 2;                // Base buffering unit
    uint32_t cb_in_units = 2 * num_cb_unit;  // 4 units total for double-buffered input
    // CB0: Input values (double-buffered for continuous DRAM streaming)
    uint32_t input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * value_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_cb_config);

    // CB1: Input indices (double-buffered, generated on-demand or read from DRAM)
    uint32_t index_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_input_intermed0_config);

    // CB24: Transposed values (single-buffered for in-place bitonic operations)
    // Holds all Wt_local tiles for complete width chunk processing
    uint32_t input_transposed_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig input_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
            .set_page_size(input_transposed_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_transposed_cb_config);

    // CB25: Transposed indices (single-buffered for in-place bitonic operations)
    uint32_t index_transposed_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig index_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
            .set_page_size(index_transposed_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_transposed_cb_config);

    // CB26: Gathered values (aggregation buffer for final core)
    // Receives local TopK results from all worker cores (Wt_final = num_cores * Kt)
    uint32_t gathered_values_cb_index = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig gathered_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * value_tile_size, {{gathered_values_cb_index, value_cb_data_format}})
            .set_page_size(gathered_values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_values_cb_config);

    // CB27: Gathered indices (aggregation buffer for final core)
    uint32_t gathered_indices_cb_index = tt::CBIndex::c_27;
    tt::tt_metal::CircularBufferConfig gathered_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * index_tile_size, {{gathered_indices_cb_index, index_cb_data_format}})
            .set_page_size(gathered_indices_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_indices_cb_config);

    // CB28: Final values (staging buffer for final compute output)
    uint32_t final_values_cb_index = tt::CBIndex::c_28;
    tt::tt_metal::CircularBufferConfig final_values_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * value_tile_size, {{final_values_cb_index, value_cb_data_format}})
            .set_page_size(final_values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_values_cb_config);

    // CB29: Final indices (staging buffer for final compute output)
    uint32_t final_indices_cb_index = tt::CBIndex::c_29;
    tt::tt_metal::CircularBufferConfig final_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * index_tile_size, {{final_indices_cb_index, index_cb_data_format}})
            .set_page_size(final_indices_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_indices_cb_config);

    // CB16: Local TopK values output (local cores → writer → final core)
    uint32_t values_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig values_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
            .set_page_size(values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, values_cb_config);

    // CB17: Local TopK indices output (local cores → writer → final core)
    uint32_t output_ind_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, output_ind_cb_config);

    // ===============================================================================
    // STEP 4: INTER-CORE SYNCHRONIZATION SETUP
    // ===============================================================================

    // Semaphore-based flow control for coordinating data transfer between local and final cores
    auto sender_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);  // Tracks data transmission completion
    auto receiver_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);  // Signals readiness to receive data

    // ===============================================================================
    // STEP 5: KERNEL PIPELINE CREATION
    // ===============================================================================
    //
    // The multicore TopK pipeline consists of 6 specialized kernels:
    //
    // LOCAL PROCESSING CORES (runs on each local core):
    // 1. Reader (Local): Streams input data and generates/reads indices
    // 2. Compute (Local): Performs bitonic sort on local width chunk
    // 3. Writer (Local): Sends local TopK results to final core
    //
    // FINAL AGGREGATION CORE (runs on final core only):
    // 4. Reader (Final): Coordinates reception of all local TopK results
    // 5. Compute (Final): Performs global bitonic merge across all results
    // 6. Writer (Final): Outputs final TopK results to DRAM

    // -------------------------------------------------------------------------------
    // KERNEL 1: LOCAL READER - Data Input and Index Generation/Reading
    // -------------------------------------------------------------------------------
    // Responsibility: Stream input tensor data from DRAM to local cores
    // Two variants: generate indices on-demand or read pre-existing index tensor
    std::vector<uint32_t> reader_local_compile_time_args = {
        input_cb_index,                // CB0: Input values destination
        index_cb_index,                // CB1: Input indices destination
        Ht,                            // Height tiles in tensor
        Wt_local,                      // Width tiles per local core
        input_shape[-1] / TILE_WIDTH,  // Total width tiles (Wt)
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_local_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_indices_buffer).append_to(reader_local_compile_time_args);

    // Select appropriate reader variant based on whether indices are provided
    std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk.cpp";
    std::map<std::string, std::string> reader_specialization_defines = {
        {"GENERATE_INDICES", tensor_args.indices.has_value() ? "0" : "1"},
    };
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        local_cores_range_set,  // Runs on all local processing cores
        tt::tt_metal::ReaderDataMovementConfig(reader_local_compile_time_args, reader_specialization_defines));

    // -------------------------------------------------------------------------------
    // KERNEL 4: FINAL READER - Local TopK Results Aggregation Coordinator
    // -------------------------------------------------------------------------------
    // Responsibility: Coordinate reception of TopK results from all local cores
    // Uses semaphore protocol to synchronize with multiple sender cores
    CoreCoord local_cores_physical_start = device->worker_core_from_logical_core(local_cores.at(0));
    CoreCoord local_cores_physical_end = device->worker_core_from_logical_core(local_cores.at(num_cores - 2u));
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)receiver_semaphore_id,         // Semaphore for coordinating data reception
        (std::uint32_t)sender_semaphore_id,           // Semaphore for tracking transmission completion
        (std::uint32_t)local_cores_physical_start.x,  // NoC coordinates of local core range
        (std::uint32_t)local_cores_physical_start.y,
        (std::uint32_t)local_cores_physical_end.x,
        (std::uint32_t)local_cores_physical_end.y,
        (std::uint32_t)Ht,             // Height tiles to process
        (std::uint32_t)Wt_final,       // Total aggregated width tiles
        (std::uint32_t)num_cores - 1,  // Number of local cores sending data
    };

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp",
        final_cores_range_set,  // Runs only on final aggregation core
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // -------------------------------------------------------------------------------
    // KERNEL 3: LOCAL WRITER - Local TopK Results Transmission
    // -------------------------------------------------------------------------------
    // Responsibility: Send local TopK results from each core to final aggregation core
    // Implements sender side of semaphore-based synchronization protocol
    CoreCoord final_cores_physical = device->worker_core_from_logical_core(final_core);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)receiver_semaphore_id,   // Semaphore to check final core readiness
        (std::uint32_t)sender_semaphore_id,     // Semaphore to signal transmission completion
        (std::uint32_t)final_cores_physical.x,  // Target final core NoC coordinates
        (std::uint32_t)final_cores_physical.y,
        Ht,      // Height tiles to send
        args.k,  // TopK value
        Kt,      // TopK in tile units
    };
    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp",
        local_cores_range_set,  // Runs on all local processing cores
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // -------------------------------------------------------------------------------
    // KERNEL 6: FINAL WRITER - Global TopK Results Output to DRAM
    // -------------------------------------------------------------------------------
    // Responsibility: Write final globally optimal TopK results to output tensors
    // Handles proper interleaved tensor formatting for host consumption
    std::vector<uint32_t> writer_compile_time_args_final = {
        values_cb_index,      // CB16: Final TopK values source
        output_ind_cb_index,  // CB17: Final TopK indices source
        Ht,                   // Height tiles to write
        Kt                    // TopK tiles per height row
    };
    tt::tt_metal::TensorAccessorArgs(values_buffer).append_to(writer_compile_time_args_final);
    tt::tt_metal::TensorAccessorArgs(index_buffer).append_to(writer_compile_time_args_final);
    tt::tt_metal::KernelHandle binary_writer_final_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_final_topk.cpp",
        final_cores_range_set,  // Runs only on final aggregation core
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_final));

    // -------------------------------------------------------------------------------
    // KERNEL 2: LOCAL COMPUTE - Local Width Chunk Bitonic Sorting
    // -------------------------------------------------------------------------------
    // Responsibility: Perform bitonic sort on assigned width chunk to extract local TopK
    // Uses iterative divide-and-conquer with log(Wt_local) merge phases
    std::vector<uint32_t> compute_args = {
        input_cb_index,                      // CB0: Input values stream
        index_cb_index,                      // CB1: Input indices stream
        input_transposed_cb_index,           // CB24: Transposed values workspace
        index_transposed_cb_index,           // CB25: Transposed indices workspace
        values_cb_index,                     // CB16: Local TopK values output
        output_ind_cb_index,                 // CB17: Local TopK indices output
        Ht,                                  // Height tiles to process
        Wt_local,                            // Width tiles per local core
        args.k,                              // TopK value
        Kt,                                  // TopK in tile units
        (std::uint32_t)std::log2(args.k),    // log2(K) for bitonic network depth
        (std::uint32_t)std::log2(Wt_local),  // log2(width) for merge iterations
        (std::uint32_t)args.largest,         // Sort direction (largest=1, smallest=0)
        (std::uint32_t)args.sorted,          // Output sorting requirement
    };
    tt::tt_metal::KernelHandle topk_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_local.cpp",
        local_cores_range_set,  // Runs on all local processing cores
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    // -------------------------------------------------------------------------------
    // KERNEL 5: FINAL COMPUTE - Global TopK Bitonic Merge
    // -------------------------------------------------------------------------------
    // Responsibility: Perform final bitonic merge of all local TopK results
    // Produces globally optimal TopK from aggregated local results
    std::vector<uint32_t> compute_args_final = {
        gathered_values_cb_index,            // CB26: Aggregated local TopK values
        gathered_indices_cb_index,           // CB27: Aggregated local TopK indices
        final_values_cb_index,               // CB28: Final processing workspace values
        final_indices_cb_index,              // CB29: Final processing workspace indices
        values_cb_index,                     // CB16: Global TopK values output
        output_ind_cb_index,                 // CB17: Global TopK indices output
        Ht,                                  // Height tiles to process
        Wt_final,                            // Total aggregated width tiles
        args.k,                              // TopK value
        Kt,                                  // TopK in tile units
        (std::uint32_t)std::log2(args.k),    // log2(K) for bitonic network depth
        (std::uint32_t)std::log2(Wt_final),  // log2(final_width) for merge iterations
        (std::uint32_t)args.largest,         // Sort direction (largest=1, smallest=0)
        (std::uint32_t)args.sorted,          // Output sorting requirement
    };

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp",
        final_cores_range_set,  // Runs only on final aggregation core
        tt::tt_metal::ComputeConfig{.compile_args = compute_args_final});

    // ===============================================================================
    // STEP 6: RUNTIME ARGUMENTS CONFIGURATION
    // ===============================================================================
    //
    // Configure each core's specific work assignment and processing parameters.
    // Local cores receive different width offsets to process disjoint chunks,
    // while the final core coordinates the global aggregation process.

    uint32_t core_w = 0;             // Width offset counter for core assignment
    bool ascending = !args.largest;  // Initial sort direction for bitonic properties

    // Configure runtime arguments for each local processing core
    for (auto core : local_cores) {
        // LOCAL READER: Assign specific width chunk to each core

        // Variant with on-demand index generation
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                input_buffer->address(),  // DRAM address of input values tensor
                0,                        // Height offset (no height parallelism currently)
                core_w * Wt_local,        // Width offset for this core's chunk
                input_indices_tensor.has_value() ? input_indices_buffer->address()
                                                 : 0u,  // DRAM address of input indices tensor (if provided)
            });

        // LOCAL WRITER: Configure transmission parameters for sending to final core
        SetRuntimeArgs(
            program,
            binary_writer_kernel_id,
            core,
            {
                core_w,  // Width position for placement in final aggregation buffer
            });

        // LOCAL COMPUTE: Configure bitonic sort direction for proper sequencing
        SetRuntimeArgs(
            program,
            topk_compute_kernel_id,
            core,
            {
                ascending,  // Sort direction for bitonic properties
            });

        core_w++;                // Advance to next width chunk
        ascending = !ascending;  // Alternate sort direction for bitonic sequence
    }

    // FINAL WRITER: Configure output tensor addresses for final results
    SetRuntimeArgs(
        program,
        binary_writer_final_kernel_id,
        final_core,
        {
            values_buffer->address(),  // DRAM address for TopK values output tensor
            index_buffer->address(),   // DRAM address for TopK indices output tensor
        });

    return cached_program_t{
        std::move(program), {unary_reader_kernel_id, binary_writer_final_kernel_id, local_cores, final_core}};
}

void TopKMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TopkParams& /*args*/,
    const TopkInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    auto& unary_reader_kernel_id = shared_vars.unary_reader_kernel_id;
    auto& binary_writer_final_kernel_id = shared_vars.binary_writer_final_kernel_id;
    auto& local_cores = shared_vars.local_cores;
    auto& final_core = shared_vars.final_core;

    auto* input_buffer = tensor_args.input.buffer();
    auto* values_buffer = std::get<0>(output_tensors).buffer();
    auto* index_buffer = std::get<1>(output_tensors).buffer();

    for (auto core : local_cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
        reader_runtime_args[0] = input_buffer->address();
        reader_runtime_args[3] = tensor_args.indices.has_value() ? tensor_args.indices->buffer()->address() : 0u;
    }

    auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_final_kernel_id, final_core);
    writer_runtime_args[0] = values_buffer->address();
    writer_runtime_args[1] = index_buffer->address();
}

}  // namespace ttnn::prim
