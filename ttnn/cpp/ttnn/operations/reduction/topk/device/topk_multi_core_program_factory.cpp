// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt_stl/assert.hpp"
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

#include <cmath>

using namespace tt::tt_metal;
using namespace std;

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
    const uint32_t index_tile_size) {
    const auto config_opt =
        find_topk_core_config(width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size);
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

TopKMultiCoreProgramFactory::cached_program_t TopKMultiCoreProgramFactory::create(
    const TopkParams& args, const TopkInputs& tensor_args, std::tuple<Tensor, Tensor>& output_tensors) {
    using namespace tt::constants;

    // Tensor references
    const auto& input_tensor = tensor_args.input;
    const auto& input_indices_tensor = tensor_args.indices;
    const auto& value_tensor = std::get<0>(output_tensors);
    const auto& index_tensor = std::get<1>(output_tensors);

    tt::tt_metal::Program program{};

    // Data format configuration for all circular buffers
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    const tt::DataFormat index_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());
    const bool is32_bit_data = index_cb_data_format == tt::DataFormat::UInt32;

    // Core grid and tile size calculations
    const auto first_core_range = args.sub_core_grids.ranges().at(0);
    const auto first_core_range_set = CoreRangeSet(first_core_range);

    const uint32_t input_tile_size = tile_size(input_cb_data_format);
    const uint32_t value_tile_size = tile_size(value_cb_data_format);
    const uint32_t index_tile_size = tile_size(index_cb_data_format);

    // DRAM buffer pointers for kernel runtime arguments
    const auto* input_buffer = input_tensor.buffer();
    const auto* values_buffer = value_tensor.buffer();
    const auto* index_buffer = index_tensor.buffer();
    const auto* input_indices_buffer = input_indices_tensor.has_value() ? input_indices_tensor->buffer() : nullptr;

    const auto* device = input_tensor.device();

    const auto input_shape = input_tensor.padded_shape();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;

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
    const uint32_t Wt_local = local_topk_input_size / TILE_WIDTH;  // Width tiles per local core
    const uint32_t Wt_final = final_topk_input_size / TILE_WIDTH;  // Total width tiles for final core
    const uint32_t Kt = args.k % TILE_WIDTH == 0 ? args.k / TILE_WIDTH : (args.k / TILE_WIDTH) + 1;  // TopK in tiles

    const uint32_t num_cb_unit = 2;                // Base buffering unit
    const uint32_t cb_in_units = 2 * num_cb_unit;  // 4 units total for double-buffered input

    // Input values (double-buffered for continuous DRAM streaming)
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * value_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_cb_config);

    // Input indices (double-buffered, generated on-demand or read from DRAM)
    constexpr uint32_t index_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_input_intermed0_config);

    // Transposed values (single-buffered for in-place bitonic operations)
    // Holds all Wt_local tiles for complete width chunk processing
    constexpr uint32_t input_transposed_cb_index = tt::CBIndex::c_2;
    const tt::tt_metal::CircularBufferConfig input_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
            .set_page_size(input_transposed_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_transposed_cb_config);

    // Transposed indices (single-buffered for in-place bitonic operations)
    constexpr uint32_t index_transposed_cb_index = tt::CBIndex::c_3;
    const tt::tt_metal::CircularBufferConfig index_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
            .set_page_size(index_transposed_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_transposed_cb_config);

    // Gathered values (aggregation buffer for final core)
    // Receives local TopK results from all worker cores (Wt_final = num_cores * Kt)
    constexpr uint32_t gathered_values_cb_index = tt::CBIndex::c_4;
    const tt::tt_metal::CircularBufferConfig gathered_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * value_tile_size, {{gathered_values_cb_index, value_cb_data_format}})
            .set_page_size(gathered_values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_values_cb_config);

    // Gathered indices (aggregation buffer for final core)
    constexpr uint32_t gathered_indices_cb_index = tt::CBIndex::c_5;
    const tt::tt_metal::CircularBufferConfig gathered_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * index_tile_size, {{gathered_indices_cb_index, index_cb_data_format}})
            .set_page_size(gathered_indices_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_indices_cb_config);

    // Final values (staging buffer for final compute output)
    constexpr uint32_t final_values_cb_index = tt::CBIndex::c_6;
    const tt::tt_metal::CircularBufferConfig final_values_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * value_tile_size, {{final_values_cb_index, value_cb_data_format}})
            .set_page_size(final_values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_values_cb_config);

    // Final indices (staging buffer for final compute output)
    constexpr uint32_t final_indices_cb_index = tt::CBIndex::c_7;
    const tt::tt_metal::CircularBufferConfig final_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * index_tile_size, {{final_indices_cb_index, index_cb_data_format}})
            .set_page_size(final_indices_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_indices_cb_config);

    // Local TopK values output (local cores → writer → final core)
    constexpr uint32_t values_cb_index = tt::CBIndex::c_8;
    const tt::tt_metal::CircularBufferConfig values_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
            .set_page_size(values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, values_cb_config);

    // Local TopK indices output (local cores → writer → final core)
    constexpr uint32_t output_ind_cb_index = tt::CBIndex::c_9;
    const tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, output_ind_cb_config);

    // Semaphore-based flow control for coordinating data transfer between local and final cores
    const uint32_t sender_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);  // Tracks data transmission completion
    const uint32_t receiver_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);  // Signals readiness to receive data

    // Local reader - Data Input and Index Generation/Reading
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
    const std::map<std::string, std::string> reader_specialization_defines = {
        {"GENERATE_INDICES", tensor_args.indices.has_value() ? "0" : "1"},
    };
    const tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk.cpp",
        local_cores_range_set,  // Runs on all local processing cores
        tt::tt_metal::ReaderDataMovementConfig(reader_local_compile_time_args, reader_specialization_defines));

    // Final reader - Local TopK Results Aggregation Coordinator
    // Responsibility: Coordinate reception of TopK results from all local cores
    // Uses semaphore protocol to synchronize with multiple sender cores
    CoreCoord local_cores_physical_start = device->worker_core_from_logical_core(local_cores.at(0));
    CoreCoord local_cores_physical_end = device->worker_core_from_logical_core(local_cores.at(num_cores - 2u));
    const std::vector<uint32_t> reader_compile_time_args = {
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
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp",
        final_cores_range_set,  // Runs only on final aggregation core
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Local writer - Local TopK Results Transmission
    // Responsibility: Send local TopK results from each core to final aggregation core
    // Implements sender side of semaphore-based synchronization protocol
    const CoreCoord final_cores_physical = device->worker_core_from_logical_core(final_core);
    const std::vector<uint32_t> writer_compile_time_args = {
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
    const tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp",
        local_cores_range_set,  // Runs on all local processing cores
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Final writer - Global TopK Results Output to DRAM
    // Responsibility: Write final globally optimal TopK results to output tensors
    // Handles proper interleaved tensor formatting for host consumption
    std::vector<uint32_t> writer_compile_time_args_final = {
        values_cb_index,      // Final TopK values source
        output_ind_cb_index,  // Final TopK indices source
        Ht,                   // Height tiles to write
        Kt                    // TopK tiles per height row
    };
    tt::tt_metal::TensorAccessorArgs(values_buffer).append_to(writer_compile_time_args_final);
    tt::tt_metal::TensorAccessorArgs(index_buffer).append_to(writer_compile_time_args_final);
    const tt::tt_metal::KernelHandle binary_writer_final_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_final_topk.cpp",
        final_cores_range_set,  // Runs only on final aggregation core
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_final));

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
    const tt::tt_metal::KernelHandle topk_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_local.cpp",
        local_cores_range_set,  // Runs on all local processing cores
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

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
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp",
        final_cores_range_set,  // Runs only on final aggregation core
        tt::tt_metal::ComputeConfig{.compile_args = compute_args_final});

    uint32_t core_id = 0;            // Width offset counter for core assignment
    bool ascending = !args.largest;  // Initial sort direction for bitonic properties

    // Configure runtime arguments for each local processing core
    for (auto core : local_cores) {
        // Local reader
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                input_buffer->address(),  // DRAM address of input values tensor
                0,                        // Height offset (no height parallelism currently)
                core_id * Wt_local,       // Width offset for this core's chunk
                is32_bit_data,            // Flag indicating if data is 32-bit
                input_indices_tensor.has_value() ? input_indices_buffer->address()
                                                 : 0u,  // DRAM address of input indices tensor (if provided)
            });

        // Local writer
        SetRuntimeArgs(
            program,
            binary_writer_kernel_id,
            core,
            {
                core_id,  // Width position for placement in final aggregation buffer
            });

        // Local compute
        SetRuntimeArgs(
            program,
            topk_compute_kernel_id,
            core,
            {
                ascending,  // Sort direction for bitonic properties
            });

        core_id++;               // Advance to next width chunk
        ascending = !ascending;  // Alternate sort direction for bitonic sequence
    }

    // Final writer
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
    // Get references to program and shared variables
    const auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;
    const auto& unary_reader_kernel_id = shared_vars.unary_reader_kernel_id;
    const auto& binary_writer_final_kernel_id = shared_vars.binary_writer_final_kernel_id;
    const auto& local_cores = shared_vars.local_cores;
    const auto& final_core = shared_vars.final_core;

    // Buffer pointers for kernel runtime arguments
    const auto* input_buffer = tensor_args.input.buffer();
    const auto* values_buffer = std::get<0>(output_tensors).buffer();
    const auto* index_buffer = std::get<1>(output_tensors).buffer();

    // Override input addresses for each local reader core
    for (const auto core : local_cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
        reader_runtime_args[0] = input_buffer->address();
        reader_runtime_args[4] = tensor_args.indices.has_value() ? tensor_args.indices->buffer()->address() : 0u;
    }

    auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_final_kernel_id, final_core);
    writer_runtime_args[0] = values_buffer->address();
    writer_runtime_args[1] = index_buffer->address();
}

}  // namespace ttnn::prim
