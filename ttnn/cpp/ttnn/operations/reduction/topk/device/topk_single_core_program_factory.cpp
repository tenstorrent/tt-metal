// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_single_core_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/work_split.hpp"

#include <string>

using namespace tt::tt_metal;
using namespace std;

namespace ttnn::prim {

TopKSingleCoreProgramFactory::cached_program_t TopKSingleCoreProgramFactory::create(
    const TopkParams& args, const TopkInputs& tensor_args, std::tuple<Tensor, Tensor>& output_tensors) {
    using namespace tt::constants;

    // std::cout << "TOPK SINGLE CORE PROGRAM FACTORY" << std::endl;

    // Tensor references
    const auto& input_tensor = tensor_args.input;
    const auto& value_tensor = std::get<0>(output_tensors);
    const auto& index_tensor = std::get<1>(output_tensors);

    // Determine index output format based on dimension size constraints
    const ttnn::Shape input_shape = input_tensor.padded_shape();
    const bool uint16_output = (input_shape[args.dim] < std::numeric_limits<uint16_t>::max());

    tt::tt_metal::Program program{};

    // Data format conversions for circular buffer configurations
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat output_val_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    const tt::DataFormat output_ind_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    // Calculate tile sizes for memory allocation
    const uint32_t input_tile_size = tile_size(input_cb_data_format);
    const uint32_t value_tile_size = tile_size(output_val_cb_data_format);
    const uint32_t index_tile_size = tile_size(output_ind_cb_data_format);

    // Device memory buffer pointers for kernel runtime arguments
    const auto* input_buffer = input_tensor.buffer();
    const auto* values_buffer = value_tensor.buffer();
    const auto* index_buffer = index_tensor.buffer();

    // Tensor shape and dimension calculations
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tt::constants::TILE_HEIGHT;
    const uint32_t Wt = input_shape[3] / tt::constants::TILE_WIDTH;

    // Single core selection from the provided core grid
    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(args.sub_core_grids, Ht, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_range, total_number_of_cores, true);

    // std::cout << "total_number_of_cores = " << total_number_of_cores << std::endl;

    // Number of tiles needed to store K top elements
    const uint32_t Ktiles = tt::div_up(args.k, tt::constants::TILE_WIDTH);

    // Pipeline Flow:
    // Input CB -> Reader Kernel -> Transposed CBs -> Compute Kernel -> Result Prep CBs -> Output CBs -> Writer Kernel
    const uint32_t num_cb_unit = 2;                         // Base unit for double buffering
    const uint32_t cb_in_units = num_cb_unit;               // 4 units total for input double buffering
    const uint32_t input_cb_tile_count = cb_in_units;       // Input stream buffer size
    const uint32_t transposed_cb_tile_count = 4;            // Transposed data staging
    const uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // Intermediate TopK results (double-buffered)
    const uint32_t output_cb_tile_count = Ktiles;           // Final output buffer

    // Circular Buffer Creations:
    const uint32_t input_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            input_cb_tile_count * input_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_cb_config);

    constexpr uint32_t index_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(
            input_cb_tile_count * index_tile_size, {{index_cb_index, output_ind_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, index_input_intermed0_config);

    constexpr uint32_t transposed_val_cb_index = tt::CBIndex::c_2;
    const tt::tt_metal::CircularBufferConfig transposed_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            transposed_cb_tile_count * input_tile_size, {{transposed_val_cb_index, input_cb_data_format}})
            .set_page_size(transposed_val_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, transposed_val_cb_config);

    constexpr uint32_t transposed_ind_cb_index = tt::CBIndex::c_3;
    const tt::tt_metal::CircularBufferConfig transposed_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            transposed_cb_tile_count * index_tile_size, {{transposed_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(transposed_ind_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, transposed_ind_cb_config);

    constexpr uint32_t result_prep_val_cb_index = tt::CBIndex::c_4;
    const tt::tt_metal::CircularBufferConfig result_prep_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            result_prep_cb_tile_count * input_tile_size, {{result_prep_val_cb_index, input_cb_data_format}})
            .set_page_size(result_prep_val_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, result_prep_val_cb_config);

    constexpr uint32_t result_prep_ind_cb_index = tt::CBIndex::c_5;
    const tt::tt_metal::CircularBufferConfig result_prep_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            result_prep_cb_tile_count * index_tile_size, {{result_prep_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(result_prep_ind_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, result_prep_ind_cb_config);

    constexpr uint32_t output_val_cb_index = tt::CBIndex::c_6;
    const tt::tt_metal::CircularBufferConfig output_val_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_cb_tile_count * value_tile_size, {{output_val_cb_index, output_val_cb_data_format}})
            .set_page_size(output_val_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, output_val_cb_config);

    constexpr uint32_t output_ind_cb_index = tt::CBIndex::c_7;
    const tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(
            output_cb_tile_count * index_tile_size, {{output_ind_cb_index, output_ind_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, output_ind_cb_config);

    constexpr uint32_t synchronization_cb_index = tt::CBIndex::c_8;
    constexpr uint32_t synchronization_cb_size = tt::constants::TILE_HW * sizeof(uint8_t);
    const tt::tt_metal::CircularBufferConfig synchronization_cb_config =
        tt::tt_metal::CircularBufferConfig(synchronization_cb_size, {{synchronization_cb_index, tt::DataFormat::UInt8}})
            .set_page_size(synchronization_cb_index, synchronization_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, synchronization_cb_config);

    // Kernel Creations:
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,                       // Input values
        index_cb_index,                       // Generated indices
        Ht,                                   // Height in tiles
        Wt,                                   // Width in tiles
        total_number_of_cores,                // Total number of cores
        static_cast<uint32_t>(uint16_output)  // Index format flag
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    if (tensor_args.indices.has_value()) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.indices->buffer()).append_to(reader_compile_time_args);
    }
    const std::map<std::string, std::string> reader_defines = {
        {"GENERATE_INDICES", "1"},  // tensor_args.indices.has_value() ? "0" : "1" - GH issue: #36329
    };
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp",
        core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    std::vector<uint32_t> writer_compile_time_args = {
        output_val_cb_index,   // CB6: Output values
        output_ind_cb_index,   // CB7: Output indices
        Ht,                    // Height in tiles
        Ktiles,                // K value in tiles
        total_number_of_cores  // Total number of cores
    };
    tt::tt_metal::TensorAccessorArgs(values_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(index_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp",
        core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const std::vector<uint32_t> compute_args = {
        input_cb_index,                            // Input values
        index_cb_index,                            // Input indices
        transposed_val_cb_index,                   // Transposed values
        transposed_ind_cb_index,                   // Transposed indices
        result_prep_val_cb_index,                  // Result prep values
        result_prep_ind_cb_index,                  // Result prep indices
        output_val_cb_index,                       // Output values
        output_ind_cb_index,                       // Output indices
        Ht,                                        // Height in tiles
        Wt,                                        // Width in tiles
        Ktiles,                                    // K value in tiles
        static_cast<std::uint32_t>(args.largest),  // Sort order: largest (true) or smallest (false)
        synchronization_cb_index,
    };
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp",
        core_range,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = !uint16_output, .compile_args = compute_args});

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                SetRuntimeArgs(
                    program,
                    unary_reader_kernel_id,
                    core,
                    {
                        input_buffer->address(),
                        id,
                        work_per_core,
                        tensor_args.indices.has_value() ? tensor_args.indices->buffer()->address()
                                                        : 0,  // Optional indices tensor
                    });
                SetRuntimeArgs(
                    program,
                    binary_writer_kernel_id,
                    core,
                    {
                        values_buffer->address(),
                        index_buffer->address(),
                        id,
                        work_per_core,
                    });
                SetRuntimeArgs(
                    program,
                    compute_kernel_id,
                    core,
                    {
                        work_per_core,
                    });
                id++;
            }
        }
    }
    return cached_program_t{std::move(program), {unary_reader_kernel_id, binary_writer_kernel_id, cores}};
}

void TopKSingleCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TopkParams& /*args*/,
    const TopkInputs& tensor_args,
    std::tuple<Tensor, Tensor>& output_tensors) {
    // Extract program and kernel information from cached program
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    auto& unary_reader_kernel_id = shared_vars.unary_reader_kernel_id;
    auto& binary_writer_kernel_id = shared_vars.binary_writer_kernel_id;

    // Get new buffer addresses for current tensor operation
    const auto* input_buffer = tensor_args.input.buffer();
    const auto* values_buffer = std::get<0>(output_tensors).buffer();
    const auto* index_buffer = std::get<1>(output_tensors).buffer();

    // Update runtime arguments with new buffer addresses
    for (const auto& core : cached_program.shared_variables.cores) {
        // Update reader kernel
        auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
        reader_runtime_args[0] = input_buffer->address();  // Input values
        if (tensor_args.indices.has_value()) {
            reader_runtime_args[3] = tensor_args.indices->buffer()->address();  // Optional indices tensor
        }

        // Update writer kernel
        auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, core);
        writer_runtime_args[0] = values_buffer->address();  // TopK values output
        writer_runtime_args[1] = index_buffer->address();   // TopK indices output
    }
}
}  // namespace ttnn::prim
