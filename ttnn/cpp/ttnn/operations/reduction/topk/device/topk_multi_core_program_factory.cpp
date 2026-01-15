// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.hpp"

#include <cmath>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

using namespace tt::tt_metal;
using namespace std;

namespace ttnn::operations::reduction::topk::program {

/**
 * Split the work along the width such that the width is divisible by min_dim and the number of cores used is less than
 * or equal to max_cores. Each core must have a minimum of two tiles - min_dim = 64 as that's the minimum size for the
 * llk. Return the number of cores utilized for the split, the size of the split along the width, the width on the
 * remainder core if any, and the remaining elements that the gather core has to process If less than the max number of
 * cores are used, then we can try splitting on height as well. Eg) if only 2 cores are used for the split and then 1
 * for the gather, we only need 3 cores per row. Then take cores_per_row = 3 and try to split the height such that the
 * number of cores used is less than or equal to max_cores.
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
    auto config_opt = topk::utils::find_topk_core_config(
        width, min_dim, max_dim, k, core_range, l1_size, value_tile_size, index_tile_size);
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
 * Split work along width dimension and compute topk values and indices for each split in parallel, on different cores.
 * Then gather the results of each split onto a single core, where the final topk values and indices are computed.
 *
 */
TopKMultiCoreProgramFactory::cached_program_t TopKMultiCoreProgramFactory::create(
    const TopkParams& args, const TopkInputs& tensor_args, tensor_return_value_t& output_tensors) {
    using namespace tt::constants;

    const auto& input_tensor = tensor_args.input;
    const auto& input_indices_tensor = tensor_args.indices;
    const auto& value_tensor = std::get<0>(output_tensors);
    const auto& index_tensor = std::get<1>(output_tensors);

    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    tt::DataFormat index_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    auto first_core_range = args.sub_core_grids.ranges().at(0);
    auto first_core_range_set = CoreRangeSet(first_core_range);

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t value_tile_size = tile_size(value_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto* input_buffer = input_tensor.buffer();
    auto* values_buffer = value_tensor.buffer();
    auto* index_buffer = index_tensor.buffer();
    auto* input_indices_buffer = input_indices_tensor.has_value() ? input_indices_tensor->buffer() : nullptr;

    auto* device = input_tensor.device();

    auto input_shape = input_tensor.padded_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    const auto& [num_cores, local_topk_input_size, rem, final_topk_input_size, selected_x, selected_y] = cores_utilized(
        input_shape[args.dim],
        64,
        input_shape[args.dim] / 2,
        args.k,
        first_core_range,
        device->l1_size_per_core(),
        value_tile_size,
        index_tile_size);

    constexpr bool select_cores_row_wise = false;

    auto local_cores_range =
        select_contiguous_range_from_corerangeset(first_core_range_set, selected_x - 1, selected_y - 1);
    auto local_cores_range_set = CoreRangeSet(local_cores_range.value());
    auto local_cores =
        corerange_to_cores(local_cores_range_set, local_cores_range_set.num_cores(), select_cores_row_wise);

    auto final_cores_range_set =
        select_from_corerangeset(first_core_range_set, selected_y, selected_y, select_cores_row_wise);
    auto final_core = corerange_to_cores(final_cores_range_set, 1u, select_cores_row_wise).at(0);

    auto all_cores_range_set = local_cores_range_set;
    all_cores_range_set = all_cores_range_set.merge(final_cores_range_set);

    uint32_t Wt_local = local_topk_input_size / TILE_WIDTH;
    uint32_t Wt_final = final_topk_input_size / TILE_WIDTH;
    uint32_t Kt = args.k % TILE_WIDTH == 0 ? args.k / TILE_WIDTH : (args.k / TILE_WIDTH) + 1;

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    // TODO: In theory if we have enough memory we could allocate 2*Wt tiles to reduce stalls
    uint32_t input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * value_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_cb_config);

    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space This CB carries the indices that are created in the reader kernel
    uint32_t index_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_input_intermed0_config);

    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig input_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
            .set_page_size(input_transposed_cb_index, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, input_transposed_cb_config);

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig index_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_local * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
            .set_page_size(index_transposed_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, index_transposed_cb_config);

    uint32_t gathered_values_cb_index = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig gathered_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * value_tile_size, {{gathered_values_cb_index, value_cb_data_format}})
            .set_page_size(gathered_values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_values_cb_config);

    uint32_t gathered_indices_cb_index = tt::CBIndex::c_27;
    tt::tt_metal::CircularBufferConfig gathered_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt_final * index_tile_size, {{gathered_indices_cb_index, index_cb_data_format}})
            .set_page_size(gathered_indices_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, gathered_indices_cb_config);

    uint32_t final_values_cb_index = tt::CBIndex::c_28;
    tt::tt_metal::CircularBufferConfig final_values_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * value_tile_size, {{final_values_cb_index, value_cb_data_format}})
            .set_page_size(final_values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_values_cb_config);

    uint32_t final_indices_cb_index = tt::CBIndex::c_29;
    tt::tt_metal::CircularBufferConfig final_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt_final * index_tile_size, {{final_indices_cb_index, index_cb_data_format}})
            .set_page_size(final_indices_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, final_indices_cb_config);

    // Output topk values
    uint32_t values_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig values_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
            .set_page_size(values_cb_index, value_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, values_cb_config);

    // Output topk indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_range_set, output_ind_cb_config);

    // Create semaphores
    auto sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);
    auto receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores_range_set, INVALID);
    std::vector<uint32_t> reader_local_compile_time_args = {
        input_cb_index,
        index_cb_index,
        Ht,
        Wt_local,
        input_shape[-1] / TILE_WIDTH,  // Wt
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_local_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_indices_buffer).append_to(reader_local_compile_time_args);
    std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk.cpp";
    if (input_indices_tensor.has_value()) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_read_index_local_topk.cpp";
    }
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        local_cores_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_local_compile_time_args));

    CoreCoord local_cores_physical_start = device->worker_core_from_logical_core(local_cores.at(0));
    CoreCoord local_cores_physical_end = device->worker_core_from_logical_core(local_cores.at(num_cores - 2u));
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)receiver_semaphore_id,
        (std::uint32_t)sender_semaphore_id,
        (std::uint32_t)local_cores_physical_start.x,
        (std::uint32_t)local_cores_physical_start.y,
        (std::uint32_t)local_cores_physical_end.x,
        (std::uint32_t)local_cores_physical_end.y,
        (std::uint32_t)Ht,
        (std::uint32_t)Wt_final,
        (std::uint32_t)num_cores - 1,
    };

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp",
        final_cores_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    CoreCoord final_cores_physical = device->worker_core_from_logical_core(final_core);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)receiver_semaphore_id,
        (std::uint32_t)sender_semaphore_id,
        (std::uint32_t)final_cores_physical.x,
        (std::uint32_t)final_cores_physical.y,
        Ht,
        args.k,
        Kt,
    };
    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp",
        local_cores_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args_final = {values_cb_index, output_ind_cb_index, Ht, Kt};
    tt::tt_metal::TensorAccessorArgs(values_buffer).append_to(writer_compile_time_args_final);
    tt::tt_metal::TensorAccessorArgs(index_buffer).append_to(writer_compile_time_args_final);
    tt::tt_metal::KernelHandle binary_writer_final_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_final_topk.cpp",
        final_cores_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_final));

    std::vector<uint32_t> compute_args = {
        input_cb_index,
        index_cb_index,
        input_transposed_cb_index,
        index_transposed_cb_index,
        values_cb_index,
        output_ind_cb_index,
        Ht,
        Wt_local,
        args.k,
        Kt,
        (std::uint32_t)std::log2(args.k),
        (std::uint32_t)std::log2(Wt_local),
        (std::uint32_t)args.largest,
        (std::uint32_t)args.sorted,
    };
    tt::tt_metal::KernelHandle topk_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_local.cpp",
        local_cores_range_set,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    std::vector<uint32_t> compute_args_final = {
        gathered_values_cb_index,
        gathered_indices_cb_index,
        final_values_cb_index,
        final_indices_cb_index,
        values_cb_index,
        output_ind_cb_index,
        Ht,
        Wt_final,
        args.k,
        Kt,
        (std::uint32_t)std::log2(args.k),
        (std::uint32_t)std::log2(Wt_final),
        (std::uint32_t)args.largest,
        (std::uint32_t)args.sorted,
    };

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp",
        final_cores_range_set,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args_final});

    uint32_t core_w = 0;
    bool ascending = !args.largest;
    for (auto core : local_cores) {
        if (input_indices_tensor.has_value()) {
            SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    input_buffer->address(),
                    input_indices_buffer->address(),
                    0,  // no height parallelism for now
                    core_w * Wt_local,
                });
        } else {
            SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {
                    input_buffer->address(),
                    0,  // no height parallelism for now
                    core_w * Wt_local,
                });
        }

        SetRuntimeArgs(
            program,
            binary_writer_kernel_id,
            core,
            {
                core_w,
            });

        SetRuntimeArgs(
            program,
            topk_compute_kernel_id,
            core,
            {
                ascending,
            });
        core_w++;
        ascending = !ascending;
    }
    SetRuntimeArgs(
        program,
        binary_writer_final_kernel_id,
        final_core,
        {
            values_buffer->address(),
            index_buffer->address(),
        });

    return cached_program_t{
        std::move(program), {unary_reader_kernel_id, binary_writer_final_kernel_id, local_cores, final_core}};
}

void TopKMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TopkParams& /*args*/,
    const TopkInputs& tensor_args,
    tensor_return_value_t& output_tensors) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    auto& unary_reader_kernel_id = shared_vars.unary_reader_kernel_id;
    auto& binary_writer_final_kernel_id = shared_vars.binary_writer_final_kernel_id;
    auto& local_cores = shared_vars.local_cores;
    auto& final_core = shared_vars.final_core;

    auto* input_buffer = tensor_args.input.buffer();
    auto* values_buffer = std::get<0>(output_tensors).buffer();
    auto* index_buffer = std::get<1>(output_tensors).buffer();

    auto* input_indices_buffer = tensor_args.indices.has_value() ? tensor_args.indices.value().buffer() : nullptr;

    for (auto core : local_cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
        reader_runtime_args[0] = input_buffer->address();
        if (tensor_args.indices.has_value()) {
            reader_runtime_args[1] = input_indices_buffer->address();
        }
    }

    auto& writer_runtime_args = GetRuntimeArgs(program, binary_writer_final_kernel_id, final_core);
    writer_runtime_args[0] = values_buffer->address();
    writer_runtime_args[1] = index_buffer->address();
}

}  // namespace ttnn::operations::reduction::topk::program
