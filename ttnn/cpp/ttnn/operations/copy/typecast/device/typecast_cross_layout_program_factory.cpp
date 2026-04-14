// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_cross_layout_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

// ─── TILE → ROW_MAJOR ────────────────────────────────────────────────────────
// Compute kernel: typecast(input_cb → intermediate_cb) + untilize(intermediate_cb → output_cb)
// Reader: reads tiles from interleaved input buffer
// Writer: writes RM sticks to interleaved output buffer

static TypecastCrossLayoutProgramFactory::cached_program_t create_tile_to_rm(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto input_dtype = args.input_dtype;
    const auto output_dtype = args.output_dtype;

    Program program{};

    const auto cb_data_format_input = datatype_to_dataformat_converter(input_dtype);
    const auto single_tile_size_input = tt::tile_size(cb_data_format_input);
    const auto cb_data_format_output = datatype_to_dataformat_converter(output_dtype);
    const auto single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();
    const auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    const auto padded_shape = input.padded_shape();
    const uint32_t tensor_width = padded_shape[-1];
    const uint32_t tensor_height = input.physical_volume() / tensor_width;
    const uint32_t ntiles_per_row = tensor_width / TILE_WIDTH;
    const uint32_t nblocks = tensor_height / TILE_HEIGHT;  // number of tile-rows

    const auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, nblocks);

    // ── Circular Buffers ─────────────────────────────────────────────────
    // Input CB: tile pages in input dtype
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    const uint32_t input_cb_num_tiles = ntiles_per_row * 2;
    CircularBufferConfig cb_input_config =
        CircularBufferConfig(input_cb_num_tiles * single_tile_size_input, {{input_cb_index, cb_data_format_input}})
            .set_page_size(input_cb_index, single_tile_size_input);
    CreateCircularBuffer(program, all_cores, cb_input_config);

    // Intermediate CB: tile pages in output dtype (typecast writes here, untilize reads from here)
    constexpr uint32_t intermediate_cb_index = tt::CBIndex::c_1;
    const uint32_t intermediate_cb_num_tiles = ntiles_per_row * 2;
    CircularBufferConfig cb_intermediate_config =
        CircularBufferConfig(
            intermediate_cb_num_tiles * single_tile_size_output, {{intermediate_cb_index, cb_data_format_output}})
            .set_page_size(intermediate_cb_index, single_tile_size_output);
    CreateCircularBuffer(program, all_cores, cb_intermediate_config);

    // Output CB: tile-sized pages in output dtype (untilize writes RM data here using tile page accounting)
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t output_cb_num_tiles = ntiles_per_row * 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(output_cb_num_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    CreateCircularBuffer(program, all_cores, cb_output_config);

    // ── Reader: tile-based interleaved reader ────────────────────────────
    std::vector<uint32_t> reader_ct_args = {(uint32_t)input_cb_index};
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
    const auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    // ── Writer: RM stick writer ──────────────────────────────────────────
    const uint32_t output_element_size = output.element_size();
    const uint32_t output_stick_size = tensor_width * output_element_size;

    std::vector<uint32_t> writer_ct_args = {
        (uint32_t)output_cb_index,
        output_stick_size,
        (uint32_t)TILE_HEIGHT,
        ntiles_per_row,
        (uint32_t)1,  // num_output_blocks_across_width (1 for interleaved)
        output_element_size,
        tensor_width,  // num_cols_per_input_block
        tensor_width,  // num_cols_per_output_block (= tensor_width for interleaved)
    };
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    const auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multi_core.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // ── Compute kernel ───────────────────────────────────────────────────
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    std::map<std::string, std::string> compute_defines;
    compute_defines["UNTILIZE_OUTPUT"] = "1";
    compute_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    compute_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const auto compute_kernel_path =
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast_cross_layout.cpp";

    // Core group 1 (full cores)
    if (!core_range.ranges().empty()) {
        std::vector<uint32_t> compute_args = {
            nblocks_per_core,       // per_core_block_cnt (tile-rows per core)
            ntiles_per_row,         // per_core_block_dim (tiles per row)
            input_cb_index,         // input_cb
            output_cb_index,        // output_cb
            intermediate_cb_index,  // intermediate_cb
        };
        CreateKernel(
            program,
            compute_kernel_path,
            core_range,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = false,
                .compile_args = compute_args,
                .defines = compute_defines});
    }

    // Core group 2 (cliff cores)
    if (!core_range_cliff.empty()) {
        std::vector<uint32_t> compute_args_cliff = {
            nblocks_per_core_cliff,
            ntiles_per_row,
            input_cb_index,
            output_cb_index,
            intermediate_cb_index,
        };
        CreateKernel(
            program,
            compute_kernel_path,
            core_range_cliff,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = false,
                .compile_args = compute_args_cliff,
                .defines = compute_defines});
    }

    // ── Runtime args ─────────────────────────────────────────────────────
    const bool has_cliff = !core_range_cliff.empty();
    const uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t block_start_id = 0;

    const auto& cores = corerange_to_cores(all_cores);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const uint32_t cur_nblocks = (i < ncores_full) ? nblocks_per_core : nblocks_per_core_cliff;
        const uint32_t num_tiles_this_core = cur_nblocks * ntiles_per_row;

        // Reader: src_addr, num_tiles, start_page_id
        SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), num_tiles_this_core, tile_start_id});

        // Writer: dst_addr, num_input_blocks_to_process, height_wise_input_block_start_index,
        //         num_unpadded_cols_per_input_block, width_wise_output_block_start_index,
        //         num_cols_already_processed_in_first_output_block
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {dst_buffer->address(), cur_nblocks, block_start_id, tensor_width, (uint32_t)0, (uint32_t)0});

        tile_start_id += num_tiles_this_core;
        block_start_id += cur_nblocks;
    }

    return TypecastCrossLayoutProgramFactory::cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, ncores, grid_size.y}};
}

// ─── ROW_MAJOR → TILE ────────────────────────────────────────────────────────
// Compute kernel: tilize(input_cb → intermediate_cb) + typecast(intermediate_cb → output_cb)
// Reader: reads RM sticks from interleaved input buffer
// Writer: writes tiles to interleaved output buffer

static TypecastCrossLayoutProgramFactory::cached_program_t create_rm_to_tile(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto input_dtype = args.input_dtype;
    const auto output_dtype = args.output_dtype;

    Program program{};

    const auto cb_data_format_input = datatype_to_dataformat_converter(input_dtype);
    const auto single_tile_size_input = tt::tile_size(cb_data_format_input);
    const auto cb_data_format_output = datatype_to_dataformat_converter(output_dtype);
    const auto single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();
    const auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    const auto logical_shape = input.logical_shape();
    const auto logical_width = logical_shape[-1];
    const uint32_t ntiles_per_row = tt::div_up(logical_width, TILE_WIDTH);
    const uint32_t ntiles = dst_buffer->num_pages();
    const uint32_t nblocks = tt::div_up(ntiles, ntiles_per_row);  // tile-rows

    const auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, nblocks);

    const uint32_t rm_page_size = src_buffer->page_size();

    // ── Circular Buffers ─────────────────────────────────────────────────
    // Input CB: RM pages (row-sized) in input dtype
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    const uint32_t input_cb_num_pages = ntiles_per_row;  // tilize needs full tile-width of RM data
    CircularBufferConfig cb_input_config =
        CircularBufferConfig(input_cb_num_pages * single_tile_size_input, {{input_cb_index, cb_data_format_input}})
            .set_page_size(input_cb_index, single_tile_size_input);
    CreateCircularBuffer(program, all_cores, cb_input_config);

    // Intermediate CB: tile pages in input dtype (tilize output, typecast input)
    constexpr uint32_t intermediate_cb_index = tt::CBIndex::c_1;
    const uint32_t intermediate_cb_num_tiles = ntiles_per_row * 2;
    CircularBufferConfig cb_intermediate_config =
        CircularBufferConfig(
            intermediate_cb_num_tiles * single_tile_size_input, {{intermediate_cb_index, cb_data_format_input}})
            .set_page_size(intermediate_cb_index, single_tile_size_input);
    CreateCircularBuffer(program, all_cores, cb_intermediate_config);

    // Output CB: tile pages in output dtype (typecast output)
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t output_cb_num_tiles = ntiles_per_row * 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(output_cb_num_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    CreateCircularBuffer(program, all_cores, cb_output_config);

    // ── Reader: RM stick reader ──────────────────────────────────────────
    const uint32_t aligned_page_size = src_buffer->aligned_page_size();
    std::vector<uint32_t> reader_ct_args = {
        aligned_page_size,
        (uint32_t)1,   // num_pages_in_row (1 for interleaved — one page = one full row)
        rm_page_size,  // size_of_valid_data_in_last_page_in_row
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
    const auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
        "reader_unary_stick_layout_split_rows_multicore.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    // ── Writer: tile-based interleaved writer ────────────────────────────
    std::vector<uint32_t> writer_ct_args = {(uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    const auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // ── Compute kernel ───────────────────────────────────────────────────
    const bool fp32_llk_acc = input_dtype == DataType::FLOAT32;
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc || args.preserve_fp32_precision) {
        unpack_to_dest_mode[input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    std::map<std::string, std::string> compute_defines;
    compute_defines["TILIZE_INPUT"] = "1";
    compute_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    compute_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const auto compute_kernel_path =
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast_cross_layout.cpp";

    // Total RM rows across all blocks = needed for tilize's total_input_pages
    const uint32_t total_rm_rows_per_core = nblocks_per_core * TILE_HEIGHT;
    const uint32_t total_rm_rows_per_cliff_core = nblocks_per_core_cliff * TILE_HEIGHT;

    if (!core_range.ranges().empty()) {
        std::vector<uint32_t> compute_args = {
            nblocks_per_core,  // per_core_block_cnt (tile-rows)
            ntiles_per_row,    // per_core_block_dim (tiles per row)
            input_cb_index,
            output_cb_index,
            intermediate_cb_index,
            total_rm_rows_per_core,  // total_input_pages for tilize
        };
        CreateKernel(
            program,
            compute_kernel_path,
            core_range,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = false,
                .compile_args = compute_args,
                .defines = compute_defines});
    }

    if (!core_range_cliff.empty()) {
        std::vector<uint32_t> compute_args_cliff = {
            nblocks_per_core_cliff,
            ntiles_per_row,
            input_cb_index,
            output_cb_index,
            intermediate_cb_index,
            total_rm_rows_per_cliff_core,
        };
        CreateKernel(
            program,
            compute_kernel_path,
            core_range_cliff,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = false,
                .compile_args = compute_args_cliff,
                .defines = compute_defines});
    }

    // ── Runtime args ─────────────────────────────────────────────────────
    const bool has_cliff = !core_range_cliff.empty();
    const uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;

    const auto& cores = corerange_to_cores(all_cores);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const uint32_t cur_nblocks = (i < ncores_full) ? nblocks_per_core : nblocks_per_core_cliff;

        // Reader runtime args (matches tilize reader):
        //   0: src_addr, 1: num_rows, 2: page_size_runtime, 3: num_tiles_per_block,
        //   4: block_width_size, 5: num_full_blocks_in_row, 6: num_leftover_tiles,
        //   7: leftover_width_in_row, 8: start_page_id
        const std::array reader_rt_args = {
            src_buffer->address(),
            cur_nblocks * TILE_HEIGHT,  // num_rows
            rm_page_size,               // page_size (runtime)
            ntiles_per_row,             // num_tiles_per_block
            rm_page_size,               // block_width_size
            (uint32_t)1,                // num_full_blocks_in_row
            (uint32_t)0,                // num_leftover_tiles
            (uint32_t)0,                // leftover_width_in_row
            page_start_id};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        // Writer runtime args: dst_addr, num_tiles, start_tile_id
        const uint32_t num_tiles_this_core = cur_nblocks * ntiles_per_row;
        SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), num_tiles_this_core, tile_start_id});

        tile_start_id += num_tiles_this_core;
        page_start_id += cur_nblocks * TILE_HEIGHT;  // RM pages (rows) per core
    }

    return TypecastCrossLayoutProgramFactory::cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, ncores, grid_size.y}};
}

// ─── Public API ──────────────────────────────────────────────────────────────

TypecastCrossLayoutProgramFactory::cached_program_t TypecastCrossLayoutProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const bool input_is_tile = (tensor_args.input.layout() == Layout::TILE);
    const bool output_is_tile = (output.layout() == Layout::TILE);

    TT_FATAL(
        input_is_tile != output_is_tile,
        "TypecastCrossLayoutProgramFactory requires different input and output layouts. "
        "Input: {}, Output: {}",
        input_is_tile ? "TILE" : "ROW_MAJOR",
        output_is_tile ? "TILE" : "ROW_MAJOR");

    if (input_is_tile && !output_is_tile) {
        return create_tile_to_rm(args, tensor_args, output);
    } else {
        return create_rm_to_tile(args, tensor_args, output);
    }
}

void TypecastCrossLayoutProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;

    auto& program = cached_program.program;

    const auto& input = tensor_args.input;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
