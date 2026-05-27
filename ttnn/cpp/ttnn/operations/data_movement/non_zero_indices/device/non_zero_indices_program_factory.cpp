// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;

// Compute geometry parameters for the given input tensor.
// Returns the input CB page size and fills:
//   - geom_args: the non-buffer runtime args starting at arg[3] (after the three Buffer* args)
//   - aligned_output_bytes: aligned byte size of the output indices buffer page
uint32_t compute_geometry(
    const Tensor& input, const Tensor& out_indices, std::vector<uint32_t>& geom_args, uint32_t& aligned_output_bytes) {
    const bool is_tile = (input.layout() == Layout::TILE);
    const auto& lshape = input.logical_shape();
    const auto& pshape = input.padded_shape();
    aligned_output_bytes = out_indices.buffer()->aligned_page_size();

    if (!is_tile) {
        // Use the buffer's physical page size so TensorAccessor gets the correct page size for
        // all layouts (interleaved, HEIGHT/WIDTH/BLOCK sharded).
        const uint32_t phys_page_bytes = input.buffer()->page_size();
        const uint32_t elements_per_page = phys_page_bytes / input.element_size();
        const uint32_t num_pages = input.buffer()->num_dev_pages();
        const uint32_t aligned_page_size = input.buffer()->aligned_page_size();

        const uint32_t logical_last_dim = lshape[3];
        const uint32_t grid_w = (elements_per_page < logical_last_dim) ? (logical_last_dim / elements_per_page) : 1;

        uint32_t pages_per_bank = 1;
        uint32_t is_col_major = 0;
        const auto mem_layout = input.memory_config().memory_layout();
        if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED || mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            const auto& shard_spec = input.memory_config().shard_spec().value();
            pages_per_bank = shard_spec.shape[0];
            is_col_major = (shard_spec.orientation == ShardOrientation::COL_MAJOR) ? 1u : 0u;
        }

        const uint32_t total_rows = num_pages / grid_w;
        const uint32_t grid_h = (pages_per_bank > 0) ? (total_rows / pages_per_bank) : 1;

        geom_args = {
            aligned_output_bytes,
            num_pages,
            elements_per_page,
            aligned_page_size,
            logical_last_dim,
            pages_per_bank,
            grid_w,
            lshape[1],  // logical_N: N-dimension for (b,n,h,c) index decomposition
            lshape[2],  // logical_H: H-dimension for (b,n,h,c) index decomposition
            grid_h,     // number of core rows (needed for COL_MAJOR bank ordering)
            is_col_major};
        return aligned_page_size;
    } else {
        const uint32_t tile_page_size = input.buffer()->aligned_page_size();

        const uint32_t B = lshape[0];
        const uint32_t N = lshape[1];
        const uint32_t logical_H = lshape[2];
        const uint32_t logical_C = lshape[3];
        const uint32_t num_tile_rows = pshape[2] / TILE_H;
        const uint32_t num_tile_cols = pshape[3] / TILE_W;

        geom_args = {aligned_output_bytes, B, N, logical_H, logical_C, num_tile_rows, num_tile_cols, tile_page_size};
        return tile_page_size;
    }
}

}  // namespace

ProgramDescriptor NonZeroIndicesProgramFactory::create_descriptor(
    const NonzeroParams& /*operation_attributes*/, const NonzeroInputs& tensor_args, NonzeroResult& output_tensors) {
    const auto& input = tensor_args.input;
    const auto& out_num_indices = std::get<0>(output_tensors);
    const auto& out_indices = std::get<1>(output_tensors);

    const bool is_tile = (input.layout() == Layout::TILE);

    std::vector<uint32_t> geom_args;
    uint32_t aligned_output_bytes = 0;
    const uint32_t input_page_size = compute_geometry(input, out_indices, geom_args, aligned_output_bytes);

    const CoreCoord core = {0, 0};
    const CoreRangeSet core_ranges{CoreRange{core, core}};

    constexpr uint32_t input_cb_index = 0;
    constexpr uint32_t output_cb_index_0 = 1;
    constexpr uint32_t output_cb_index_1 = 2;

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(DataType::UINT32);

    ProgramDescriptor desc;

    // Input CB: single-buffered (barrier after each read; no double-buffering benefit)
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_page_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_page_size,
        }}},
    });

    // Output CB 0: count tensor (32 bytes fixed)
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * 32,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index_0,
            .data_format = output_cb_data_format,
            .page_size = 32,
        }}},
    });

    // Output CB 1: indices staging buffer (worst case: all elements non-zero → volume * 4 * 4 bytes)
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * aligned_output_bytes,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index_1,
            .data_format = output_cb_data_format,
            .page_size = aligned_output_bytes,
        }}},
    });

    KernelDescriptor::Defines defines = {{"NUM_BYTES", std::to_string(input.element_size())}};
    if (is_tile) {
        defines.push_back({"INPUT_IS_TILE", "1"});
    }

    std::vector<uint32_t> compile_time_args = {
        static_cast<uint32_t>(input_cb_index),
        static_cast<uint32_t>(output_cb_index_0),
        static_cast<uint32_t>(output_cb_index_1),
    };
    TensorAccessorArgs(*input.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*out_num_indices.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*out_indices.buffer()).append_to(compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/non_zero_indices/device/kernels/dataflow/"
        "non_zero_indices_sc_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(compile_time_args);
    reader_desc.defines = std::move(defines);
    reader_desc.config = ReaderConfigDescriptor{};

    // Buffer* slots are auto-patched by the framework on program cache hits.
    // Geometry uint32_t args are fixed by the tensor spec (part of the program cache key).
    std::vector<std::variant<uint32_t, Buffer*>> runtime_args_list;
    runtime_args_list.push_back(input.buffer());
    runtime_args_list.push_back(out_num_indices.buffer());
    runtime_args_list.push_back(out_indices.buffer());
    for (uint32_t v : geom_args) {
        runtime_args_list.push_back(v);
    }
    reader_desc.emplace_runtime_args(core, runtime_args_list);

    desc.kernels.push_back(std::move(reader_desc));
    return desc;
}

}  // namespace ttnn::prim
