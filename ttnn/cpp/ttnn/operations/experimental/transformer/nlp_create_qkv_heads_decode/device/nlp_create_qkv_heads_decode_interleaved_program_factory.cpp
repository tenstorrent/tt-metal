// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_interleaved_program_factory.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor NLPCreateQKVHeadsDecodeInterleavedProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& num_q_heads = operation_attributes.num_q_heads;
    const auto& num_kv_heads = operation_attributes.num_kv_heads;
    const auto& head_dim = operation_attributes.head_dim;

    ProgramDescriptor desc;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    uint32_t q_output_cb_index = CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = q_num_tiles * single_tile_size,
        .core_ranges = q_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(q_output_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output[0].buffer(),
    });

    auto k_shard_spec = output[1].shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t k_output_cb_index = CBIndex::c_17;
    desc.cbs.push_back(CBDescriptor{
        .total_size = k_num_tiles * single_tile_size,
        .core_ranges = k_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(k_output_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output[1].buffer(),
    });

    auto v_shard_spec = output[2].shard_spec().value();
    auto v_cores = q_shard_spec.grid;
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    uint32_t v_output_cb_index = CBIndex::c_18;
    desc.cbs.push_back(CBDescriptor{
        .total_size = v_num_tiles * single_tile_size,
        .core_ranges = v_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(v_output_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output[2].buffer(),
    });

    Buffer* in_buffer = input_tensor.buffer();

    // The reader kernel reads each face row as a single 16-element noc_async_read transaction
    // (`16 * element_size` bytes). When the input is DRAM-interleaved and that read size is below
    // the device DRAM read alignment (Blackhole bf16: 32 < 64), the NOC alignment rule
    // ((src & (alignment-1)) == (dst & (alignment-1))) is violated for half the (batch, head)
    // parities and the read silently returns wrong data (issue #43270). When that condition
    // holds, switch the kernel to a DRAM-aligned scratch+memcpy path; otherwise the original
    // direct-read fast path runs unchanged. Sharded inputs do not go through this factory.
    const bool is_dram = in_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const bool use_aligned_path = is_dram && (sub_tile_line_bytes < dram_alignment);

    // Per-RISC scratch CB sized for one DRAM-aligned chunk per tile in a single head. The two
    // RISCs read different phases concurrently, so they need independent scratch slots — assign
    // distinct CB indices. The kernel reads DRAM-aligned chunks into this CB; the NOC requires
    // (src & (alignment-1)) == (dst & (alignment-1)). Since the source addresses are aligned to
    // dram_alignment, the destination addresses inside the scratch must also be aligned to
    // dram_alignment. CBs in L1 are only allocated at L1 alignment (16 B on BH), so oversize the
    // CB by one dram_alignment chunk and have the kernel round its base up.
    constexpr uint32_t reader_scratch_cb_index = CBIndex::c_0;
    constexpr uint32_t writer_scratch_cb_index = CBIndex::c_1;
    if (use_aligned_path) {
        const uint32_t scratch_size_bytes = (head_tiles + 1) * dram_alignment;
        // Float16_b is just a placeholder DataFormat for this scratch CB — the kernel only treats
        // it as raw L1 storage and copies bytes via memcpy.
        desc.cbs.push_back(CBDescriptor{
            .total_size = scratch_size_bytes,
            .core_ranges = q_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(reader_scratch_cb_index),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = dram_alignment,
            }}},
        });
        desc.cbs.push_back(CBDescriptor{
            .total_size = scratch_size_bytes,
            .core_ranges = q_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(writer_scratch_cb_index),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = dram_alignment,
            }}},
        });
    }

    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2
    // of a tile respectively)
    std::vector<uint32_t> reader_compile_time_args = {
        element_size,
        sub_tile_line_bytes,
        q_output_cb_index,
        k_output_cb_index,
        v_output_cb_index,
        head_size,
        num_q_heads,
        num_kv_heads,
        head_tiles,
        1,  // read the first phase
        static_cast<uint32_t>(use_aligned_path),
        dram_alignment,
        reader_scratch_cb_index,
    };
    tt::tt_metal::TensorAccessorArgs(in_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = q_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = reader_compile_time_args;
    writer_compile_time_args[9] = 2;  // read the second phase
    writer_compile_time_args[12] = writer_scratch_cb_index;

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = q_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * sub_tile_line_bytes : ((i - 16) * sub_tile_line_bytes) + (512 * element_size);

        const auto& core = cores[i];
        reader_desc.emplace_runtime_args(core, {in_tile_offset_by_batch, in_buffer});
        writer_desc.emplace_runtime_args(core, {in_tile_offset_by_batch, in_buffer});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
