// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <optional>

#include "ttnn/operations/embedding_backward/device/embedding_backward_device_operation.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Scratch for the output-stationary reader: indices (uint32), the pair list (uint16), three uint32 arrays per
// output tile row and a 16 byte tail.
uint32_t onehot_scratch_bytes(uint32_t num_tokens, uint32_t vocab_tiles) {
    return tt::round_up(num_tokens * 4, 16) + tt::round_up(num_tokens * 2, 16) + (3 * vocab_tiles * 4) + 16;
}

}  // namespace

bool embedding_backward_uses_onehot(const Tensor& index_tensor, const Tensor& grad_tensor, uint32_t num_embeddings) {
    if (grad_tensor.dtype() != DataType::BFLOAT16) {
        return false;
    }
    const uint32_t num_tokens = index_tensor.padded_shape()[0] * index_tensor.padded_shape()[-1];
    // The pair list stores grad tile rows as uint16.
    if (num_tokens / TILE_HEIGHT >= (1u << 16)) {
        return false;
    }
    return onehot_scratch_bytes(num_tokens, tt::div_up(num_embeddings, TILE_HEIGHT)) <= 512 * 1024;
}

namespace {

// Output-stationary path: every (output tile row, hidden chunk) unit is owned by one core, which accumulates
// onehot(indices) x grad for the grad tile rows that hit it in DEST and writes each output tile once. Returns
// std::nullopt when the shape or dtypes are outside what it handles, and the caller falls back to the scatter path.
std::optional<ProgramDescriptor> create_onehot_descriptor(
    const EmbeddingBackwardParams& operation_attributes,
    const EmbeddingBackwardInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& index_tensor = tensor_args.index_tensor;
    const auto& grad_tensor = tensor_args.grad_tensor;
    if (!embedding_backward_uses_onehot(index_tensor, grad_tensor, operation_attributes.num_embeddings) ||
        tensor_return_value.dtype() != DataType::BFLOAT16) {
        return std::nullopt;
    }
    const uint32_t num_index_pages = index_tensor.padded_shape()[0];
    const uint32_t index_page_len = index_tensor.padded_shape()[-1];
    const uint32_t num_tokens = num_index_pages * index_page_len;
    const uint32_t hidden_tiles = grad_tensor.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t vocab_tiles = tt::div_up(operation_attributes.num_embeddings, TILE_HEIGHT);

    // FP32 DEST holds 4 tiles in half-sync mode. Every chunk has the same width, so the grad and output rings (two
    // chunks each) always take whole windows and never wrap inside one.
    uint32_t chunk_tiles = 4;
    while (hidden_tiles % chunk_tiles != 0) {
        --chunk_tiles;
    }
    const uint32_t num_chunks = hidden_tiles / chunk_tiles;
    const uint32_t num_units = vocab_tiles * num_chunks;

    auto* device = grad_tensor.device();
    auto grid_size = device->compute_with_storage_grid_size();
    // Units are dealt round-robin (unit u on core u % num_cores): with contiguous ranges, skewed token
    // distributions put every frequent (low-index) row on one core. Each core buckets all output rows.
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, num_units);
    const uint32_t scratch_bytes = onehot_scratch_bytes(num_tokens, vocab_tiles);

    const tt::DataFormat bf16 = tt::DataFormat::Float16_b;
    const tt::DataFormat out_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const uint32_t tile_bytes = tt::tile_size(bf16);
    const uint32_t out_tile_bytes = tt::tile_size(out_format);
    ProgramDescriptor desc;
    auto add_cb = [&](CBIndex index, uint32_t total, uint32_t page, tt::DataFormat format) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = total,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = format,
                .page_size = page,
            }}},
        });
    };
    // The reader keeps up to 4 pairs in flight, so both pair rings hold 8 per-pair windows.
    add_cb(CBIndex::c_0, 8 * chunk_tiles * tile_bytes, tile_bytes, bf16);                 // grad tiles
    add_cb(CBIndex::c_1, 8 * tile_bytes, tile_bytes, bf16);                               // one-hot tiles
    add_cb(CBIndex::c_2, 2 * 16, 16, bf16);                                               // per-unit count
    add_cb(CBIndex::c_16, 2 * chunk_tiles * out_tile_bytes, out_tile_bytes, out_format);  // finished output tiles
    add_cb(CBIndex::c_25, scratch_bytes, scratch_bytes, bf16);                            // index and bucket scratch

    std::vector<uint32_t> reader_ct_args = {
        hidden_tiles,
        chunk_tiles,
        num_chunks,
        num_index_pages,
        index_page_len,
        static_cast<uint32_t>(index_tensor.dtype() == DataType::BFLOAT16)};
    TensorAccessorArgs(*grad_tensor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*index_tensor.buffer()).append_to(reader_ct_args);
    std::vector<uint32_t> writer_ct_args = {hidden_tiles, chunk_tiles, num_chunks};
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/dataflow/reader_embedding_backward_onehot.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/dataflow/writer_embedding_backward_onehot.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/compute/embedding_backward_onehot.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {hidden_tiles, chunk_tiles, num_chunks};
    compute_desc.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};

    uint32_t first_unit = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        reader_desc.emplace_runtime_args(
            core, {grad_tensor.buffer(), index_tensor.buffer(), first_unit, num_cores, num_units});
        writer_desc.emplace_runtime_args(core, {tensor_return_value.buffer(), first_unit, num_cores, num_units});
        compute_desc.emplace_runtime_args(core, {first_unit, num_cores, num_units});
        ++first_unit;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace

ProgramDescriptor EmbeddingBackwardDeviceOperation::create_descriptor(
    const EmbeddingBackwardParams& operation_attributes,
    const EmbeddingBackwardInputs& tensor_args,
    Tensor& tensor_return_value) {
    if (auto onehot = create_onehot_descriptor(operation_attributes, tensor_args, tensor_return_value)) {
        return std::move(*onehot);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* index_tensor_buffer = tensor_args.index_tensor.buffer();
    tt_metal::Buffer* grad_tensor_buffer = tensor_args.grad_tensor.buffer();
    tt_metal::Buffer* out_buffer = tensor_return_value.buffer();

    auto* device = tensor_args.grad_tensor.device();

    const auto& index_tensor = tensor_args.index_tensor;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t index_element_size_bytes = index_tensor.element_size();
    constexpr uint32_t INPUT_SIZE = 32;

    tt::DataFormat grad_cb_data_format = datatype_to_dataformat_converter(tensor_args.grad_tensor.dtype());
    uint32_t grad_single_tile_size = tt::tile_size(grad_cb_data_format);

    tt::DataFormat index_cb_data_format = datatype_to_dataformat_converter(tensor_args.index_tensor.dtype());
    uint32_t index_single_page_size =
        INPUT_SIZE * index_element_size_bytes;  // Only need 32 at most at a time, which is less than full page size
    uint32_t index_page_size = index_tensor.padded_shape()[-1] * index_element_size_bytes;

    tt::DataFormat mask_cb_data_format = tt::DataFormat::UInt8;
    uint32_t mask_single_page_size = INPUT_SIZE * 1;  // UInt8 is 1 byte per element

    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t embedding_dim = tensor_args.grad_tensor.padded_shape()[-1];
    uint32_t embedding_tiles = embedding_dim / TILE_WIDTH;

    uint32_t batch_size = tensor_args.index_tensor.padded_shape()[0];
    uint32_t seq_len_tiles = tensor_args.index_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t input_height_tiles = batch_size * seq_len_tiles;

    // Round up: the output is tile-padded, so a weight table whose row count is not a
    // multiple of TILE_HEIGHT still owns a full trailing tile row. Truncating here left
    // that tile row out of the reader's output zero-init loop, so it kept whatever was
    // in the freshly allocated DRAM and the accumulation added on top of garbage.
    uint32_t num_embeddings_tiles = tt::div_up(operation_attributes.num_embeddings, TILE_HEIGHT);

    // We split work based on the number of tiles in the embedding dimension
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid_size, embedding_tiles);
    uint32_t max_tiles_per_core = std::max(num_tiles_per_core_group_1, num_tiles_per_core_group_2);

    log_debug(LogType::LogOp, "Embedding hidden size tiles: {}", embedding_tiles);
    log_debug(LogType::LogOp, "Num parallel cores: {}", num_cores);
    log_debug(LogType::LogOp, "Max hidden size tiles per core: {}", max_tiles_per_core);

    ////////////////////////////////////////////////////////////////////////////
    //                 Circular buffers
    ////////////////////////////////////////////////////////////////////////////

    ProgramDescriptor desc;

    // To read from grad tensor
    desc.cbs.push_back(CBDescriptor{
        .total_size = max_tiles_per_core * grad_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = grad_cb_data_format,
            .page_size = grad_single_tile_size,
        }}},
    });

    // To store index values for a single tile
    desc.cbs.push_back(CBDescriptor{
        .total_size = index_single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
            .data_format = index_cb_data_format,
            .page_size = index_single_page_size,
        }}},
    });

    // To read from output tensor: two sets of tiles, so the next chunk is read while the current one is computed
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * max_tiles_per_core * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // To store mask values for a single tile
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = mask_cb_data_format,
            .page_size = mask_single_page_size,
        }}},
    });

    // L1 scratch space to pass chunk_count from reader to UNPACK
    desc.cbs.push_back(CBDescriptor{
        .total_size = 16,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_25),
            .data_format = grad_cb_data_format,
            .page_size = 16,
        }}},
    });  // grad_cb_data_format doesn't matter here

    // For tiles to be written to the output: two sets, so compute can start the next chunk while one is written back
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * max_tiles_per_core * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    ////////////////////////////////////////////////////////////////////////////
    //                 Kernels
    ////////////////////////////////////////////////////////////////////////////

    // reader

    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)max_tiles_per_core,
        (uint32_t)batch_size,
        (uint32_t)seq_len_tiles,
        (uint32_t)num_embeddings_tiles,
        (uint32_t)index_page_size,
        (uint32_t)(tensor_args.index_tensor.dtype() == DataType::BFLOAT16)};
    TensorAccessorArgs(*grad_tensor_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*index_tensor_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*out_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/dataflow/reader_embedding_backward.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/embedding_backward/device/kernels/compute/embedding_backward.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    // A FLOAT32 output is accumulated in 32-bit DEST. The partial output tile is unpacked straight to DEST so it is
    // not rounded to TF32 on its way through SrcA. A bfloat16 or bfloat8_b gradient is exact in TF32, so it goes
    // through SrcA into the FP32 DEST without a separate cast pass; only a FLOAT32 gradient needs the direct unpack.
    const bool fp32_accumulate = tensor_return_value.dtype() == DataType::FLOAT32;
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_accumulate) {
        if (tensor_args.grad_tensor.dtype() == DataType::FLOAT32) {
            unpack_to_dest_mode[CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
        }
        unpack_to_dest_mode[CBIndex::c_2] = UnpackToDestMode::UnpackToDestFp32;
    }
    compute_desc.compile_time_args = {max_tiles_per_core, input_height_tiles};
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_accumulate,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                 Run-time arguments
    ////////////////////////////////////////////////////////////////////////////

    auto cores = corerange_to_cores(all_cores);
    uint32_t offset = 0;
    for (auto core : cores) {
        uint32_t tiles_this_core = 0;
        if (core_group_1.contains(core)) {
            tiles_this_core = num_tiles_per_core_group_1;
        } else {
            tiles_this_core = num_tiles_per_core_group_2;
        }
        // Buffer* entries register for fast cache-hit address patching (see PR #42992).
        reader_desc.emplace_runtime_args(
            core, {grad_tensor_buffer, index_tensor_buffer, out_buffer, embedding_tiles, offset, tiles_this_core});
        compute_desc.emplace_runtime_args(core, {tiles_this_core});

        offset += tiles_this_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
