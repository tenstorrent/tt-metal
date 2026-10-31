// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flash_mqa_decode_device_operation.hpp"

#include <bit>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tile.hpp>

namespace ttnn::operations::experimental::deepseek::flash_mqa_decode {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr const char* kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/dataflow/"
    "reader_flash_mqa_decode.cpp";
constexpr const char* kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/dataflow/"
    "writer_flash_mqa_decode.cpp";
constexpr const char* kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/compute/flash_mqa_decode.cpp";

// Half-sync bf16 dst holds 8 tiles; the tail reduction works in blocks of that size.
constexpr uint32_t kDstSize = 8;
constexpr uint32_t kReducerSemaphoreId = 0;

}  // namespace

tt::tt_metal::ProgramDescriptor FlashMqaDecodeDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q = tensor_args.q;
    const auto& kv = tensor_args.kv;
    const auto& lengths = tensor_args.lengths;
    const bool has_selected = tensor_args.selected_kv.has_value();
    const bool has_sink = tensor_args.attention_sink.has_value();
    auto* device = q.device();

    const uint32_t heads = q.logical_shape()[2];
    const uint32_t head_dim = q.logical_shape()[3];
    const uint32_t DHt = head_dim / TILE_WIDTH;
    const uint32_t vDHt = DHt;
    const uint32_t Sk_chunk_t = args.k_chunk_size / TILE_HEIGHT;
    const uint32_t num_head_groups = heads / kHeadsPerCore;
    const uint32_t num_s_blocks = args.num_s_blocks;
    uint32_t num_tree_steps = 0;
    while ((1u << num_tree_steps) < num_s_blocks) {
        ++num_tree_steps;
    }

    const uint32_t kv_rows = kv.logical_shape()[2];
    const uint32_t ring_capacity = args.ring_rows ? args.ring_rows : kv_rows;
    const bool has_ring_offset = tensor_args.lengths.logical_shape()[-1] >= 3;
    const uint32_t selected_capacity =
        has_selected ? tensor_args.selected_kv->logical_shape()[2] - args.selected_row_offset : 0;
    const uint32_t row_bytes = head_dim * static_cast<uint32_t>(q.element_size());

    const tt::DataFormat bf16_df = tt::DataFormat::Float16_b;
    const tt::DataFormat kv_df = datatype_to_dataformat_converter(kv.dtype());
    const Tile tiny_tile({kHeadsPerCore, TILE_WIDTH});
    const Tile full_tile({TILE_HEIGHT, TILE_WIDTH});
    const uint32_t tiny_tile_bytes = tiny_tile.get_tile_size(bf16_df);
    const uint32_t k_tile_bytes = full_tile.get_tile_size(kv_df);
    const uint32_t sink_tile_bytes = full_tile.get_tile_size(bf16_df);
    const std::optional<TileDescriptor> tiny_desc{TileDescriptor{tiny_tile}};
    const std::optional<TileDescriptor> full_desc{TileDescriptor{full_tile}};

    std::vector<CoreCoord> cores;
    for (uint32_t g = 0; g < num_head_groups; ++g) {
        for (uint32_t b = 0; b < num_s_blocks; ++b) {
            cores.emplace_back(b, g);
        }
    }
    const CoreRangeSet all_cores(CoreRange({0, 0}, {num_s_blocks - 1, num_head_groups - 1}));

    constexpr uint8_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint8_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint8_t cb_mask = tt::CBIndex::c_2;
    constexpr uint8_t cb_ms_in = tt::CBIndex::c_3;
    constexpr uint8_t cb_out_in = tt::CBIndex::c_4;
    constexpr uint8_t cb_out_o = tt::CBIndex::c_5;
    constexpr uint8_t cb_out_ms = tt::CBIndex::c_6;
    constexpr uint8_t cb_interm_out = tt::CBIndex::c_7;
    constexpr uint8_t cb_interm_ms = tt::CBIndex::c_8;
    constexpr uint8_t cb_out_rm = tt::CBIndex::c_9;
    constexpr uint8_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint8_t cb_lengths = tt::CBIndex::c_11;
    constexpr uint8_t cb_reader_lengths = tt::CBIndex::c_12;
    constexpr uint8_t cb_writer_lengths = tt::CBIndex::c_13;
    constexpr uint8_t cb_sink_scratch = tt::CBIndex::c_14;

    tt::tt_metal::ProgramDescriptor desc;
    const auto add_cb = [&](uint8_t index,
                            uint32_t pages,
                            tt::DataFormat df,
                            uint32_t page_size,
                            const std::optional<TileDescriptor>& tile,
                            std::optional<FaceGeometry> face_geometry = std::nullopt) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = pages * page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index,
                .data_format = df,
                .page_size = page_size,
                .tile = tile,
                .face_geometry = face_geometry}}},
        });
    };
    const uint32_t receive_slots = num_tree_steps + 1;
    add_cb(cb_q_in, DHt, bf16_df, tiny_tile_bytes, tiny_desc);
    // Double-buffered so the next chunk's DRAM read overlaps this chunk's compute.
    add_cb(cb_k_in, 2 * Sk_chunk_t * DHt, kv_df, k_tile_bytes, full_desc);
    add_cb(cb_mask, 2, bf16_df, tiny_tile_bytes, std::nullopt);
    add_cb(cb_ms_in, receive_slots, bf16_df, tiny_tile_bytes, tiny_desc);
    add_cb(cb_out_in, receive_slots * vDHt, bf16_df, tiny_tile_bytes, tiny_desc);
    add_cb(cb_out_o, vDHt, bf16_df, tiny_tile_bytes, tiny_desc);
    add_cb(cb_out_ms, 2, bf16_df, tiny_tile_bytes, tiny_desc);
    add_cb(cb_interm_out, vDHt, bf16_df, tiny_tile_bytes, tiny_desc);
    add_cb(cb_interm_ms, 2, bf16_df, tiny_tile_bytes, tiny_desc);
    add_cb(
        cb_out_rm,
        vDHt,
        bf16_df,
        tiny_tile_bytes,
        tiny_desc,
        FaceGeometry{.face_r_dim = kHeadsPerCore, .num_faces = tiny_tile.get_num_faces()});
    add_cb(cb_q_rm, DHt, bf16_df, tiny_tile_bytes, tiny_desc);
    constexpr uint32_t lengths_page_bytes = 32;
    add_cb(cb_lengths, 1, tt::DataFormat::UInt32, lengths_page_bytes, std::nullopt);
    add_cb(cb_reader_lengths, 1, tt::DataFormat::UInt32, lengths_page_bytes, std::nullopt);
    add_cb(cb_writer_lengths, 1, tt::DataFormat::UInt32, lengths_page_bytes, std::nullopt);
    add_cb(cb_sink_scratch, 1, bf16_df, sink_tile_bytes, full_desc);

    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = kReducerSemaphoreId, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});

    Buffer* selected_buffer = has_selected ? tensor_args.selected_kv->buffer() : nullptr;
    Buffer* sink_buffer = has_sink ? tensor_args.attention_sink->buffer() : nullptr;

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderKernelPath;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.named_compile_time_args = {
        {"DHt", DHt},
        {"Sk_chunk_t", Sk_chunk_t},
        {"k_chunk_size", args.k_chunk_size},
        {"num_s_blocks", num_s_blocks},
        {"heads_per_core", kHeadsPerCore},
        {"q_row_bytes", row_bytes},
        {"ring_capacity", ring_capacity},
        {"has_ring_offset", has_ring_offset ? 1u : 0u},
        {"max_ring_first_tile_row", (kv_rows - ring_capacity) / TILE_HEIGHT},
        {"selected_capacity", selected_capacity},
        {"selected_first_tile_row", args.selected_row_offset / TILE_HEIGHT},
        {"cb_q_rm", cb_q_rm},
        {"cb_k_in", cb_k_in},
        {"cb_mask", cb_mask},
        {"cb_lengths", cb_lengths},
        {"cb_lengths_scratch", cb_reader_lengths},
    };
    TensorAccessorArgs(q.buffer()).append_to(reader_desc.compile_time_args);
    TensorAccessorArgs(kv.buffer()).append_to(reader_desc.compile_time_args);
    TensorAccessorArgs(selected_buffer).append_to(reader_desc.compile_time_args);
    TensorAccessorArgs(lengths.buffer()).append_to(reader_desc.compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterKernelPath;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.named_compile_time_args = {
        {"vDHt", vDHt},
        {"k_chunk_size", args.k_chunk_size},
        {"num_s_blocks", num_s_blocks},
        {"num_tree_steps", num_tree_steps},
        {"heads_per_core", kHeadsPerCore},
        {"out_row_bytes", row_bytes},
        {"ring_capacity", ring_capacity},
        {"selected_capacity", selected_capacity},
        {"has_sink", has_sink ? 1u : 0u},
        {"reducer_semaphore_id", kReducerSemaphoreId},
        {"cb_ms_in", cb_ms_in},
        {"cb_out_in", cb_out_in},
        {"cb_out_o", cb_out_o},
        {"cb_out_ms", cb_out_ms},
        {"cb_out_rm", cb_out_rm},
        {"cb_lengths_scratch", cb_writer_lengths},
        {"cb_sink_scratch", cb_sink_scratch},
    };
    TensorAccessorArgs(output.buffer()).append_to(writer_desc.compile_time_args);
    TensorAccessorArgs(lengths.buffer()).append_to(writer_desc.compile_time_args);
    TensorAccessorArgs(sink_buffer).append_to(writer_desc.compile_time_args);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeKernelPath;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .math_approx_mode = math_approx_mode,
    };
    compute_desc.named_compile_time_args = {
        {"DHt", DHt},
        {"vDHt", vDHt},
        {"Sk_chunk_t", Sk_chunk_t},
        {"k_chunk_size", args.k_chunk_size},
        {"num_s_blocks", num_s_blocks},
        {"num_tree_steps", num_tree_steps},
        {"scale_fp32", std::bit_cast<uint32_t>(args.scale)},
        {"dst_size", kDstSize},
        {"cb_q_in", cb_q_in},
        {"cb_q_rm", cb_q_rm},
        {"cb_k_in", cb_k_in},
        {"cb_mask", cb_mask},
        {"cb_ms_in", cb_ms_in},
        {"cb_out_in", cb_out_in},
        {"cb_out_o", cb_out_o},
        {"cb_out_ms", cb_out_ms},
        {"cb_interm_out", cb_interm_out},
        {"cb_interm_ms", cb_interm_ms},
        {"cb_out_rm", cb_out_rm},
        {"cb_lengths", cb_lengths},
    };

    for (const auto& core : cores) {
        const uint32_t s_block = core.x;
        const uint32_t head_group = core.y;
        reader_desc.emplace_runtime_args(
            core,
            std::vector<std::variant<uint32_t, Buffer*>>{
                q.buffer(),
                kv.buffer(),
                selected_buffer != nullptr ? std::variant<uint32_t, Buffer*>{selected_buffer}
                                           : std::variant<uint32_t, Buffer*>{0u},
                lengths.buffer(),
                s_block,
                head_group});

        std::vector<std::variant<uint32_t, Buffer*>> writer_args{
            output.buffer(),
            lengths.buffer(),
            sink_buffer != nullptr ? std::variant<uint32_t, Buffer*>{sink_buffer} : std::variant<uint32_t, Buffer*>{0u},
            s_block,
            head_group};
        for (uint32_t b = 0; b < num_s_blocks; ++b) {
            const auto noc = device->worker_core_from_logical_core(CoreCoord(b, head_group));
            writer_args.emplace_back(static_cast<uint32_t>(noc.x));
            writer_args.emplace_back(static_cast<uint32_t>(noc.y));
        }
        writer_desc.emplace_runtime_args(core, writer_args);

        compute_desc.emplace_runtime_args(core, {s_block});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::flash_mqa_decode
