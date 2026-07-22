// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "nlp_create_qkv_heads_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor NlpCreateHeadsDeviceOperation::Interleaved::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor_q;
    std::optional<const Tensor> input_tensor_kv = tensor_args.input_tensor_kv;
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k_heads = operation_attributes.transpose_k_heads;
    auto& output = tensor_return_value;
    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    const auto& input_shape = input_tensor.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = input_tensor.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    tt_metal::Buffer* in1_buffer = nullptr;
    if (read_from_input_tensor_kv) {
        in1_buffer = input_tensor_kv.value().buffer();
        TT_ASSERT(in1_buffer->size() % single_tile_size == 0);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = input_tensor_kv.value().padded_shape()[3] / TILE_WIDTH;
    }

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    // Output shape for K/V is: [B, num_kv_heads, s, head_dim], shuffled from [B, 1, s, num_kv_heads * head_dim]
    // NOTE: Output h and w dims are identical for Q, K, V, so any arg that is related to these dims for q_* can be
    // shared for K, V
    uint32_t q_out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // tiles along head_dim
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of in0_w_tiles per core
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Tensor& q = std::get<0>(output);
    tt_metal::Tensor& k = std::get<1>(output);
    tt_metal::Tensor& v = std::get<2>(output);

    tt_metal::Buffer* q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer* k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer* v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)q_num_tiles,
        (std::uint32_t)kv_num_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args);
    // Always append placeholder/accessor for in1 to keep offsets stable
    tt::tt_metal::TensorAccessorArgs(read_from_input_tensor_kv ? in1_buffer : nullptr)
        .append_to(reader_compile_time_args);

    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)q_out_h_tiles,
        (std::uint32_t)q_out_w_tiles,
        (std::uint32_t)q_out_HtWt,
        (std::uint32_t)num_q_heads,   // q_out_c
        (std::uint32_t)num_kv_heads,  // kv_out_c
    };
    tt::tt_metal::TensorAccessorArgs(q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;
    if (transpose_k_heads) {
        // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
        // resolves the unpack-dst CB to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
        // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
        const bool fp32_dest_acc_en = input_tensor.dtype() == tt_metal::DataType::FLOAT32;

        std::vector<uint32_t> compute_args_core_group_1 = {num_blocks_per_core_group_1 * kv_num_tiles};
        KernelDescriptor compute_desc_1;
        compute_desc_1.kernel_source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
        compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_1.core_ranges = core_group_1;
        compute_desc_1.compile_time_args = std::move(compute_args_core_group_1);
        compute_desc_1.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};
        desc.kernels.push_back(std::move(compute_desc_1));

        if (core_group_2.num_cores() > 0) {
            std::vector<uint32_t> compute_args_core_group_2 = {num_blocks_per_core_group_2 * kv_num_tiles};
            KernelDescriptor compute_desc_2;
            compute_desc_2.kernel_source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
            compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc_2.core_ranges = core_group_2;
            compute_desc_2.compile_time_args = std::move(compute_args_core_group_2);
            compute_desc_2.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};
            desc.kernels.push_back(std::move(compute_desc_2));
        }

        reader_defines.emplace_back("TRANSPOSE_K_HEADS", "1");
        writer_defines.emplace_back("TRANSPOSE_K_HEADS", "1");
    }
    if (read_from_input_tensor_kv) {
        reader_defines.emplace_back("READ_FROM_INPUT_TENSOR_KV", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "writer_tm_tile_layout_nlp_create_qkv_heads.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    // Create circular buffers
    uint32_t micro_block_size = 1;                 // Num tiles to read/wait for in reader and writer
    uint32_t cb_num_tiles = micro_block_size * 4;  // Quadruple buffer everything

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t cb1_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    uint32_t src1_cb_index = 1;  // cb0 is needed for compute if we want to use generic transpose_wh compute kernel
    uint32_t cb1_num_tiles = cb_num_tiles;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb1_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // If we transpose_k_heads:
    // - reader will write to cb0, instead of cb1
    // - compute will wait on cb0 and write to cb16
    // - writer will wait on cb 16, instead of cb1
    if (transpose_k_heads) {
        uint32_t src0_cb_index = 0;
        uint32_t cb0_num_tiles = cb_num_tiles;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb0_num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src0_cb_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });

        uint32_t out_cb_index = 16;
        uint32_t out_cb_num_tiles = cb_num_tiles;
        desc.cbs.push_back(CBDescriptor{
            .total_size = out_cb_num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(out_cb_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    }

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        KernelDescriptor::RTArgList reader_rt;
        reader_rt.reserve(5);
        reader_rt.push_back(in0_buffer);
        if (in1_buffer != nullptr) {
            reader_rt.push_back(in1_buffer);
        } else {
            reader_rt.push_back(uint32_t{0});
        }
        reader_rt.push_back(num_blocks_per_core);
        reader_rt.push_back(num_blocks_written * in0_w_tiles);
        reader_rt.push_back(num_blocks_written * in1_w_tiles);
        reader_desc.emplace_runtime_args(core, reader_rt);

        writer_desc.emplace_runtime_args(
            core,
            {
                q_buffer,              // q_tensor_addr
                k_buffer,              // k_tensor_addr
                v_buffer,              // v_tensor_addr
                num_blocks_per_core,   // num_blocks
                q_out_h_dim,           // q_out_h_dim
                q_out_tensor_tile_id,  // q_out_tensor_tile_id
                k_out_tensor_tile_id,  // k_out_tensor_tile_id
                v_out_tensor_tile_id,  // v_out_tensor_tile_id
            });

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

namespace {

// Runtime-arg slot layout for the Sharded reader/writer.  Slots 6 and 15 carry the *clean* input-buffer
// bases (bound as Buffer* in create_descriptor(), so the framework patches them on every cache hit); slots
// 7 and 16 carry the *byte offsets* of the Q and K/V regions within a remote input shard.  The kernel adds
// base + region offset + per-core head offset itself — no host-folded `base + offset` address is passed, so
// no get_dynamic_runtime_args() re-application hook is needed.
constexpr uint32_t kQBaseAddrIdx = 6;    // q buffer base   -> Buffer* binding
constexpr uint32_t kKVBaseAddrIdx = 15;  // k/v buffer base -> Buffer* binding

struct ShardedCoreArgs {
    CoreCoord core;
    std::vector<uint32_t> reader_args;
    std::vector<uint32_t> writer_args;
};

// Single source of truth for the Sharded per-core reader/writer runtime args.  The base-address slots
// (6, 15) hold a placeholder 0 here and are replaced by the input-buffer Buffer* in create_descriptor();
// slots 7/16 hold the Q / K-or-V region byte offset within a shard, which the kernel adds to the base.
std::vector<ShardedCoreArgs> build_sharded_core_args(
    const NlpCreateHeadsDeviceOperation::operation_attributes_t& operation_attributes,
    const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args,
    NlpCreateHeadsDeviceOperation::tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto head_dim = operation_attributes.head_dim;
    auto num_q_heads = operation_attributes.num_q_heads;
    auto num_kv_heads = operation_attributes.num_kv_heads;

    tt_metal::IDevice* device = input_tensor.device();
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    auto q_shard_spec = std::get<0>(output).shard_spec().value();
    auto q_cores = q_shard_spec.grid;

    uint32_t per_core_out_q_heads = num_q_heads / q_cores.num_cores();
    uint32_t per_risc0_out_q_heads = div_up(per_core_out_q_heads, 2);
    uint32_t per_risc1_out_q_heads = per_core_out_q_heads / 2;
    uint32_t per_core_in_q_heads = num_q_heads / input_tensor.shard_spec().value().num_cores();

    auto k_shard_spec = std::get<1>(output).shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t per_core_out_kv_heads = num_kv_heads / k_cores.num_cores();
    uint32_t per_core_in_kv_heads =
        num_kv_heads / (read_from_input_tensor_kv ? input_tensor_kv.value().shard_spec().value().num_cores()
                                                  : input_tensor.shard_spec().value().num_cores());

    // Region byte-offsets of Q / K / V within a remote input shard.  Q always starts at offset 0.
    // Fused-QKV shard layout is [Q | K | V]; with a separate KV tensor the KV shard is [K | V].  These
    // replace the old host-folded `base + region` addresses: the kernel adds base + region offset itself.
    uint32_t q_region_offset = 0;
    uint32_t k_region_offset = read_from_input_tensor_kv ? 0u : per_core_in_q_heads * head_size;
    uint32_t v_region_offset = read_from_input_tensor_kv ? per_core_in_kv_heads * head_size
                                                         : (per_core_in_q_heads + per_core_in_kv_heads) * head_size;

    uint32_t num_cores = std::max(q_cores.num_cores(), k_cores.num_cores());
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(num_cores_x);
    for (uint32_t x = 0; x < num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(num_cores_y);
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    uint32_t remote_q_head_start_idx = 0;
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t q_x = 0, q_y = 0, kv_x = 0, kv_y = 0;

    uint32_t remote_q_read = 0;
    uint32_t remote_kv_read = 0;

    std::vector<ShardedCoreArgs> result;
    result.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        bool read_kv_heads = i < k_cores.num_cores();
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(18 + num_cores_x + num_cores_y);
        reader_runtime_args = {
            head_size,                 // 0
            per_risc0_out_q_heads,     // 1
            per_core_in_q_heads,       // 2
            remote_q_head_start_idx,   // 3
            q_x,                       // 4
            q_y,                       // 5
            0,                         // 6  q buffer base -> Buffer* binding (placeholder)
            q_region_offset,           // 7  Q region byte offset within shard (0)
            0,                         // 8  Q L1 write offset (reader = 0)
            read_kv_heads,             // 9
            per_core_out_kv_heads,     // 10
            per_core_in_kv_heads,      // 11
            remote_kv_head_start_idx,  // 12
            kv_x,                      // 13
            kv_y,                      // 14
            0,                         // 15 k/v buffer base -> Buffer* binding (placeholder)
            k_region_offset,           // 16 K region byte offset within shard
            k_num_tiles,               // 17
            num_cores_x,               // 18
        };
        reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

        remote_q_read += per_risc0_out_q_heads;
        q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
        q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;

        // Reader gets the args as built above (risc0 values); writer gets the same vector with the
        // risc1 q values patched, and (for kv cores) the V region offset over slot 16.  Slots 6/15
        // stay 0 here — create_descriptor() binds the input buffers there as Buffer* for both.
        std::vector<uint32_t> writer_runtime_args = reader_runtime_args;

        writer_runtime_args[1] = per_risc1_out_q_heads;
        writer_runtime_args[3] = remote_q_head_start_idx;
        writer_runtime_args[4] = q_x;
        writer_runtime_args[5] = q_y;
        writer_runtime_args[8] = per_risc0_out_q_heads * head_size;  // risc1 writes Q after risc0's heads

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
            q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
        }

        if (read_kv_heads) {
            // Writer reads V where reader read K: same clean buffer base (slot 15), but the V region
            // sits after K within the shard.
            writer_runtime_args[16] = v_region_offset;
            remote_kv_read += per_core_out_kv_heads;
            kv_y = (remote_kv_read / per_core_in_kv_heads) / num_cores_x;
            kv_x = (remote_kv_read / per_core_in_kv_heads) % num_cores_x;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
        }

        result.push_back({core, std::move(reader_runtime_args), std::move(writer_runtime_args)});
    }

    return result;
}

}  // namespace

ProgramDescriptor NlpCreateHeadsDeviceOperation::Sharded::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    auto& output = tensor_return_value;

    ProgramDescriptor desc;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    auto q_shard_spec = std::get<0>(output).shard_spec().value();
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
        .buffer = std::get<0>(output).buffer(),
    });

    auto k_shard_spec = std::get<1>(output).shard_spec().value();
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
        .buffer = std::get<1>(output).buffer(),
    });

    auto v_shard_spec = std::get<2>(output).shard_spec().value();
    auto v_cores = v_shard_spec.grid;
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
        .buffer = std::get<2>(output).buffer(),
    });

    std::vector<uint32_t> reader_compile_time_args = {q_output_cb_index, k_output_cb_index};
    std::vector<uint32_t> writer_compile_time_args = {q_output_cb_index, v_output_cb_index};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = q_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = q_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Build the per-core reader/writer runtime args, then bind the input buffers as Buffer* at the
    // base-address slots (6 = Q input, 15 = K/V input).  The framework registers these as BufferBindings
    // and re-patches their addresses on every cache hit, so no get_dynamic_runtime_args() hook is needed;
    // every other slot (region offsets, head counts, NoC coords) is deterministic from the hashed shard
    // specs and is baked once.  The output CBs are patched via their `.buffer` bindings.
    tt_metal::Buffer* q_input_buffer = input_tensor.buffer();
    tt_metal::Buffer* kv_input_buffer =
        tensor_args.input_tensor_kv.has_value() ? tensor_args.input_tensor_kv.value().buffer() : input_tensor.buffer();

    auto per_core_args = build_sharded_core_args(operation_attributes, tensor_args, tensor_return_value);
    for (auto& e : per_core_args) {
        std::vector<std::variant<uint32_t, Buffer*>> reader_rt(e.reader_args.begin(), e.reader_args.end());
        reader_rt[kQBaseAddrIdx] = q_input_buffer;
        reader_rt[kKVBaseAddrIdx] = kv_input_buffer;
        reader_desc.emplace_runtime_args(e.core, reader_rt);

        std::vector<std::variant<uint32_t, Buffer*>> writer_rt(e.writer_args.begin(), e.writer_args.end());
        writer_rt[kQBaseAddrIdx] = q_input_buffer;
        writer_rt[kKVBaseAddrIdx] = kv_input_buffer;
        writer_desc.emplace_runtime_args(e.core, writer_rt);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::transformer
