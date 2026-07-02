// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor CreateQKVHeadsSeparateTensorsProgramFactory::create_descriptor(
    const CreateQKVHeadsFromSeparateTensorsParams& operation_attributes,
    const CreateQKVHeadsFromSeparateTensorsInputs& tensor_args,
    CreateQKVHeadsFromSeparateTensorsResult& tensor_return_value) {
    ProgramDescriptor desc;

    const auto& input_tensor_q = tensor_args.input_tensor;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto& output_q = std::get<0>(tensor_return_value);
    auto& output_k = std::get<1>(tensor_return_value);
    auto& output_v = std::get<2>(tensor_return_value);

    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k = operation_attributes.transpose_k_heads;
    const auto& q_shape = input_tensor_q.padded_shape();
    const auto& kv_shape = input_tensor_kv.padded_shape();
    auto shard_spec = input_tensor_q.shard_spec().value();
    auto all_cores = shard_spec.grid;
    auto bbox = all_cores.bounding_box();
    ShardOrientation shard_orientation = shard_spec.orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

    uint32_t q_shard_wt =
        (q_shape[3]) /
        (num_w_cores * TILE_WIDTH);  // number of tiles in width dimension  - multiple tiles per head, multiple heads
                                     // per group, multiple tensors in group, multiple groups per cores
    uint32_t q_shard_ht = (q_shape[0] * q_shape[2]) / (num_h_cores * TILE_HEIGHT);

    uint32_t k_shard_wt = (kv_shape[3] / (2 * num_w_cores * TILE_WIDTH));
    uint32_t k_shard_ht = (kv_shape[0] * kv_shape[2]) / (num_h_cores * TILE_HEIGHT);

    uint32_t per_core_q_tiles = q_shard_ht * q_shard_wt;
    uint32_t per_core_k_tiles = k_shard_ht * k_shard_wt;

    const auto q_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    const auto kv_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_kv.dtype());
    uint32_t single_tile_size = tile_size(q_data_format);

    uint32_t q_heads_per_core = num_q_heads / num_w_cores;
    uint32_t k_heads_per_core = num_kv_heads / num_w_cores;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)q_shard_ht,
        (std::uint32_t)q_shard_wt,
        (std::uint32_t)k_shard_ht,
        (std::uint32_t)k_shard_wt,  // shard width for k and v individually, times two for entire kv tensor
        (std::uint32_t)q_heads_per_core,
        (std::uint32_t)k_heads_per_core,
        (std::uint32_t)head_dim / TILE_WIDTH,  // tiles per head
    };

    std::map<std::string, std::string> reader_defines;
    if (transpose_k) {
        reader_defines["TRANSPOSE_K_HEADS"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/device/kernels/"
        "reader_create_qkv_heads_sharded_separate.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = KernelDescriptor::Defines{reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(reader_desc));

    if (transpose_k) {
        std::vector<uint32_t> compute_args = {
            (std::uint32_t)(per_core_k_tiles),  // number of K tiles
        };
        // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
        // resolves the unpack-dst CB to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
        // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
        const bool fp32_dest_acc_en = input_tensor_kv.dtype() == tt_metal::DataType::FLOAT32;

        KernelDescriptor compute_desc;
        compute_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
            "compute/transpose_wh_sharded.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = std::move(compute_args);
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        desc.kernels.push_back(std::move(compute_desc));
    }

    uint32_t q_size = per_core_q_tiles * single_tile_size;
    uint32_t k_size = per_core_k_tiles * single_tile_size;
    uint32_t v_size = k_size;
    uint32_t kv_size = 2 * k_size;

    // qkv tensor (input shards)
    desc.cbs.push_back(CBDescriptor{
        .total_size = q_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_0,
            .data_format = q_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = input_tensor_q.buffer(),
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = kv_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_1,
            .data_format = kv_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = input_tensor_kv.buffer(),
    });

    // q sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = q_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_16,
            .data_format = q_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output_q.buffer(),
    });
    // k sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = k_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_17,
            .data_format = kv_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output_k.buffer(),
    });
    // v sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = v_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_18,
            .data_format = kv_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output_v.buffer(),
    });

    if (transpose_k) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = k_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = CBIndex::c_24,
                .data_format = kv_data_format,
                .page_size = single_tile_size,
            }}},
        });
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
