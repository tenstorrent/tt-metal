// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_sharded_program_factory.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor NLPCreateQKVHeadsDecodeShardedProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& batch_offset = tensor_args.batch_offset;
    const auto& num_q_heads = operation_attributes.num_q_heads;
    const auto& num_kv_heads = operation_attributes.num_kv_heads;
    const auto& head_dim = operation_attributes.head_dim;
    const auto& overlap_qk_coregrid = operation_attributes.overlap_qk_coregrid;

    ProgramDescriptor desc;

    IDevice* device = input_tensor.device();
    // Create CBs for reader/writer for batch_offset
    uint32_t batch_offset_cb_index_reader = CBIndex::c_15;
    uint32_t batch_offset_cb_index_writer = CBIndex::c_14;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto k_shard_spec = output[1].shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;
    auto in_shard_spec = input_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;
    uint32_t batch_offset_index_stick_size = 0;
    auto qk_cores = q_cores;
    if (!overlap_qk_coregrid) {
        auto qk_cores_set = std::set<CoreRange>();
        qk_cores_set.insert(q_cores.ranges().begin(), q_cores.ranges().end());
        qk_cores_set.insert(k_cores.ranges().begin(), k_cores.ranges().end());
        qk_cores = CoreRangeSet(qk_cores_set);
    }
    // if batch_offset is provided we need to allocate a buffer for it
    if (batch_offset.has_value()) {
        tt::DataFormat cb_batch_offset_data_format = datatype_to_dataformat_converter(batch_offset.value().dtype());
        uint32_t single_batch_offset_tile_size = tt::tile_size(cb_batch_offset_data_format);
        batch_offset_index_stick_size = batch_offset.value().buffer()->aligned_page_size();

        desc.cbs.push_back(CBDescriptor{
            .total_size = single_batch_offset_tile_size,
            .core_ranges = qk_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(batch_offset_cb_index_reader),
                .data_format = cb_batch_offset_data_format,
                .page_size = 1,
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = single_batch_offset_tile_size,
            .core_ranges = qk_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(batch_offset_cb_index_writer),
                .data_format = cb_batch_offset_data_format,
                .page_size = 1,
            }}},
        });
    }

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
        .buffer = output[2].buffer(),
    });

    Buffer* in_buffer = input_tensor.buffer();
    Buffer* batch_offset_buffer = batch_offset.has_value() ? batch_offset.value().buffer() : nullptr;

    // cores for q
    uint32_t q_num_cores = q_cores.num_cores();  // number of cores of the output
    auto q_core_grid = q_cores.bounding_box();
    uint32_t q_num_cores_x = q_core_grid.end_coord.x + 1, q_num_cores_y = q_core_grid.end_coord.y + 1;
    const auto& q_cores_vector = grid_to_cores(q_num_cores, q_num_cores_x, q_num_cores_y, true);

    // cores for k
    uint32_t k_num_cores = k_cores.num_cores();  // number of cores of the output
    const auto& k_cores_vector = corerange_to_cores(k_cores, k_num_cores, true);

    // cores for input
    auto in_core_grid = in_cores.bounding_box();
    uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores_x);
    for (uint32_t x = 0; x < in_num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores_y);
    for (uint32_t y = 0; y < in_num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    uint32_t process_qv = 1, process_k = 1;
    // In case of overlapping qk coregrid, we create a single set of kernels for q which also process k and v heads
    // from the input and write to the respective output buffers while if q and k are not overlapped, we create two
    // sets of kernels in different coregrids one set of kernels for q which also process v heads but skips k heads
    // from the input and write to the respective output buffers another set of kernels for k which reads k heads from
    // the input and write to the respective output buffers while skipping q and v heads
    if (!overlap_qk_coregrid) {
        process_qv = 1;
        process_k = 0;
    }

    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2
    // of a tile respectively)
    std::vector<uint32_t> q_reader_compile_time_args = {
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        k_output_cb_index,
        v_output_cb_index,
        head_size,
        num_q_heads,
        num_kv_heads,
        head_tiles,
        1,  // read the first phase
        in_num_cores_x,
        in_num_cores_y,
        process_qv,                        // read and write q and v heads
        process_k,                         // read and write k heads
        batch_offset.has_value() ? 1 : 0,  // use_batch_offset
        batch_offset_index_stick_size,
        batch_offset_cb_index_reader};
    tt::tt_metal::TensorAccessorArgs(batch_offset_buffer).append_to(q_reader_compile_time_args);

    KernelDescriptor q_reader_desc;
    q_reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";
    q_reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    q_reader_desc.core_ranges = q_cores;
    q_reader_desc.compile_time_args = q_reader_compile_time_args;
    q_reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> q_writer_compile_time_args = q_reader_compile_time_args;
    q_writer_compile_time_args[9] = 2;  // read the second phase

    KernelDescriptor q_writer_desc;
    q_writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";
    q_writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    q_writer_desc.core_ranges = q_cores;
    q_writer_desc.compile_time_args = std::move(q_writer_compile_time_args);
    q_writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor k_reader_desc;
    KernelDescriptor k_writer_desc;
    if (!overlap_qk_coregrid) {
        // Switch process_qv and process_k for k kernels
        process_qv = 0;
        process_k = 1;
        std::vector<uint32_t> k_reader_compile_time_args = q_reader_compile_time_args;
        k_reader_compile_time_args[12] = process_qv;
        k_reader_compile_time_args[13] = process_k;

        k_reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
            "reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";
        k_reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        k_reader_desc.core_ranges = k_cores;
        k_reader_desc.compile_time_args = k_reader_compile_time_args;
        k_reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> k_writer_compile_time_args = k_reader_compile_time_args;
        k_writer_compile_time_args[9] = 2;  // read the second phase

        k_writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
            "reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp";
        k_writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        k_writer_desc.core_ranges = k_cores;
        k_writer_desc.compile_time_args = std::move(k_writer_compile_time_args);
        k_writer_desc.config = WriterConfigDescriptor{};
    }

    auto push_batch_offset = [&](KernelDescriptor::RTArgList& rt) {
        if (batch_offset_buffer != nullptr) {
            rt.push_back(batch_offset_buffer);
        } else {
            rt.push_back(uint32_t{0});
        }
    };

    for (uint32_t i = 0; i < q_num_cores; ++i) {
        const auto& core = q_cores_vector[i];
        KernelDescriptor::RTArgList rt;
        rt.reserve(3 + in_num_cores_x + in_num_cores_y);
        rt.push_back(in_buffer);  // q_start_addr (= input_tensor base)
        push_batch_offset(rt);
        rt.push_back(i);
        rt.append(noc_x_coords);
        rt.append(noc_y_coords);

        q_reader_desc.emplace_runtime_args(core, rt);
        q_writer_desc.emplace_runtime_args(core, rt);
    }

    if (!overlap_qk_coregrid) {
        for (uint32_t i = 0; i < k_num_cores; ++i) {
            const auto& core = k_cores_vector[i];
            KernelDescriptor::RTArgList rt;
            rt.reserve(3 + in_num_cores_x + in_num_cores_y);
            rt.push_back(in_buffer);
            push_batch_offset(rt);
            rt.push_back(i);
            rt.append(noc_x_coords);
            rt.append(noc_y_coords);

            k_reader_desc.emplace_runtime_args(core, rt);
            k_writer_desc.emplace_runtime_args(core, rt);
        }
    }

    desc.kernels.push_back(std::move(q_reader_desc));
    desc.kernels.push_back(std::move(q_writer_desc));
    if (!overlap_qk_coregrid) {
        desc.kernels.push_back(std::move(k_reader_desc));
        desc.kernels.push_back(std::move(k_writer_desc));
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
