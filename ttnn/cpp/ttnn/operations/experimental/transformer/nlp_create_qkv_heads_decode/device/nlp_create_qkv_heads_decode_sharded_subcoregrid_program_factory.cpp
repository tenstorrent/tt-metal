// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_sharded_subcoregrid_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt;

namespace ttnn::experimental::prim {

NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory::cached_program_t
NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory::create(
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

    Program program = CreateProgram();

    IDevice* device = input_tensor.device();
    // Create CBs for reader/writer for batch_offset
    uint32_t batch_offset_cb_index_reader = CBIndex::c_15;
    uint32_t batch_offset_cb_index_writer = CBIndex::c_14;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const uint32_t head_tiles = head_dim / TILE_WIDTH;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t sub_tile_line_bytes = 16 * element_size;
    const auto q_shard_spec = output[0].shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    const auto k_shard_spec = output[1].shard_spec().value();
    const auto k_cores = k_shard_spec.grid;
    const auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;
    const auto in_shard_spec = input_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;
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

        CircularBufferConfig cb_batch_offset_config_reader =
            CircularBufferConfig(
                single_batch_offset_tile_size, {{batch_offset_cb_index_reader, cb_batch_offset_data_format}})
                .set_page_size(batch_offset_cb_index_reader, 1);
        CreateCircularBuffer(program, qk_cores, cb_batch_offset_config_reader);

        CircularBufferConfig cb_batch_offset_config_writer =
            CircularBufferConfig(
                single_batch_offset_tile_size, {{batch_offset_cb_index_writer, cb_batch_offset_data_format}})
                .set_page_size(batch_offset_cb_index_writer, 1);
        CreateCircularBuffer(program, qk_cores, cb_batch_offset_config_writer);
    }

    uint32_t q_output_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_q_output_config =
        CircularBufferConfig(q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[0].buffer());
    auto cb_q_output = CreateCircularBuffer(program, q_cores, cb_q_output_config);

    uint32_t k_output_cb_index = CBIndex::c_17;
    CircularBufferConfig cb_k_output_config =
        CircularBufferConfig(k_num_tiles * single_tile_size, {{k_output_cb_index, cb_data_format}})
            .set_page_size(k_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[1].buffer());
    auto cb_k_output = CreateCircularBuffer(program, k_cores, cb_k_output_config);

    const auto v_shard_spec = output[0].shard_spec().value();
    const auto v_cores = q_shard_spec.grid;
    const auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    uint32_t v_output_cb_index = CBIndex::c_18;
    CircularBufferConfig cb_v_output_config =
        CircularBufferConfig(v_num_tiles * single_tile_size, {{v_output_cb_index, cb_data_format}})
            .set_page_size(v_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[2].buffer());
    auto cb_v_output = CreateCircularBuffer(program, v_cores, cb_v_output_config);

    uint32_t q_base_addr = input_tensor.buffer()->address();

    // cores for q
    const uint32_t q_num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& q_cores_vector = corerange_to_cores(q_cores, q_num_cores, true);

    // cores for k
    const uint32_t k_num_cores = k_cores.num_cores();  // number of cores of the output
    const auto& k_cores_vector = corerange_to_cores(k_cores, k_num_cores, true);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    auto in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> noc_x_coords, noc_y_coords;
    noc_x_coords.reserve(in_num_cores);
    noc_y_coords.reserve(in_num_cores);

    for (uint32_t i = 0; i < in_num_cores; ++i) {
        auto worker_core = device->worker_core_from_logical_core(in_cores_vec[i]);
        noc_x_coords.push_back(worker_core.x);
        noc_y_coords.push_back(worker_core.y);
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
        in_num_cores,
        process_qv,                        // read and write q and v heads
        process_k,                         // read and write k heads
        batch_offset.has_value() ? 1 : 0,  // use_batch_offset
        batch_offset_index_stick_size,
        batch_offset_cb_index_reader};

    tt::tt_metal::TensorAccessorArgs(batch_offset.has_value() ? batch_offset.value().buffer() : nullptr)
        .append_to(q_reader_compile_time_args);

    auto q_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp",
        q_cores,
        ReaderDataMovementConfig(q_reader_compile_time_args));
    std::vector<uint32_t> q_writer_compile_time_args = q_reader_compile_time_args;
    q_writer_compile_time_args[9] = 2;  // read the second phase
    q_writer_compile_time_args[15] = batch_offset_cb_index_writer;
    auto q_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp",
        q_cores,
        WriterDataMovementConfig(q_writer_compile_time_args));

    tt::tt_metal::KernelHandle k_reader_kernel_id = 0, k_writer_kernel_id = 0;
    if (!overlap_qk_coregrid) {
        // Switch process_qv and process_k for k kernels
        process_qv = 0;
        process_k = 1;
        std::vector<uint32_t> k_reader_compile_time_args = q_reader_compile_time_args;
        k_reader_compile_time_args[11] = process_qv;
        k_reader_compile_time_args[12] = process_k;
        k_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
            "reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp",
            k_cores,
            ReaderDataMovementConfig(k_reader_compile_time_args));

        std::vector<uint32_t> k_writer_compile_time_args = k_reader_compile_time_args;
        k_writer_compile_time_args[9] = 2;  // read the second phase
        k_writer_compile_time_args[15] = batch_offset_cb_index_writer;
        k_writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
            "reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp",
            k_cores,
            WriterDataMovementConfig(k_writer_compile_time_args));
    }

    uint32_t q_start_addr = q_base_addr;
    bool use_batch_offset = batch_offset.has_value();

    for (uint32_t i = 0; i < q_num_cores; ++i) {
        const auto& core = q_cores_vector[i];
        std::vector<uint32_t> q_reader_runtime_args;
        q_reader_runtime_args.reserve(3 + (2 * in_num_cores));
        q_reader_runtime_args = {q_start_addr, use_batch_offset ? batch_offset.value().buffer()->address() : 0, i};
        q_reader_runtime_args.insert(q_reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        q_reader_runtime_args.insert(q_reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());
        SetRuntimeArgs(program, q_reader_kernel_id, core, q_reader_runtime_args);
        SetRuntimeArgs(program, q_writer_kernel_id, core, q_reader_runtime_args);
    }

    if (!overlap_qk_coregrid) {
        for (uint32_t i = 0; i < k_num_cores; ++i) {
            const auto& core = k_cores_vector[i];
            std::vector<uint32_t> k_reader_runtime_args;
            k_reader_runtime_args.reserve(3 + (2 * in_num_cores));
            k_reader_runtime_args = {q_start_addr, use_batch_offset ? batch_offset.value().buffer()->address() : 0, i};
            k_reader_runtime_args.insert(k_reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
            k_reader_runtime_args.insert(k_reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());
            SetRuntimeArgs(program, k_reader_kernel_id, core, k_reader_runtime_args);
            SetRuntimeArgs(program, k_writer_kernel_id, core, k_reader_runtime_args);
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .q_reader_kernel_id = q_reader_kernel_id,
            .q_writer_kernel_id = q_writer_kernel_id,
            .k_reader_kernel_id = k_reader_kernel_id,
            .k_writer_kernel_id = k_writer_kernel_id,
            .q_num_cores = q_num_cores,
            .k_num_cores = k_num_cores,
            .cb_q_output = cb_q_output,
            .cb_k_output = cb_k_output,
            .cb_v_output = cb_v_output,
            .q_cores_vector = q_cores_vector,
            .k_cores_vector = k_cores_vector,
            .element_size = element_size,
            .sub_tile_line_bytes = sub_tile_line_bytes,
            .overlap_qk_coregrid = overlap_qk_coregrid,
            .use_batch_offset = use_batch_offset}};
}

void NLPCreateQKVHeadsDecodeShardedSubcoregridProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& q_reader_kernel_id = cached_program.shared_variables.q_reader_kernel_id;
    const auto& q_writer_kernel_id = cached_program.shared_variables.q_writer_kernel_id;
    const auto& k_reader_kernel_id = cached_program.shared_variables.k_reader_kernel_id;
    const auto& k_writer_kernel_id = cached_program.shared_variables.k_writer_kernel_id;
    const auto& q_num_cores = cached_program.shared_variables.q_num_cores;
    const auto& k_num_cores = cached_program.shared_variables.k_num_cores;
    const auto& cb_q_output = cached_program.shared_variables.cb_q_output;
    const auto& cb_k_output = cached_program.shared_variables.cb_k_output;
    const auto& cb_v_output = cached_program.shared_variables.cb_v_output;
    const auto& q_cores_vector = cached_program.shared_variables.q_cores_vector;
    const auto& k_cores_vector = cached_program.shared_variables.k_cores_vector;
    const auto& overlap_qk_coregrid = cached_program.shared_variables.overlap_qk_coregrid;
    const auto& use_batch_offset = cached_program.shared_variables.use_batch_offset;

    auto *dst_buffer_query = output_tensors.at(0).buffer();
    auto *dst_buffer_key = output_tensors.at(1).buffer();
    auto *dst_buffer_value = output_tensors.at(2).buffer();

    UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);
    UpdateDynamicCircularBufferAddress(program, cb_k_output, *dst_buffer_key);
    UpdateDynamicCircularBufferAddress(program, cb_v_output, *dst_buffer_value);

    uint32_t q_base_addr = tensor_args.input_tensor.buffer()->address();
    uint32_t q_start_addr = q_base_addr;

    auto& q_reader_args_by_core = GetRuntimeArgs(program, q_reader_kernel_id);
    auto& q_writer_args_by_core = GetRuntimeArgs(program, q_writer_kernel_id);

    for (uint32_t i = 0; i < q_num_cores; ++i) {
        const auto& core = q_cores_vector[i];
        auto& runtime_args = q_reader_args_by_core[core.x][core.y];
        runtime_args[0] = q_start_addr;
        runtime_args[1] = use_batch_offset ? tensor_args.batch_offset.value().buffer()->address() : 0;
        runtime_args[2] = i;

        auto& runtime_args_writer = q_writer_args_by_core[core.x][core.y];
        runtime_args_writer[0] = q_start_addr;
        runtime_args_writer[1] = use_batch_offset ? tensor_args.batch_offset.value().buffer()->address() : 0;
        runtime_args_writer[2] = i;
    }

    if (!overlap_qk_coregrid) {
        auto& k_reader_args_by_core = GetRuntimeArgs(program, k_reader_kernel_id);
        auto& k_writer_args_by_core = GetRuntimeArgs(program, k_writer_kernel_id);

        for (uint32_t i = 0; i < k_num_cores; ++i) {
            const auto& core = k_cores_vector[i];
            auto& runtime_args = k_reader_args_by_core[core.x][core.y];
            runtime_args[0] = q_start_addr;
            runtime_args[1] = use_batch_offset ? tensor_args.batch_offset.value().buffer()->address() : 0;
            runtime_args[2] = i;

            auto& runtime_args_writer = k_writer_args_by_core[core.x][core.y];
            runtime_args_writer[0] = q_start_addr;
            runtime_args_writer[1] = use_batch_offset ? tensor_args.batch_offset.value().buffer()->address() : 0;
            runtime_args_writer[2] = i;
        }
    }
}

}  // namespace ttnn::experimental::prim
