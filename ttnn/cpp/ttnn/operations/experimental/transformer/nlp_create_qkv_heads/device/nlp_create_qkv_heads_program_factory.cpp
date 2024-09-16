// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "nlp_create_qkv_heads_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

NlpCreateHeadsDeviceOperation::Interleaved::cached_program_t NlpCreateHeadsDeviceOperation::Interleaved::create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {

    const Tensor &input_tensor = tensor_args.input_tensor_q;
    std::optional<const Tensor> input_tensor_kv = tensor_args.input_tensor_kv;
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k_heads = operation_attributes.transpose_k_heads;
    auto& output = tensor_return_value;
    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    const auto& input_shape = input_tensor.get_legacy_shape();

    tt_metal::Device *device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    tt_metal::Buffer *in1_buffer;
    uint32_t in1_buffer_addr = 0;
    tt_metal::BufferType in1_buffer_type = tt_metal::BufferType::DRAM;
    if (read_from_input_tensor_kv) {
        in1_buffer = input_tensor_kv.value().buffer();
        TT_ASSERT(in1_buffer->size() % single_tile_size == 0);
        in1_buffer_addr = in1_buffer->address();
        in1_buffer_type = in1_buffer->buffer_type();
    }


    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = input_tensor_kv.value().get_legacy_shape()[3] / TILE_WIDTH;
    }

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    // Output shape for K/V is: [B, num_kv_heads, s, head_dim], shuffled from [B, 1, s, num_kv_heads * head_dim]
    // NOTE: Output h and w dims are identical for Q, K, V, so any arg that is related to these dims for q_* can be shared for K, V
    uint32_t q_out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH; // tiles along head_dim
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of in0_w_tiles per core
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] = tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Tensor& q = std::get<0>(output);
    tt_metal::Tensor& k = std::get<1>(output);
    tt_metal::Tensor& v = std::get<2>(output);

    tt_metal::Buffer *q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer *k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer *v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    bool tile_dtype_is_bfloat16 = input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = false;
    if (read_from_input_tensor_kv) {
        in1_is_dram = in1_buffer_type == tt_metal::BufferType::DRAM ? 1 : 0;
    }

    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    bool out_is_dram = q_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) in0_is_dram,
            (std::uint32_t) in1_is_dram,
            (std::uint32_t) q_num_tiles,
            (std::uint32_t) kv_num_tiles,
    };
    std::vector<uint32_t> writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) out_is_dram,
            (std::uint32_t) q_out_h_tiles,
            (std::uint32_t) q_out_w_tiles,
            (std::uint32_t) q_out_HtWt,
            (std::uint32_t) num_q_heads, // q_out_c
            (std::uint32_t) num_kv_heads, // kv_out_c
    };

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
    if (transpose_k_heads) {
        std::vector<uint32_t> compute_args_core_group_1 = {num_blocks_per_core_group_1 * kv_num_tiles};
        auto compute_kernel_id_group_1 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/transpose_wh.cpp",
            core_group_1,
            tt_metal::ComputeConfig{.compile_args = compute_args_core_group_1}
        );

        if (core_group_2.num_cores() > 0) {
            std::vector<uint32_t> compute_args_core_group_2 = {num_blocks_per_core_group_2 * kv_num_tiles};
            auto compute_kernel_id_group_2 = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/transpose_wh.cpp",
                core_group_2,
                tt_metal::ComputeConfig{.compile_args = compute_args_core_group_2}
            );
        }

        reader_defines["TRANSPOSE_K_HEADS"] = "1";
        writer_defines["TRANSPOSE_K_HEADS"] = "1";
    }
    if (read_from_input_tensor_kv) {
        reader_defines["READ_FROM_INPUT_TENSOR_KV"] = "1";
    }

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));


    // Create circular buffers
    uint32_t micro_block_size = 1; // Num tiles to read/wait for in reader and writer
    uint32_t cb_num_tiles = micro_block_size * 4; // Quadruple buffer everything

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t cb1_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    uint32_t src1_cb_index = 1; // cb0 is needed for compute if we want to use generic transpose_wh compute kernel
    uint32_t cb1_num_tiles = cb_num_tiles;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(cb1_num_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // If we transpose_k_heads:
    // - reader will write to cb0, instead of cb1
    // - compute will wait on cb0 and write to cb16
    // - writer will wait on cb 16, instead of cb1
    if (transpose_k_heads) {
        uint32_t src0_cb_index = 0;
        uint32_t cb0_num_tiles = cb_num_tiles;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(cb0_num_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		    .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

        uint32_t out_cb_index = 16;
        uint32_t out_cb_num_tiles = cb_num_tiles;
        tt_metal::CircularBufferConfig cb_out_config = tt_metal::CircularBufferConfig(out_cb_num_tiles * single_tile_size, {{out_cb_index, cb_data_format}})
		    .set_page_size(out_cb_index, single_tile_size);
        auto cb_out = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    }

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        std::vector<uint32_t> reader_runtime_args = {
            (std::uint32_t) in0_buffer->address(),
            (std::uint32_t) in1_buffer_addr,
            num_blocks_per_core,
            num_blocks_written * in0_w_tiles,
            num_blocks_written * in1_w_tiles,
        };

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id = num_blocks_written / q_out_h_tiles * q_out_CHtWt + q_out_h_dim * q_out_w_tiles;
        uint32_t v_out_tensor_tile_id = num_blocks_written / q_out_h_tiles * kv_out_CHtWt + q_out_h_dim * q_out_w_tiles;
        uint32_t k_out_tensor_tile_id = transpose_k_heads ? num_blocks_written / q_out_h_tiles * kv_out_CHtWt + q_out_h_dim : v_out_tensor_tile_id;

        std::vector<uint32_t> writer_runtime_args = {
            (std::uint32_t) q_buffer->address(), // q_tensor_addr
            (std::uint32_t) k_buffer->address(), // k_tensor_addr
            (std::uint32_t) v_buffer->address(), // v_tensor_addr
            num_blocks_per_core, // num_blocks
            q_out_h_dim, // q_out_h_dim
            q_out_tensor_tile_id, // q_out_tensor_tile_id
            k_out_tensor_tile_id, // k_out_tensor_tile_id
            v_out_tensor_tile_id, // v_out_tensor_tile_id
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        num_blocks_written += num_blocks_per_core;
    }

    auto override_runtime_arguments_callback = [
            reader_kernel_id,
            writer_kernel_id,
            num_cores,
            num_cores_y,
            read_from_input_tensor_kv=read_from_input_tensor_kv
        ]
    (
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();

        uint32_t src_kv_buffer_addr = 0;
        if (read_from_input_tensor_kv) {
            src_kv_buffer_addr = optional_input_tensors.at(0).value().buffer()->address();
        }

        auto dst_buffer_query = output_tensors.at(0).buffer();
        auto dst_buffer_key = output_tensors.at(1).buffer();
        auto dst_buffer_value = output_tensors.at(2).buffer();

        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();

                if (read_from_input_tensor_kv) {
                    runtime_args[1] = src_kv_buffer_addr;
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer_query->address();
                runtime_args[1] = dst_buffer_key->address();
                runtime_args[2] = dst_buffer_value->address();
            }
        }
    };

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, num_cores_y, read_from_input_tensor_kv}};
}

void NlpCreateHeadsDeviceOperation::Interleaved::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {

        auto src_buffer = tensor_args.input_tensor_q.buffer();

        uint32_t src_kv_buffer_addr = 0;
        if (cached_program.shared_variables.read_from_input_tensor_kv) {
            src_kv_buffer_addr = tensor_args.input_tensor_kv.value().buffer()->address();
        }

        auto dst_buffer_query = std::get<0>(tensor_return_value).buffer();
        auto dst_buffer_key = std::get<1>(tensor_return_value).buffer();
        auto dst_buffer_value = std::get<2>(tensor_return_value).buffer();

        for (uint32_t i = 0, num_blocks_written = 0; i < cached_program.shared_variables.num_cores; i++){
            CoreCoord core = {i / cached_program.shared_variables.num_cores_y, i % cached_program.shared_variables.num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();

                if (cached_program.shared_variables.read_from_input_tensor_kv) {
                    runtime_args[1] = src_kv_buffer_addr;
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
                runtime_args[0] = dst_buffer_query->address();
                runtime_args[1] = dst_buffer_key->address();
                runtime_args[2] = dst_buffer_value->address();
            }
        }
    }

NlpCreateHeadsDeviceOperation::Sharded::cached_program_t NlpCreateHeadsDeviceOperation::Sharded::create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
    auto& input_tensor = tensor_args.input_tensor_q;
    auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto& output = tensor_return_value;
    auto head_dim = operation_attributes.head_dim;
    auto num_q_heads = operation_attributes.num_q_heads;
    auto num_kv_heads = operation_attributes.num_kv_heads;

    tt_metal::Program program = tt_metal::CreateProgram();

    const auto& input_shape = input_tensor.get_legacy_shape();

    tt_metal::Device *device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    auto q_shard_spec = std::get<0>(output).shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    uint32_t per_core_out_q_heads = num_q_heads / q_cores.num_cores();
    uint32_t per_risc0_out_q_heads = div_up(per_core_out_q_heads, 2);
    uint32_t per_risc1_out_q_heads = per_core_out_q_heads / 2;
    uint32_t per_core_in_q_heads = num_q_heads / input_tensor.shard_spec().value().num_cores();

    uint32_t q_output_cb_index = CB::c_out0;
    tt_metal::CircularBufferConfig cb_q_output_config =
        tt_metal::CircularBufferConfig(
            q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size).set_globally_allocated_address(*std::get<0>(output).buffer());
    auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    auto k_shard_spec = std::get<1>(output).shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t k_output_cb_index = CB::c_out1;
    tt_metal::CircularBufferConfig cb_k_output_config =
        tt_metal::CircularBufferConfig(
            k_num_tiles * single_tile_size, {{k_output_cb_index, cb_data_format}})
            .set_page_size(k_output_cb_index, single_tile_size).set_globally_allocated_address(*std::get<1>(output).buffer());
    auto cb_k_output = tt_metal::CreateCircularBuffer(program, k_cores, cb_k_output_config);

    auto v_shard_spec = std::get<0>(output).shard_spec().value();
    auto v_cores = q_shard_spec.grid;
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    uint32_t v_output_cb_index = CB::c_out2;
    tt_metal::CircularBufferConfig cb_v_output_config =
        tt_metal::CircularBufferConfig(
            v_num_tiles * single_tile_size, {{v_output_cb_index, cb_data_format}})
            .set_page_size(v_output_cb_index, single_tile_size).set_globally_allocated_address(*std::get<2>(output).buffer());
    auto cb_v_output = tt_metal::CreateCircularBuffer(program, v_cores, cb_v_output_config);

    uint32_t per_core_out_kv_heads = num_kv_heads / k_cores.num_cores();
    uint32_t per_core_in_kv_heads = num_kv_heads / (read_from_input_tensor_kv ? input_tensor_kv.value().shard_spec().value().num_cores() : input_tensor.shard_spec().value().num_cores());

    uint32_t q_base_addr = input_tensor.buffer()->address();
    uint32_t k_base_addr = 0;
    if (read_from_input_tensor_kv) {
        k_base_addr = input_tensor_kv.value().buffer()->address();
    } else {
        k_base_addr = q_base_addr + per_core_in_q_heads * head_tiles * single_tile_size;
    }
    uint32_t v_base_addr = k_base_addr + per_core_in_kv_heads * head_tiles * single_tile_size;

    std::vector<uint32_t> reader_compile_time_args = {
        q_output_cb_index,
        k_output_cb_index
    };
    std::vector<uint32_t> writer_compile_time_args = {
        q_output_cb_index,
        v_output_cb_index
    };
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp",
        q_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp",
        q_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t num_cores = std::max(q_cores.num_cores(), k_cores.num_cores());

    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    uint32_t num_kv_cores = k_cores.num_cores();

    const auto &cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

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
    uint32_t q_start_addr = q_base_addr;
    uint32_t k_start_addr = k_base_addr;
    uint32_t v_start_addr = v_base_addr;

    uint32_t remote_q_read = 0;
    uint32_t remote_kv_read = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        bool read_kv_heads = i < k_cores.num_cores();
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(18 + num_cores_x + num_cores_y);
        reader_runtime_args = {
            head_size,
            per_risc0_out_q_heads,
            per_core_in_q_heads,
            remote_q_head_start_idx,
            q_x,
            q_y,
            q_base_addr,
            q_start_addr,
            0,
            read_kv_heads,
            per_core_out_kv_heads,
            per_core_in_kv_heads,
            remote_kv_head_start_idx,
            kv_x,
            kv_y,
            k_base_addr,
            k_start_addr,
            k_num_tiles,
            num_cores_x,
        };
        reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

        remote_q_read += per_risc0_out_q_heads;
        q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
        q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;
        q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        reader_runtime_args[1] = per_risc1_out_q_heads;
        reader_runtime_args[3] = remote_q_head_start_idx;
        reader_runtime_args[4] = q_x;
        reader_runtime_args[5] = q_y;
        reader_runtime_args[7] = q_start_addr;
        reader_runtime_args[8] = per_risc0_out_q_heads * head_size;

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
            q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
            q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
        }

        if (read_kv_heads) {
            reader_runtime_args[15] = v_base_addr;
            reader_runtime_args[16] = v_start_addr;
            remote_kv_read += per_core_out_kv_heads;
            kv_y = (remote_kv_read / per_core_in_kv_heads) / num_cores_x;
            kv_x = (remote_kv_read / per_core_in_kv_heads) % num_cores_x;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
            k_start_addr = k_base_addr + remote_kv_head_start_idx * head_size;
            v_start_addr = v_base_addr + remote_kv_head_start_idx * head_size;
        }

        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, reader_runtime_args);
    }

    // auto override_runtime_arguments_callback = [
    //         reader_kernel_id,
    //         writer_kernel_id,
    //         num_cores,
    //         num_cores_y,
    //         read_from_input_tensor_kv=read_from_input_tensor_kv,
    //         cb_q_output,
    //         cb_k_output,
    //         cb_v_output,
    //         cores,
    //         head_size,
    //         per_risc0_out_q_heads,
    //         per_risc1_out_q_heads,
    //         per_core_in_q_heads,
    //         per_core_out_kv_heads,
    //         per_core_in_kv_heads,
    //         head_tiles,
    //         num_kv_cores,
    //         single_tile_size
    //     ]
    // (
    //     const void* operation,
    //     Program &program,
    //     const std::vector<Tensor>& input_tensors,
    //     const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    //     const std::vector<Tensor>& output_tensors
    // ) {

    //     auto src_buffer = input_tensors.at(0).buffer();

    //     uint32_t src_kv_buffer_addr = 0;
    //     if (read_from_input_tensor_kv) {
    //         src_kv_buffer_addr = optional_input_tensors.at(0).value().buffer()->address();
    //     }

    //     auto dst_buffer_query = output_tensors.at(0).buffer();
    //     auto dst_buffer_key = output_tensors.at(1).buffer();
    //     auto dst_buffer_value = output_tensors.at(2).buffer();

    //     UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);
    //     UpdateDynamicCircularBufferAddress(program, cb_k_output, *dst_buffer_key);
    //     UpdateDynamicCircularBufferAddress(program, cb_v_output, *dst_buffer_value);

    //     uint32_t q_base_addr = input_tensors[0].buffer()->address();
    //     uint32_t k_base_addr = 0;
    //     if (read_from_input_tensor_kv) {
    //         k_base_addr = input_tensor_kv.value().buffer()->address();
    //     } else {
    //         k_base_addr = q_base_addr + per_core_in_q_heads * head_tiles * single_tile_size;
    //     }
    //     uint32_t v_base_addr = k_base_addr + per_core_in_kv_heads * head_tiles * single_tile_size;

    //     uint32_t remote_q_head_start_idx = 0;
    //     uint32_t remote_kv_head_start_idx = 0;
    //     uint32_t q_start_addr = q_base_addr;
    //     uint32_t k_start_addr = k_base_addr;
    //     uint32_t v_start_addr = v_base_addr;

    //     for (uint32_t i = 0; i < num_cores; ++i) {
    //         const auto& core = cores[i];
    //         bool read_kv_heads = i < num_kv_cores;
    //         {
    //             auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
    //             runtime_args[6] = q_base_addr;
    //             runtime_args[7] = q_start_addr;
    //             runtime_args[15] = k_base_addr;
    //             runtime_args[16] = k_start_addr;
    //             remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;
    //             q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
    //         }
    //         {
    //             auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
    //             runtime_args[6] = q_base_addr;
    //             runtime_args[15] = v_base_addr;
    //             if (per_risc1_out_q_heads > 0) {
    //                 runtime_args[7] = q_start_addr;
    //                 remote_q_head_start_idx = (remote_q_head_start_idx + per_risc1_out_q_heads) % per_core_in_q_heads;
    //                 q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
    //             }
    //             if (read_kv_heads) {
    //                 runtime_args[16] = v_start_addr;
    //                 remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
    //                 k_start_addr = k_base_addr + remote_kv_head_start_idx * head_size;
    //                 v_start_addr = v_base_addr + remote_kv_head_start_idx * head_size;
    //             }
    //         }
    //     }
    // };

    return {std::move(program),
    {
        reader_kernel_id,
        writer_kernel_id,
        num_cores,
        num_cores_y,
        read_from_input_tensor_kv,
        cb_q_output,
        cb_k_output,
        cb_v_output,
        cores,
        head_size,
        per_risc0_out_q_heads,
        per_risc1_out_q_heads,
        per_core_in_q_heads,
        per_core_out_kv_heads,
        per_core_in_kv_heads,
        head_tiles,
        num_kv_cores,
        single_tile_size
    }
    };
}

void NlpCreateHeadsDeviceOperation::Sharded::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
        auto src_buffer = tensor_args.input_tensor_q.buffer();

        uint32_t src_kv_buffer_addr = 0;
        if (cached_program.shared_variables.read_from_input_tensor_kv) {
            src_kv_buffer_addr = tensor_args.input_tensor_kv.value().buffer()->address();
        }

        auto dst_buffer_query = std::get<0>(tensor_return_value).buffer();
        auto dst_buffer_key = std::get<1>(tensor_return_value).buffer();
        auto dst_buffer_value = std::get<2>(tensor_return_value).buffer();

        UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_q_output, *dst_buffer_query);
        UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_k_output, *dst_buffer_key);
        UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_v_output, *dst_buffer_value);

        uint32_t q_base_addr = tensor_args.input_tensor_q.buffer()->address();
        uint32_t k_base_addr = 0;
        if (cached_program.shared_variables.read_from_input_tensor_kv) {
            k_base_addr = tensor_args.input_tensor_kv.value().buffer()->address();
        } else {
            k_base_addr = q_base_addr + cached_program.shared_variables.per_core_in_q_heads * cached_program.shared_variables.head_tiles * cached_program.shared_variables.single_tile_size;
        }
        uint32_t v_base_addr = k_base_addr + cached_program.shared_variables.per_core_in_kv_heads * cached_program.shared_variables.head_tiles * cached_program.shared_variables.single_tile_size;

        uint32_t remote_q_head_start_idx = 0;
        uint32_t remote_kv_head_start_idx = 0;
        uint32_t q_start_addr = q_base_addr;
        uint32_t k_start_addr = k_base_addr;
        uint32_t v_start_addr = v_base_addr;

        for (uint32_t i = 0; i < cached_program.shared_variables.num_cores; ++i) {
            const auto& core = cached_program.shared_variables.cores[i];
            bool read_kv_heads = i < cached_program.shared_variables.num_kv_cores;
            {
                auto &runtime_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
                runtime_args[6] = q_base_addr;
                runtime_args[7] = q_start_addr;
                runtime_args[15] = k_base_addr;
                runtime_args[16] = k_start_addr;
                remote_q_head_start_idx = (remote_q_head_start_idx + cached_program.shared_variables.per_risc0_out_q_heads) % cached_program.shared_variables.per_core_in_q_heads;
                q_start_addr = q_base_addr + remote_q_head_start_idx * cached_program.shared_variables.head_size;
            }
            {
                auto &runtime_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
                runtime_args[6] = q_base_addr;
                runtime_args[15] = v_base_addr;
                if (cached_program.shared_variables.per_risc1_out_q_heads > 0) {
                    runtime_args[7] = q_start_addr;
                    remote_q_head_start_idx = (remote_q_head_start_idx + cached_program.shared_variables.per_risc1_out_q_heads) % cached_program.shared_variables.per_core_in_q_heads;
                    q_start_addr = q_base_addr + remote_q_head_start_idx * cached_program.shared_variables.head_size;
                }
                if (read_kv_heads) {
                    runtime_args[16] = v_start_addr;
                    remote_kv_head_start_idx = (remote_kv_head_start_idx + cached_program.shared_variables.per_core_out_kv_heads) % cached_program.shared_variables.per_core_in_kv_heads;
                    k_start_addr = k_base_addr + remote_kv_head_start_idx * cached_program.shared_variables.head_size;
                    v_start_addr = v_base_addr + remote_kv_head_start_idx * cached_program.shared_variables.head_size;
                }
            }
        }

    }

}  // ttnn::operations::experimental::transformer
