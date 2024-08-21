// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_qkv_heads_from_separate_tensors_device_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace ttnn::operations::experimental::transformer {

    static inline operation::ProgramWithCallbacks create_qkv_separate(const Tensor &input_tensor_q, const Tensor &input_tensor_kv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, std::vector<Tensor> &output, bool transpose_k) {

        const auto &q_shape = input_tensor_q.get_legacy_shape();
        const auto &kv_shape = input_tensor_kv.get_legacy_shape();
        auto shard_spec = input_tensor_q.shard_spec().value();
        auto all_cores = shard_spec.grid;
        auto bbox = all_cores.bounding_box();
        ShardOrientation shard_orientation = shard_spec.orientation;
        bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
        uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
        uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;

        uint32_t q_shard_wt = (q_shape[3]) / (num_w_cores * TILE_WIDTH); // number of tiles in width dimension  - multiple tiles per head, multiple heads per group, multiple tensors in group, multiple groups per cores
        uint32_t q_shard_ht = (q_shape[0] * q_shape[2])/ (num_h_cores * TILE_HEIGHT);

        uint32_t k_shard_wt = (kv_shape[3] / (2 * num_w_cores * TILE_WIDTH));
        uint32_t k_shard_ht = (kv_shape[0] * kv_shape[2])/ (num_h_cores * TILE_HEIGHT);

        uint32_t per_core_q_tiles = q_shard_ht * q_shard_wt;
        uint32_t per_core_k_tiles = k_shard_ht * k_shard_wt;

        const auto q_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_q.get_dtype());
        const auto kv_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_kv.get_dtype());
        uint32_t single_tile_size = tile_size(q_data_format);

        uint32_t q_heads_per_core = num_q_heads / num_w_cores;
        uint32_t k_heads_per_core = num_kv_heads / num_w_cores;

        Program program = CreateProgram();
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) q_shard_ht,
            (std::uint32_t) q_shard_wt,
            (std::uint32_t) k_shard_ht,
            (std::uint32_t) k_shard_wt, // shard width for k and v individually, times two for entire kv tensor
            (std::uint32_t) q_heads_per_core,
            (std::uint32_t) k_heads_per_core,
            (std::uint32_t) head_dim / TILE_WIDTH, // tiles per head
        };

        std::map<string, string> reader_defines;
        if (transpose_k) {
            reader_defines["TRANSPOSE_K_HEADS"] = "1";
        }
        auto reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors/device/kernels/reader_create_qkv_heads_sharded_separate.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

        if (transpose_k)    {
            std::vector<uint32_t> compute_args = {
                (std::uint32_t) (per_core_k_tiles), // number of K tiles
            };
            auto compute_kernel_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp",
                all_cores,
                tt_metal::ComputeConfig{.compile_args = compute_args});
        }

        uint32_t q_size = per_core_q_tiles * single_tile_size;
        uint32_t k_size = per_core_k_tiles * single_tile_size;
        uint32_t v_size = k_size;
        uint32_t kv_size = 2*k_size;

        // qkv tensor
        auto c_in0_config = CircularBufferConfig(q_size,
        {{CB::c_in0, q_data_format}}).set_page_size(CB::c_in0, single_tile_size).set_globally_allocated_address(*input_tensor_q.buffer());
        auto cb_in0_id = CreateCircularBuffer(program, all_cores, c_in0_config);

        auto c_in1_config = CircularBufferConfig(kv_size, {
            {CB::c_in1, kv_data_format}}
            ).set_page_size(CB::c_in1, single_tile_size).set_globally_allocated_address(*input_tensor_kv.buffer());
        auto cb_in1_id = CreateCircularBuffer(program, all_cores, c_in1_config);

        // q sharded
        auto c_out0_config = CircularBufferConfig(q_size, {{CB::c_out0, q_data_format}})
            .set_page_size(CB::c_out0, single_tile_size).set_globally_allocated_address(*output[0].buffer());
        auto cb_out0_id = CreateCircularBuffer( program, all_cores, c_out0_config );
        // k sharded
        auto c_out1_config = CircularBufferConfig(k_size, {{CB::c_out1, kv_data_format}})
            .set_page_size(CB::c_out1, single_tile_size).set_globally_allocated_address(*output[1].buffer());
        auto cb_out1_id = CreateCircularBuffer( program, all_cores, c_out1_config );
        // v sharded
        auto c_out2_config = CircularBufferConfig(v_size, {{CB::c_out2, kv_data_format}})
            .set_page_size(CB::c_out2, single_tile_size).set_globally_allocated_address(*output[2].buffer());
        auto cb_out2_id = CreateCircularBuffer( program, all_cores, c_out2_config );

        if (transpose_k) {
            auto c_im0_config = CircularBufferConfig(k_size, {{CB::c_intermed0, kv_data_format}})
                .set_page_size(CB::c_intermed0, single_tile_size);
            auto cb_im0_id = CreateCircularBuffer(program, all_cores, c_im0_config);
        }


        auto override_runtime_args_callback = [
            cb_in0_id,
            cb_in1_id,
            cb_out0_id,
            cb_out1_id,
            cb_out2_id
            ]
        (
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors
        ) {
            auto in0_buffer = input_tensors.at(0).buffer();
            auto in1_buffer = input_tensors.at(1).buffer();
            auto out0_buffer = output_tensors.at(0).buffer();
            auto out1_buffer = output_tensors.at(1).buffer();
            auto out2_buffer = output_tensors.at(2).buffer();

            UpdateDynamicCircularBufferAddress(program, cb_in0_id, *in0_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_in1_id, *in1_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_out0_id, *out0_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_out1_id, *out1_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_out2_id, *out2_buffer);
        };

        return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
    }


    /**
     * Combined QKV
     *
     * m = num_KV_heads
     * p = num_Q_heads
     * n = p/m
     * i = head index for K/V, corresponding Q group for K/V
     *
     * input: [nQi Ki Vi for i in [0, m)] repeated for the sequence length
     * dims: [B, 1, S, H] Hi = [nQi Ki Vi] for all i kv heads = ((n + 2)*head_dim)
     *
     * output: 3 separate tensors organized by heads
     * [Qs,j for j in [0, p] and s in [0,S)]
     * dims: [B, p, S, head_dim] (all nQi in the KVi group are stacked together, p = n*m)
     *
     * [Ks,i for i in [0, m) and s in [0,S)]
     * dims: [B, m, S, head_dim]
     *
     * [Vs,i for i in [0, m) and s in [0,S)]
     * dims: [B, m, S, head_dim]
     *
     * Tiles stay the same Vi,s[x:x+32] to Vi+32,s[x:x+32] stays in one tile, but now instead of nQi,s and Ki,s tiles there is Vi,s-1 and Vi,s+1
     *
     * Shard across each i kv head group (width sharding) and then shard across each token s (height sharding)
     * Each block: B x [nQi Ki Vi]s (shard across flattened heads and sequence length)
     *
     * Combined batch/sequence sharding is possible too...that may best be left as an extension
    */
    operation::ProgramWithCallbacks multi_core_create_q_and_kv_heads_sharded(const Tensor &input_tensor_q, const Tensor &input_tensor_kv, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, const bool transpose_k_heads, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size) {
        return create_qkv_separate(input_tensor_q, input_tensor_kv, num_q_heads, num_kv_heads, head_dim, output, transpose_k_heads);
    }
} // namespace ttnn::operations::experimental::transformer
