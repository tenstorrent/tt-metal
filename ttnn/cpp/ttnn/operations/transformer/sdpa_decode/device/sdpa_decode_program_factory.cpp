// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_program_factory.hpp"

#include <optional>
#include <string>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

SdpaDecodeProgramFactory::cached_program_t SdpaDecodeProgramFactory::create(
    const SdpaDecodeParams& operation_attributes, const SdpaDecodeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    bool use_mla = operation_attributes.use_mla.value_or(false);
    if (!use_mla) {
        TT_FATAL(tensor_args.v.has_value(), "V tensor must be provided when MLA is disabled.");
    }
    const auto& input_tensor_v = use_mla ? input_tensor_k : tensor_args.v.value();

    const auto& cur_pos_tensor = tensor_args.cur_pos_tensor;
    const auto& page_table_tensor = tensor_args.page_table_tensor;
    const auto& attn_mask = tensor_args.attn_mask;
    const auto& attention_sink = tensor_args.attention_sink;

    const auto& output_tensor = tensor_return_value;

    auto scale = operation_attributes.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1]));
    }
    auto sliding_window_size = operation_attributes.sliding_window_size;
    if (not sliding_window_size.has_value()) {
        sliding_window_size = 0;
    }
    const auto is_causal = operation_attributes.is_causal;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const auto& program_config = operation_attributes.program_config;
    const auto& k_chunk_size = operation_attributes.k_chunk_size;
    std::optional<bool> share_cache = operation_attributes.share_cache;
    const auto& head_dim_v = operation_attributes.head_dim_v.value_or(0);
    const auto& cur_pos_ids = operation_attributes.cur_pos;

    /*
    Q: 1 x B x PNH x DH
    K: B x NKV x S x DH
    V: B x NKV x S x DH
    */

    /*
    Initially during compile time, we compile the kernel based on the longest sequence length in the batch.
    During runtime, we may override the number of chunks being processed based on the actual sequence length of the
    current batch.
    */

    const bool is_paged_attention = page_table_tensor.has_value();

    auto q_shape = input_tensor_q.padded_shape();
    const bool tilize_q = input_tensor_q.layout() == Layout::ROW_MAJOR;
    q_shape[2] = tt::round_up(q_shape[2], tt::constants::TILE_HEIGHT);  // round up for row major Q tensor.
    const auto& q_shape_unpadded = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.padded_shape();
    // Use k_shape for S and DH since Q might be different for decode
    uint32_t B = q_shape[1], PNH = q_shape[2], S = k_shape[2], DH = k_shape[3];

    uint32_t num_kv_heads = k_shape[1];
    uint32_t num_q_heads = q_shape_unpadded[2];
    uint32_t page_block_size_t = 0;

    bool is_q_sharded = input_tensor_q.is_sharded();
    bool is_output_sharded = output_tensor.is_sharded();

    // balance the number of cores to use based on batch
    uint32_t q_heads_parallel_factor = 1;
    if (is_q_sharded && use_mla) {
        uint32_t q_shard_height = input_tensor_q.memory_config().shard_spec()->shape[0];
        q_heads_parallel_factor = std::max((uint32_t)1, (num_q_heads + q_shard_height - 1) / q_shard_height);

        if (q_heads_parallel_factor > 1) {
            TT_FATAL(
                num_kv_heads == 1,
                "If parallelizing over Q num heads (with parallelization factor q_heads_parallel_factor: {}), then "
                "num_kv_heads must be 1, but got num_kv_heads: {}",
                q_heads_parallel_factor,
                num_kv_heads);
        }

        B *= q_heads_parallel_factor;  // adjust batch size to account for Q sharding
    }

    if (is_paged_attention) {
        uint32_t block_size = k_shape[2];
        page_block_size_t = block_size / TILE_HEIGHT;
        // get real S using the page_table_tensor
        S = page_table_tensor.value().padded_shape()[-1] * S;
    }
    uint32_t Bkv = k_shape[0];
    uint32_t St = S / TILE_HEIGHT;
    uint32_t DHt = DH / TILE_WIDTH;
    uint32_t vDHt = use_mla ? head_dim_v / TILE_WIDTH : DHt;
    uint32_t PNHt = PNH / q_heads_parallel_factor / TILE_HEIGHT;

    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;

    if (!share_cache.has_value()) {
        // default share_cache to false
        share_cache = false;
    }
    if (share_cache.value()) {
        TT_FATAL(B % Bkv == 0, "Batch dim in Q must be divisible by batch dim in KV if sharing cache");
    }

    // log_debug all of the above
    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "PNH: {}", PNH);
    log_debug(tt::LogOp, "S: {}", S);
    log_debug(tt::LogOp, "DH: {}", DH);
    log_debug(tt::LogOp, "num_kv_heads: {}", num_kv_heads);
    log_debug(tt::LogOp, "Bkv: {}", Bkv);
    log_debug(tt::LogOp, "St: {}", St);
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "vDHt: {}", vDHt);
    log_debug(tt::LogOp, "PNHt: {}", PNHt);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);

    Program program = CreateProgram();

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    bool exp_approx_mode =
        program_config.has_value()
            ? (program_config->exp_approx_mode.has_value() ? program_config->exp_approx_mode.value() : true)
            : true;

    auto* q_buffer = input_tensor_q.buffer();
    auto* k_buffer = input_tensor_k.buffer();
    auto* v_buffer = input_tensor_v.buffer();
    auto* out0_buffer = output_tensor.buffer();

    bool use_cur_pos_tensor = cur_pos_tensor.has_value();
    bool use_attention_mask = attn_mask.has_value();
    bool use_attention_sink = attention_sink.has_value();

    log_debug(tt::LogOp, "use_cur_pos_tensor: {}", use_cur_pos_tensor);
    log_debug(tt::LogOp, "use_attention_mask: {}", use_attention_mask);
    log_debug(tt::LogOp, "use_attention_sink: {}", use_attention_sink);

    // Parallelization scheme
    // We will assign cores to batches
    // Split to cores
    CoreCoord grid_size = program_config.has_value() ? program_config->compute_with_storage_grid_size
                                                     : device->compute_with_storage_grid_size();

    uint32_t num_cores_available = grid_size.x * grid_size.y;

    CoreRangeSet core_grid;
    bool on_subcoregrid = false;
    if (program_config.has_value() && program_config->sub_core_grids.has_value()) {
        core_grid = program_config->sub_core_grids.value();
        TT_FATAL(
            core_grid.num_cores() == num_cores_available,
            "Number of cores in sub_core_grids must match the number of cores available");
        on_subcoregrid = true;
    } else {
        core_grid = CoreRangeSet(std::vector{CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
    }

    uint32_t num_cores_in_grid =
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
    TT_FATAL(
        num_cores_available <= num_cores_in_grid,
        "Expected number of cores available to be less than or equal to the number of cores in the grid, got {} and {}",
        num_cores_available,
        num_cores_in_grid);
    TT_FATAL(
        num_cores_available >= B,
        "Expect number of cores available to be greater or equal to batch size, got {} and {}",
        num_cores_available,
        B);

    // balance the number of cores to use based on batch
    uint32_t max_num_cores_for_compute =
        program_config.has_value() ? program_config->max_cores_per_head_batch * B * num_kv_heads : num_cores_available;
    uint32_t num_cores_per_batch = std::min(num_cores_available, max_num_cores_for_compute) / B;
    //// for core assignment, it is the same whether there's 1 core for head or 1 core for many heads
    uint32_t num_cores_per_head = std::max((uint32_t)1, num_cores_per_batch / num_kv_heads);
    uint32_t num_heads_per_core = std::max((uint32_t)1, (uint32_t)std::ceil((float)num_kv_heads / num_cores_per_batch));
    uint32_t num_reducer_cores = num_kv_heads * B / num_heads_per_core;
    uint32_t num_output_cores = B;
    uint32_t num_active_cores = num_cores_per_head * num_kv_heads * B / num_heads_per_core;
    //// recalculate num_cores_per_batch based on num_active_cores
    num_cores_per_batch = num_active_cores / B;

    TT_FATAL(
        ((num_cores_per_head >= 1) && (num_heads_per_core == 1)) ||
            ((num_cores_per_head == 1) && (num_heads_per_core >= 1)),
        "This assertion should always be true, unless core assignment logic is wrong");

    // create core group, assume n batch and k_heads:
    // this is a 1D list of cores sorted by batch_output1, worker, ..., batch_output2, worker, ..., batch_output n,
    // worker, ... Within each batch, we will assign head reducers. e.g. the following mapping:
    // (batch_output1, worker1,   worker2),   (worker3,       worker4,   worker5),   ..., (... worker3*k-1, worker3*k)
    // (head_reducer1, h_worker1, h_worker2), (head_reducer2, h_worker1, h_worker2), ..., (head_reducerk, h_worker1,
    // h_worker2) head_reducer2 to head_reducerk then send the result to head_reducer1, which is also the batch_output1
    std::vector<CoreCoord> core_group;
    std::vector<CoreCoord> core_group_idle;
    if (on_subcoregrid) {
        if (is_q_sharded || is_output_sharded) {
            auto cores_vec = corerange_to_cores(core_grid, num_cores_available, true);
            int reducer_idx = 0;
            int worker_idx = num_output_cores;
            for (int i = 0; i < num_cores_available; ++i) {
                if (i % num_cores_per_batch == 0 && reducer_idx < num_output_cores) {
                    i < num_active_cores ? core_group.push_back(cores_vec[reducer_idx])
                                         : core_group_idle.push_back(cores_vec[reducer_idx]);
                    reducer_idx++;
                } else {
                    i < num_active_cores ? core_group.push_back(cores_vec[worker_idx])
                                         : core_group_idle.push_back(cores_vec[worker_idx]);
                    worker_idx++;
                }
            }
        } else {
            TT_FATAL(false, "We only support SDPA on subcoregrids with sharded Q and sharded output");
        }
    } else {
        if (is_q_sharded || is_output_sharded) {
            int reducer_idx = 0;
            int worker_idx = num_output_cores;

            for (int i = 0; i < num_cores_available; ++i) {
                CoreCoord core;
                if (i % num_cores_per_batch == 0 && reducer_idx < num_output_cores) {
                    core = {reducer_idx % grid_size.x, reducer_idx / grid_size.x};
                    reducer_idx++;
                } else {
                    core = {worker_idx % grid_size.x, worker_idx / grid_size.x};
                    worker_idx++;
                }
                if (i < num_active_cores) {
                    core_group.push_back(core);
                } else {
                    core_group_idle.push_back(core);
                }
            }
        } else {
            for (int i = 0; i < num_cores_available; ++i) {
                CoreCoord core = {i % grid_size.x, i / grid_size.x};
                if (i < num_active_cores) {
                    core_group.push_back(core);
                } else {
                    core_group_idle.push_back(core);
                }
            }
        }
    }

    log_debug(tt::LogOp, "Parallelization scheme:");
    log_debug(tt::LogOp, "num_cores_available: {}", num_cores_available);
    log_debug(tt::LogOp, "num_cores_per_batch: {}", num_cores_per_batch);
    log_debug(tt::LogOp, "num_cores_per_head: {}", num_cores_per_head);
    log_debug(tt::LogOp, "num_heads_per_core: {}", num_heads_per_core);
    log_debug(tt::LogOp, "num_active_cores: {}", num_active_cores);
    log_debug(tt::LogOp, "num_reducer_cores: {}", num_reducer_cores);
    log_debug(tt::LogOp, "num_output_cores: {}", num_output_cores);
    log_debug(tt::LogOp, "core_group: {}", core_group);
    log_debug(tt::LogOp, "core_group_idle: {}", core_group_idle);
    log_debug(tt::LogOp, "max_num_cores_for_compute: {}", max_num_cores_for_compute);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)

    // If using dyanmic chunk size, set it to some max number of tiles (less than DST for now)
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t max_dynamic_chunk_size = dst_size;
    const uint32_t Sk_chunk_t_cb_size = Sk_chunk_t == 0 ? max_dynamic_chunk_size : Sk_chunk_t;

    uint32_t q_tiles = PNHt * DHt;
    uint32_t k_tiles = Sk_chunk_t_cb_size * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t_cb_size * vDHt * 2;  // double buffer
    uint32_t qk_tiles = PNHt * Sk_chunk_t_cb_size;
    uint32_t out_im_tiles = PNHt * vDHt;
    uint32_t out0_t = PNHt * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = PNHt;  // Single column of values in each iteration

    // log all values
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t qk_in0_block_w = DHt;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    uint32_t qk_out_subblock_w = 0;
    uint32_t qk_out_subblock_h = 0;
    uint32_t qk_in0_num_subblocks = 0;
    uint32_t qk_in1_num_subblocks = 0;
    if (Sk_chunk_t > 0) {
        // max of Sk_chunk_t and dst_size
        qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
        // If qk_out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to
        // maintain row-major intermediate buffer.
        qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t) ? (std::min(PNHt, dst_size / qk_out_subblock_w)) : 1;

        qk_in0_num_subblocks = PNHt / qk_out_subblock_h;
        qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    }

    // now for out0
    uint32_t out_in0_block_w = 0;
    uint32_t out_num_blocks = 0;
    if (Sk_chunk_t > 0) {
        out_in0_block_w = Sk_chunk_t;
        out_num_blocks = Sk_chunk_t / out_in0_block_w;
    }

    const uint32_t out_out_subblock_w = std::min(vDHt, dst_size);
    const uint32_t out_out_subblock_h =
        (out_out_subblock_w == vDHt) ? (std::min(PNHt, dst_size / out_out_subblock_w)) : 1;

    const uint32_t out_in0_num_subblocks = PNHt / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;

    uint32_t dht_granularity = std::min(DHt, dst_size);
    uint32_t log2_dht_granularity = std::log2(dht_granularity);
    // Sometimes DHt is not a power of 2, so granularity should be 1
    if (dht_granularity != (1 << log2_dht_granularity)) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }
    TT_FATAL(
        dht_granularity == (1 << log2_dht_granularity),
        "dht_granularity must be a power of 2. Got {}.",
        dht_granularity);

    // log all values
    log_debug(tt::LogOp, "dst_size: {}", dst_size);
    log_debug(tt::LogOp, "qk_in0_block_w: {}", qk_in0_block_w);
    log_debug(tt::LogOp, "qk_out_subblock_w: {}", qk_out_subblock_w);
    log_debug(tt::LogOp, "qk_out_subblock_h: {}", qk_out_subblock_h);
    log_debug(tt::LogOp, "qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    log_debug(tt::LogOp, "qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    log_debug(tt::LogOp, "qk_num_blocks: {}", qk_num_blocks);
    log_debug(tt::LogOp, "out_in0_block_w: {}", out_in0_block_w);
    log_debug(tt::LogOp, "out_out_subblock_w: {}", out_out_subblock_w);
    log_debug(tt::LogOp, "out_out_subblock_h: {}", out_out_subblock_h);
    log_debug(tt::LogOp, "out_in0_num_subblocks: {}", out_in0_num_subblocks);
    log_debug(tt::LogOp, "out_in1_num_subblocks: {}", out_in1_num_subblocks);
    log_debug(tt::LogOp, "out_num_blocks: {}", out_num_blocks);
    log_debug(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_debug(tt::LogOp, "log2_dht_granularity: {}", log2_dht_granularity);

    // Create circular buffers
    tt::DataFormat q_df = tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    tt::DataFormat v_df = tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    tt::DataFormat mask_df = use_attention_mask ? tt_metal::datatype_to_dataformat_converter(attn_mask.value().dtype())
                                                : tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
    // tt::DataFormat im_df = tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = tt::DataFormat::Float16_b;

    const auto half_tile = tt::tt_metal::Tile({16, 32});
    const auto full_tile = tt::tt_metal::Tile({32, 32});

    auto q_tile = full_tile;
    const auto k_tile = full_tile;
    const auto v_tile = full_tile;
    auto mask_tile = full_tile;

    auto out_tile = full_tile;

    auto scalar_tile = full_tile;
    auto im_tile = full_tile;
    auto stats_tile = full_tile;

    // TODO: Directly get q input as tensor with 16x32 tiny tiles #25059
    // For now, use this flag in reader differentiate
    // - In non-causal mode, mask can be an input tensor which needs proper handling to read as 16x32 tiles
    // - Only support Float16_b since block float w/ shared exp needs special handling to read as 16x32 tiles
    // In compute, need to find a proper way to get num_faces for sfpu functions
    const bool use_half_tile = (is_causal and num_q_heads <= 16 and q_df == tt::DataFormat::Float16_b);

    if (use_half_tile) {
        q_tile = half_tile;
        mask_tile = half_tile;

        // TODO: out_tile is re-packed as full 32x32 with PACK for now #25060
        // out_tile = half_tile;

        scalar_tile = half_tile;
        im_tile = half_tile;
        stats_tile = half_tile;
    }

    uint32_t q_tile_size = q_tile.get_tile_size(q_df);
    uint32_t k_tile_size = k_tile.get_tile_size(k_df);
    uint32_t v_tile_size = v_tile.get_tile_size(v_df);
    uint32_t mask_tile_size = mask_tile.get_tile_size(mask_df);
    uint32_t out_tile_size = out_tile.get_tile_size(out_df);
    uint32_t scalar_tile_size = scalar_tile.get_tile_size(scalar_df);
    uint32_t im_tile_size = im_tile.get_tile_size(im_df);
    uint32_t stats_tile_size = stats_tile.get_tile_size(stats_df);

    uint32_t intermed_output_tiles = (out0_t + 2 * PNHt) * (num_cores_per_head - 1);

    uint32_t index_stick_size = 0;
    bool is_cur_pos_tensor_sharded = false;
    CBHandle cb_in8_id = 0;
    if (use_cur_pos_tensor) {
        auto* pos_buffer = cur_pos_tensor.value().buffer();
        tt::DataFormat pos_df = tt_metal::datatype_to_dataformat_converter(cur_pos_tensor.value().dtype());
        index_stick_size = pos_buffer->aligned_page_size();

        // cb pos
        auto c_in8_config = CircularBufferConfig(index_stick_size, {{CBIndex::c_8, pos_df}})
                                .set_page_size(CBIndex::c_8, index_stick_size);
        if (cur_pos_tensor.value().is_sharded()) {
            is_cur_pos_tensor_sharded = true;
            c_in8_config.set_globally_allocated_address(*pos_buffer);
        }
        cb_in8_id = CreateCircularBuffer(program, core_grid, c_in8_config);
    }

    uint32_t page_table_stick_size = 0;
    uint32_t shard_size = 0;
    bool is_page_table_sharded = false;
    CBHandle cb_in9_id = 0;
    if (is_paged_attention) {
        auto* page_table_buffer = page_table_tensor.value().buffer();
        is_page_table_sharded = page_table_tensor.value().is_sharded();
        tt::DataFormat page_table_df = tt_metal::datatype_to_dataformat_converter(page_table_tensor.value().dtype());
        page_table_stick_size = page_table_buffer->aligned_page_size();
        shard_size = is_page_table_sharded ? B * page_table_stick_size : page_table_stick_size;
        // cb page_table
        auto c_in9_config = CircularBufferConfig(shard_size, {{CBIndex::c_9, page_table_df}})
                                .set_page_size(CBIndex::c_9, page_table_stick_size);

        if (is_page_table_sharded) {
            c_in9_config.set_globally_allocated_address(*page_table_buffer);
        }
        cb_in9_id = CreateCircularBuffer(program, core_grid, c_in9_config);
    }

    log_debug(tt::LogOp, "q_data_format: {}", q_df);
    log_debug(tt::LogOp, "k_data_format: {}", k_df);
    log_debug(tt::LogOp, "v_data_format: {}", v_df);
    log_debug(tt::LogOp, "out_data_format: {}", out_df);
    log_debug(tt::LogOp, "scalar_data_format: {}", scalar_df);
    log_debug(tt::LogOp, "intermediate_data_format: {}", im_df);
    log_debug(tt::LogOp, "statistics_data_format: {}", stats_df);

    // CBs
    // Q input
    auto c_in0_config = CircularBufferConfig(q_tiles * q_tile_size, {{CBIndex::c_0, q_df}})
                            .set_page_size(CBIndex::c_0, q_tile_size)
                            .set_tile_dims(CBIndex::c_0, q_tile);
    CreateCircularBuffer(program, core_grid, c_in0_config);

    // K input
    auto c_in1_config =
        CircularBufferConfig(k_tiles * k_tile_size, {{CBIndex::c_1, k_df}}).set_page_size(CBIndex::c_1, k_tile_size);
    CreateCircularBuffer(program, core_grid, c_in1_config);

    // V input
    auto c_in2_config =
        CircularBufferConfig(v_tiles * v_tile_size, {{CBIndex::c_2, v_df}}).set_page_size(CBIndex::c_2, v_tile_size);
    CreateCircularBuffer(program, core_grid, c_in2_config);

    // attn_mask input
    auto c_in3_config = CircularBufferConfig(qk_tiles * mask_tile_size, {{CBIndex::c_3, mask_df}})
                            .set_page_size(CBIndex::c_3, mask_tile_size)
                            .set_tile_dims(CBIndex::c_3, mask_tile);
    CreateCircularBuffer(program, core_grid, c_in3_config);

    // attention_sink input (conditionally created based on use_attention_sink)
    if (use_attention_sink) {
        auto c_in4_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_4, stats_df}})
                                .set_page_size(CBIndex::c_4, stats_tile_size)
                                .set_tile_dims(CBIndex::c_4, stats_tile);
        CreateCircularBuffer(program, core_grid, c_in4_config);
    }

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{CBIndex::c_5, scalar_df}})
                            .set_page_size(CBIndex::c_5, scalar_tile_size)
                            .set_tile_dims(CBIndex::c_5, scalar_tile);
    CreateCircularBuffer(program, core_grid, c_in5_config);

    // cb_m_in
    auto c_in6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_6, stats_df}})
                            .set_page_size(CBIndex::c_6, stats_tile_size)
                            .set_tile_dims(CBIndex::c_6, stats_tile);
    CreateCircularBuffer(program, core_grid, c_in6_config);

    // cb_l_in
    auto c_in7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_7, stats_df}})
                            .set_page_size(CBIndex::c_7, stats_tile_size)
                            .set_tile_dims(CBIndex::c_7, stats_tile);
    CreateCircularBuffer(program, core_grid, c_in7_config);

    // tilizedQ input

    auto c_tilized_q_config = CircularBufferConfig(q_tiles * q_tile_size, {{CBIndex::c_10, q_df}})
                                  .set_page_size(CBIndex::c_10, q_tile_size)
                                  .set_tile_dims(CBIndex::c_10, q_tile);
    CreateCircularBuffer(program, core_grid, c_tilized_q_config);

    // cb_col_identity
    auto col_identity_tile = full_tile;
    auto col_identity_tile_size = col_identity_tile.get_tile_size(scalar_df);

    auto c_in11_config = CircularBufferConfig(scale_tiles * col_identity_tile_size, {{CBIndex::c_11, scalar_df}})
                             .set_page_size(CBIndex::c_11, col_identity_tile_size)
                             .set_tile_dims(CBIndex::c_11, col_identity_tile);
    CreateCircularBuffer(program, core_grid, c_in11_config);

    // cb zero config
    auto c_zero_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{CBIndex::c_12, scalar_df}})
                             .set_page_size(CBIndex::c_12, scalar_tile_size)
                             .set_tile_dims(CBIndex::c_12, scalar_tile);
    CreateCircularBuffer(program, core_grid, c_zero_config);

    // sliding window mask input (conditionally created based on sliding_window_size)
    if (sliding_window_size.has_value() && sliding_window_size.value() > 0) {
        auto c_sliding_window_mask_config = CircularBufferConfig(qk_tiles * mask_tile_size, {{CBIndex::c_13, mask_df}})
                                                .set_page_size(CBIndex::c_13, mask_tile_size)
                                                .set_tile_dims(CBIndex::c_13, mask_tile);
        CreateCircularBuffer(program, core_grid, c_sliding_window_mask_config);
    }

    // cb_qk_im
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{CBIndex::c_24, im_df}})
                                  .set_page_size(CBIndex::c_24, im_tile_size)
                                  .set_tile_dims(CBIndex::c_24, im_tile);
    CreateCircularBuffer(program, core_grid, c_intermed0_config);

    // cb_out_im
    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{CBIndex::c_25, im_df}})
                                  .set_page_size(CBIndex::c_25, im_tile_size)
                                  .set_tile_dims(CBIndex::c_25, im_tile);
    CreateCircularBuffer(program, core_grid, c_intermed1_config);

    // cb_out_accumulate_im
    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{CBIndex::c_26, im_df}})
                                  .set_page_size(CBIndex::c_26, im_tile_size)
                                  .set_tile_dims(CBIndex::c_26, im_tile);
    CreateCircularBuffer(program, core_grid, c_intermed2_config);

    // cb_cur_max
    auto c_intermed3_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_27, stats_df}})
                                  .set_page_size(CBIndex::c_27, stats_tile_size)
                                  .set_tile_dims(CBIndex::c_27, stats_tile);
    CreateCircularBuffer(program, core_grid, c_intermed3_config);

    // cb_prev_max
    auto c_intermed4_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_28, stats_df}})
                                  .set_page_size(CBIndex::c_28, stats_tile_size)
                                  .set_tile_dims(CBIndex::c_28, stats_tile);
    CreateCircularBuffer(program, core_grid, c_intermed4_config);

    // cb_cur_sum
    auto c_intermed5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_29, stats_df}})
                                  .set_page_size(CBIndex::c_29, stats_tile_size)
                                  .set_tile_dims(CBIndex::c_29, stats_tile);
    CreateCircularBuffer(program, core_grid, c_intermed5_config);

    // cb_prev_sum
    auto c_intermed6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_30, stats_df}})
                                  .set_page_size(CBIndex::c_30, stats_tile_size)
                                  .set_tile_dims(CBIndex::c_30, stats_tile);
    CreateCircularBuffer(program, core_grid, c_intermed6_config);

    // cb_exp_max_diff
    auto c_intermed7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_31, stats_df}})
                                  .set_page_size(CBIndex::c_31, stats_tile_size)
                                  .set_tile_dims(CBIndex::c_31, stats_tile);
    CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // cb_prev_sum_2
    auto c_out5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_21, stats_df}})
                             .set_page_size(CBIndex::c_21, stats_tile_size)
                             .set_tile_dims(CBIndex::c_21, stats_tile);
    CreateCircularBuffer(program, core_grid, c_out5_config);

    // cb_exp_max_diff_2
    auto c_out6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_22, stats_df}})
                             .set_page_size(CBIndex::c_22, stats_tile_size)
                             .set_tile_dims(CBIndex::c_22, stats_tile);
    CreateCircularBuffer(program, core_grid, c_out6_config);

    // cb_out_accumulate_im_2
    auto c_out7_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{CBIndex::c_23, im_df}})
                             .set_page_size(CBIndex::c_23, im_tile_size)
                             .set_tile_dims(CBIndex::c_23, im_tile);
    CreateCircularBuffer(program, core_grid, c_out7_config);

    // Output
    // cb_out_o
    auto c_out0_config = CircularBufferConfig(out0_t * stats_tile_size, {{CBIndex::c_16, stats_df}})
                             .set_page_size(CBIndex::c_16, stats_tile_size)
                             .set_tile_dims(CBIndex::c_16, stats_tile);
    CreateCircularBuffer(program, core_grid, c_out0_config);

    // cb_out_m
    auto c_out1_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_17, stats_df}})
                             .set_page_size(CBIndex::c_17, stats_tile_size)
                             .set_tile_dims(CBIndex::c_17, stats_tile);
    CreateCircularBuffer(program, core_grid, c_out1_config);

    // cb_out_l
    auto c_out2_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{CBIndex::c_18, stats_df}})
                             .set_page_size(CBIndex::c_18, stats_tile_size)
                             .set_tile_dims(CBIndex::c_18, stats_tile);
    CreateCircularBuffer(program, core_grid, c_out2_config);

    // when there are worker cores
    if (intermed_output_tiles > 0) {
        // cb_intermed_out
        auto c_out3_config = CircularBufferConfig(intermed_output_tiles * stats_tile_size, {{CBIndex::c_19, stats_df}})
                                 .set_page_size(CBIndex::c_19, stats_tile_size)
                                 .set_tile_dims(CBIndex::c_19, stats_tile);
        CreateCircularBuffer(program, core_grid, c_out3_config);
    }

    // cb_out_final
    auto c_out4_config = CircularBufferConfig(out0_t * out_tile_size, {{CBIndex::c_20, out_df}})
                             .set_page_size(CBIndex::c_20, out_tile_size)
                             .set_tile_dims(CBIndex::c_20, out_tile);
    if (is_output_sharded) {
        c_out4_config.set_globally_allocated_address(*out0_buffer);
    }
    auto cb_out4_id = CreateCircularBuffer(program, core_grid, c_out4_config);

    // *** Create Kernels and Compile Time Args ***
    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    class bfloat16 bfloat_zero_scalar(0.0f);
    uint32_t packed_zero_scalar = pack_two_bfloat16_into_uint32({bfloat_zero_scalar, bfloat_zero_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale.value_or(1.0f);

    // Create core groups for reduce cores
    std::vector<uint32_t> reduce_core_physical_xs;
    std::vector<uint32_t> reduce_core_physical_ys;
    uint32_t reduce_core_noc_x{};
    uint32_t reduce_core_noc_y{};
    reduce_core_physical_xs.reserve(num_reducer_cores);
    reduce_core_physical_ys.reserve(num_reducer_cores);

    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        uint32_t worker_id_for_reduce = (i % num_cores_per_head) - 1;
        bool do_reduce = (worker_id_for_reduce == -1);
        if (do_reduce) {
            reduce_core_noc_x = core.x;
            reduce_core_noc_y = core.y;
            // get physical core
            CoreCoord reduce_core = {(std::size_t)reduce_core_noc_x, (std::size_t)reduce_core_noc_y};
            auto reduce_core_physical = device->worker_core_from_logical_core(reduce_core);
            reduce_core_physical_xs.push_back((uint32_t)reduce_core_physical.x);
            reduce_core_physical_ys.push_back((uint32_t)reduce_core_physical.y);
        }
    }

    log_debug(tt::LogOp, "reduce_core_physical_xs: {}", reduce_core_physical_xs);
    log_debug(tt::LogOp, "reduce_core_physical_ys: {}", reduce_core_physical_ys);

    // Create core ggroups for output cores
    std::vector<uint32_t> output_core_physical_xs;
    std::vector<uint32_t> output_core_physical_ys;
    uint32_t output_core_noc_x{};
    uint32_t output_core_noc_y{};
    output_core_physical_xs.reserve(num_output_cores);  // num output cores is equal to batch size
    output_core_physical_ys.reserve(num_output_cores);

    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        uint32_t worker_id_for_output = (i % num_cores_per_batch) - 1;
        bool do_output = (worker_id_for_output == -1);
        if (do_output) {
            output_core_noc_x = core.x;
            output_core_noc_y = core.y;
            // get physical core
            CoreCoord output_core = {(std::size_t)output_core_noc_x, (std::size_t)output_core_noc_y};
            auto output_core_physical = device->worker_core_from_logical_core(output_core);
            output_core_physical_xs.push_back((uint32_t)output_core_physical.x);
            output_core_physical_ys.push_back((uint32_t)output_core_physical.y);
        }
    }

    log_debug(tt::LogOp, "output_core_physical_xs: {}", output_core_physical_xs);
    log_debug(tt::LogOp, "output_core_physical_ys: {}", output_core_physical_ys);

    // Common Compile time Args
    auto reducer_semaphore_id = tt_metal::CreateSemaphore(program, core_grid, 0);
    auto output_semaphore_id = tt_metal::CreateSemaphore(program, core_grid, 0);

    // If q is sharded, directly read in q_chunk_size_bytes if q is row major or tilized but with full tiles
    // If q is tilized and want to use tiny tiles, this is ignored since we need to skip bottom half of tiles
    const uint32_t q_chunk_size_bytes =
        q_tiles * (tilize_q ? num_q_heads * TILE_WIDTH * input_tensor_q.element_size() : q_tile_size);

    std::vector<uint32_t> reader_compile_time_args_common = {
        B,
        PNHt,
        St,
        DHt,
        vDHt,
        Sk_chunk_t,
        num_active_cores,
        is_q_sharded,
        num_cores_per_batch,
        k_chunk_size,
        index_stick_size,
        (uint32_t)is_paged_attention,
        num_kv_heads,
        page_block_size_t,
        Bkv,
        q_heads_parallel_factor,
        num_cores_per_head,
        num_heads_per_core,
        num_output_cores,
        is_causal,
        use_attention_mask,
        use_attention_sink,
        max_dynamic_chunk_size,
        tilize_q,
        (uint32_t)use_mla,
        use_half_tile,
        q_chunk_size_bytes,
        is_cur_pos_tensor_sharded,
        is_page_table_sharded,
        full_tile.get_tile_size(q_df),
        sliding_window_size.value_or(0),
    };
    tt_metal::TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(attn_mask ? attn_mask->buffer() : nullptr).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(cur_pos_tensor ? cur_pos_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(page_table_tensor ? page_table_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args_common);

    if (use_attention_sink) {
        tt_metal::TensorAccessorArgs(*attention_sink->buffer()).append_to(reader_compile_time_args_common);
    } else {
        reader_compile_time_args_common.push_back(0);
    }

    std::vector<uint32_t> writer_compile_time_args_common = {
        B,
        PNHt,
        St,
        DHt,
        vDHt,
        Sk_chunk_t,
        packed_identity_scalar,
        packed_zero_scalar,
        scale_union.u,
        num_cores_per_batch,
        num_active_cores,
        reducer_semaphore_id,
        output_semaphore_id,
        is_output_sharded,
        k_chunk_size,
        num_q_heads,
        num_kv_heads,
        num_cores_per_head,
        num_heads_per_core,
        num_reducer_cores,
        num_output_cores,
        output_tensor.element_size(),
        is_causal,
        max_dynamic_chunk_size,
        q_heads_parallel_factor,
        sliding_window_size.value_or(0),
    };
    tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args_common);

    std::vector<uint32_t> compute_compile_time_args_common = {
        St,
        DHt,
        vDHt,
        PNHt,
        Sk_chunk_t,
        qk_in0_block_w,
        qk_out_subblock_w,
        qk_out_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,
        out_in0_block_w,
        out_out_subblock_w,
        out_out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,
        num_cores_per_batch,
        k_chunk_size,
        num_cores_per_head,
        num_heads_per_core,
        is_causal,
        use_attention_mask,
        use_attention_sink,
        max_dynamic_chunk_size,
        tilize_q,
        q_heads_parallel_factor,
        use_half_tile,
        scale_union.u,
        sliding_window_size.value_or(0),
    };

    // Determine granularity for compute loops
    std::map<std::string, std::string> compute_defines;
    if (Sk_chunk_t > 0) {
        const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
        const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
        TT_FATAL(
            sub_exp_granularity == (1 << log2_sub_exp_granularity),
            "Sub-exp granularity ({}) must be a power of 2 (2^{})",
            sub_exp_granularity,
            log2_sub_exp_granularity);

        const uint32_t mul_bcast_granularity = std::min(PNHt * Sk_chunk_t, dst_size);
        const uint32_t log2_mul_bcast_granularity = std::log2(mul_bcast_granularity);
        TT_FATAL(
            mul_bcast_granularity == (1 << log2_mul_bcast_granularity),
            "Mul-bcast granularity ({}) must be a power of 2 (2^{})",
            mul_bcast_granularity,
            log2_mul_bcast_granularity);

        compute_defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
        compute_defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
        compute_defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
        compute_defines["LOG2_MUL_BCAST_GRANULARITY"] = std::to_string(log2_mul_bcast_granularity);

        // Log these
        log_debug(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
        log_debug(tt::LogOp, "log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
        log_debug(tt::LogOp, "mul_bcast_granularity: {}", mul_bcast_granularity);
        log_debug(tt::LogOp, "log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);
    } else {
        compute_defines["DYNAMIC_CHUNK_SIZE"] = "1";
    }
    compute_defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    compute_defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    compute_defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);

    if (Sk_chunk_t > 0) {
        // Determine granularity for statistics computation
        const uint32_t stats_granularity = std::min(Sk_chunk_t, dst_size);
        // Find log2 of stats_granularity using std
        const uint32_t log2_stats_granularity = std::log2(stats_granularity);
        // Assert that this is a power of 2
        TT_FATAL(
            stats_granularity == (1 << log2_stats_granularity),
            "stats_granularity must be a power of 2. Got {}.",
            stats_granularity);

        compute_defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
        compute_defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
    }

    // Compute
    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp",
        core_grid,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args_common,
            .defines = compute_defines});

    // Reader
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp",
        core_grid,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args_common));

    // Writer
    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp",
        core_grid,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args_common));

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t pos_addr = use_cur_pos_tensor ? cur_pos_tensor.value().buffer()->address() : 0;
    uint32_t page_table_addr = is_paged_attention ? page_table_tensor.value().buffer()->address() : 0;
    uint32_t attn_mask_addr = use_attention_mask ? attn_mask.value().buffer()->address() : 0;
    uint32_t attention_sink_addr = use_attention_sink ? attention_sink.value().buffer()->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    // Set rt args
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        uint32_t worker_id_for_reduce = (i % num_cores_per_head) - 1;
        uint32_t worker_id_for_output = (i % num_cores_per_batch) - 1;
        bool do_reduce = (worker_id_for_reduce == -1);
        bool do_output = (worker_id_for_output == -1);

        uint32_t cur_head = (i % num_cores_per_batch) / num_cores_per_head;
        uint32_t cur_batch = i / num_cores_per_batch;
        uint32_t core_num_in_reduce = i % num_cores_per_head;
        uint32_t core_num_in_output = i % num_cores_per_batch;

        uint32_t cur_pos =
            (use_cur_pos_tensor || !is_causal) ? -1 : cur_pos_ids.at((uint32_t)(cur_batch / q_heads_parallel_factor));

        log_debug(tt::LogOp, "---- core_id: {}, coord: {} ----", i, core);
        log_debug(tt::LogOp, "worker_id_for_reduce: {}", worker_id_for_reduce);
        log_debug(tt::LogOp, "worker_id_for_output: {}", worker_id_for_output);
        log_debug(tt::LogOp, "do_reduce: {}", do_reduce);
        log_debug(tt::LogOp, "do_output: {}", do_output);
        log_debug(tt::LogOp, "cur_head: {}", cur_head);
        log_debug(tt::LogOp, "cur_batch: {}", cur_batch);
        log_debug(tt::LogOp, "core_num_in_reduce: {}", core_num_in_reduce);
        log_debug(tt::LogOp, "core_num_in_output: {}", core_num_in_output);
        log_debug(tt::LogOp, "cur_pos: {}", cur_pos);

        // reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            q_addr,
            k_addr,
            v_addr,
            pos_addr,
            page_table_addr,
            attn_mask_addr,
            attention_sink_addr,
            page_table_stick_size,
            do_reduce,
            do_output,
            cur_head,
            cur_batch,
            core_num_in_reduce,
            core_num_in_output,
            cur_pos};
        reader_rt_args.insert(reader_rt_args.end(), output_core_physical_xs.begin(), output_core_physical_xs.end());
        reader_rt_args.insert(reader_rt_args.end(), output_core_physical_ys.begin(), output_core_physical_ys.end());

        // writer runtime args
        std::vector<uint32_t> writer_rt_args = {
            out_addr,
            worker_id_for_reduce,
            worker_id_for_output,
            do_reduce,
            do_output,
            cur_head,
            cur_batch,
            core_num_in_reduce,
            core_num_in_output,
            cur_pos};
        writer_rt_args.insert(writer_rt_args.end(), reduce_core_physical_xs.begin(), reduce_core_physical_xs.end());
        writer_rt_args.insert(writer_rt_args.end(), reduce_core_physical_ys.begin(), reduce_core_physical_ys.end());
        writer_rt_args.insert(writer_rt_args.end(), output_core_physical_xs.begin(), output_core_physical_xs.end());
        writer_rt_args.insert(writer_rt_args.end(), output_core_physical_ys.begin(), output_core_physical_ys.end());

        // compute runtime args
        std::vector<uint32_t> compute_rt_args = {
            do_reduce, do_output, cur_head, cur_batch, core_num_in_reduce, core_num_in_output, cur_pos};

        SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);
        SetRuntimeArgs(program, writer_kernels_id, core, writer_rt_args);
        SetRuntimeArgs(program, compute_kernels_id, core, compute_rt_args);
    }
    if (num_active_cores < num_cores_available) {
        log_debug(tt::LogOp, "idle cores {}", core_group_idle.size());
        // Set the rest of the cores to idle
        for (auto core : core_group_idle) {
            log_debug(tt::LogOp, "Setting core {} to idle", core);
            // reader runtime args
            std::vector<uint32_t> reader_rt_args = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            // writer runtime args
            std::vector<uint32_t> writer_rt_args = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);
            SetRuntimeArgs(program, writer_kernels_id, core, writer_rt_args);
            SetRuntimeArgs(program, compute_kernels_id, core, {65, 0, 0, 0, 0, 0, 0});
        }
    }

    return cached_program_t{
        std::move(program),
        {.num_active_cores = num_active_cores,
         .core_group = core_group,
         .reader_kernels_id = reader_kernels_id,
         .writer_kernels_id = writer_kernels_id,
         .compute_kernels_id = compute_kernels_id,
         .num_cores_per_batch = num_cores_per_batch,
         .num_cores_per_head = num_cores_per_head,
         .num_output_cores = num_output_cores,
         .cb_in8_id = cb_in8_id,
         .cb_in9_id = cb_in9_id,
         .is_output_sharded = is_output_sharded,
         .cb_out4_id = cb_out4_id,
         .B = B,
         .q_heads_parallel_factor = q_heads_parallel_factor,
         .use_cur_pos_tensor = use_cur_pos_tensor,
         .use_attention_mask = use_attention_mask,
         .use_attention_sink = use_attention_sink,
         .is_paged_attention = is_paged_attention,
         .is_causal = is_causal,
         .use_mla = use_mla}};
}

void SdpaDecodeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SdpaDecodeParams& operation_attributes,
    const SdpaDecodeInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;

    const auto& shared_variables = cached_program.shared_variables;
    const auto& num_active_cores = shared_variables.num_active_cores;
    const auto& core_group = shared_variables.core_group;
    const auto& reader_kernels_id = shared_variables.reader_kernels_id;
    const auto& writer_kernels_id = shared_variables.writer_kernels_id;
    const auto& compute_kernels_id = shared_variables.compute_kernels_id;
    const auto& num_cores_per_batch = shared_variables.num_cores_per_batch;
    const auto& num_cores_per_head = shared_variables.num_cores_per_head;
    const auto& cb_in8_id = shared_variables.cb_in8_id;
    const auto& cb_in9_id = shared_variables.cb_in9_id;
    const auto& is_output_sharded = shared_variables.is_output_sharded;
    const auto& cb_out4_id = shared_variables.cb_out4_id;
    const auto& q_heads_parallel_factor = shared_variables.q_heads_parallel_factor;
    const auto& cur_pos_ids = operation_attributes.cur_pos;
    const bool use_cur_pos_tensor = shared_variables.use_cur_pos_tensor;
    const bool use_attention_mask = shared_variables.use_attention_mask;
    const bool use_attention_sink = shared_variables.use_attention_sink;
    const bool is_paged_attention = shared_variables.is_paged_attention;
    const bool is_causal = shared_variables.is_causal;
    const bool use_mla = shared_variables.use_mla;

    auto* q_buffer = tensor_args.q.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* v_buffer = use_mla ? k_buffer : tensor_args.v.value().buffer();

    auto* out0_buffer = tensor_return_value.buffer();

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t out_addr = out0_buffer->address();

    const auto& cur_pos_tensor = tensor_args.cur_pos_tensor;
    const auto& page_table_tensor = tensor_args.page_table_tensor;
    uint32_t pos_addr = use_cur_pos_tensor ? cur_pos_tensor.value().buffer()->address() : 0;

    uint32_t page_table_addr = is_paged_attention ? page_table_tensor.value().buffer()->address() : 0;
    uint32_t attn_mask_addr = use_attention_mask ? tensor_args.attn_mask.value().buffer()->address() : 0;
    uint32_t attention_sink_addr = use_attention_sink ? tensor_args.attention_sink.value().buffer()->address() : 0;
    auto* page_table_buffer = is_paged_attention ? page_table_tensor.value().buffer() : nullptr;
    uint32_t page_table_stick_size = is_paged_attention ? page_table_buffer->aligned_page_size() : 0;

    auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);
    auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernels_id);

    // Set rt args
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        uint32_t worker_id_for_reduce = (num_cores_per_head == 0) ? UINT32_MAX : (i % num_cores_per_head) - 1;
        uint32_t worker_id_for_output = ((i % num_cores_per_batch) == 0) ? UINT32_MAX : (i % num_cores_per_batch) - 1;
        bool do_reduce = (worker_id_for_reduce == UINT32_MAX);
        bool do_output = (worker_id_for_output == UINT32_MAX);
        uint32_t cur_head = (num_cores_per_head == 0) ? 0 : (i % num_cores_per_batch) / num_cores_per_head;
        uint32_t cur_batch = i / num_cores_per_batch;
        uint32_t core_num_in_reduce = (num_cores_per_head == 0) ? 0 : i % num_cores_per_head;
        uint32_t core_num_in_output = i % num_cores_per_batch;
        uint32_t cur_pos =
            (use_cur_pos_tensor || !is_causal) ? -1 : cur_pos_ids.at((uint32_t)(cur_batch / q_heads_parallel_factor));

        auto& reader_args = reader_args_by_core[core.x][core.y];
        auto& writer_args = writer_args_by_core[core.x][core.y];
        auto& compute_args = compute_args_by_core[core.x][core.y];

        // reader runtime args
        uint32_t arg_idx = 0;
        reader_args[arg_idx++] = q_addr;
        reader_args[arg_idx++] = k_addr;
        reader_args[arg_idx++] = v_addr;
        reader_args[arg_idx++] = pos_addr;
        reader_args[arg_idx++] = page_table_addr;
        reader_args[arg_idx++] = attn_mask_addr;
        reader_args[arg_idx++] = attention_sink_addr;
        reader_args[arg_idx++] = page_table_stick_size;
        reader_args[arg_idx++] = do_reduce;
        reader_args[arg_idx++] = do_output;
        reader_args[arg_idx++] = cur_head;
        reader_args[arg_idx++] = cur_batch;
        reader_args[arg_idx++] = core_num_in_reduce;
        reader_args[arg_idx++] = core_num_in_output;
        reader_args[arg_idx++] = cur_pos;

        // writer runtime args
        arg_idx = 0;
        writer_args[arg_idx++] = out_addr;
        writer_args[arg_idx++] = worker_id_for_reduce;
        writer_args[arg_idx++] = worker_id_for_output;
        writer_args[arg_idx++] = do_reduce;
        writer_args[arg_idx++] = do_output;
        writer_args[arg_idx++] = cur_head;
        writer_args[arg_idx++] = cur_batch;
        writer_args[arg_idx++] = core_num_in_reduce;
        writer_args[arg_idx++] = core_num_in_output;
        writer_args[arg_idx++] = cur_pos;

        // compute runtime args
        arg_idx = 0;
        compute_args[arg_idx++] = do_reduce;
        compute_args[arg_idx++] = do_output;
        compute_args[arg_idx++] = cur_head;
        compute_args[arg_idx++] = cur_batch;
        compute_args[arg_idx++] = core_num_in_reduce;
        compute_args[arg_idx++] = core_num_in_output;
        compute_args[arg_idx++] = cur_pos;
    }
    if (use_cur_pos_tensor and cur_pos_tensor.value().is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, cb_in8_id, *cur_pos_tensor.value().buffer());
    }
    if (is_paged_attention and page_table_tensor.value().is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, cb_in9_id, *page_table_tensor.value().buffer());
    }
    if (is_output_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_out4_id, *out0_buffer);
    }
}

}  // namespace ttnn::prim
