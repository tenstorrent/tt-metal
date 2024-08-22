// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_program_factory.hpp"

#include <optional>

#include "impl/buffers/buffer.hpp"
#include "sdpa_decode_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;

namespace {
uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<int>> get_runtime_args(
    const std::vector<uint32_t>& cur_pos, int B, int num_cores_per_batch, uint32_t k_chunk_size) {
    std::vector<uint32_t> all_PSt;
    std::vector<uint32_t> all_num_chunks;
    std::vector<int> all_chunk_assignments;
    all_PSt.reserve(cur_pos.size());
    all_num_chunks.reserve(cur_pos.size());
    all_chunk_assignments.reserve(B * num_cores_per_batch * 2);

    for (const auto& pos : cur_pos) {
        uint32_t valid_seq_len = nearest_n(pos + 1, k_chunk_size);
        uint32_t pst_value = valid_seq_len / TILE_HEIGHT;
        uint32_t num_chunks_value = valid_seq_len / k_chunk_size;
        all_PSt.push_back(pst_value);
        all_num_chunks.push_back(num_chunks_value);
    }

    for (int b = 0; b < B; ++b) {
        int chunks_per_core = all_num_chunks[b] / num_cores_per_batch;

        for (int i = 0; i < num_cores_per_batch; ++i) {
            all_chunk_assignments.push_back((num_cores_per_batch - i - 1) * chunks_per_core);
            all_chunk_assignments.push_back((num_cores_per_batch - i) * chunks_per_core);
        }
        all_chunk_assignments[2 * b * num_cores_per_batch + 1] += (all_num_chunks[b] % num_cores_per_batch);

        // chunk_assignment = chunk_assignment[::-1]
        // reduction core is the first core, and we always want the reduction core to deal with the residual chunks
        // residual chunks exists when other chunks are 0, and has less chunks than the other cores when other chunks
        // are not 0
    }

    return {all_PSt, all_num_chunks, all_chunk_assignments};
}
}  // namespace

namespace ttnn::operations::transformer::detail {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks sdpa_decode_multi_core(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& output_tensor,
    const std::vector<uint32_t>& cur_pos,
    std::optional<float> scale,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config,
    const uint32_t k_chunk_size) {
    /*
    Q: 1 x B x PNH x DH
    K: 1 x B x S x DH
    V: 1 x B x S x DH
    */

    /*
    Initially during compile time, we compile the kernel based on the longest sequence length in the batch.
    During runtime, we may override the number of chunks being processed based on the actual sequence length of the
    current batch.
    */

    const auto q_shape = input_tensor_q.get_legacy_shape();
    const auto k_shape = input_tensor_k.get_legacy_shape();
    // Use k_shape for S and DH since Q might be different for decode
    uint32_t B = q_shape[1], PNH = q_shape[2], S = k_shape[2], DH = k_shape[3];
    uint32_t St = S / TILE_HEIGHT;
    uint32_t DHt = DH / TILE_WIDTH;
    uint32_t PNHt = PNH / TILE_HEIGHT;
    uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    bool is_q_sharded = input_tensor_q.is_sharded();
    bool is_output_sharded = output_tensor.is_sharded();

    // log_debug all of the above
    tt::log_debug("B: {}", B);
    tt::log_debug("S: {}", S);
    tt::log_debug("DH: {}", DH);
    tt::log_debug("St: {}", St);
    tt::log_debug("DHt: {}", DHt);
    tt::log_debug("PNHt: {}", PNHt);
    tt::log_debug("Sk_chunk_t: {}", Sk_chunk_t);
    tt::log_debug("k_chunk_size: {}", k_chunk_size);

    std::shared_ptr<Program> program = CreateProgram();

    Device* device = input_tensor_q.device();

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;

    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                TT_FATAL(device->arch() == tt::ARCH::GRAYSKULL, "kernel config is not for graykull");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = false;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                TT_FATAL(
                    ttnn::device::is_wormhole_or_blackhole(device->arch()),
                    "kernel config is not for wormhole_b0 or blackhole");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
            } else {
                TT_FATAL("arch not supported");
            }
        },
        compute_kernel_config);

    auto q_buffer = input_tensor_q.buffer();
    auto k_buffer = input_tensor_k.buffer();
    auto v_buffer = input_tensor_v.buffer();

    auto out0_buffer = output_tensor.buffer();

    // Parallelization scheme
    // We will assign cores to batches
    // Split to cores
    CoreCoord grid_size = program_config.has_value() ? program_config->compute_with_storage_grid_size
                                                     : device->compute_with_storage_grid_size();

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    uint32_t num_cores_available = grid_size.x * grid_size.y;

    TT_FATAL(
        num_cores_available <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    // balance the number of cores to use based on batch
    uint32_t num_cores_per_batch = num_cores_available / B;
    uint32_t num_active_cores = num_cores_per_batch * B;

    // create core group, which is a 1D list of cores sorted by reducer1, worker, ..., reducer2, worker, ..., reducer n,
    // worker, ...
    std::vector<CoreCoord> core_group;
    std::vector<CoreCoord> core_group_idle;
    uint32_t num_reducers = B;
    if (is_q_sharded || is_output_sharded) {
        int reducer_idx = 0;
        int worker_idx = num_reducers;

        for (int i = 0; i < num_cores_available; ++i) {
            CoreCoord core;
            if (i % num_cores_per_batch == 0) {
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
        for (int i = 0; i < num_active_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};
            if (i < num_active_cores) {
                core_group.push_back(core);
            } else {
                core_group_idle.push_back(core);
            }
        }
    }

    tt::log_debug("Parallelization scheme:");
    tt::log_debug("num_cores_available: {}", num_cores_available);
    tt::log_debug("num_cores_per_batch: {}", num_cores_per_batch);
    tt::log_debug("num_active_cores: {}", num_active_cores);
    tt::log_debug("core_group: {}", core_group);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = PNHt * DHt;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t qk_tiles = PNHt * Sk_chunk_t;
    uint32_t out_im_tiles = PNHt * DHt;
    uint32_t out0_t = PNHt * DHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = PNHt;  // Single column of values in each iteration

    // log all values
    tt::log_debug("q_tiles: {}", q_tiles);
    tt::log_debug("k_tiles: {}", k_tiles);
    tt::log_debug("v_tiles: {}", v_tiles);
    tt::log_debug("qk_tiles: {}", qk_tiles);
    tt::log_debug("out0_t: {}", out0_t);
    tt::log_debug("scale_tiles: {}", scale_tiles);
    tt::log_debug("statistics_tiles: {}", statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_in0_block_w = DHt;
    // max of Sk_chunk_t and dst_size
    const uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
    // If qk_out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // row-major intermediate buffer.
    const uint32_t qk_out_subblock_h =
        (qk_out_subblock_w == Sk_chunk_t) ? (std::min(PNHt, dst_size / qk_out_subblock_w)) : 1;

    const uint32_t qk_in0_num_subblocks = PNHt / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;
    const uint32_t out_out_subblock_w = std::min(DHt, dst_size);
    const uint32_t out_out_subblock_h =
        (out_out_subblock_w == DHt) ? (std::min(PNHt, dst_size / out_out_subblock_w)) : 1;

    const uint32_t out_in0_num_subblocks = PNHt / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // log all values
    tt::log_debug("dst_size: {}", dst_size);
    tt::log_debug("qk_in0_block_w: {}", qk_in0_block_w);
    tt::log_debug("qk_out_subblock_w: {}", qk_out_subblock_w);
    tt::log_debug("qk_out_subblock_h: {}", qk_out_subblock_h);
    tt::log_debug("qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    tt::log_debug("qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    tt::log_debug("qk_num_blocks: {}", qk_num_blocks);
    tt::log_debug("out_in0_block_w: {}", out_in0_block_w);
    tt::log_debug("out_out_subblock_w: {}", out_out_subblock_w);
    tt::log_debug("out_out_subblock_h: {}", out_out_subblock_h);
    tt::log_debug("out_in0_num_subblocks: {}", out_in0_num_subblocks);
    tt::log_debug("out_in1_num_subblocks: {}", out_in1_num_subblocks);
    tt::log_debug("out_num_blocks: {}", out_num_blocks);

    // Determine granularity for statistics computation
    const uint32_t stats_granularity = std::min(PNHt, dst_size);
    // Find log2 of stats_granularity using std
    const uint32_t log2_stats_granularity = std::log2(stats_granularity);
    // Assert that this is a power of 2
    TT_FATAL(stats_granularity == (1 << log2_stats_granularity));

    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_FATAL(sub_exp_granularity == (1 << log2_sub_exp_granularity));

    const uint32_t mul_bcast_granularity = std::min(PNHt * Sk_chunk_t, dst_size);
    const uint32_t log2_mul_bcast_granularity = std::log2(mul_bcast_granularity);
    TT_FATAL(mul_bcast_granularity == (1 << log2_mul_bcast_granularity));

    const uint32_t dht_granularity = std::min(DHt, dst_size);
    const uint32_t log2_dht_granularity = std::log2(dht_granularity);

    // Log these
    tt::log_debug("stats_granularity: {}", stats_granularity);
    tt::log_debug("log2_stats_granularity: {}", log2_stats_granularity);
    tt::log_debug("sub_exp_granularity: {}", sub_exp_granularity);
    tt::log_debug("log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    tt::log_debug("mul_bcast_granularity: {}", mul_bcast_granularity);
    tt::log_debug("log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);
    tt::log_debug("dht_granularity: {}", dht_granularity);
    tt::log_debug("log2_dht_granularity: {}", log2_dht_granularity);

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.get_dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.get_dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.get_dtype());
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
    // tt::DataFormat im_df = tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = tt::DataFormat::Float16_b;

    uint32_t q_tile_size = tt::tt_metal::detail::TileSize(q_df);
    uint32_t k_tile_size = tt::tt_metal::detail::TileSize(k_df);
    uint32_t v_tile_size = tt::tt_metal::detail::TileSize(v_df);
    uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_df);
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_df);
    uint32_t stats_tile_size = tt::tt_metal::detail::TileSize(stats_df);
    uint32_t intermed_output_tiles = (out0_t + 2 * PNHt) * (num_cores_per_batch - 1);

    log_debug("q_data_format: {}", q_df);
    log_debug("k_data_format: {}", k_df);
    log_debug("v_data_format: {}", v_df);
    log_debug("out_data_format: {}", out_df);
    log_debug("scalar_data_format: {}", scalar_df);
    log_debug("intermediate_data_format: {}", im_df);
    log_debug("statistics_data_format: {}", stats_df);

    // CBs
    // Q input
    auto c_in0_config =
        CircularBufferConfig(q_tiles * q_tile_size, {{tt::CB::c_in0, q_df}}).set_page_size(tt::CB::c_in0, q_tile_size);
    auto cb_in0_id = CreateCircularBuffer(program, core_grid, c_in0_config);

    // K input
    auto c_in1_config =
        CircularBufferConfig(k_tiles * k_tile_size, {{tt::CB::c_in1, k_df}}).set_page_size(tt::CB::c_in1, k_tile_size);
    auto cb_in1_id = CreateCircularBuffer(program, core_grid, c_in1_config);

    // V input
    auto c_in2_config =
        CircularBufferConfig(v_tiles * v_tile_size, {{tt::CB::c_in2, v_df}}).set_page_size(tt::CB::c_in2, v_tile_size);
    auto cb_in2_id = CreateCircularBuffer(program, core_grid, c_in2_config);

    // attn_mask input
    auto c_in3_config = CircularBufferConfig(qk_tiles * stats_tile_size, {{tt::CB::c_in3, stats_df}})
                            .set_page_size(tt::CB::c_in3, stats_tile_size);
    auto cb_in3_id = CreateCircularBuffer(program, core_grid, c_in3_config);

    // scale input
    auto c_in4_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CB::c_in4, scalar_df}})
                            .set_page_size(tt::CB::c_in4, scalar_tile_size);
    auto cb_in4_id = CreateCircularBuffer(program, core_grid, c_in4_config);

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CB::c_in5, scalar_df}})
                            .set_page_size(tt::CB::c_in5, scalar_tile_size);
    auto cb_in5_id = CreateCircularBuffer(program, core_grid, c_in5_config);

    // cb_m_in
    auto c_in6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_in6, stats_df}})
                            .set_page_size(tt::CB::c_in6, stats_tile_size);
    auto cb_in6_id = CreateCircularBuffer(program, core_grid, c_in6_config);

    // cb_l_in
    auto c_in7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_in7, stats_df}})
                            .set_page_size(tt::CB::c_in7, stats_tile_size);
    auto c_in7_id = CreateCircularBuffer(program, core_grid, c_in7_config);

    // cb_qk_im
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{tt::CB::c_intermed0, im_df}})
                                  .set_page_size(tt::CB::c_intermed0, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer(program, core_grid, c_intermed0_config);

    // cb_out_im
    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CB::c_intermed1, im_df}})
                                  .set_page_size(tt::CB::c_intermed1, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer(program, core_grid, c_intermed1_config);

    // cb_out_accumulate_im
    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CB::c_intermed2, im_df}})
                                  .set_page_size(tt::CB::c_intermed2, im_tile_size);
    auto cb_intermed2_id = CreateCircularBuffer(program, core_grid, c_intermed2_config);

    // cb_cur_max
    auto c_intermed3_config =
        CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_intermed3, stats_df}})
            .set_page_size(tt::CB::c_intermed3, stats_tile_size);
    auto cb_intermed3_id = CreateCircularBuffer(program, core_grid, c_intermed3_config);

    // cb_prev_max
    auto c_intermed4_config =
        CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_intermed4, stats_df}})
            .set_page_size(tt::CB::c_intermed4, stats_tile_size);
    auto cb_intermed4_id = CreateCircularBuffer(program, core_grid, c_intermed4_config);

    // cb_cur_sum
    auto c_intermed5_config =
        CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_intermed5, stats_df}})
            .set_page_size(tt::CB::c_intermed5, stats_tile_size);
    auto cb_intermed5_id = CreateCircularBuffer(program, core_grid, c_intermed5_config);

    // cb_prev_sum
    auto c_intermed6_config =
        CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_intermed6, stats_df}})
            .set_page_size(tt::CB::c_intermed6, stats_tile_size);
    auto cb_intermed6_id = CreateCircularBuffer(program, core_grid, c_intermed6_config);

    // cb_exp_max_diff
    auto c_intermed7_config =
        CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_intermed7, stats_df}})
            .set_page_size(tt::CB::c_intermed7, stats_tile_size);
    auto cb_intermed7_id = CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // cb_prev_sum_2
    auto c_out5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_out5, stats_df}})
                             .set_page_size(tt::CB::c_out5, stats_tile_size);
    auto c_out5_id = CreateCircularBuffer(program, core_grid, c_out5_config);

    // cb_exp_max_diff_2
    auto c_out6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_out6, stats_df}})
                             .set_page_size(tt::CB::c_out6, stats_tile_size);
    auto c_out6_id = CreateCircularBuffer(program, core_grid, c_out6_config);

    // cb_out_accumulate_im_2
    auto c_out7_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CB::c_out7, im_df}})
                             .set_page_size(tt::CB::c_out7, im_tile_size);
    auto c_out7_id = CreateCircularBuffer(program, core_grid, c_out7_config);

    // Output
    // cb_out_o
    auto c_out0_config = CircularBufferConfig(out0_t * stats_tile_size, {{tt::CB::c_out0, stats_df}})
                             .set_page_size(tt::CB::c_out0, stats_tile_size);
    auto cb_out0_id = CreateCircularBuffer(program, core_grid, c_out0_config);

    // cb_out_m
    auto c_out1_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_out1, stats_df}})
                             .set_page_size(tt::CB::c_out1, stats_tile_size);
    auto cb_out1_id = CreateCircularBuffer(program, core_grid, c_out1_config);

    // cb_out_l
    auto c_out2_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CB::c_out2, stats_df}})
                             .set_page_size(tt::CB::c_out2, stats_tile_size);
    auto c_out2_id = CreateCircularBuffer(program, core_grid, c_out2_config);

    // when there are worker cores
    if (intermed_output_tiles > 0) {
        // cb_intermed_out
        auto c_out3_config = CircularBufferConfig(intermed_output_tiles * stats_tile_size, {{tt::CB::c_out3, stats_df}})
                                 .set_page_size(tt::CB::c_out3, stats_tile_size);
        auto c_out3_id = CreateCircularBuffer(program, core_grid, c_out3_config);
    }

    // cb_out_final
    auto c_out4_config = CircularBufferConfig(out0_t * out_tile_size, {{tt::CB::c_out4, out_df}})
                             .set_page_size(tt::CB::c_out4, out_tile_size);
    if (is_output_sharded) {
        c_out4_config.set_globally_allocated_address(*out0_buffer);
    }
    auto cb_out4_id = CreateCircularBuffer(program, core_grid, c_out4_config);

    // *** Create Kernels and Compile Time Args ***
    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union;
    scale_union.f = scale.value_or(1.0f);

    std::vector<uint32_t> reduce_core_physical_xs;
    std::vector<uint32_t> reduce_core_physical_ys;
    uint32_t reduce_core_noc_x{};
    uint32_t reduce_core_noc_y{};
    reduce_core_physical_xs.reserve(B);  // num reducers is equal to batch size
    reduce_core_physical_ys.reserve(B);

    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        uint32_t worker_id = i % num_cores_per_batch - 1;
        bool do_reduce = (worker_id == -1);
        uint32_t cur_batch = i / num_cores_per_batch;
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

    tt::log_debug("reduce_core_physical_xs: {}", reduce_core_physical_xs);
    tt::log_debug("reduce_core_physical_ys: {}", reduce_core_physical_ys);

    // Common Compile time Args
    auto in0_mcast_reducer_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, 0);

    std::vector<uint32_t> reader_compile_time_args_common = {
        B, PNHt, St, DHt, Sk_chunk_t, num_active_cores, is_q_sharded};

    std::vector<uint32_t> writer_compile_time_args_common = {
        B,
        PNHt,
        St,
        DHt,
        packed_identity_scalar,
        scale_union.u,
        num_cores_per_batch,
        num_active_cores,
        in0_mcast_reducer_semaphore_id,
        is_output_sharded,
    };

    std::vector<uint32_t> compute_compile_time_args_common = {
        St,
        DHt,
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
        num_cores_per_batch};

    std::map<string, string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["LOG2_MUL_BCAST_GRANULARITY"] = std::to_string(log2_mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);

    // Compute
    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args_common,
            .defines = defines});

    // Reader
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_common, defines));

    // Writer
    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_common, defines));

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t out_addr = out0_buffer->address();

    uint32_t PSt_i;
    uint32_t k_num_chunks_i;
    uint32_t k_chunk_start_i;
    uint32_t k_chunk_end_i;

    // Sequence length assignment
    auto [all_PSt, all_num_chunks, all_chunk_assignments] =
        get_runtime_args(cur_pos, B, num_cores_per_batch, k_chunk_size);

    tt::log_debug("num_cores_per_batch: {}", num_cores_per_batch);
    tt::log_debug("cur_pos: {}", cur_pos);
    tt::log_debug("all_PSt: {}", all_PSt);
    tt::log_debug("all_num_chunks: {}", all_num_chunks);
    tt::log_debug("all_chunk_assignments: {}", all_chunk_assignments);

    int batch_id = -1;
    // Set reader rt args
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        uint32_t cur_pos_i = cur_pos[i / num_cores_per_batch];
        uint32_t worker_id = i % num_cores_per_batch - 1;
        bool do_reduce = (worker_id == -1);

        uint32_t cur_batch = i / num_cores_per_batch;
        if (do_reduce) {
            batch_id++;
            // reduce core and its workers share the same runtime args
            PSt_i = all_PSt[batch_id];
            k_num_chunks_i = all_num_chunks[batch_id];
        }
        k_chunk_start_i = all_chunk_assignments[2 * i];
        k_chunk_end_i = all_chunk_assignments[2 * i + 1];

        // reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            q_addr, k_addr, v_addr, PSt_i, k_num_chunks_i, k_chunk_start_i, k_chunk_end_i, cur_batch, !do_reduce};
        reader_rt_args.insert(reader_rt_args.end(), reduce_core_physical_xs.begin(), reduce_core_physical_xs.end());
        reader_rt_args.insert(reader_rt_args.end(), reduce_core_physical_ys.begin(), reduce_core_physical_ys.end());

        // writer runtime args
        std::vector<uint32_t> writer_rt_args = {
            out_addr,
            cur_pos_i,
            PSt_i,
            k_num_chunks_i,
            k_chunk_start_i,
            k_chunk_end_i,
            cur_batch,
            worker_id,
            !do_reduce};
        writer_rt_args.insert(writer_rt_args.end(), reduce_core_physical_xs.begin(), reduce_core_physical_xs.end());
        writer_rt_args.insert(writer_rt_args.end(), reduce_core_physical_ys.begin(), reduce_core_physical_ys.end());

        SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);
        SetRuntimeArgs(program, writer_kernels_id, core, writer_rt_args);
        SetRuntimeArgs(program, compute_kernels_id, core, {k_num_chunks_i, k_chunk_start_i, k_chunk_end_i, do_reduce});
    }
    if (num_active_cores < num_cores_available) {
        // Set the rest of the cores to idle
        for (uint32_t i = 0; i < core_group_idle.size(); ++i) {
            CoreCoord core = core_group_idle[i];
            // reader runtime args
            std::vector<uint32_t> reader_rt_args = {0, 0, 0, 0, 0, 0, 0, 0, 0};

            // writer runtime args
            std::vector<uint32_t> writer_rt_args = {0, 0, 0, 0, 0, 0, 0, 0, 0};

            SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);
            SetRuntimeArgs(program, writer_kernels_id, core, writer_rt_args);
            SetRuntimeArgs(program, compute_kernels_id, core, {0, 0, 0, 0});
        }
    }

    auto override_runtime_arguments_callback =
        [num_active_cores,
         core_group,
         reader_kernels_id,
         writer_kernels_id,
         compute_kernels_id,
         num_cores_per_batch,
         is_output_sharded,
         cb_out4_id,
         B,
         k_chunk_size](
            const void* operation,
            std::shared_ptr<Program> program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {

            const auto cur_pos =
                static_cast<const ttnn::operations::transformer::ScaledDotProductAttentionDecode*>(operation)->cur_pos;

            auto q_buffer = input_tensors.at(0).buffer();
            auto k_buffer = input_tensors.at(1).buffer();
            auto v_buffer = input_tensors.at(2).buffer();

            auto out0_buffer = output_tensors.at(0).buffer();
            uint32_t q_addr = q_buffer->address();
            uint32_t k_addr = k_buffer->address();
            uint32_t v_addr = v_buffer->address();
            uint32_t out_addr = out0_buffer->address();

            uint32_t PSt_i;
            uint32_t k_num_chunks_i;
            uint32_t k_chunk_start_i;
            uint32_t k_chunk_end_i;

            // Sequence length assignment
            auto [all_PSt, all_num_chunks, all_chunk_assignments] =
                get_runtime_args(cur_pos, B, num_cores_per_batch, k_chunk_size);

            tt::log_debug("num_cores_per_batch: {}", num_cores_per_batch);
            tt::log_debug("cur_pos: {}", cur_pos);
            tt::log_debug("all_PSt: {}", all_PSt);
            tt::log_debug("all_num_chunks: {}", all_num_chunks);
            tt::log_debug("all_chunk_assignments: {}", all_chunk_assignments);

            int batch_id = -1;

            auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);
            auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernels_id);

            // Set rt args
            for (uint32_t i = 0; i < num_active_cores; ++i) {
                CoreCoord core = core_group[i];
                uint32_t cur_pos_i = cur_pos[i / num_cores_per_batch];
                uint32_t worker_id = i % num_cores_per_batch - 1;
                bool do_reduce = (worker_id == -1);

                uint32_t cur_batch = i / num_cores_per_batch;
                if (do_reduce) {
                    batch_id++;
                    // reduce core and its workers share the same runtime args
                    PSt_i = all_PSt[batch_id];
                    k_num_chunks_i = all_num_chunks[batch_id];
                }
                k_chunk_start_i = all_chunk_assignments[2 * i];
                k_chunk_end_i = all_chunk_assignments[2 * i + 1];

                auto& reader_args = reader_args_by_core[core.x][core.y];
                auto& writer_args = writer_args_by_core[core.x][core.y];
                auto& compute_args = compute_args_by_core[core.x][core.y];

                // reader runtime args
                reader_args[0] = q_addr;
                reader_args[1] = k_addr;
                reader_args[2] = v_addr;
                reader_args[3] = PSt_i;
                reader_args[4] = k_num_chunks_i;
                reader_args[5] = k_chunk_start_i;
                reader_args[6] = k_chunk_end_i;
                reader_args[7] = cur_batch;
                reader_args[8] = !do_reduce;

                // writer runtime args
                writer_args[0] = out_addr;
                writer_args[1] = cur_pos_i;
                writer_args[2] = PSt_i;
                writer_args[3] = k_num_chunks_i;
                writer_args[4] = k_chunk_start_i;
                writer_args[5] = k_chunk_end_i;
                writer_args[6] = cur_batch;
                writer_args[7] = worker_id;
                writer_args[8] = !do_reduce;

                // compute runtime args
                compute_args[0] = k_num_chunks_i;
                compute_args[1] = k_chunk_start_i;
                compute_args[2] = k_chunk_end_i;
                compute_args[3] = do_reduce;
            }

            if (is_output_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_out4_id, *out0_buffer);
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::transformer::detail
