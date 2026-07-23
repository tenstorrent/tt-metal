# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor and inline kernels for fused non-causal flash attention."""

import ttnn

from .config import FlashAttentionProgramConfig, resolve_block_tiles, resolve_output_subblock


TILE = 32

# Streamed operands.
CB_Q = 0
CB_K = 1
CB_V = 2
CB_SCALER = 3
CB_Q_SCALED = 4

# Online-softmax scratch/state. Persistent max, denominator, and output state is
# BF16; only the final reciprocal remains an FP32 intermediate.
CB_SCORES = 5
CB_PROBS = 6
CB_BLOCK_M = 7
CB_M0 = 8
CB_M1 = 9
CB_ALPHA = 10
CB_L0 = 12
CB_L1 = 13
CB_OUT = 16
CB_O0 = 18
CB_O1 = 19
CB_INV_L = 21

_NOC = {"noc0": ttnn.NOC.NOC_0, "noc1": ttnn.NOC.NOC_1}

_COMPUTE_PROFILE_PHASES = {
    "FA_Q_SCALE",
    "FA_QK_MATMUL",
    "FA_BLOCK_MAX",
    "FA_ONLINE_RESCALE",
    "FA_PROBS_EXP",
    "FA_BLOCK_SUM",
    "FA_PV_MATMUL",
    "FA_STATE_O_UPDATE",
    "FA_FINAL_NORMALIZE",
}


_DIRECT_READER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

#ifdef FA_PROFILE_DIRECT_Q_READ
#define FA_DIRECT_Q_READ_SCOPE() MaybeDeviceZoneScope("FA_DIRECT_Q_READ")
#else
#define FA_DIRECT_Q_READ_SCOPE()
#endif
#ifdef FA_PROFILE_DIRECT_KV_READ
#define FA_DIRECT_KV_READ_SCOPE() MaybeDeviceZoneScope("FA_DIRECT_KV_READ")
#else
#define FA_DIRECT_KV_READ_SCOPE()
#endif

void kernel_main() {
    constexpr uint32_t q_block = get_compile_time_arg_val(0);
    constexpr uint32_t k_block = get_compile_time_arg_val(1);
    constexpr uint32_t q_seq_t = get_compile_time_arg_val(2);
    constexpr uint32_t kv_seq_t = get_compile_time_arg_val(3);
    constexpr uint32_t d_t = get_compile_time_arg_val(4);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t barrier_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t q_args_at = get_compile_time_arg_val(7);
    constexpr uint32_t k_args_at = get_compile_time_arg_val(8);
    constexpr uint32_t v_args_at = get_compile_time_arg_val(9);
    constexpr auto q_args = TensorAccessorArgs<q_args_at>();
    constexpr auto k_args = TensorAccessorArgs<k_args_at>();
    constexpr auto v_args = TensorAccessorArgs<v_args_at>();
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_scaler = 3;

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_job = get_arg_val<uint32_t>(3);
    const uint32_t num_jobs = get_arg_val<uint32_t>(4);
    const uint32_t tile_bytes = get_tile_size(cb_q);
    const auto q = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v = TensorAccessor(v_args, v_addr, tile_bytes);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (uint32_t local = 0; local < num_jobs; ++local) {
        const uint32_t job = start_job + local;
        const uint32_t q_blocks_per_head = q_seq_t / q_block;
        const uint32_t bh = job / q_blocks_per_head;
        const uint32_t qb = job - bh * q_blocks_per_head;

        {
            FA_DIRECT_Q_READ_SCOPE();
            cb_reserve_back(cb_q, q_block * d_t);
            const uint32_t q_dst = get_write_ptr(cb_q);
            const uint32_t q_base = bh * q_seq_t * d_t + qb * q_block * d_t;
            uint32_t issued = 0;
            for (uint32_t i = 0; i < q_block * d_t; ++i) {
                noc_async_read_tile(q_base + i, q, q_dst + i * tile_bytes);
                if (++issued == barrier_tiles) { noc_async_read_barrier(); issued = 0; }
            }
            if (issued != 0) { noc_async_read_barrier(); }
            cb_push_back(cb_q, q_block * d_t);
        }

        const uint32_t kv_head_base = bh * kv_seq_t * d_t;
        for (uint32_t block = 0; block < num_kv_blocks; ++block) {
            {
                FA_DIRECT_KV_READ_SCOPE();
                cb_reserve_back(cb_k, d_t * k_block);
                cb_reserve_back(cb_v, k_block * d_t);
                const uint32_t k_dst = get_write_ptr(cb_k);
                const uint32_t v_dst = get_write_ptr(cb_v);
                const uint32_t key0 = block * k_block;
                uint32_t issued = 0;

                // K is emitted as a [D_t, K_t] tile grid. matmul_block(transpose=true)
                // handles the within-tile transpose, so no transpose compute pass is needed.
                for (uint32_t dt = 0; dt < d_t; ++dt) {
                    for (uint32_t kt = 0; kt < k_block; ++kt) {
                        const uint32_t src = kv_head_base + (key0 + kt) * d_t + dt;
                        const uint32_t dst_i = dt * k_block + kt;
                        noc_async_read_tile(src, k, k_dst + dst_i * tile_bytes);
                        if (++issued == barrier_tiles) { noc_async_read_barrier(); issued = 0; }
                    }
                }
                for (uint32_t kt = 0; kt < k_block; ++kt) {
                    for (uint32_t dt = 0; dt < d_t; ++dt) {
                        const uint32_t src = kv_head_base + (key0 + kt) * d_t + dt;
                        const uint32_t dst_i = kt * d_t + dt;
                        noc_async_read_tile(src, v, v_dst + dst_i * tile_bytes);
                        if (++issued == barrier_tiles) { noc_async_read_barrier(); issued = 0; }
                    }
                }
                if (issued != 0) { noc_async_read_barrier(); }
                cb_push_back(cb_k, d_t * k_block);
                cb_push_back(cb_v, k_block * d_t);
            }
        }
    }
}
"""


_MCAST_SENDER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

#ifdef FA_PROFILE_SENDER_Q_READ
#define FA_SENDER_Q_READ_SCOPE() MaybeDeviceZoneScope("FA_SENDER_Q_READ")
#else
#define FA_SENDER_Q_READ_SCOPE()
#endif
#ifdef FA_PROFILE_SENDER_KV_RESERVE
#define FA_SENDER_KV_RESERVE_SCOPE() MaybeDeviceZoneScope("FA_SENDER_KV_RESERVE")
#else
#define FA_SENDER_KV_RESERVE_SCOPE()
#endif
#ifdef FA_PROFILE_SENDER_KV_DRAM
#define FA_SENDER_KV_DRAM_SCOPE() MaybeDeviceZoneScope("FA_SENDER_KV_DRAM")
#else
#define FA_SENDER_KV_DRAM_SCOPE()
#endif
#ifdef FA_PROFILE_SENDER_KV_MCAST
#define FA_SENDER_KV_MCAST_SCOPE() MaybeDeviceZoneScope("FA_SENDER_KV_MCAST")
#else
#define FA_SENDER_KV_MCAST_SCOPE()
#endif

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t s = mc.next_compile_time_args_offset();
    constexpr uint32_t q_block = get_compile_time_arg_val(s + 0);
    constexpr uint32_t k_block = get_compile_time_arg_val(s + 1);
    constexpr uint32_t q_seq_t = get_compile_time_arg_val(s + 2);
    constexpr uint32_t kv_seq_t = get_compile_time_arg_val(s + 3);
    constexpr uint32_t d_t = get_compile_time_arg_val(s + 4);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(s + 5);
    constexpr uint32_t q_blocks_per_head = get_compile_time_arg_val(s + 6);
    constexpr uint32_t group_width = get_compile_time_arg_val(s + 7);
    constexpr uint32_t barrier_tiles = get_compile_time_arg_val(s + 8);
    constexpr uint32_t q_args_at = get_compile_time_arg_val(s + 9);
    constexpr uint32_t k_args_at = get_compile_time_arg_val(s + 10);
    constexpr uint32_t v_args_at = get_compile_time_arg_val(s + 11);
    constexpr auto q_args = TensorAccessorArgs<q_args_at>();
    constexpr auto k_args = TensorAccessorArgs<k_args_at>();
    constexpr auto v_args = TensorAccessorArgs<v_args_at>();
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_scaler = 3;
    constexpr uint32_t rounds = (q_blocks_per_head + group_width - 1) / group_width;
    constexpr uint32_t r = mc.next_runtime_args_offset();

    const uint32_t q_addr = get_arg_val<uint32_t>(r + 0);
    const uint32_t k_addr = get_arg_val<uint32_t>(r + 1);
    const uint32_t v_addr = get_arg_val<uint32_t>(r + 2);
    const uint32_t bh = get_arg_val<uint32_t>(r + 3);
    const uint32_t lane = get_arg_val<uint32_t>(r + 4);
    const uint32_t tile_bytes = get_tile_size(cb_q);
    const auto q = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v = TensorAccessor(v_args, v_addr, tile_bytes);

    Noc noc;
    CircularBuffer k_buf(cb_k), v_buf(cb_v);
    auto pipe = mc.sender(noc);

    calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (uint32_t round = 0; round < rounds; ++round) {
        const uint32_t qb = round * group_width + lane;
        if (qb < q_blocks_per_head) {
            FA_SENDER_Q_READ_SCOPE();
            cb_reserve_back(cb_q, q_block * d_t);
            const uint32_t q_dst = get_write_ptr(cb_q);
            const uint32_t q_base = bh * q_seq_t * d_t + qb * q_block * d_t;
            uint32_t issued = 0;
            for (uint32_t i = 0; i < q_block * d_t; ++i) {
                noc_async_read_tile(q_base + i, q, q_dst + i * tile_bytes);
                if (++issued == barrier_tiles) { noc_async_read_barrier(); issued = 0; }
            }
            if (issued != 0) { noc_async_read_barrier(); }
            cb_push_back(cb_q, q_block * d_t);
        }

        const uint32_t kv_head_base = bh * kv_seq_t * d_t;
        for (uint32_t block = 0; block < num_kv_blocks; ++block) {
            uint32_t k_dst;
            uint32_t v_dst;
            {
                FA_SENDER_KV_RESERVE_SCOPE();
                k_buf.reserve_back(d_t * k_block);
                v_buf.reserve_back(k_block * d_t);
                k_dst = k_buf.get_write_ptr();
                v_dst = v_buf.get_write_ptr();
            }
            const uint32_t key0 = block * k_block;
            {
                FA_SENDER_KV_DRAM_SCOPE();
                uint32_t issued = 0;
                for (uint32_t dt = 0; dt < d_t; ++dt) {
                    for (uint32_t kt = 0; kt < k_block; ++kt) {
                        const uint32_t src = kv_head_base + (key0 + kt) * d_t + dt;
                        noc_async_read_tile(src, k, k_dst + (dt * k_block + kt) * tile_bytes);
                        if (++issued == barrier_tiles) { noc_async_read_barrier(); issued = 0; }
                    }
                }
                for (uint32_t kt = 0; kt < k_block; ++kt) {
                    for (uint32_t dt = 0; dt < d_t; ++dt) {
                        const uint32_t src = kv_head_base + (key0 + kt) * d_t + dt;
                        noc_async_read_tile(src, v, v_dst + (kt * d_t + dt) * tile_bytes);
                        if (++issued == barrier_tiles) { noc_async_read_barrier(); issued = 0; }
                    }
                }
                if (issued != 0) { noc_async_read_barrier(); }
            }
            {
                FA_SENDER_KV_MCAST_SCOPE();
                pipe.send(k_dst, k_dst, d_t * k_block * tile_bytes);
                k_buf.push_back(d_t * k_block);
                pipe.send(v_dst, v_dst, k_block * d_t * tile_bytes);
                v_buf.push_back(k_block * d_t);
            }
        }
    }
}
"""


_MCAST_RECEIVER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

#ifdef FA_PROFILE_RECEIVER_Q_READ
#define FA_RECEIVER_Q_READ_SCOPE() MaybeDeviceZoneScope("FA_RECEIVER_Q_READ")
#else
#define FA_RECEIVER_Q_READ_SCOPE()
#endif
#ifdef FA_PROFILE_RECEIVER_KV_MCAST
#define FA_RECEIVER_KV_MCAST_SCOPE() MaybeDeviceZoneScope("FA_RECEIVER_KV_MCAST")
#else
#define FA_RECEIVER_KV_MCAST_SCOPE()
#endif

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t s = mc.next_compile_time_args_offset();
    constexpr uint32_t q_block = get_compile_time_arg_val(s + 0);
    constexpr uint32_t k_block = get_compile_time_arg_val(s + 1);
    constexpr uint32_t q_seq_t = get_compile_time_arg_val(s + 2);
    constexpr uint32_t d_t = get_compile_time_arg_val(s + 4);
    constexpr uint32_t num_kv_blocks = get_compile_time_arg_val(s + 5);
    constexpr uint32_t q_blocks_per_head = get_compile_time_arg_val(s + 6);
    constexpr uint32_t group_width = get_compile_time_arg_val(s + 7);
    constexpr uint32_t barrier_tiles = get_compile_time_arg_val(s + 8);
    constexpr uint32_t q_args_at = get_compile_time_arg_val(s + 9);
    constexpr auto q_args = TensorAccessorArgs<q_args_at>();
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_scaler = 3;
    constexpr uint32_t rounds = (q_blocks_per_head + group_width - 1) / group_width;
    constexpr uint32_t r = mc.next_runtime_args_offset();

    const uint32_t q_addr = get_arg_val<uint32_t>(r + 0);
    const uint32_t bh = get_arg_val<uint32_t>(r + 3);
    const uint32_t lane = get_arg_val<uint32_t>(r + 4);
    const uint32_t tile_bytes = get_tile_size(cb_q);
    const auto q = TensorAccessor(q_args, q_addr, tile_bytes);

    Noc noc;
    CircularBuffer k_buf(cb_k), v_buf(cb_v);
    auto pipe = mc.receiver(noc);

    calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    for (uint32_t round = 0; round < rounds; ++round) {
        const uint32_t qb = round * group_width + lane;
        if (qb < q_blocks_per_head) {
            FA_RECEIVER_Q_READ_SCOPE();
            cb_reserve_back(cb_q, q_block * d_t);
            const uint32_t q_dst = get_write_ptr(cb_q);
            const uint32_t q_base = bh * q_seq_t * d_t + qb * q_block * d_t;
            uint32_t issued = 0;
            for (uint32_t i = 0; i < q_block * d_t; ++i) {
                noc_async_read_tile(q_base + i, q, q_dst + i * tile_bytes);
                if (++issued == barrier_tiles) { noc_async_read_barrier(); issued = 0; }
            }
            if (issued != 0) { noc_async_read_barrier(); }
            cb_push_back(cb_q, q_block * d_t);
        }

        for (uint32_t block = 0; block < num_kv_blocks; ++block) {
            FA_RECEIVER_KV_MCAST_SCOPE();
            k_buf.reserve_back(d_t * k_block);
            pipe.receive();
            k_buf.push_back(d_t * k_block);
            v_buf.reserve_back(k_block * d_t);
            pipe.receive();
            v_buf.push_back(k_block * d_t);
        }
    }
}
"""


_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace compute_kernel_lib;
using ckernel::PoolType;
using ckernel::ReduceDim;

#ifdef FA_PROFILE_Q_SCALE
#define FA_Q_SCALE_SCOPE() MaybeDeviceZoneScope("FA_Q_SCALE")
#else
#define FA_Q_SCALE_SCOPE()
#endif
#ifdef FA_PROFILE_QK_MATMUL
#define FA_QK_MATMUL_SCOPE() MaybeDeviceZoneScope("FA_QK_MATMUL")
#else
#define FA_QK_MATMUL_SCOPE()
#endif
#ifdef FA_PROFILE_BLOCK_MAX
#define FA_BLOCK_MAX_SCOPE() MaybeDeviceZoneScope("FA_BLOCK_MAX")
#else
#define FA_BLOCK_MAX_SCOPE()
#endif
#ifdef FA_PROFILE_ONLINE_RESCALE
#define FA_ONLINE_RESCALE_SCOPE() MaybeDeviceZoneScope("FA_ONLINE_RESCALE")
#else
#define FA_ONLINE_RESCALE_SCOPE()
#endif
#ifdef FA_PROFILE_PROBS_EXP
#define FA_PROBS_EXP_SCOPE() MaybeDeviceZoneScope("FA_PROBS_EXP")
#else
#define FA_PROBS_EXP_SCOPE()
#endif
#ifdef FA_PROFILE_BLOCK_SUM
#define FA_BLOCK_SUM_SCOPE() MaybeDeviceZoneScope("FA_BLOCK_SUM")
#else
#define FA_BLOCK_SUM_SCOPE()
#endif
#ifdef FA_PROFILE_PV_MATMUL
#define FA_PV_MATMUL_SCOPE() MaybeDeviceZoneScope("FA_PV_MATMUL")
#else
#define FA_PV_MATMUL_SCOPE()
#endif
#ifdef FA_PROFILE_STATE_O_UPDATE
#define FA_STATE_O_UPDATE_SCOPE() MaybeDeviceZoneScope("FA_STATE_O_UPDATE")
#else
#define FA_STATE_O_UPDATE_SCOPE()
#endif
#ifdef FA_PROFILE_FINAL_NORMALIZE
#define FA_FINAL_NORMALIZE_SCOPE() MaybeDeviceZoneScope("FA_FINAL_NORMALIZE")
#else
#define FA_FINAL_NORMALIZE_SCOPE()
#endif

constexpr uint32_t QB = get_compile_time_arg_val(0);
constexpr uint32_t KB = get_compile_time_arg_val(1);
constexpr uint32_t DT = get_compile_time_arg_val(2);
constexpr uint32_t NUM_KV_BLOCKS = get_compile_time_arg_val(3);
constexpr uint32_t Q_BLOCKS_PER_HEAD = get_compile_time_arg_val(4);
constexpr bool MCAST_SCHEDULE = get_compile_time_arg_val(5) != 0;
constexpr uint32_t GROUP_WIDTH = get_compile_time_arg_val(6);
constexpr uint32_t SOFTMAX_BLOCK = get_compile_time_arg_val(7);
constexpr uint32_t QK_SB_H = get_compile_time_arg_val(8);
constexpr uint32_t QK_SB_W = get_compile_time_arg_val(9);
constexpr uint32_t PV_SB_H = get_compile_time_arg_val(10);
constexpr uint32_t PV_SB_W = get_compile_time_arg_val(11);
constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(12);
constexpr bool PROBS_APPROX_EXP = get_compile_time_arg_val(13) != 0;
constexpr bool PROBS_UNCLAMPED_EXP = get_compile_time_arg_val(14) != 0;
constexpr bool RESCALE_APPROX_EXP = get_compile_time_arg_val(15) != 0;
constexpr bool RESCALE_UNCLAMPED_EXP = get_compile_time_arg_val(16) != 0;
constexpr auto PROBS_EXP_MODE = PROBS_APPROX_EXP ? Approx::Fast : Approx::Exact;
constexpr auto PROBS_EXP_INPUT_CLAMPING =
    PROBS_UNCLAMPED_EXP ? ExpInputClamping::None : ExpInputClamping::ClampToNegative;
constexpr auto PROBS_EXP_PACK_RELU = PROBS_UNCLAMPED_EXP ? PackRelu::Zero : PackRelu::None;
constexpr auto RESCALE_EXP_MODE = RESCALE_APPROX_EXP ? Approx::Fast : Approx::Exact;
constexpr auto RESCALE_EXP_INPUT_CLAMPING =
    RESCALE_UNCLAMPED_EXP ? ExpInputClamping::None : ExpInputClamping::ClampToNegative;
constexpr auto RESCALE_EXP_PACK_RELU = RESCALE_UNCLAMPED_EXP ? PackRelu::Zero : PackRelu::None;

constexpr uint32_t Q = 0, K = 1, V = 2, SCALER = 3, Q_SCALED = 4;
constexpr uint32_t SCORES = 5, PROBS = 6, BLOCK_M = 7, M0 = 8, M1 = 9, ALPHA = 10;
constexpr uint32_t L0 = 12, L1 = 13, OUT = 16;
constexpr uint32_t O0 = 18, O1 = 19, INV_L = 21;

template <uint32_t InCb, uint32_t OutCb>
ALWI void block_row_max() {
    if constexpr (KB <= 16) {
        cb_wait_front(InCb, QB * KB);
        cb_reserve_back(OutCb, QB);
        reduce_block_max_row_init<KB>(OutCb);
        for (uint32_t row0 = 0; row0 < QB; row0 += SOFTMAX_BLOCK) {
            const uint32_t rows_this_batch = (row0 + SOFTMAX_BLOCK <= QB) ? SOFTMAX_BLOCK : (QB - row0);
            tile_regs_acquire();
            for (uint32_t local_row = 0; local_row < rows_this_batch; ++local_row) {
                reduce_block_max_row<KB>(InCb, SCALER, (row0 + local_row) * KB, local_row);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t local_row = 0; local_row < rows_this_batch; ++local_row) {
                pack_tile(local_row, OutCb, row0 + local_row);
            }
            tile_regs_release();
        }
        reduce_block_max_row_uninit(InCb);
        cb_push_back(OutCb, QB);
    } else {
        reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            InCb,
            SCALER,
            OutCb,
            ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(QB, KB));
    }
}

template <uint32_t InCb, uint32_t MOld, uint32_t MNew>
ALWI void block_row_max_online_rescale() {
    static_assert(KB <= 16, "fused block max/rescale requires the direct block-max path");
    static_assert(2 * QB <= DEST_AUTO_LIMIT, "fused block max/rescale requires two DEST tiles per query row");

    cb_wait_front(InCb, QB * KB);
    cb_wait_front(MOld, QB);
    cb_reserve_back(MNew, QB);
    cb_reserve_back(ALPHA, QB);
    tile_regs_acquire();

    // Reduce every score row into D[0:QB]. Keeping the complete query block in
    // one DEST window lets the online update consume the maxima without the
    // former BLOCK_M pack/unpack round trip.
    {
        FA_BLOCK_MAX_SCOPE();
        reduce_block_max_row_init<KB>(MNew);
        for (uint32_t row = 0; row < QB; ++row) {
            reduce_block_max_row<KB>(InCb, SCALER, row * KB, row);
        }
        reduce_block_max_row_uninit(InCb);
    }

    // D[row] starts as block_max and D[QB + row] starts as m_old:
    //   D[row]      = m_new = max(block_max, m_old)
    //   D[QB + row] = alpha = exp(m_old - m_new)
    // Only the reduced column is semantically live; the resulting tiles are
    // consumed as column-broadcast operands by the following phases.
    {
        FA_ONLINE_RESCALE_SCOPE();
        reconfig_data_format_srca(InCb, MOld);
        copy_tile_init(MOld);
        for (uint32_t row = 0; row < QB; ++row) {
            copy_tile(MOld, row, QB + row);
        }

        binary_max_tile_init();
        for (uint32_t row = 0; row < QB; ++row) {
            binary_max_tile(row, QB + row, row);
        }

        sub_binary_tile_init();
        for (uint32_t row = 0; row < QB; ++row) {
            sub_binary_tile(QB + row, row, QB + row);
        }

        Exp<RESCALE_EXP_MODE, Approx::Fast, Dst::D0, RESCALE_EXP_INPUT_CLAMPING>::init();
        for (uint32_t row = 0; row < QB; ++row) {
            Exp<RESCALE_EXP_MODE, Approx::Fast, Dst::D0, RESCALE_EXP_INPUT_CLAMPING>::exec_impl(QB + row);
        }

        tile_regs_commit();
        tile_regs_wait();

        pack_reconfig_data_format(MNew);
        for (uint32_t row = 0; row < QB; ++row) {
            pack_tile(row, MNew);
        }
        cb_push_back(MNew, QB);

        pack_reconfig_data_format(MNew, ALPHA);
        if constexpr (RESCALE_EXP_PACK_RELU == PackRelu::Zero) {
            ckernel::pack_relu_config(ckernel::ReluConfig::zero());
        }
        for (uint32_t row = 0; row < QB; ++row) {
            pack_tile(QB + row, ALPHA);
        }
        if constexpr (RESCALE_EXP_PACK_RELU == PackRelu::Zero) {
            ckernel::pack_relu_config(ckernel::ReluConfig::none());
        }
        cb_push_back(ALPHA, QB);
        tile_regs_release();
    }

    cb_pop_front(MOld, QB);
}

template <uint32_t OutCb>
ALWI void qk_matmul() {
    FA_QK_MATMUL_SCOPE();
    CircularBuffer q(Q_SCALED), k(K), out(OutCb);
    matmul_block<
        /*transpose in1 tiles=*/true,
        /*packer_l1_acc=*/false,
        LastBlockTarget::Out,
        OutputCBLayout::TileRowMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndRetainOnLastBlock,
        InputPolicy::WaitAndPopPerKBlock>(
        q,
        k,
        out,
        out,
        MatmulBlockShape::of(QB / QK_SB_H, KB / QK_SB_W, QK_SB_H, QK_SB_W, DT, 1));
}

template <uint32_t OutCb>
ALWI void pv_matmul() {
    FA_PV_MATMUL_SCOPE();
    CircularBuffer p(PROBS), v(V), out(OutCb);
    matmul_block<
        /*transpose=*/false,
        /*packer_l1_acc=*/false,
        LastBlockTarget::Out,
        OutputCBLayout::TileRowMajor,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::WaitAndPopPerKBlock>(
        p,
        v,
        out,
        out,
        MatmulBlockShape::of(QB / PV_SB_H, DT / PV_SB_W, PV_SB_H, PV_SB_W, KB, 1));
}

template <uint32_t OOld, uint32_t ONew>
ALWI void pv_matmul_update() {
    // Seed the caller-owned output region with alpha*O_old, then accumulate P@V
    // into that region in the matmul packer. This removes the block-O round trip.
    {
        FA_STATE_O_UPDATE_SCOPE();
        cb_reserve_back(ONew, QB * DT);
        eltwise_chain(
            EltwiseShape::grid(QB, DT, SOFTMAX_BLOCK),
            BinaryFpu<
                OOld,
                ALPHA,
                BinaryFpuOp::Mul,
                BroadcastDim::Col,
                InputLifecycle::Bulk,
                InputLifecycle::Bulk,
                BinaryDataFormatReconfig::Input,
                Dst::D0,
                OperandKind::Block,
                OperandKind::Col>{},
            PackTile<ONew, OutputLifecycle::CallerManaged, PackTileReconfig::Output>{});
    }

    {
        FA_PV_MATMUL_SCOPE();
        CircularBuffer p(PROBS), v(V), out(ONew);
        matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/true,
            LastBlockTarget::Interm,
            OutputCBLayout::TileRowMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndPopPerKBlock,
            InputPolicy::WaitAndPopPerKBlock,
            NoPostCompute,
            NoPreKBlock,
            NoPostKBlock,
            /*untilize_block_ct_dim=*/0,
            NoKBlockInnerDimFn,
            NoIn0Source,
            NoIn1BaseOffset,
            /*caller_owns_pack_target=*/true,
            NoneActivation,
            matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT,
            /*accumulate_first_k_block=*/true>(
            p,
            v,
            out,
            out,
            MatmulBlockShape::of(QB / PV_SB_H, DT / PV_SB_W, PV_SB_H, PV_SB_W, KB, 1));
        cb_push_back(ONew, QB * DT);
    }
}

template <uint32_t MaxCb>
ALWI void scores_to_probabilities() {
    FA_PROBS_EXP_SCOPE();
    if constexpr (KB % SOFTMAX_BLOCK == 0) {
        cb_wait_front(SCORES, QB * KB);
        cb_wait_front(MaxCb, QB);
        cb_reserve_back(PROBS, QB * KB);
        sub_bcast_cols_init_short_custom(SCORES, MaxCb, SOFTMAX_BLOCK);
        Exp<PROBS_EXP_MODE, Approx::Fast, Dst::D0, PROBS_EXP_INPUT_CLAMPING>::init();
        pack_reconfig_data_format(PROBS);
        if constexpr (PROBS_EXP_PACK_RELU == PackRelu::Zero) {
            ckernel::pack_relu_config(ckernel::ReluConfig::zero());
        }
        for (uint32_t row = 0; row < QB; ++row) {
            for (uint32_t col = 0; col < KB; col += SOFTMAX_BLOCK) {
                tile_regs_acquire();
                sub_tiles_bcast_cols_custom(SCORES, MaxCb, row * KB + col, row, 0, SOFTMAX_BLOCK);
                for (uint32_t tile = 0; tile < SOFTMAX_BLOCK; ++tile) {
                    Exp<PROBS_EXP_MODE, Approx::Fast, Dst::D0, PROBS_EXP_INPUT_CLAMPING>::exec_impl(tile);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t tile = 0; tile < SOFTMAX_BLOCK; ++tile) {
                    pack_tile(tile, PROBS);
                }
                tile_regs_release();
            }
        }
        if constexpr (PROBS_EXP_PACK_RELU == PackRelu::Zero) {
            ckernel::pack_relu_config(ckernel::ReluConfig::none());
        }
        cb_push_back(PROBS, QB * KB);
        cb_pop_front(SCORES, QB * KB);
    } else {
    eltwise_chain(
        EltwiseShape::grid(QB, KB, SOFTMAX_BLOCK),
        BinaryFpu<
            SCORES,
            MaxCb,
            BinaryFpuOp::Sub,
            BroadcastDim::Col,
            InputLifecycle::Bulk,
            InputLifecycle::HeldBulk,
            BinaryDataFormatReconfig::Input,
            Dst::D0,
            OperandKind::Block,
            OperandKind::Col>{},
        Exp<PROBS_EXP_MODE, Approx::Fast, Dst::D0, PROBS_EXP_INPUT_CLAMPING>{},
        PackTile<
            PROBS,
            OutputLifecycle::Bulk,
            PackTileReconfig::Output,
            Dst::D0,
            TileOffset::Unset,
            PackTileL1Accumulation::Disabled,
            PROBS_EXP_PACK_RELU>{});
    }
}

template <uint32_t MaxOut, uint32_t LOut, uint32_t OOut>
ALWI void first_kv_block() {
    qk_matmul<SCORES>();
    {
        FA_BLOCK_MAX_SCOPE();
        block_row_max<SCORES, MaxOut>();
    }
    scores_to_probabilities<MaxOut>();
    {
        FA_BLOCK_SUM_SCOPE();
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            PROBS,
            SCALER,
            LOut,
            ReduceInputPolicy::WaitUpfrontNoPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(QB, KB));
    }
    pv_matmul<OOut>();
}

template <uint32_t MOld, uint32_t LOld, uint32_t OOld, uint32_t MNew, uint32_t LNew, uint32_t ONew>
ALWI void update_kv_block() {
    qk_matmul<SCORES>();
    if constexpr (KB <= 16 && 2 * QB <= DEST_AUTO_LIMIT) {
        block_row_max_online_rescale<SCORES, MOld, MNew>();
    } else {
        {
            FA_BLOCK_MAX_SCOPE();
            block_row_max<SCORES, BLOCK_M>();
        }

        // Fallback for configurations which cannot hold block_max and m_old
        // together in DEST.
        {
            FA_ONLINE_RESCALE_SCOPE();
            eltwise_chain(
                EltwiseShape::tiles(QB, SOFTMAX_BLOCK),
                CopyTile<MOld, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
                CopyTile<BLOCK_M, Dst::D1, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
                BinaryMax<Dst::D0, Dst::D1, Dst::D1>{},
                PackTile<MNew, OutputLifecycle::Bulk, PackTileReconfig::Output, Dst::D1>{},
                SubBinary<>{},
                Exp<RESCALE_EXP_MODE, Approx::Fast, Dst::D0, RESCALE_EXP_INPUT_CLAMPING>{},
                PackTile<
                    ALPHA,
                    OutputLifecycle::Bulk,
                    PackTileReconfig::Output,
                    Dst::D0,
                    TileOffset::Unset,
                    PackTileL1Accumulation::Disabled,
                    RESCALE_EXP_PACK_RELU>{});
        }
    }

    scores_to_probabilities<MNew>();
    {
        FA_BLOCK_SUM_SCOPE();
        cb_wait_front(ALPHA, QB);
        cb_wait_front(LOld, QB);
        uint32_t row = 0;
        reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            PROBS,
            SCALER,
            LNew,
            ReduceInputPolicy::WaitUpfrontNoPop,
            ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceAlgorithm::AccumulateViaAdd>(
            ReduceInputBlockShape::of(QB, KB),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            [&](uint32_t dst_idx) {
                mul_tiles_init(ALPHA, LOld, /*acc_to_dest=*/1, __builtin_LINE());
                mul_tiles(ALPHA, LOld, row, row, dst_idx);
                ++row;
            });
        cb_pop_front(LOld, QB);
    }
    pv_matmul_update<OOld, ONew>();
}

template <uint32_t MFinal, uint32_t LFinal, uint32_t OFinal>
ALWI void finish_attention() {
    FA_FINAL_NORMALIZE_SCOPE();
    unary<Recip<>, LFinal, INV_L, InputLifecycle::Bulk, OutputLifecycle::Bulk,
        CopyTileReconfig::Input, PackTileReconfig::Output, OperandKind::Block>(
        EltwiseShape::tiles(QB, SOFTMAX_BLOCK));
    mul<
        OFinal,
        INV_L,
        OUT,
        BroadcastDim::Col,
        InputLifecycle::Bulk,
        InputLifecycle::Bulk,
        OutputLifecycle::Bulk,
        BinaryDataFormatReconfig::Input,
        PackTileReconfig::Output,
        OperandKind::Block,
        OperandKind::Col>(EltwiseShape::grid(QB, DT, SOFTMAX_BLOCK));
    cb_wait_front(MFinal, QB);
    cb_pop_front(MFinal, QB);
}

ALWI void run_one_query_block() {
    // Pre-scale Q once, then retain it through every QK block. Scaling Q rather
    // than every score tile removes an O(sequence) SFPU pass.
    {
        FA_Q_SCALE_SCOPE();
        eltwise_chain(
            EltwiseShape::grid(QB, DT, SOFTMAX_BLOCK),
            CopyTile<Q, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
            MulUnary<>(SCALE_BITS),
            PackTile<Q_SCALED, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
    }

    first_kv_block<M0, L0, O0>();
    for (uint32_t block = 1; block < NUM_KV_BLOCKS; ++block) {
        if (block & 1u) {
            update_kv_block<M0, L0, O0, M1, L1, O1>();
        } else {
            update_kv_block<M1, L1, O1, M0, L0, O0>();
        }
    }

    if constexpr (NUM_KV_BLOCKS & 1u) {
        finish_attention<M0, L0, O0>();
    } else {
        finish_attention<M1, L1, O1>();
    }
    cb_wait_front(Q_SCALED, QB * DT);
    cb_pop_front(Q_SCALED, QB * DT);
}

ALWI void drain_inactive_round() {
    for (uint32_t block = 0; block < NUM_KV_BLOCKS; ++block) {
        cb_wait_front(K, DT * KB);
        cb_pop_front(K, DT * KB);
        cb_wait_front(V, KB * DT);
        cb_pop_front(V, KB * DT);
    }
}

void kernel_main() {
    // One heavy hardware configure for the whole fused kernel. Every helper below
    // owns only its short per-phase init/reconfig.
    compute_kernel_hw_startup<SrcOrder::Reverse>(Q_SCALED, K, SCORES);
    matmul_block_init(Q_SCALED, K, /*transpose=*/1, QK_SB_W, QK_SB_H, DT);

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t second = get_arg_val<uint32_t>(1);
    if constexpr (MCAST_SCHEDULE) {
        constexpr uint32_t rounds = (Q_BLOCKS_PER_HEAD + GROUP_WIDTH - 1) / GROUP_WIDTH;
        const uint32_t lane = second;
        for (uint32_t round = 0; round < rounds; ++round) {
            const uint32_t qb = round * GROUP_WIDTH + lane;
            if (qb < Q_BLOCKS_PER_HEAD) {
                run_one_query_block();
            } else {
                drain_inactive_round();
            }
        }
    } else {
        const uint32_t num_jobs = second;
        for (uint32_t local = 0; local < num_jobs; ++local) {
            run_one_query_block();
        }
    }
}
"""


_WRITER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef FA_PROFILE_WRITER_WAIT
#define FA_WRITER_WAIT_SCOPE() MaybeDeviceZoneScope("FA_WRITER_WAIT")
#else
#define FA_WRITER_WAIT_SCOPE()
#endif
#ifdef FA_PROFILE_WRITER_DRAM
#define FA_WRITER_DRAM_SCOPE() MaybeDeviceZoneScope("FA_WRITER_DRAM")
#else
#define FA_WRITER_DRAM_SCOPE()
#endif

void kernel_main() {
    constexpr uint32_t q_block = get_compile_time_arg_val(0);
    constexpr uint32_t q_seq_t = get_compile_time_arg_val(1);
    constexpr uint32_t d_t = get_compile_time_arg_val(2);
    constexpr uint32_t q_blocks_per_head = get_compile_time_arg_val(3);
    constexpr bool mcast_schedule = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t group_width = get_compile_time_arg_val(5);
    constexpr uint32_t barrier_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t out_args_at = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<out_args_at>();
    constexpr uint32_t cb_out = 16;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t second = get_arg_val<uint32_t>(2);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    const uint32_t count = mcast_schedule
        ? (q_blocks_per_head + group_width - 1) / group_width
        : second;
    for (uint32_t local = 0; local < count; ++local) {
        uint32_t bh;
        uint32_t qb;
        if constexpr (mcast_schedule) {
            bh = first;
            qb = local * group_width + second;
            if (qb >= q_blocks_per_head) { continue; }
        } else {
            const uint32_t job = first + local;
            bh = job / q_blocks_per_head;
            qb = job - bh * q_blocks_per_head;
        }

        {
            FA_WRITER_WAIT_SCOPE();
            cb_wait_front(cb_out, q_block * d_t);
        }
        const uint32_t src = get_read_ptr(cb_out);
        const uint32_t page0 = bh * q_seq_t * d_t + qb * q_block * d_t;
        {
            FA_WRITER_DRAM_SCOPE();
            uint32_t issued = 0;
            for (uint32_t i = 0; i < q_block * d_t; ++i) {
                noc_async_write_tile(page0 + i, out, src + i * tile_bytes);
                if (++issued == barrier_tiles) { noc_async_write_barrier(); issued = 0; }
            }
            if (issued != 0) { noc_async_write_barrier(); }
            cb_pop_front(cb_out, q_block * d_t);
        }
    }
}
"""


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])


def _normal_cb(cb_id, core_ranges, num_pages, dtype):
    page_size = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def _assign_contiguous(count, workers):
    base, remainder = divmod(count, workers)
    result = []
    start = 0
    for index in range(workers):
        local = base + (index < remainder)
        result.append((start, int(local)))
        start += int(local)
    return result


def _ordered_row_major_cores(device, count):
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(index % grid.x, index // grid.x) for index in range(count)]


def _resolve_direct_schedule(device, total_jobs, config):
    grid = device.compute_with_storage_grid_size()
    available = grid.x * grid.y
    requested = available if config.num_cores is None else config.num_cores
    if requested > available:
        raise ValueError(f"flash_attention: num_cores={requested} exceeds the {grid.x}x{grid.y} worker grid")
    count = min(requested, total_jobs)
    cores = _ordered_row_major_cores(device, count)
    return cores, _assign_contiguous(total_jobs, count)


def _resolve_mcast_group_width(device, batch_heads, q_blocks_per_head, config):
    grid = device.compute_with_storage_grid_size()
    available = grid.x * grid.y
    if batch_heads > available:
        return 1
    core_cap = available if config.num_cores is None else config.num_cores
    if core_cap > available:
        raise ValueError(f"flash_attention: num_cores={core_cap} exceeds the {grid.x}x{grid.y} worker grid")
    width_cap = min(q_blocks_per_head, grid.x, core_cap // batch_heads)
    if config.q_parallel_group_size is not None:
        width_cap = min(width_cap, config.q_parallel_group_size)
    for width in range(width_cap, 0, -1):
        groups_per_row = grid.x // width
        if groups_per_row * grid.y >= batch_heads:
            return width
    return 1


def _make_mcast_groups(device, batch_heads, width, spread_senders=True):
    grid = device.compute_with_storage_grid_size()
    groups_per_row = grid.x // width
    groups = []
    for bh in range(batch_heads):
        row = bh // groups_per_row
        segment = bh % groups_per_row
        x0 = segment * width
        cores = [ttnn.CoreCoord(x0 + lane, row) for lane in range(width)]
        sender_lane = bh % width if spread_senders else 0
        groups.append((bh, cores, cores[sender_lane]))
    return groups


def _dm_config(processor, noc):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=_NOC[noc])


def create_program_descriptor(q, k, v, output, *, scale, program_config: FlashAttentionProgramConfig):
    """Create the fused reader/compute/writer program for validated tensors."""
    q_shape, k_shape = list(q.shape), list(k.shape)
    batch, heads, q_seq, head_dim = q_shape
    kv_seq = k_shape[-2]
    batch_heads = batch * heads
    q_seq_t, kv_seq_t, d_t = q_seq // TILE, kv_seq // TILE, head_dim // TILE
    q_block = resolve_block_tiles(q_seq_t, program_config.query_block_tiles, 4)
    k_block = resolve_block_tiles(kv_seq_t, program_config.key_block_tiles, 16)
    q_blocks_per_head = q_seq_t // q_block
    num_kv_blocks = kv_seq_t // k_block
    total_jobs = batch_heads * q_blocks_per_head
    qk_h, qk_w = resolve_output_subblock(
        q_block,
        k_block,
        program_config.qk_output_subblock,
        program_config.dest_tile_capacity,
        prefer_wide=False,
    )
    pv_h, pv_w = resolve_output_subblock(
        q_block, d_t, program_config.pv_output_subblock, program_config.dest_tile_capacity
    )

    device = q.device()
    group_width = _resolve_mcast_group_width(device, batch_heads, q_blocks_per_head, program_config)
    use_mcast = program_config.use_kv_multicast and group_width > 1
    profile_phase = program_config.profile_phase
    profile_define = [] if profile_phase is None else [(f"FA_PROFILE_{profile_phase.removeprefix('FA_')}", "1")]
    direct_defines = profile_define if profile_phase is not None and profile_phase.startswith("FA_DIRECT_") else []
    sender_defines = profile_define if profile_phase is not None and profile_phase.startswith("FA_SENDER_") else []
    receiver_defines = profile_define if profile_phase is not None and profile_phase.startswith("FA_RECEIVER_") else []
    writer_defines = profile_define if profile_phase is not None and profile_phase.startswith("FA_WRITER_") else []
    compute_defines = profile_define if profile_phase in _COMPUTE_PROFILE_PHASES else []

    q_ct = list(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    k_ct = list(ttnn.TensorAccessorArgs(k).get_compile_time_args())
    v_ct = list(ttnn.TensorAccessorArgs(v).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    q_addr, k_addr, v_addr, out_addr = (
        q.buffer_address(),
        k.buffer_address(),
        v.buffer_address(),
        output.buffer_address(),
    )

    semaphores = []
    kernels = []
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    if use_mcast:
        groups = _make_mcast_groups(device, batch_heads, group_width, program_config.spread_kv_readers)
        cores = [core for _, group, _ in groups for core in group]
        senders = [sender for _, _, sender in groups]
        receivers = [core for _, group, sender in groups for core in group if core != sender]
        all_ranges = _core_range_set(cores)
        sender_rt = ttnn.RuntimeArgs()
        receiver_rt = ttnn.RuntimeArgs()

        mcasts = []
        for bh, group, sender_core in groups:
            ranges = _core_range_set(group)
            mc = ttnn.Mcast2D(
                device,
                ranges,
                sender_core,
                ttnn.McastConfig(handshake=True, base_sem_id=0),
            )
            mcasts.append(mc)
            semaphores.extend(mc.owned_semaphores())
            for lane, core in enumerate(group):
                args = [*mc.runtime_args(core), q_addr, k_addr, v_addr, bh, lane]
                if core == sender_core:
                    sender_rt[core.x][core.y] = args
                else:
                    receiver_rt[core.x][core.y] = args
                compute_rt[core.x][core.y] = [bh, lane]
                writer_rt[core.x][core.y] = [out_addr, bh, lane]

        mc_ct = list(mcasts[0].compile_time_args())
        scalar_count = 12
        q_offset = len(mc_ct) + scalar_count
        k_offset = q_offset + len(q_ct)
        v_offset = k_offset + len(k_ct)
        reader_ct = [
            *mc_ct,
            q_block,
            k_block,
            q_seq_t,
            kv_seq_t,
            d_t,
            num_kv_blocks,
            q_blocks_per_head,
            group_width,
            program_config.read_barrier_tiles,
            q_offset,
            k_offset,
            v_offset,
            *q_ct,
            *k_ct,
            *v_ct,
        ]
        sender = ttnn.KernelDescriptor(
            kernel_source=_MCAST_SENDER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_core_range_set(senders),
            compile_time_args=reader_ct,
            runtime_args=sender_rt,
            defines=sender_defines,
            config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, program_config.reader_noc),
        )
        receiver = ttnn.KernelDescriptor(
            kernel_source=_MCAST_RECEIVER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_core_range_set(receivers),
            compile_time_args=reader_ct,
            runtime_args=receiver_rt,
            defines=receiver_defines,
            config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, program_config.reader_noc),
        )
        kernels.extend([sender, receiver])
    else:
        cores, assignment = _resolve_direct_schedule(device, total_jobs, program_config)
        all_ranges = _core_range_set(cores)
        group_width = 1
        q_offset = 10
        k_offset = q_offset + len(q_ct)
        v_offset = k_offset + len(k_ct)
        reader_ct = [
            q_block,
            k_block,
            q_seq_t,
            kv_seq_t,
            d_t,
            num_kv_blocks,
            program_config.read_barrier_tiles,
            q_offset,
            k_offset,
            v_offset,
            *q_ct,
            *k_ct,
            *v_ct,
        ]
        for core, (start, count) in zip(cores, assignment):
            reader_rt[core.x][core.y] = [q_addr, k_addr, v_addr, start, count]
            compute_rt[core.x][core.y] = [start, count]
            writer_rt[core.x][core.y] = [out_addr, start, count]
        reader = ttnn.KernelDescriptor(
            kernel_source=_DIRECT_READER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=all_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=direct_defines,
            config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, program_config.reader_noc),
        )
        kernels.append(reader)

    probs_exp_mode = program_config.exp_approx_mode
    rescale_exp_mode = program_config.resolved_rescale_exp_approx_mode
    compute_ct = [
        q_block,
        k_block,
        d_t,
        num_kv_blocks,
        q_blocks_per_head,
        int(use_mcast),
        group_width,
        program_config.resolved_softmax_block_tiles,
        qk_h,
        qk_w,
        pv_h,
        pv_w,
        _float_to_u32(scale),
        int(probs_exp_mode == "fast"),
        int(probs_exp_mode in ("fast", "accurate_fast")),
        int(rescale_exp_mode == "fast"),
        int(rescale_exp_mode in ("fast", "accurate_fast")),
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        defines=compute_defines,
        config=ttnn.ComputeConfigDescriptor(
            fp32_dest_acc_en=program_config.fp32_dest_acc_en,
            math_fidelity=program_config.math_fidelity,
        ),
    )
    kernels.append(compute)

    writer_scalar_count = 8
    writer_ct = [
        q_block,
        q_seq_t,
        d_t,
        q_blocks_per_head,
        int(use_mcast),
        group_width,
        program_config.write_barrier_tiles,
        writer_scalar_count,
        *out_ct,
    ]
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        defines=writer_defines,
        config=_dm_config(ttnn.DataMovementProcessor.RISCV_0, program_config.writer_noc),
    )
    kernels.append(writer)

    bf16, fp32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        _normal_cb(CB_Q, all_ranges, q_block * d_t, bf16),
        _normal_cb(CB_K, all_ranges, program_config.kv_buffer_depth * d_t * k_block, bf16),
        _normal_cb(CB_V, all_ranges, program_config.kv_buffer_depth * k_block * d_t, bf16),
        _normal_cb(CB_SCALER, all_ranges, 1, bf16),
        _normal_cb(CB_Q_SCALED, all_ranges, q_block * d_t, bf16),
        _normal_cb(CB_SCORES, all_ranges, q_block * k_block, bf16),
        _normal_cb(CB_PROBS, all_ranges, q_block * k_block, bf16),
        _normal_cb(CB_BLOCK_M, all_ranges, q_block, bf16),
        _normal_cb(CB_M0, all_ranges, q_block, bf16),
        _normal_cb(CB_M1, all_ranges, q_block, bf16),
        _normal_cb(CB_ALPHA, all_ranges, q_block, bf16),
        _normal_cb(CB_L0, all_ranges, q_block, bf16),
        _normal_cb(CB_L1, all_ranges, q_block, bf16),
        _normal_cb(CB_OUT, all_ranges, program_config.output_buffer_depth * q_block * d_t, bf16),
        _normal_cb(CB_O0, all_ranges, q_block * d_t, bf16),
        _normal_cb(CB_O1, all_ranges, q_block * d_t, bf16),
        _normal_cb(CB_INV_L, all_ranges, q_block, fp32),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def _float_to_u32(value):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(value)))[0]
