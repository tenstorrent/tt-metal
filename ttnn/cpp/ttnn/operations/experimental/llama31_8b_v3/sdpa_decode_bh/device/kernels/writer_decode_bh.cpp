// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/debug/assert.h"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/rt_args_common_dec.hpp"
#include "dataflow_common_dec.hpp"

#define MAX_TREE_REDUCTION_ROUNDS 6

void kernel_main() {
    Noc noc;

    constexpr uint32_t PNHt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t reducer_semaphore_id = get_compile_time_arg_val(7);
    uint32_t reducer_semaphore_addr = get_semaphore(reducer_semaphore_id);
    constexpr uint32_t output_semaphore_id = get_compile_time_arg_val(8);
    constexpr bool is_out_sharded = get_compile_time_arg_val(9);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(10);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(12);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t num_reducer_cores = get_compile_time_arg_val(14);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(15);
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(16);
    constexpr bool is_causal = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(18);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(19);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(20);
    constexpr uint32_t original_block_size = get_compile_time_arg_val(21);
    constexpr bool has_block_padding = original_block_size > 0 && original_block_size < 32;

    constexpr uint32_t spec_multi_pos_T = get_compile_time_arg_val(22);
    constexpr bool spec_multi_pos = spec_multi_pos_T > 0;

    constexpr auto out_args = TensorAccessorArgs<23>();

    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;

    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_sliding_window_mask_in = tt::CBIndex::c_13;
    constexpr uint32_t cb_block_pad_mask = tt::CBIndex::c_14;
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_worker = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_intermed_out = tt::CBIndex::c_19;
    constexpr uint32_t cb_out = tt::CBIndex::c_20;

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_at_round = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t step_semaphore_inc[6] = {1, 16, 256, 4096, 65536, 1048576};

    constexpr uint32_t step_semaphore_shift[6] = {0, 4, 8, 12, 16, 20};

    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }

    tt_l1_ptr uint32_t* reduction_group_core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;
    tt_l1_ptr uint32_t* reduction_group_core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;

    if (out_addr == 0) {
        return;
    }

    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;

    constexpr uint32_t spec_pos_n = spec_multi_pos ? spec_multi_pos_T : 1;
    uint32_t spec_pos[spec_pos_n];
#ifdef SDB_M
    static_assert(!spec_multi_pos && PNHt == 1 && !is_out_sharded && is_causal);
    uint32_t sdm_pos[SDB_M];
#endif

    if constexpr (is_causal) {
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            CircularBuffer cb_index(cb_cur_pos);
            cb_index.wait_front(1);
            uint32_t index_cb_ptr = cb_index.get_read_ptr();
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
            if constexpr (spec_multi_pos) {
                const uint32_t spec_pos_base = cur_batch * spec_multi_pos_T;
                for (uint32_t j = 0; j < spec_multi_pos_T; ++j) {
                    spec_pos[j] = index_ptr[spec_pos_base + j];
                }
#if ASSERT_ENABLED

                for (uint32_t j = 0; j + 1 < spec_multi_pos_T; ++j) {
                    ASSERT(spec_pos[j] <= spec_pos[j + 1]);
                }
#endif

                cur_pos = spec_pos[spec_multi_pos_T - 1];
            } else {
                cur_pos = index_ptr[(uint32_t)(cur_batch / q_heads_parallel_factor)];
#ifdef SDB_M
                for (uint32_t b = 0; b < SDB_M; ++b) {
                    sdm_pos[b] = index_ptr[b];
                }
#endif
            }
            cb_index.pop_front(1);
        }

        if (cur_pos == UINT32_MAX) {
            return;
        }
    }

    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

#ifdef SDB_M
    bool sdm_ok[SDB_M];
    for (uint32_t b = 0; b < SDB_M; ++b) {
        const uint32_t pb = sdm_pos[b];
        sdm_ok[b] = false;
        if (pb != UINT32_MAX) {
            const uint32_t skb = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(pb);
            sdm_ok[b] = skb == Sk_chunk_t_dynamic &&
                        nearest_n(pb + 1, k_chunk_size_dynamic) / k_chunk_size_dynamic == k_num_chunks;
        }
    }
#endif

    bool has_local_data = (k_chunk_start != k_chunk_end);

    if (!has_local_data) {
        return;
    }

    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;

    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }

    const bool should_send_to_parent = (parent_core_in_group != UINT32_MAX);

    tt_l1_ptr uint32_t* all_reducer_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t* all_reducer_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    dataflow_kernel_lib::prepare_zero_tile<cb_zero_in>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    if (k_chunk_start == window_start_chunk && window_start_unaligned > 0) {
        generate_sliding_window_mask<cb_sliding_window_mask_in, PNHt>(Sk_chunk_t_dynamic, window_start_unaligned);
    }

    if constexpr (has_block_padding) {
        generate_block_padding_mask<cb_block_pad_mask, PNHt>(Sk_chunk_t_dynamic, original_block_size);
    }

    if constexpr (spec_multi_pos) {
        const uint32_t spec_pos_min = spec_pos[0];
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            const uint32_t chunk_start_pos = k_chunk * k_chunk_size_dynamic;
            const uint32_t chunk_last_pos = chunk_start_pos + k_chunk_size_dynamic - 1;
            if (chunk_last_pos > spec_pos_min) {
                generate_mask_spec_multi_pos<cb_mask_in, PNHt>(Sk_chunk_t_dynamic, chunk_start_pos, spec_pos);
            }
        }
    } else if constexpr (is_causal) {
#ifdef SDB_M
        uint32_t sdm_bound[32];
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t b = r / 4;
            sdm_bound[r] = (b < SDB_M && sdm_ok[b] ? sdm_pos[b] : cur_pos) % k_chunk_size_dynamic;
        }
        sdk_generate_mask_rows<cb_mask_in>(Sk_chunk_t_dynamic, sdm_bound);
#else

        generate_mask<cb_mask_in, PNHt>(k_num_chunks, Sk_chunk_t_dynamic, cur_pos);
#endif
    }

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);

    const auto out_writer = TensorAccessor(out_args, out_addr);

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    noc.async_write_barrier();

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        if (num_active_children > 0) {
            ASSERT(num_heads_per_core == 1);

            for (uint32_t round = 0; round < num_active_rounds; ++round) {
                uint32_t child_id = active_children_per_round[round];
                if (child_id != UINT32_MAX) {
                    while (true) {
                        invalidate_l1_cache();
                        uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                        uint8_t step_sem = (sem_val >> step_semaphore_shift[round]) & 0x0F;
                        if (step_sem >= 1) {
                            break;
                        }
                    }

                    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
                    constexpr uint32_t o_read_size = out_chunk_tiles * tile_bytes_intermed;
                    constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;

                    uint32_t block_offset = round * (out_chunk_tiles + 2 * PNHt) * tile_bytes_intermed;
                    CircularBuffer cb_intermed(cb_intermed_out);
                    const uint8_t noc_id = noc.get_noc_id();
                    const uint32_t my_noc_x = my_x[noc_id];
                    const uint32_t my_noc_y = my_y[noc_id];
                    uint32_t intermed_l1_read_addr = cb_intermed.get_read_ptr() + block_offset;
                    UnicastEndpoint intermed_src;

                    {
                        CircularBuffer cb_l(cb_l_in);
                        cb_l.reserve_back(PNHt);
                        uint32_t l_write_ptr = cb_l.get_write_ptr();
                        noc.async_read(
                            intermed_src,
                            CoreLocalMem<uint32_t>(l_write_ptr),
                            ml_read_size,
                            {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                            {});
                        intermed_l1_read_addr += ml_read_size;
                        noc.async_read_barrier();
                        cb_l.push_back(PNHt);

                        CircularBuffer cb_m(cb_m_in);
                        cb_m.reserve_back(PNHt);
                        uint32_t m_write_ptr = cb_m.get_write_ptr();
                        noc.async_read(
                            intermed_src,
                            CoreLocalMem<uint32_t>(m_write_ptr),
                            ml_read_size,
                            {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                            {});
                        intermed_l1_read_addr += ml_read_size;
                        noc.async_read_barrier();
                        cb_m.push_back(PNHt);

                        CircularBuffer cb_o(cb_out_o);
                        cb_o.reserve_back(out_chunk_tiles);
                        uint32_t o_write_ptr = cb_o.get_write_ptr();
                        noc.async_read(
                            intermed_src,
                            CoreLocalMem<uint32_t>(o_write_ptr),
                            o_read_size,
                            {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                            {});
                        noc.async_read_barrier();
                        cb_o.push_back(out_chunk_tiles);
                    }
                }
            }
        }

        if (!is_tree_root && should_send_to_parent) {
            CircularBuffer cb_out_w(cb_out_worker);
            CircularBuffer cb_out_m_buf(cb_out_m);
            CircularBuffer cb_out_l_buf(cb_out_l);
            CircularBuffer cb_intermed(cb_intermed_out);

            cb_out_w.wait_front(out_chunk_tiles);
            cb_out_m_buf.wait_front(PNHt);
            cb_out_l_buf.wait_front(PNHt);

            constexpr uint32_t tile_bytes = get_tile_size(cb_out_worker);
            uint32_t block_offset = send_at_round * (out_chunk_tiles + 2 * PNHt) * tile_bytes;
            constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes;
            constexpr uint32_t ml_write_size = PNHt * tile_bytes;

            uint32_t parent_noc_x = reduction_group_core_xs[parent_core_in_group];
            uint32_t parent_noc_y = reduction_group_core_ys[parent_core_in_group];
            uint32_t output_write_addr = cb_intermed.get_write_ptr() + block_offset;
            UnicastEndpoint output_dst;

            noc.async_write(
                CoreLocalMem<uint32_t>(cb_out_l_buf.get_read_ptr()),
                output_dst,
                ml_write_size,
                {},
                {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
            output_write_addr += ml_write_size;
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_out_m_buf.get_read_ptr()),
                output_dst,
                ml_write_size,
                {},
                {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
            output_write_addr += ml_write_size;
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_out_w.get_read_ptr()),
                output_dst,
                o_write_size,
                {},
                {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
            noc.async_write_barrier();

            Semaphore<>(reducer_semaphore_id).up(noc, parent_noc_x, parent_noc_y, step_semaphore_inc[send_at_round]);

            cb_out_w.pop_front(out_chunk_tiles);
            cb_out_m_buf.pop_front(PNHt);
            cb_out_l_buf.pop_front(PNHt);
            noc.async_atomic_barrier();

            return;
        }

        if (!is_tree_root) {
            return;
        }

        uint32_t out_tile_id = cur_batch * out_chunk_tiles;
        CircularBuffer cb_out_buf(cb_out);
        if constexpr (num_kv_heads > 1 || !is_out_sharded) {
            cb_out_buf.wait_front(out_chunk_tiles);
        }
        noc.async_writes_flushed();

        if constexpr (num_kv_heads > 1) {
            constexpr uint32_t num_heads_to_write = num_q_heads / num_kv_heads;
            if (!is_out_sharded) {
#ifdef SDB_M
                {
                    constexpr uint32_t m_tile = get_tile_size(cb_out);
                    constexpr uint32_t m_line = 16 * ELEMENT_SIZE;
                    constexpr uint32_t m_face = 256 * ELEMENT_SIZE;
                    static_assert(num_heads_to_write == 4 && m_tile == 4 * m_face);
                    const uint32_t d_row = cur_head * num_heads_to_write;
                    const uint32_t d_off = d_row < 16 ? d_row * m_line : (d_row + 16) * m_line;
                    const uint32_t m_l1 = cb_out_buf.get_read_ptr();
                    for (uint32_t b = 0; b < SDB_M; ++b) {
                        if (!sdm_ok[b]) {
                            continue;
                        }
                        const uint32_t s_row = b * 4;
                        const uint32_t s_off = s_row < 16 ? s_row * m_line : (s_row + 16) * m_line;
                        for (uint32_t t = 0; t < out_chunk_tiles; ++t) {
                            const uint32_t src = m_l1 + t * m_tile + s_off;
                            for (uint32_t f = 0; f < 2; ++f) {
                                noc.async_write(
                                    CoreLocalMem<uint32_t>(src + f * m_face),
                                    out_writer,
                                    num_heads_to_write * m_line,
                                    {},
                                    {.page_id = b * out_chunk_tiles + t, .offset_bytes = d_off + f * m_face});
                            }
                        }
                    }
                }
                if constexpr (false)
#endif
#ifdef SDB_W0
                    if constexpr (PNHt == 1 && (16 % num_heads_to_write) == 0) {
                        constexpr uint32_t owm_tile_bytes = get_tile_size(cb_out);
                        constexpr uint32_t owm_line = 16 * ELEMENT_SIZE;
                        constexpr uint32_t owm_face = 256 * ELEMENT_SIZE;
                        const uint32_t owm_row = cur_head * num_heads_to_write;
                        const uint32_t owm_off = owm_row < 16 ? owm_row * owm_line : (owm_row + 16) * owm_line;
                        const uint32_t owm_l1 = cb_out_buf.get_read_ptr();
                        for (uint32_t t = 0; t < out_chunk_tiles; ++t) {
                            const uint32_t src = owm_l1 + t * owm_tile_bytes + owm_off;
                            noc.async_write(
                                CoreLocalMem<uint32_t>(src),
                                out_writer,
                                num_heads_to_write * owm_line,
                                {},
                                {.page_id = out_tile_id + t, .offset_bytes = owm_off});
                            noc.async_write(
                                CoreLocalMem<uint32_t>(src + owm_face),
                                out_writer,
                                num_heads_to_write * owm_line,
                                {},
                                {.page_id = out_tile_id + t, .offset_bytes = owm_off + owm_face});
                        }
                        out_tile_id += out_chunk_tiles;
                    } else
#endif
                        barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold, PNHt>(
                            out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, vDHt);
            }

            else if (do_output) {
                constexpr uint32_t SUBTILE_LINE_BYTES = 16 * ELEMENT_SIZE;

                constexpr uint32_t num_reducers_per_output = get_named_compile_time_arg_val("reducers_per_output");
                constexpr uint32_t num_reducers_to_wait = num_reducers_per_output - 1;
                Semaphore<>(output_semaphore_id).wait(num_reducers_to_wait);

                uint32_t reduce_core_read_index_start = (cur_batch * num_cores_per_batch) / num_cores_per_head;

                UnicastEndpoint out_src;
                const uint32_t peer_l1_base = cb_out_buf.get_read_ptr();
                for (uint32_t reduce_core_read_index = reduce_core_read_index_start + 1;
                     reduce_core_read_index < reduce_core_read_index_start + num_reducers_per_output;
                     reduce_core_read_index++) {
                    uint32_t reduce_core_read_noc_x = all_reducer_noc_x[reduce_core_read_index];
                    uint32_t reduce_core_read_noc_y = all_reducer_noc_y[reduce_core_read_index];

                    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                        uint32_t l1_write_addr = cb_out_buf.get_write_ptr() + tile * tile_bytes;
                        uint32_t peer_tile_base = peer_l1_base + tile * tile_bytes;

                        for (uint32_t head = 0; head < num_heads_to_write; ++head) {
                            uint32_t starting_row = cur_head * num_heads_to_write + head;
                            uint32_t in_tile_offset_by_starting_head =
                                starting_row < 16 ? starting_row * SUBTILE_LINE_BYTES
                                                  : (starting_row - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;
                            uint32_t peer_addr_head = peer_tile_base + in_tile_offset_by_starting_head;
                            uint32_t l1_write_addr_head = l1_write_addr + in_tile_offset_by_starting_head;

                            noc.async_read(
                                out_src,
                                CoreLocalMem<uint32_t>(l1_write_addr_head),
                                SUBTILE_LINE_BYTES,
                                {.noc_x = reduce_core_read_noc_x,
                                 .noc_y = reduce_core_read_noc_y,
                                 .addr = peer_addr_head},
                                {});

                            noc.async_read(
                                out_src,
                                CoreLocalMem<uint32_t>(l1_write_addr_head + 256 * ELEMENT_SIZE),
                                SUBTILE_LINE_BYTES,
                                {.noc_x = reduce_core_read_noc_x,
                                 .noc_y = reduce_core_read_noc_y,
                                 .addr = peer_addr_head + 256 * ELEMENT_SIZE},
                                {});

                            if (++barrier_count == barrier_threshold) {
                                noc.async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                    }
                }
                noc.async_read_barrier();
            } else {
                uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
                uint32_t output_core_noc_y = all_output_noc_y[cur_batch];
                Semaphore<>(output_semaphore_id).up(noc, output_core_noc_x, output_core_noc_y, 1);
            }
        } else {
            if (!is_out_sharded) {
                barrier_count = write_tiles_to_memory<cb_out, out_chunk_tiles, barrier_threshold>(
                    out_tile_id, out_writer, barrier_count);
            }
        }
        if constexpr (num_kv_heads > 1 || !is_out_sharded) {
            noc.async_write_barrier();
            cb_out_buf.pop_front(out_chunk_tiles);
        }
    }
}
