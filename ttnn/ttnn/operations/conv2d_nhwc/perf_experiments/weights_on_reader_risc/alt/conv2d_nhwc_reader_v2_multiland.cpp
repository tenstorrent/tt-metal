// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc reader (NCRISC) — regimes R0, R3 and R4/R5: the window / halo fetch and im2col gather
// (conv2d_nhwc_gather.hpp). With the split gather (R3) it takes the even m-blocks; the writer the odd.
//
// weights_on_reader_risc (define WR_WEIGHTS, BLOCK weight multicast only): the streamed R4/R5 weight K-block ring (DRAM
// reads + rotating column multicast / receive) runs HERE instead of on the writer, as a non-blocking state machine
// stepped while the gather waits for cb_act_rm space (and once per tile-row), then drained. Bypasses nothing in
// kernel_lib except the McastArgs blocking waits: readiness (delivery flag / consumer acks) is PEEKED on the channel's
// own semaphores so a step never blocks the gather; the pipe's send / wait_delivery are then called with their
// condition already met.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

#define GATHER_CT_BASE 0
#include "conv2d_nhwc_gather.hpp"

#if defined(WR_WEIGHTS)
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

constexpr uint32_t cb_weight_tiles = get_named_compile_time_arg_val("wr_cb_w");
constexpr uint32_t w_num_k_blocks = get_named_compile_time_arg_val("wr_nkb");  // global K-blocks
constexpr uint32_t w_num_n_blocks = get_named_compile_time_arg_val("wr_nnb");
constexpr uint32_t w_bk = get_named_compile_time_arg_val("wr_bk");
constexpr uint32_t w_bn = get_named_compile_time_arg_val("wr_bn");
constexpr uint32_t w_N_t = get_named_compile_time_arg_val("wr_nt");
constexpr uint32_t w_kps = get_named_compile_time_arg_val("wr_kps");  // K-blocks per slice
constexpr bool w_mcast = get_named_compile_time_arg_val("wr_mcast") != 0;
constexpr uint32_t w_depth = get_named_compile_time_arg_val("wr_depth");
constexpr uint32_t w_k_rows = get_named_compile_time_arg_val("wr_krows");
constexpr uint32_t w_senders = get_named_compile_time_arg_val("wr_senders");
constexpr uint32_t w_slice_tiles = get_named_compile_time_arg_val("wr_slice_tiles");
constexpr uint32_t w_ack_count = get_named_compile_time_arg_val("wr_ack_count");
constexpr bool w_act_mcast = get_named_compile_time_arg_val("wr_act_mcast") != 0;
constexpr uint32_t W_RT = 4;  // this kernel's own RT args; the gather's follow
constexpr auto weight_args = TensorAccessorArgs<conv_gather::input_args.next_compile_time_args_offset()>();
constexpr auto weight_channel = dataflow_kernel_lib::McastArgs<
    get_named_compile_time_arg_val("w_mcast_ct_offset"),
    get_named_compile_time_arg_val("w_mcast_rt_offset")>();

constexpr uint32_t W_CT = get_named_compile_time_arg_val("w_mcast_ct_offset");
constexpr dataflow_kernel_lib::McastSemaphoreBinding W_DATA_READY{
    dataflow_kernel_lib::detail::positional_mcast_semaphore<W_CT, dataflow_kernel_lib::mcast_wire::DATA_READY>()};
constexpr dataflow_kernel_lib::McastSemaphoreBinding W_CONSUMER_READY{
    dataflow_kernel_lib::detail::positional_mcast_semaphore<W_CT, dataflow_kernel_lib::mcast_wire::CONSUMER_READY>()};
constexpr uint32_t w_num_slices = w_act_mcast ? w_num_k_blocks / w_kps : 1;
constexpr uint32_t w_slice_k_tiles = w_kps * w_bk;
constexpr uint32_t w_land_lead = w_depth > 2 ? 1 : 0;
constexpr uint32_t w_read_ahead = w_depth > w_land_lead + 3 ? w_depth - w_land_lead - 2 : 1;
static_assert(w_read_ahead < conv_gather::GATHER_TRID, "weight trids 1..read_ahead below the gather's");
constexpr uint32_t w_block_tiles = w_k_rows * w_bn;
static_assert(w_mcast && weight_channel.active, "WR_WEIGHTS: the BLOCK weight multicast channel");

FORCE_INLINE uint32_t w_first_row(uint32_t k_block) {
    return w_act_mcast ? (k_block % w_num_slices) * w_slice_k_tiles + (k_block / w_num_slices) * w_bk : k_block * w_bk;
}

FORCE_INLINE void w_zero_fill(uint64_t zeros_noc, uint32_t dst, uint32_t bytes) {
    while (bytes > 0) {
        const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros_noc, dst, chunk);
        dst += chunk;
        bytes -= chunk;
    }
}

void kernel_main() {
    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t weight_phase = get_arg_val<uint32_t>(2);  // this core's row (rotating) / 1 = fixed reader
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(3);
    const bool reads_weights = !w_mcast || weight_phase != 0;
    const auto weights = TensorAccessor(weight_args, weight_addr);
    const uint32_t tile_bytes = get_tile_size(cb_weight_tiles);
    const uint32_t slot_bytes = w_block_tiles * tile_bytes;
    const uint32_t cb_limit = get_local_cb_interface(cb_weight_tiles).fifo_limit;
    const uint32_t cb_bytes = get_local_cb_interface(cb_weight_tiles).fifo_size;
    const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
    Noc noc;
    auto sender = weight_channel.optional_sender(noc);
    auto receiver = weight_channel.optional_receiver(noc);
    using WCh = decltype(weight_channel);
    // Peeked semaphores (same bindings the pipes use).
    auto data_ready = dataflow_kernel_lib::detail::make_mcast_semaphore<W_DATA_READY>();
    auto consumer_ready = dataflow_kernel_lib::detail::make_mcast_semaphore<W_CONSUMER_READY>();

    // Global weight block sequence over every (m, n, k) block: g -> k = g % nkb, n = (g / nkb) % nnb.
    const uint32_t G = num_m_blocks * w_num_n_blocks * w_num_k_blocks;
    uint32_t g_push = 0, g_read = 0, g_ack = 0;
    auto slot_of = [&](uint32_t g) {
        uint32_t a = get_write_ptr(cb_weight_tiles) + (g - g_push) * slot_bytes;
        return a >= cb_limit ? a - cb_bytes : a;
    };
    auto read_block = [&](uint32_t g, uint32_t dst) {
        const uint32_t k_block = g % w_num_k_blocks;
        const uint32_t n0 = n_tile_start + ((g / w_num_k_blocks) % w_num_n_blocks) * w_bn;
        if constexpr (w_slice_tiles > 0) {
            constexpr uint32_t k_rows_total = w_num_slices * w_slice_k_tiles;
            const uint32_t p0 =
                (n0 / w_slice_tiles) * k_rows_total * w_slice_tiles + w_first_row(k_block) * w_slice_tiles;
            constexpr uint32_t np = w_block_tiles;
            const uint32_t bank_stride = weights.get_aligned_page_size();
            for (uint32_t b = 0; b < NUM_DRAM_BANKS && b < np; ++b) {
                const uint64_t a = weights.get_noc_addr(p0 + b);
                noc_async_read_set_state(a);
                uint32_t src = static_cast<uint32_t>(a);
                uint32_t d = dst + b * tile_bytes;
                for (uint32_t q = b; q < np; q += NUM_DRAM_BANKS) {
                    noc_async_read_with_state(src, d, tile_bytes);
                    src += bank_stride;
                    d += NUM_DRAM_BANKS * tile_bytes;
                }
            }
            return;
        }
        uint32_t row_page = w_first_row(k_block) * w_N_t + n0;
        for (uint32_t kk = 0; kk < w_k_rows; ++kk) {
            for (uint32_t j = 0; j < w_bn; ++j) {
                if (n0 + j < w_N_t) {
                    noc_async_read(weights.get_noc_addr(row_page + j), dst, tile_bytes);
                } else {
                    w_zero_fill(zeros_noc, dst, tile_bytes);
                }
                dst += tile_bytes;
            }
            row_page += w_N_t;
        }
    };
    auto delivered = [&](uint32_t g) {
        if constexpr (WCh::signal == dataflow_kernel_lib::DataReadySignal::Counter) {
            return data_ready.value() >= g + 1;
        } else {
            return data_ready.value() == VALID;
        }
    };
    // Rotating phases tracked incrementally (no per-step division).
    uint32_t push_phase = 0, read_phase = 0;  // g % w_senders
    uint32_t push_trid = 0, read_trid = 0;    // g % w_read_ahead
    auto bump = [](uint32_t& v, uint32_t n) { v = v + 1 == n ? 0 : v + 1; };
    auto owns_phase = [&](uint32_t ph) {
        if constexpr (w_senders > 0) {
            return ph == weight_phase;
        } else {
            return reads_weights;
        }
    };
    auto issue_reads = [&]() -> bool {
        bool issued = false;
        for (; g_read < G && g_read < g_push + w_read_ahead;
             ++g_read, bump(read_phase, w_senders ? w_senders : 1), bump(read_trid, w_read_ahead)) {
            if (!owns_phase(read_phase)) {
                continue;
            }
            const uint32_t need = (g_read - g_push + 1) * w_block_tiles;
            if (!cb_pages_reservable_at_back(cb_weight_tiles, need)) {
                break;
            }
            cb_reserve_back(cb_weight_tiles, need);
            noc_async_read_set_trid(1 + read_trid);
            read_block(g_read, slot_of(g_read));
            issued = true;
        }
        if (issued) {
            noc_async_read_set_trid(0);
        }
        return issued;
    };
    // land g_push if its condition is met (never blocks)
    auto try_land = [&]() -> bool {
        if (g_push >= G) {
            return false;
        }
        if (owns_phase(push_phase)) {
            if (g_read <= g_push || !ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + push_trid)) {
                return false;
            }
            if constexpr (WCh::pre_handshake) {
                if (consumer_ready.value() < w_ack_count) {
                    return false;
                }
            }
            const uint32_t a = slot_of(g_push);
            sender->send(a, a, slot_bytes);  // self-excluded; acks already in
            g_ack = g_push + 1;
        } else {
            if (g_ack <= g_push || !delivered(g_push)) {
                return false;
            }
            receiver->wait_delivery(g_push);  // condition met: clears the flag
        }
        cb_push_back(cb_weight_tiles, w_block_tiles);
        ++g_push;
        bump(push_phase, w_senders ? w_senders : 1);
        bump(push_trid, w_read_ahead);
        return true;
    };
    // receiver ack of g_push (Flag order: after the previous delivery)
    auto try_ack = [&]() {
        if (g_ack == g_push && g_push < G && !owns_phase(push_phase) &&
            cb_pages_reservable_at_back(cb_weight_tiles, w_block_tiles)) {
            cb_reserve_back(cb_weight_tiles, w_block_tiles);
            receiver->acknowledge(g_push);
            ++g_ack;
        }
    };
    // One non-blocking step: lands every block that is ready; true if it issued NoC reads (the
    // caller restores its read state).
    auto step = [&]() -> bool {
        bool issued = issue_reads();
        while (try_land()) {
            try_ack();
            issued |= issue_reads();
        }
        try_ack();
        return issued;
    };

    step();
    conv_gather::gather_m_blocks(W_RT, 0, step);
    while (g_push < G) {
        step();
    }
    noc_async_write_barrier();
}
#else
void kernel_main() { conv_gather::gather_m_blocks(0, 0); }
#endif
