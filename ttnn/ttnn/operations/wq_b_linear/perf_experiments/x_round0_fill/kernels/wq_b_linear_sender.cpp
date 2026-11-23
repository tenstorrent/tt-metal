// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-multicaster (x_round0_fill experiment copy; ordered-signal path only, the op's default).
// HELPER DEVIATION (same as the op): Counter-mode SenderPipe::send() issues the payload unlinked and blocks
// on a full write barrier before its counter atomic; this kernel multicasts the payload linked and publishes
// by multicasting the running count into the receivers' Counter cell right behind it (same path + VC).
// Additions:
//   * chunked delivery: each round is sent as nch(r) chunks (see the x-reader); every chunk is published
//     with its own count (count = chunks sent so far), so receivers wait for count >= cum_chunks(r).
//   * r0_brisc_chunks: this RISC reads the last r0_brisc_chunks chunks of round 0 from DRAM itself on its
//     own NoC at kernel start (splitting round 0's read over both RISCs / both NoCs), and multicasts
//     them after the x-reader's chunks.

#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t first_round = get_compile_time_arg_val(4);
    constexpr uint32_t round_stride = get_compile_time_arg_val(5);
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t x_chunks_per_round = get_compile_time_arg_val(7);
    constexpr bool x_chunk_r0_only = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t r0_brisc_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr uint32_t my_rounds = (num_k_blocks - first_round + round_stride - 1) / round_stride;
    using McA = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>;
    using McB = McastArgs<
        get_named_compile_time_arg_val("x_mcast_b_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_b_rt_offset")>;
    static_assert(McA::active != McB::active, "a sender kernel carries exactly one x channel");
    constexpr std::conditional_t<McA::active, McA, McB> mc{};
    Noc noc;
    auto pipe = mc.sender(noc);  // as in the op (channel setup); the ordered path publishes by hand
    (void)pipe;

    const uint32_t dst_x_start = get_arg_val<uint32_t>(0);
    const uint32_t dst_y_start = get_arg_val<uint32_t>(1);
    const uint32_t dst_x_end = get_arg_val<uint32_t>(2);
    const uint32_t dst_y_end = get_arg_val<uint32_t>(3);
    const uint32_t num_dests = get_arg_val<uint32_t>(4);
    const uint32_t x_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const uint32_t round_bytes = round_tiles * x_tile_bytes;
    const uint32_t base = get_read_ptr(cb_x_resident);

    if constexpr (r0_brisc_chunks > 0) {
        // Read the tail chunks of round 0 on this RISC's NoC (the x-reader skips them).
        const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);
        constexpr uint32_t chunk_tiles = round_tiles / x_chunks_per_round;
        // One transaction id per chunk (1..), so each chunk is multicast as soon as it has landed.
        uint32_t dst = base + (x_chunks_per_round - r0_brisc_chunks) * chunk_tiles * x_tile_bytes;
        for (uint32_t c = x_chunks_per_round - r0_brisc_chunks; c < x_chunks_per_round; ++c) {
            dataflow_kernel_lib::set_read_trid(noc, c + 1);
            for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
                const uint32_t m = t / block_k_tiles, kk = t % block_k_tiles;
                noc_async_read_tile(m * tensor_k_tiles + kk, x_acc, dst);
                dst += x_tile_bytes;
            }
        }
        dataflow_kernel_lib::set_read_trid(noc, 0);
    }

    auto issue_payload = [&](uint32_t addr, uint32_t bytes) {
#ifdef ABLATE_X_MCAST
        return;
#endif
        for (uint32_t off = 0; off < bytes; off += NOC_MAX_BURST_SIZE) {
            const uint32_t len = bytes - off < NOC_MAX_BURST_SIZE ? bytes - off : NOC_MAX_BURST_SIZE;
            noc_async_write_multicast(
                addr + off,
                get_noc_multicast_addr(dst_x_start, dst_y_start, dst_x_end, dst_y_end, addr + off),
                len,
                num_dests,
                /*linked=*/true);
        }
    };

    const uint32_t cell_addr = get_semaphore(data_ready_sem_id);
    volatile tt_l1_ptr uint32_t* cell = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cell_addr);
    const uint64_t cell_mcast = get_noc_multicast_addr(dst_x_start, dst_y_start, dst_x_end, dst_y_end, cell_addr);
    uint32_t count = 0;
    for (uint32_t i = 0; i < my_rounds; ++i) {
        const uint32_t r = first_round + i * round_stride;
        const uint32_t nch = (r == 0 || !x_chunk_r0_only) ? x_chunks_per_round : 1;
        const uint32_t chunk_tiles = round_tiles / nch;
        for (uint32_t c = 0; c < nch; ++c) {
            {
                MaybeDeviceZoneScope("x_mcast_wait");
                cb_wait_front(cb_x_resident, chunk_tiles);
                if (r0_brisc_chunks > 0 && r == 0 && c >= nch - r0_brisc_chunks) {
                    dataflow_kernel_lib::async_read_barrier_with_trid(noc, c + 1);  // this RISC's own chunk landed
                }
            }
            {
                MaybeDeviceZoneScope("x_mcast_issue");
                issue_payload(base + r * round_bytes + c * chunk_tiles * x_tile_bytes, chunk_tiles * x_tile_bytes);
                noc_async_writes_flushed();  // the previous count's source read is done before rewriting it
                *cell = ++count;
                noc_async_write_multicast(cell_addr, cell_mcast, sizeof(uint32_t), num_dests, /*linked=*/false);
            }
            cb_pop_front(cb_x_resident, chunk_tiles);
        }
    }
    {
        MaybeDeviceZoneScope("x_mcast_barrier");
        noc_async_write_barrier();
    }
}
