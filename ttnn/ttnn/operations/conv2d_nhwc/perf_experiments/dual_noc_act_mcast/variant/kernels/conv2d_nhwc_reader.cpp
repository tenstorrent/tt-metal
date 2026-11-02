// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc reader (NCRISC) — regimes R0, R3 and R4/R5: the window / halo fetch and im2col gather
// (conv2d_nhwc_gather.hpp). With the split gather (R3) it takes the even m-blocks; the writer the odd.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// ---- dual_noc_act_mcast (perf bench, dna_mode 3) ----
// RAW NoC (no dataflow_kernel_lib Mcast channel): the owner's writer (BRISC, NOC1) posts the round's
// tail part (src, landing) in an L1 mailbox after the receivers' acks; this RISC multicasts it on its
// own NOC0 as a linked chain closed by data-ready flag B (receivers wait flags A and B), and
// publishes done once the writes are acked (the owner's writer then pushes its own landing / pops).
constexpr uint32_t dna_mode = get_named_compile_time_arg_val("dna_mode");
constexpr bool dna_rb = dna_mode == 3 && get_named_compile_time_arg_val("dna_b16") > 0;
constexpr uint32_t dna_sem_fb = get_named_compile_time_arg_val("dna_sem_fb");
constexpr uint32_t dna_sem_one = get_named_compile_time_arg_val("dna_sem_one");
constexpr uint32_t dna_rd_rt = get_named_compile_time_arg_val("dna_rd_rt");
constexpr uint32_t dna_sem_post = get_named_compile_time_arg_val("dna_sem_post");
constexpr uint32_t dna_num_slices = get_named_compile_time_arg_val("dna_num_slices");
constexpr uint32_t cb_dna_post = get_named_compile_time_arg_val("dna_cb_post");
constexpr uint32_t cb_dna_done = get_named_compile_time_arg_val("dna_cb_done");
namespace dna {
// Doorbell = a 1-page CB (writer -> here: {src, dst, bytes}) polled through its stream counter (a
// register read: no L1-cache invalidate on the hot path). The owner's writer waits flag B on itself
// (loopback, ordered behind the data) instead of a done handshake.
uint32_t served = 0;
uint64_t b_base = 0;
FORCE_INLINE void service() {
    if constexpr (dna_rb) {
        if (cb_pages_available_at_front(cb_dna_post, 1)) {
            invalidate_l1_cache();
            auto m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_dna_post));
            const uint32_t src = m[0], dst = m[1], bytes = m[2];
            cb_pop_front(cb_dna_post, 1);
            MaybeDeviceZoneScope("reader_b_issue");
            if (bytes) {
                noc_async_write_multicast_loopback_src(src, b_base | dst, bytes, dna_num_slices, true, noc_index);
            }
            noc_semaphore_set_multicast_loopback_src(
                get_semaphore(dna_sem_one), b_base | get_semaphore(dna_sem_fb), dna_num_slices, false, noc_index);
            ++served;
        }
    }
}
// The per-tile-row read barrier, servicing the doorbell while the reads drain.
FORCE_INLINE void read_barrier() {
    while (!ncrisc_noc_reads_flushed(noc_index)) {
        service();
    }
    invalidate_l1_cache();
}
}  // namespace dna
constexpr bool dna_fine = get_named_compile_time_arg_val("dna_fine") != 0;
#define GATHER_HOOK()       \
    if constexpr (dna_fine) \
    dna::service()
#define GATHER_HOOK_COARSE() dna::service()
#define GATHER_HOOK_BARRIER() dna::read_barrier()

#define GATHER_CT_BASE 0
#include "conv2d_nhwc_gather.hpp"

void kernel_main() {
    if constexpr (dna_rb) {
        const uint32_t xlo = get_arg_val<uint32_t>(dna_rd_rt + 0), ylo = get_arg_val<uint32_t>(dna_rd_rt + 1);
        const uint32_t xhi = get_arg_val<uint32_t>(dna_rd_rt + 2), yhi = get_arg_val<uint32_t>(dna_rd_rt + 3);
        dna::b_base = get_noc_multicast_addr(xlo, ylo, xhi, yhi, 0, noc_index);
    }
    conv_gather::gather_m_blocks(0, 0);
    if constexpr (dna_rb) {
        const uint32_t own = get_arg_val<uint32_t>(3) * conv_gather::num_n_blocks * conv_gather::num_k_blocks;
        while (dna::served < own) {
            dna::service();
        }
        noc_async_write_barrier();
    }
}
