// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH — reader-side transaction ISSUE cost (lever B13, stateful NoC).
// NOT part of the rms_norm op.  Reconstructs the op's reader issue loop and nothing
// else: N whole-page reads from an interleaved DRAM tensor into one CB, ONE barrier
// per chunk (the op's lever B7 shape), no compute, no gamma.
//
// ---------------------------------------------------------------------------
// RAW-API JUSTIFICATION (why this bypasses noc_async_read_page / read_tile)
// ---------------------------------------------------------------------------
// `noc_async_read_page(id, acc, dst)` (which `noc_async_read_tile` forwards to)
// costs, per transaction:
//   * TensorAccessor::get_bank_and_offset(id)   — a per-dim %,/ loop over the
//     page-id -> page-coord decomposition, then the shard->bank mapping;
//   * ncrisc_noc_fast_read()                    — SIX command-buffer register
//     writes (RET_ADDR_LO, TARG_ADDR_LO, TARG_ADDR_MID, TARG_ADDR_COORDINATE,
//     AT_LEN_BE, CMD_CTRL), because the full 64-bit address is re-published
//     every time;
//   * plus the any-len dispatch (`max_page_size = NOC_MAX_BURST_SIZE + 1` makes
//     `noc_async_read` take the runtime `len_bytes > NOC_MAX_BURST_SIZE` path).
// Three of those six registers (MID, COORDINATE, AT_LEN_BE) are INVARIANT across
// every read of one bank, and the page size is invariant across the whole loop.
// `noc_async_read_one_packet_set_state` / `..._with_state` publish them once and
// leave 3 writes per transaction — but they are only usable if consecutive
// transactions share a NoC coordinate, which round-robin interleaving breaks.
// Variant 3 fixes that by walking the chunk BANK-MAJOR (page p and p+NUM_BANKS
// live in the same bank), which is legal precisely because the whole chunk is
// covered by ONE barrier, so issue order is unobservable.
//
// There is no dataflow_kernel_lib helper for "issue this page set with state
// reuse" — the gap is a CAPABILITY gap, not ergonomics (see the experiment
// README).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

constexpr uint32_t cb_in = 0;

// 0 = whole-page (tile) reads; 1 = partial-page (row-major stick) reads.
constexpr uint32_t MODE = get_compile_time_arg_val(0);
// Reader variant, see the menu in bench_stateful_noc.py.
constexpr uint32_t RVAR = get_compile_time_arg_val(1);
constexpr uint32_t READ_BYTES = get_compile_time_arg_val(2);  // bytes per transaction
constexpr uint32_t CB_PAGE_BYTES = get_compile_time_arg_val(3);
constexpr uint32_t CHUNK = get_compile_time_arg_val(4);   // pages per barrier group
constexpr uint32_t BYTE_OFF = get_compile_time_arg_val(5);  // in-page offset (stick mode)
// /perf-measure ablation: keep every CB op, loop trip and barrier, issue no NoC
// transfer.  Used to price ONE half of the bench without the other half's NoC
// traffic competing for the same core's bandwidth.  Correctness for the variant
// is established by the matching un-stubbed copy arm.
constexpr uint32_t SKIP_PAYLOAD = get_compile_time_arg_val(6);
// NUM_DRAM_BANKS is a kernel-build define (the same constant
// interleaved_addr_gen::get_bank_index uses), so the bank stride is compile-time.
constexpr uint32_t NUM_BANKS = NUM_DRAM_BANKS;

constexpr auto in_args = TensorAccessorArgs<7>();

// Variant ids
constexpr uint32_t V_BASELINE = 0;    // noc_async_read_page (== the op today)
constexpr uint32_t V_ONE_PACKET = 1;  // skip the any-len dispatch only
constexpr uint32_t V_AFFINE = 2;      // + skip the accessor's bank arithmetic
constexpr uint32_t V_BANK_STATE = 3;  // + reuse cmd-buf state, bank-major order
constexpr uint32_t V_BANK_TRID = 4;   // V_BANK_STATE issued with a transaction id
// V_BANK_STATE walks the chunk bank-major, so every core starts on the SAME bank
// (a core's first page id is a multiple of Wt, hence of NUM_BANKS in practice) and
// they all queue on one DRAM channel at a time.  V_BANK_ROT rotates each core's
// bank order by its own index so the cores spread across channels again.
constexpr uint32_t V_BANK_ROT = 5;

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_chunks = get_arg_val<uint32_t>(2);
    const uint32_t rot = get_arg_val<uint32_t>(3);  // per-core bank rotation (V_BANK_ROT)

    const auto acc = TensorAccessor(in_args, src_addr);
    static_assert(decltype(acc)::DSpec::is_dram, "bench is DRAM-interleaved only");
    static_assert(decltype(acc)::DSpec::is_interleaved, "bench is DRAM-interleaved only");
    // Bank stride: page p and p + NUM_BANKS share a bank and differ by exactly
    // one aligned page in that bank's local address space (InterleavedAddrGen::get_addr).
    const uint32_t APS = acc.get_aligned_page_size();

    // Bank-base table for V_AFFINE: page k (k < NUM_BANKS) sits in bank k at
    // bank_page_offset 0, so get_noc_addr(k) IS bank k's base address.
    uint64_t bank_base[NUM_BANKS];
    if constexpr (RVAR == V_AFFINE) {
        for (uint32_t k = 0; k < NUM_BANKS; ++k) {
            bank_base[k] = acc.get_noc_addr(k, BYTE_OFF);
        }
    }
    // V_BANK_TRID prices ONLY the per-transaction cost of the trid form
    // (`ncrisc_noc_fast_read_with_transaction_id` adds a NIU_MST_REQS_OUTSTANDING_ID
    // status-register POLL per transaction on top of the same 3 register writes).
    // It barriers on its OWN chunk, so it is correctness-identical to
    // V_BANK_STATE and gains no overlap: the question it answers is whether
    // lever B8's per-id barrier is affordable at all before anyone builds the
    // software pipeline that would actually cash it in.
    constexpr uint32_t TRID = 1;

    for (uint32_t c = 0; c < num_chunks; ++c) {
        const uint32_t p0 = start_page + c * CHUNK;
        {
            MaybeDeviceZoneScope("rd_reserve");
            cb_reserve_back(cb_in, CHUNK);
        }
        const uint32_t dst0 = get_write_ptr(cb_in);
        {
            MaybeDeviceZoneScope("rd_issue");
            if constexpr (SKIP_PAYLOAD) {
                // nothing issued; the CB ops and the barrier below still run
            } else if constexpr (RVAR == V_BASELINE) {
                uint32_t dst = dst0;
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    if constexpr (MODE == 0) {
                        noc_async_read_page(p0 + i, acc, dst);
                    } else {
                        noc_async_read(acc.get_noc_addr(p0 + i, BYTE_OFF), dst, READ_BYTES);
                    }
                    dst += CB_PAGE_BYTES;
                }
            } else if constexpr (RVAR == V_ONE_PACKET) {
                uint32_t dst = dst0;
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    noc_async_read_one_packet(acc.get_noc_addr(p0 + i, BYTE_OFF), dst, READ_BYTES);
                    dst += CB_PAGE_BYTES;
                }
            } else if constexpr (RVAR == V_AFFINE) {
                uint32_t bi = p0 % NUM_BANKS;
                uint64_t off = static_cast<uint64_t>(p0 / NUM_BANKS) * APS;
                uint32_t dst = dst0;
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    const uint64_t a = bank_base[bi] + off;
                    ASSERT(a == acc.get_noc_addr(p0 + i, BYTE_OFF));
                    noc_async_read_one_packet(a, dst, READ_BYTES);
                    dst += CB_PAGE_BYTES;
                    if (++bi == NUM_BANKS) {
                        bi = 0;
                        off += APS;
                    }
                }
            } else {  // V_BANK_STATE / V_BANK_TRID / V_BANK_ROT — bank-major, state reused
                if constexpr (RVAR == V_BANK_TRID) {
                    noc_async_read_set_trid(TRID);
                }
                const uint32_t nb = NUM_BANKS < CHUNK ? NUM_BANKS : CHUNK;
                for (uint32_t kk = 0; kk < nb; ++kk) {
                    const uint32_t k = (RVAR == V_BANK_ROT) ? ((kk + rot) % nb) : kk;
                    const uint64_t base = acc.get_noc_addr(p0 + k, BYTE_OFF);
                    noc_async_read_one_packet_set_state(base, READ_BYTES);
                    uint32_t lo = static_cast<uint32_t>(base);
                    uint32_t dst = dst0 + k * CB_PAGE_BYTES;
                    for (uint32_t i = k; i < CHUNK; i += NUM_BANKS) {
                        // The with_state form publishes only the LOW 32 address
                        // bits; the anchor's MID/COORDINATE must still hold.
                        ASSERT(
                            (acc.get_noc_addr(p0 + i, BYTE_OFF) >> 32) == (base >> 32) &&
                            static_cast<uint32_t>(acc.get_noc_addr(p0 + i, BYTE_OFF)) == lo);
                        if constexpr (RVAR == V_BANK_TRID) {
                            noc_async_read_one_packet_with_state_with_trid(0, lo, dst, TRID);
                        } else {
                            noc_async_read_one_packet_with_state(lo, dst);
                        }
                        lo += APS;
                        dst += NUM_BANKS * CB_PAGE_BYTES;
                    }
                }
            }
        }  // rd_issue
        {
            MaybeDeviceZoneScope("rd_barrier");
            if constexpr (RVAR == V_BANK_TRID) {
                noc_async_read_barrier_with_trid(TRID);
            } else {
                noc_async_read_barrier();
            }
        }
        cb_push_back(cb_in, CHUNK);
    }
}
