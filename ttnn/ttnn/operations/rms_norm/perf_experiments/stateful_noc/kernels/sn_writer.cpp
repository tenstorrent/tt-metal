// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED PERF BENCH — writer twin of sn_reader.cpp (lever B13, stateful NoC).
// NOT part of the rms_norm op.
//
// RAW-API JUSTIFICATION: `noc_async_write_page` / `noc_async_write_tile` pays
// per transaction the accessor's bank arithmetic plus SEVEN command-buffer
// register writes (NOC_CTRL, TARG_ADDR_LO, RET_ADDR_LO, RET_ADDR_MID,
// RET_ADDR_COORDINATE, AT_LEN_BE, CMD_CTRL) — one more than the read path,
// which is why the op measures ~36 ns/tile on `wr_issue` vs ~30 on
// `rd_in_issue`.  Four of the seven (CTRL, MID, COORDINATE, AT_LEN_BE) are
// invariant per bank + size.  `noc_async_write_one_packet_set_state` /
// `..._with_state` publish them once, leaving 3 per transaction; the bank-major
// walk is what makes the coordinate invariant under round-robin interleaving,
// and is legal because one barrier covers the whole chunk.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

constexpr uint32_t cb_in = 0;

constexpr uint32_t MODE = get_compile_time_arg_val(0);
constexpr uint32_t WVAR = get_compile_time_arg_val(1);
constexpr uint32_t WRITE_BYTES = get_compile_time_arg_val(2);
constexpr uint32_t CB_PAGE_BYTES = get_compile_time_arg_val(3);
constexpr uint32_t CHUNK = get_compile_time_arg_val(4);
constexpr uint32_t BYTE_OFF = get_compile_time_arg_val(5);
// /perf-measure ablation — see sn_reader.cpp.
constexpr uint32_t SKIP_PAYLOAD = get_compile_time_arg_val(6);
// NUM_DRAM_BANKS is a kernel-build define (the same constant
// interleaved_addr_gen::get_bank_index uses), so the bank stride is compile-time.
constexpr uint32_t NUM_BANKS = NUM_DRAM_BANKS;

constexpr auto out_args = TensorAccessorArgs<7>();

constexpr uint32_t V_BASELINE = 0;
constexpr uint32_t V_ONE_PACKET = 1;
constexpr uint32_t V_AFFINE = 2;
constexpr uint32_t V_BANK_STATE = 3;
// See sn_reader.cpp: rotate each core's bank order so the cores do not all queue
// on the same DRAM channel at the same time.
constexpr uint32_t V_BANK_ROT = 4;

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_chunks = get_arg_val<uint32_t>(2);
    const uint32_t rot = get_arg_val<uint32_t>(3);  // per-core bank rotation (V_BANK_ROT)

    const auto acc = TensorAccessor(out_args, dst_addr);
    static_assert(decltype(acc)::DSpec::is_dram, "bench is DRAM-interleaved only");
    static_assert(decltype(acc)::DSpec::is_interleaved, "bench is DRAM-interleaved only");
    const uint32_t APS = acc.get_aligned_page_size();

    uint64_t bank_base[NUM_BANKS];
    if constexpr (WVAR == V_AFFINE) {
        for (uint32_t k = 0; k < NUM_BANKS; ++k) {
            bank_base[k] = acc.get_noc_addr(k, BYTE_OFF);
        }
    }

    for (uint32_t c = 0; c < num_chunks; ++c) {
        const uint32_t p0 = start_page + c * CHUNK;
        {
            MaybeDeviceZoneScope("wr_wait");
            cb_wait_front(cb_in, CHUNK);
        }
        const uint32_t src0 = get_read_ptr(cb_in);
        {
            MaybeDeviceZoneScope("wr_issue");
            if constexpr (SKIP_PAYLOAD) {
                // nothing issued; the CB ops and the barrier below still run
            } else if constexpr (WVAR == V_BASELINE) {
                uint32_t src = src0;
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    if constexpr (MODE == 0) {
                        noc_async_write_page(p0 + i, acc, src);
                    } else {
                        noc_async_write(src, acc.get_noc_addr(p0 + i, BYTE_OFF), WRITE_BYTES);
                    }
                    src += CB_PAGE_BYTES;
                }
            } else if constexpr (WVAR == V_ONE_PACKET) {
                uint32_t src = src0;
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    noc_async_write_one_packet(src, acc.get_noc_addr(p0 + i, BYTE_OFF), WRITE_BYTES);
                    src += CB_PAGE_BYTES;
                }
            } else if constexpr (WVAR == V_AFFINE) {
                uint32_t bi = p0 % NUM_BANKS;
                uint64_t off = static_cast<uint64_t>(p0 / NUM_BANKS) * APS;
                uint32_t src = src0;
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    const uint64_t a = bank_base[bi] + off;
                    ASSERT(a == acc.get_noc_addr(p0 + i, BYTE_OFF));
                    noc_async_write_one_packet(src, a, WRITE_BYTES);
                    src += CB_PAGE_BYTES;
                    if (++bi == NUM_BANKS) {
                        bi = 0;
                        off += APS;
                    }
                }
            } else {  // V_BANK_STATE / V_BANK_ROT — bank-major, cmd-buf state reused
                const uint32_t nb = NUM_BANKS < CHUNK ? NUM_BANKS : CHUNK;
                for (uint32_t kk = 0; kk < nb; ++kk) {
                    const uint32_t k = (WVAR == V_BANK_ROT) ? ((kk + rot) % nb) : kk;
                    const uint64_t base = acc.get_noc_addr(p0 + k, BYTE_OFF);
                    noc_async_write_one_packet_set_state(base, WRITE_BYTES);
                    uint32_t lo = static_cast<uint32_t>(base);
                    uint32_t src = src0 + k * CB_PAGE_BYTES;
                    for (uint32_t i = k; i < CHUNK; i += NUM_BANKS) {
                        ASSERT(
                            (acc.get_noc_addr(p0 + i, BYTE_OFF) >> 32) == (base >> 32) &&
                            static_cast<uint32_t>(acc.get_noc_addr(p0 + i, BYTE_OFF)) == lo);
                        noc_async_write_one_packet_with_state(src, lo);
                        lo += APS;
                        src += NUM_BANKS * CB_PAGE_BYTES;
                    }
                }
            }
        }  // wr_issue
        {
            MaybeDeviceZoneScope("wr_barrier");
            noc_async_write_barrier();
        }
        cb_pop_front(cb_in, CHUNK);
    }
}
