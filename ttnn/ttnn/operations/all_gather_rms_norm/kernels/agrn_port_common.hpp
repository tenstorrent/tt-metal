// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port cores — shared compile-time layout and the chunk walk of one slot (TRANSPORT).
// Must match _port_ct() / _walk_chunks() in all_gather_rms_norm_program_descriptor.py.
//
// A slot is one chip's normalized slice at its FINAL destination pages: page id = q * slot_stride + rr * row_stride
// + jj for rr < rows_eff, jj < cols_eff (aligned W: the slice's columns inside the gathered output rows; R_U: the
// slot-contiguous staging scratch, one flat row). A chunk is a run of <= run_pages pages consecutive in ONE bank of
// the destination (page ids P, P + nb, P + 2 nb, ...), so one packet writes it with one NoC write; the same run is
// contiguous in the local copy a relay reads. Link l owns banks l, l + L, ...; part 1 / 2 = the first / second half of
// those banks (the far slot of a balanced ring). The walk is bank-major, so part 1 is a prefix of the part-0 walk.
#pragma once

#include <cstdint>

#include "agrn_common.hpp"

namespace agrn_port {

constexpr uint32_t cb_chunks = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t run_pages = get_compile_time_arg_val(2);
constexpr uint32_t group = get_compile_time_arg_val(3);      // chunks per batch (the port CB holds 2 x group)
constexpr uint32_t inc_every = get_compile_time_arg_val(4);  // arrival increment every N chunks (+ each entry's last)
constexpr uint32_t nb = get_compile_time_arg_val(5);         // banks of the destination buffer
constexpr uint32_t rows_eff = get_compile_time_arg_val(6);
constexpr uint32_t cols_eff = get_compile_time_arg_val(7);
constexpr uint32_t row_stride = get_compile_time_arg_val(8);
constexpr uint32_t slot_stride = get_compile_time_arg_val(9);
constexpr bool flat_slot = get_compile_time_arg_val(10) != 0;  // R_U: one flat row of R * Wlt pages
constexpr uint32_t Wlt = get_compile_time_arg_val(11);
constexpr uint32_t K = get_compile_time_arg_val(12);
constexpr uint32_t n_groups = get_compile_time_arg_val(13);
constexpr uint32_t rows_hi = get_compile_time_arg_val(14);
constexpr uint32_t big = get_compile_time_arg_val(15);
constexpr uint32_t cw_nom = get_compile_time_arg_val(16);
constexpr uint32_t payload_bytes = get_compile_time_arg_val(17);  // router max payload (stats packets)
constexpr uint32_t scatter_runs = get_compile_time_arg_val(18);   // runs per bulk packet (1..4, scatter write)
constexpr uint32_t PORT_CT_ARGS = 19;                             // TensorAccessorArgs(dst) follow
constexpr uint32_t chunk_bytes = run_pages * page_bytes;

static_assert(scatter_runs >= 1 && scatter_runs <= 4, "NOC_SCATTER_WRITE_MAX_CHUNKS = 4");

// One bulk packet: up to scatter_runs runs (each consecutive in one destination bank), <= run_pages pages in total,
// packed back to back in the port CB slot. run i = pages first[i], first[i] + nb, ... (n[i] of them).
struct Packet {
    uint32_t nr;
    uint32_t pages;
    uint32_t first[4];
    uint32_t n[4];
    uint32_t rr[4];
    uint32_t jj[4];
};

// f(const Packet&): the packets of slot q for the lane (first_bank, bank_stride), part 0 / 1 / 2. Runs are visited
// bank-major; a packet closes when full (scatter_runs runs or run_pages pages) and always at the half-lane bank
// boundary, so the part-1 walk is a prefix of the part-0 walk and part 2 a suffix (balanced-ring relays).
template <typename F>
inline void for_each_packet(uint32_t q, uint32_t part, uint32_t first_bank, uint32_t bank_stride, F&& f) {
    if (first_bank >= nb) {
        return;
    }
    const uint32_t nl = (nb - first_bank + bank_stride - 1) / bank_stride;
    const uint32_t lo = part == 2 ? nl / 2 : 0;
    const uint32_t hi = part == 1 ? nl / 2 : nl;
    const uint32_t base = q * slot_stride;
    Packet pk;
    pk.nr = 0;
    pk.pages = 0;
    auto flush = [&]() {
        if (pk.nr > 0) {
            f(pk);
            pk.nr = 0;
            pk.pages = 0;
        }
    };
    for (uint32_t i = lo; i < hi; ++i) {
        if (i == nl / 2) {
            flush();
        }
        const uint32_t b = first_bank + i * bank_stride;
        for (uint32_t rr = 0; rr < rows_eff; ++rr) {
            const uint32_t rb = base + rr * row_stride;
            uint32_t jj = (b + nb - rb % nb) % nb;  // first column of this row in bank b
            while (jj < cols_eff) {
                const uint32_t avail = (cols_eff - 1 - jj) / nb + 1;
                const uint32_t n = avail < run_pages ? avail : run_pages;
                if (pk.nr == scatter_runs || pk.pages + n > run_pages) {
                    flush();
                }
                pk.first[pk.nr] = rb + jj;
                pk.n[pk.nr] = n;
                pk.rr[pk.nr] = rr;
                pk.jj[pk.nr] = jj;
                ++pk.nr;
                pk.pages += n;
                jj += n * nb;
            }
        }
    }
    flush();
}

}  // namespace agrn_port
