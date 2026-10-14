// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert, DRISC mode (se_drisc.hpp): reader slot r's Tensix core posts this call's gate/up job to its
// DRISC. Builds the schedule (se_dyn_load, as the reader did), writes the job descriptor into the DRISC's L1, then
// bumps its job counter (after the descriptor has landed).
//
// CT: 0 CHUNK_BYTES, 1 BLK_BYTES, 2 RG, 3 G, 4 PER_E, 5 GRANT_SEM (the receivers' grant counter), 6 DATA_SEM, 7 NUM_E
// RT: 0 DRISC working base, 1 gate/up weights address, 2 landing (arena), 3 NS slots, 4 R, 5 MPFE word (SD_MPFE), then
//     per slot: DRISC xy, job index on it, region offset in its bank; then the slots' R receiver xy each; then the
//     se_dyn.hpp args
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"
#include "se_drisc.hpp"

void kernel_main() {
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t blk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t rg = get_compile_time_arg_val(2);
    constexpr uint32_t groups = get_compile_time_arg_val(3);
    constexpr uint32_t per_e = get_compile_time_arg_val(4);
    constexpr uint32_t grant_sem = get_compile_time_arg_val(5);
    constexpr uint32_t data_sem = get_compile_time_arg_val(6);
    constexpr uint32_t num_e = get_compile_time_arg_val(7);
    const uint32_t base = get_arg_val<uint32_t>(0), gu = get_arg_val<uint32_t>(1), land = get_arg_val<uint32_t>(2);
    const uint32_t ns = get_arg_val<uint32_t>(3), R = get_arg_val<uint32_t>(4);
    const uint32_t rx0 = 6 + 3 * ns;

    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_7);
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, rx0 + ns * R, scratch, 1);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);  // (rows read: reuse)
    d[SD_CHUNK] = chunk_bytes;
    d[SD_BLK] = blk_bytes;
    d[SD_RG] = rg;
    d[SD_R] = R;
    d[SD_G] = groups;
    d[SD_PER_E] = per_e;
    d[SD_LAND] = land;
    d[SD_DATA_SEM] = get_semaphore(data_sem);
    d[SD_N_LOAD] = dyn.n_load;
    d[SD_GRANTS] = get_semaphore(grant_sem);
    d[SD_MPFE] = get_arg_val<uint32_t>(5);
    for (uint32_t l = 0; l < dyn.n_load; ++l) {
        d[SD_LOADS + l] = dyn.load_eid[l] | (uint32_t(dyn.region[l]) << 8);
    }
    const uint32_t bytes = ((SD_LOADS + dyn.n_load) * 4 + 15) / 16 * 16;
    for (uint32_t s = 0; s < ns; ++s) {  // the slots' descriptors differ in the region and the receivers only
        const uint32_t dxy = get_arg_val<uint32_t>(6 + 3 * s), job = get_arg_val<uint32_t>(7 + 3 * s);
        d[SD_GDDR] = gu + get_arg_val<uint32_t>(8 + 3 * s);
        for (uint32_t j = 0; j < R; ++j) {
            d[SD_RXY + j] = get_arg_val<uint32_t>(rx0 + s * R + j);
        }
        se_drisc_write(scratch, dxy, base + se_drisc_desc(job), bytes);
        noc_async_write_barrier();  // (landed before the job count moves; and the buffer is reused)
        se_drisc_inc(dxy, base + se_drisc_jobs(job), 1);
    }
    noc_async_atomic_barrier();
    noc_async_full_barrier();
}
