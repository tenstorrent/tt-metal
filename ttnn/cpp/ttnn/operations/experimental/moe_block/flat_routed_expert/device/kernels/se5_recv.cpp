// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Gate/up receiver data movement (BRISC, NOC1) on a gate/up core of M-group G. A non-blocking event loop that
//   * runs the in1 (weight) ring: grants each free landing slot (this core's grant counter) to the weight reads
//     (se_gu_read.hpp: this core's NCRISC on NOC0, or this loop on NOC1) and publishes blocks as their DATA count
//     moves; the ring holds the weight slices of the schedule's loads, compute frees blocks after their last use,
//   * publishes x blocks to compute as their pieces arrive along the snake (XARR, incremented by the predecessor or
//     the relay after each piece's write is acknowledged) and posts the consumed count (SE_X_FWD_NC_CONS) for the
//     NCRISC forwarder (se5_xfwd.cpp), which forwards x and reports freed slots upstream,
//   * sends its h slice of each virtual expert v ([MT x NP] tiles, h K-tiles CG * NP + p) into the h_all of every down
//     chain head (both M-groups, group G at G * HALF) with a GATH increment there, once go >= v (counted by the down
//     cores' coordinator).
//
// CT: 0 X_CB, 1 X_BLK_TILES, 2 IN1_CB, 3 SLOT_TILES, 4 TOTAL_W_BLOCKS, 5 NUM_V, 6 W_SLOTS, 7 OUT_CB, 8 OUT_TILES,
//     9 H_LOCAL_CB, 10 H_ALL_CB, 11 MT, 12 H_TILE_BYTES, 13 NCC, 14 DATA_SEM, 15 HARR_SEM, 16 GO_SEM, 17 DONE_SEM,
//     18 KBLK, 19 HARR1_SEM, 20 SFREE_SEM, 21 HBUF, 22 XARR_SEM, 23 HFREE_SEM, 24 X_SLOTS, 25 X_BYTES, 26 S, 27 NP,
//     28 HSFREE_SEM, 29 H_PIECES, 30 X_PER_V (x blocks per virtual expert), 31-32 GATH
//     ids of h buffers 1, 2, 33 X_PIECES (x blocks travel the chain cut-through in this many pieces; XARR counts
//     pieces), 34 G (M-groups sharing the down cores' h_all)
// RT: 0 unused, 1 the grant counter's sem id here, 2 column group CG, 3 NH (chain heads of the
// group), 4 unused,
//     5 coordinator xy, 6 group's GATH sem id, 7 x ring address, 8 is coordinator, 9 predecessor xy (0: chain
//     head), 10 successor xy (0: chain tail), 11 h predecessor xy (0: chain head), 12 sem id of this core's h-free word
//     on the h predecessor, 13 h_all address, 14 M-group, 15 partner xy, 16 AVAIL sem id (on B), 17 BCOPY
//     sem id (on A), 18 second x successor xy (0: none; x travels down a tree), 19 sem id of this core's x-free word
//     on its predecessor (SFREE or SFREE2 there), 20 sem id of the second successor's free word here, 21.. NH head xy,
//     then NCC compute-core xy, then the se_dyn.hpp args, then the se_gu_read.hpp args
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"
#include "se_gu_read.hpp"
#define SE_H_WRITE(...) noc_async_write(__VA_ARGS__)

// The h exchange's unicast VC (gate/up -> down heads, the down chains): its own, off the default write VC (1)
constexpr uint8_t h_vc = 0;

void kernel_main() {
    constexpr uint32_t x_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_blk = get_compile_time_arg_val(1);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t slot = get_compile_time_arg_val(3);
    constexpr uint32_t total_w_ct = get_compile_time_arg_val(4);
    constexpr uint32_t num_v_ct = get_compile_time_arg_val(5);
    constexpr uint32_t w_slots = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t h_local_cb = get_compile_time_arg_val(9);
    constexpr uint32_t h_all_cb = get_compile_time_arg_val(10);
    constexpr uint32_t mt = get_compile_time_arg_val(11);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t ncc = get_compile_time_arg_val(13);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(16);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(17);
    constexpr uint32_t kblk = get_compile_time_arg_val(18);
    constexpr uint32_t harr1_sem_id = get_compile_time_arg_val(19);
    constexpr uint32_t hbuf = get_compile_time_arg_val(21);
    constexpr uint32_t xarr_sem_id = get_compile_time_arg_val(22);
    constexpr uint32_t x_slots = get_compile_time_arg_val(24);
    constexpr uint32_t total_x = get_compile_time_arg_val(25);
    constexpr uint32_t sfree_sem_id = get_compile_time_arg_val(20);
    constexpr uint32_t hfree_sem_id = get_compile_time_arg_val(23);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(25);
    constexpr uint32_t sub = get_compile_time_arg_val(26);
    constexpr uint32_t np = get_compile_time_arg_val(27);
    constexpr uint32_t hsfree_sem_id = get_compile_time_arg_val(28);
    constexpr uint32_t h_pieces = get_compile_time_arg_val(29);
    constexpr uint32_t x_per_v = get_compile_time_arg_val(30);
    constexpr uint32_t xp = get_compile_time_arg_val(33);
    constexpr uint32_t groups = get_compile_time_arg_val(34);
#ifdef SE_HG_NU  // 16-column units: NCC / G cores x NU units = the group's K-tiles x 2
    constexpr uint32_t hg_nu = SE_HG_NU, hpc = SE_HPC;
    constexpr uint32_t h_all_tiles = (ncc / groups) * hg_nu * mt / 2;
    static_assert(((ncc / groups) * hg_nu) % 2 == 0);
#else
    constexpr uint32_t hpc = np;                                // h tiles a row
    constexpr uint32_t h_all_tiles = (ncc / groups) * np * mt;  // the group's h: NCC / G cores x NP K-tiles, MT rows
#endif
    constexpr uint32_t half_bytes = h_all_tiles * h_tile_bytes;
    constexpr uint32_t h_piece_bytes = half_bytes / h_pieces;
    static_assert(hbuf >= 1 && hbuf <= 4);

    const uint32_t credit_sem_id = get_arg_val<uint32_t>(1);
    const uint32_t cg = get_arg_val<uint32_t>(2);
    const uint32_t nh = get_arg_val<uint32_t>(3);
    const uint32_t cxy = get_arg_val<uint32_t>(5);
    const uint32_t gath_sem_id = get_arg_val<uint32_t>(6);
    const uint32_t x_ring = get_arg_val<uint32_t>(7);
    const uint32_t pxy = get_arg_val<uint32_t>(9), sxy = get_arg_val<uint32_t>(10);
    const bool is_coord = get_arg_val<uint32_t>(8) != 0;
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    // the landing slots granted so far (RT 1: the semaphore counting them, zero at launch): the weight reads
    // (se_gu_read.hpp) read into granted slots only
    volatile tt_l1_ptr uint32_t* granted_word = sem(credit_sem_id);
    auto grant = [&](uint32_t n) { *granted_word = *granted_word + n; };
    const uint64_t done_noc = get_noc_addr(cxy >> 16, cxy & 0xFFFF, get_semaphore(done_sem_id));
    volatile tt_l1_ptr uint32_t* gath_sem = sem(gath_sem_id);
    const uint32_t peer0 = 21 + nh;
    // Dynamic counts: the active experts' weights and sub-blocks only (CT 4 is then blocks per expert x NUM_E, CT 5
    // unused); RT 21 + NH.. are the se_dyn.hpp args, CB 7 this RISC's scratch; the compute gets them in CB 6.
    constexpr uint32_t num_e_dyn = get_compile_time_arg_val(35);
    SeDyn dyn;
    se_dyn_load<num_e_dyn>(dyn, peer0, get_write_ptr(tt::CBIndex::c_7), mt * 32);
    se_dyn_publish(dyn, tt::CBIndex::c_6);
    const uint32_t total_w = dyn.n_load * (total_w_ct / num_e_dyn);  // gate/up loads
    const uint32_t num_v = dyn.num_v;
    const bool is_b = false;  // no pairs: every core runs its own ring (RT 14 is then the M-group, SE_GU_ONLY)
    const uint32_t partner = get_arg_val<uint32_t>(15);
    const uint64_t partner_avail =
        get_noc_addr(partner >> 16, partner & 0xFFFF, get_semaphore(get_arg_val<uint32_t>(16)));
    volatile tt_l1_ptr uint32_t* bcopy = sem(get_arg_val<uint32_t>(17));
    constexpr bool no_partner = true;
    const uint32_t hpxy = 0;  // RT 11/12 name the x relay; the gate/up cores all send h straight to the down heads
    const uint64_t hpred_free = get_noc_addr(hpxy >> 16, hpxy & 0xFFFF, get_semaphore(get_arg_val<uint32_t>(12)));
    const uint32_t h_all_addr = get_arg_val<uint32_t>(13);
    volatile tt_l1_ptr uint32_t* hsfree = sem(hsfree_sem_id);
    const uint64_t succ_harr = get_noc_addr(sxy >> 16, sxy & 0xFFFF, get_semaphore(harr_sem_id));
    const uint64_t succ_hall = get_noc_addr(sxy >> 16, sxy & 0xFFFF, h_all_addr);
    uint32_t h_cons = 0, h_fwd = 0, h_iss = 0, h_rep = 0;
    const bool h_head = hpxy == 0;
    constexpr uint32_t group_ncc = ncc / 2;
    constexpr uint32_t link_depth = 3;
    auto write_trid = [](uint32_t src, uint64_t dst, uint32_t bytes, uint32_t trid) {
        for (uint32_t o = 0; o < bytes; o += NOC_MAX_BURST_SIZE) {
            const uint32_t n = bytes - o < NOC_MAX_BURST_SIZE ? bytes - o : NOC_MAX_BURST_SIZE;
            noc_async_write_one_packet_with_trid(src + o, dst + o, n, trid);
        }
    };
    volatile tt_l1_ptr uint32_t* sfree = sem(sfree_sem_id);
    volatile tt_l1_ptr uint32_t* hfree = sem(hfree_sem_id);
    const uint64_t pred_sfree = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(get_arg_val<uint32_t>(19)));
    // Up to two x successors (a tree keeps the delivery skew across the group low); each has its own free word here.
    const uint32_t s_xy[2] = {sxy, get_arg_val<uint32_t>(18)};
    volatile tt_l1_ptr uint32_t* s_free[2] = {sem(sfree_sem_id), sem(get_arg_val<uint32_t>(20))};
    uint64_t s_ring[2], s_xarr[2];
    for (uint32_t k = 0; k < 2; ++k) {
        s_ring[k] = get_noc_addr(s_xy[k] >> 16, s_xy[k] & 0xFFFF, x_ring);
        s_xarr[k] = get_noc_addr(s_xy[k] >> 16, s_xy[k] & 0xFFFF, get_semaphore(xarr_sem_id));
    }
    const uint32_t nsucc = s_xy[1] ? 2 : (s_xy[0] ? 1 : 0);
    volatile tt_l1_ptr uint32_t* data_sem = sem(data_sem_id);
    volatile tt_l1_ptr uint32_t* done_sem = sem(done_sem_id);
    volatile tt_l1_ptr uint32_t* go_sem = sem(go_sem_id);
    volatile tt_l1_ptr uint32_t* xarr_sem = sem(xarr_sem_id);
    volatile tt_l1_ptr uint32_t* harr_sem[2] = {sem(harr_sem_id), sem(harr1_sem_id)};

    // the gate/up weights read by this core (se_gu_read.hpp; RT after the se_dyn.hpp args): the loads list handed to
    // the NCRISC (mode 1) in its half of CB 7, or read from here (mode 2, NOC1)
    const uint32_t ga = peer0 + SE_DYN_NARGS;
    const uint32_t gmode = get_arg_val<uint32_t>(ga);
    SeGuRead gr;
    bool g_done = true;
    {
        volatile tt_l1_ptr uint32_t* mb =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_7) + 2 * SE_DYN_HALF);
        mb[0] = dyn.n_load;
        for (uint32_t l = 0; l < dyn.n_load; ++l) {
            mb[1 + l] = dyn.load_eid[l] | (uint32_t(dyn.region[l]) << 8);
        }
        if (gmode == 2) {
            gr.init(ga, SE_GU_BLK, SE_GU_PER_E, SE_GU_DEPTH, dyn.n_load, mb + 1, granted_word, data_sem);
            g_done = gr.total == 0;
        } else if (gmode == 1) {
            *sem(SE_GU_MBOX_SEM) = 1;
        }
    }
    if (!is_b) {
        grant(w_slots);  // every landing slot starts free
    }
    uint32_t granted = w_slots, pushed = 0;
    uint32_t x_pub = 0, x_cons = 0, x_rep = 0;
    uint32_t x_iss[2] = {0, 0}, x_fwd[2] = {0, 0};
    uint32_t h_sent = 0, h_pending = 0, h_pub = 0;
    uint32_t out_done = 0, out_popped = 0, go_sent = 0;

    const uint32_t my_group = get_arg_val<uint32_t>(14);
    constexpr bool all_x = false;
    while (h_sent < num_v || (nsucc > 0 && x_fwd[0] < num_v * x_per_v * xp) ||
           (nsucc > 1 && x_fwd[1] < num_v * x_per_v * xp) || (!is_b && pushed < total_w) ||
           (all_x && x_pub < num_v * x_per_v)) {
        invalidate_l1_cache();
        while (!is_b && granted < total_w && cb_pages_reservable_at_back(in1_cb, (granted - pushed + 1) * slot) &&
               (no_partner || *bcopy >= granted + 1 - w_slots)) {
            grant(1);
            ++granted;
        }
        if (!g_done) {
            g_done = gr.step();
        }
        while (!is_b && *data_sem > pushed) {
            cb_push_back(in1_cb, slot);
            ++pushed;
            if (!no_partner) {
                noc_semaphore_inc(partner_avail, 1);  // the forwarder's write is acknowledged before DATA moves
            }
        }
        const uint32_t arrived_p = *xarr_sem;     // pieces
        const uint32_t arrived = arrived_p / xp;  // whole blocks
        while (arrived > x_pub) {
            cb_push_back(x_cb, x_blk);
            ++x_pub;
        }
        if (x_cons < x_pub && cb_pages_reservable_at_back(x_cb, (x_slots - (x_pub - x_cons) + 1) * x_blk)) {
            ++x_cons;  // compute has consumed one more block
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SE_X_FWD_NC_CONS)) = x_cons;
        // Chain links keep up to LINK_DEPTH blocks in flight, each tagged with its own NoC transaction id and completed
        // in order (write acknowledged -> successor's counter); one block per round trip only reached ~13 GB/s.
        for (uint32_t k = 0; k < nsucc; ++k) {
            // x_iss / x_fwd count pieces: each piece is forwarded as soon as it has arrived (cut-through), once the
            // successor has freed the block's slot.
            const uint32_t trid0 = 1 + 8 * k;  // successor 0: ids 1..6, successor 1: ids 9..14
            constexpr uint32_t x_depth = SE_X_FWD_DEPTH;
            constexpr uint32_t piece = x_bytes / xp;
            while (x_iss[k] < arrived_p && x_iss[k] - x_fwd[k] < x_depth &&
                   (x_iss[k] / xp < x_slots || *s_free[k] >= x_iss[k] / xp + 1 - x_slots)) {
                const uint32_t off = ((x_iss[k] / xp) % x_slots) * x_bytes + (x_iss[k] % xp) * piece;
                write_trid(x_ring + off, s_ring[k] + off, piece, trid0 + x_iss[k] % x_depth);
                ++x_iss[k];
            }
            // (the write command buffer drained first: an id's outstanding count only covers requests the NIU has
            // taken - right after issuing, the last write can still sit in the command buffer and the id reads "done")
            while (x_fwd[k] < x_iss[k] && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            }
            while (x_fwd[k] < x_iss[k] &&
                   ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid0 + x_fwd[k] % x_depth)) {
                noc_semaphore_inc(s_xarr[k], 1);
                ++x_fwd[k];
            }
        }
        uint32_t freed = x_cons;
        for (uint32_t k = 0; k < nsucc; ++k) {
            freed = x_fwd[k] / xp < freed ? x_fwd[k] / xp : freed;
        }
        // (the freed reports to the predecessor / relay: the NCRISC forwarder's, se5_xfwd.cpp, from x_cons)
        if (!h_pending && h_sent < num_v && *go_sem + hbuf >= h_sent + 1 &&
            cb_pages_available_at_front(h_local_cb, mt * hpc)) {
            const uint32_t src = get_read_ptr(h_local_cb);
            const uint32_t dst = ((h_sent % hbuf) * groups + my_group) * half_bytes;  // down h_all: [HBUF][group]
            for (uint32_t hd = 0; hd < nh; ++hd) {
                const uint32_t hxy = get_arg_val<uint32_t>(21 + hd);
                const uint64_t bc_h = get_noc_addr(hxy >> 16, hxy & 0xFFFF, h_all_addr);
#ifdef SE_HG_NU
                // 16-column units cg NU..: h tiles from K-tile cg NU / 2, the first one's right half only when cg NU is
                // odd (lead), the last one's left half only when (cg + 1) NU is odd (tail); the full ones are one run,
                // the halves go face by face (bf16: faces 512 B; bfp8: 16 B exponents of each face, then 256 B data)
                {
                    static_assert(h_tile_bytes == 2048 || h_tile_bytes == 1088, "SE_HG_NU: bf16 or bfp8 h tiles");
                    const uint32_t u0 = cg * hg_nu, k0 = u0 / 2, lead = u0 & 1, tail = (u0 + hg_nu) & 1;
                    const uint32_t j0 = lead, j1 = hpc - tail;
                    const uint64_t base = bc_h + dst + k0 * mt * h_tile_bytes;
                    if (j1 > j0) {
                        SE_H_WRITE(
                            src + j0 * mt * h_tile_bytes,
                            base + j0 * mt * h_tile_bytes,
                            (j1 - j0) * mt * h_tile_bytes,
                            noc_index,
                            h_vc);
                    }
                    auto half = [&](uint32_t j, uint32_t right) {  // tile j's left (faces 0 / 2) or right (1 / 3) half
                        for (uint32_t m = 0; m < mt; ++m) {
                            const uint32_t s = src + (j * mt + m) * h_tile_bytes;
                            const uint64_t d = base + (j * mt + m) * h_tile_bytes;
                            for (uint32_t f = right; f < 4; f += 2) {
                                if constexpr (h_tile_bytes == 2048) {
                                    SE_H_WRITE(s + f * 512, d + f * 512, 512, noc_index, h_vc);
                                } else {
                                    SE_H_WRITE(s + f * 16, d + f * 16, 16, noc_index, h_vc);
                                    SE_H_WRITE(s + 64 + f * 256, d + 64 + f * 256, 256, noc_index, h_vc);
                                }
                            }
                        }
                    };
                    if (lead) {
                        half(0, 1);
                    }
                    if (tail) {
                        half(hpc - 1, 0);
                    }
                }
#else
                // Down cores keep h K-tile-major ([K][rows] per group): this core's NP K-tiles are one run.
                SE_H_WRITE(src, bc_h + dst + cg * np * mt * h_tile_bytes, np * mt * h_tile_bytes, noc_index, h_vc);
#endif
            }
            h_pending = 1;
        }
        if (h_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            // Slices of h(v) are counted per down-core buffer v % HBUF (CT 31.. give buffers 1, 2).
            const uint32_t gid = h_sent % hbuf == 0   ? gath_sem_id
                                 : h_sent % hbuf == 1 ? get_compile_time_arg_val(31)
                                 : h_sent % hbuf == 2 ? get_compile_time_arg_val(32)
                                                      : 0;
            for (uint32_t hd = 0; hd < nh; ++hd) {
                const uint32_t hxy = get_arg_val<uint32_t>(21 + hd);
                noc_semaphore_inc(get_noc_addr(hxy >> 16, hxy & 0xFFFF, get_semaphore(gid)), 1);
            }
            cb_pop_front(h_local_cb, mt * hpc);
            ++h_sent;
            h_pending = 0;
        }
    }
    // x_cons follows compute at most a block a loop pass, so the loop can end (all h sent) with it short of the call's
    // blocks; the end count must be the total (the forwarder ends at fwd == (end - 1) x pieces: a short count left it
    // spinning past it - a 1-in-13k whole-model hang - or would end it before the successor has every block)
    for (const uint32_t x_total = num_v * x_per_v; x_cons < x_total;) {
        invalidate_l1_cache();
        if (cb_pages_reservable_at_back(x_cb, (x_slots - (x_pub - x_cons) + 1) * x_blk)) {
            ++x_cons;
        }
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SE_X_FWD_NC_CONS)) = x_cons;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SE_X_FWD_NC_END)) = x_cons + 1;
    if (gmode == 2) {
        gr.finish();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    // the write command buffer's packet tag back to transaction ID 0: the next program's plain writes (implicit ID
    // 0) would otherwise carry this kernel's last ID and their barriers / flush checks hang (watcher: "write-capable
    // NOC packet tags must be zero")
    noc_async_write_set_trid(0);
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
