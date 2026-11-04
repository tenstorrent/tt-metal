// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Port sender: injects this port's rows (one fabric connection, multi-hop 2D unicasts
// straight into the home accumulator's landing ring), returns credits of the rows its chip's accumulators consumed
// from the sources whose first hop back is this port, and forwards those sources their rings' layouts for the call.
// Defines: REC_FUSED (a port that is also an
// accumulator, acc_on_ports: this RISC also runs the port's L1 hand-off and the accumulator's output writes),
// REC_ACC_LUT (per-chip roles: a home chip's accumulator cells from a DRAM table).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "combine_zero_gate.hpp"
using namespace tt::tt_fabric;

constexpr uint32_t cb_inj = get_compile_time_arg_val(0);
constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
constexpr uint32_t SLOT = get_compile_time_arg_val(2);
constexpr uint32_t PSL = get_compile_time_arg_val(3);  // landing pool slots at each accumulator (carved per call)
constexpr uint32_t A = get_compile_time_arg_val(4);
constexpr uint32_t AP = (A + 3) / 4 * 4;  // credit words per home (16-byte aligned arrays: they travel as NoC writes)
constexpr uint32_t N = get_compile_time_arg_val(5);
constexpr uint32_t L = get_compile_time_arg_val(6);
constexpr uint32_t ISLOT = get_compile_time_arg_val(7);
constexpr uint32_t CRED = get_compile_time_arg_val(8), USED = get_compile_time_arg_val(9),
                   RET = get_compile_time_arg_val(10), MISC = get_compile_time_arg_val(11),
                   HDRS = get_compile_time_arg_val(12), RINGS = get_compile_time_arg_val(13),
                   LAYR = get_compile_time_arg_val(14), LAYO = get_compile_time_arg_val(15),
                   U0 = get_compile_time_arg_val(16);
constexpr uint32_t CRED_THR = get_compile_time_arg_val(17);  // rows consumed from a source before its credits go back
constexpr uint32_t CRED_IDLE = get_compile_time_arg_val(18),
                   CRED_AGE = get_compile_time_arg_val(19);  // (idle: CRED_IDLE rows, or CRED_AGE cycles old)
constexpr uint32_t LASTPB = row_bytes;                       // the row's 32 B trailer (one fabric packet a row)
constexpr uint32_t NH = get_compile_time_arg_val(20);  // rotating row headers: flush (and pop the sent pages) every NH
constexpr uint32_t HSZ = get_compile_time_arg_val(21);
static_assert(sizeof(PACKET_HEADER_TYPE) <= HSZ, "header stride");
constexpr uint32_t HW = sizeof(PACKET_HEADER_TYPE) / 4;
// barrier: every port core counts one atomic increment from every other chip in a persistent word (bar[0], cumulative;
// bar[1] its calls) and sends nothing remote before all N - 1 of this call are in; each chip's port facing a chip
// increments that chip's ports. A chip's ports increment only after its zeroing (combine_zero_gate.hpp).
constexpr uint32_t MRTOT = get_compile_time_arg_val(22), MRDONE = get_compile_time_arg_val(23);
constexpr uint32_t BAR_SLOT = get_compile_time_arg_val(24);
constexpr uint32_t ZERO = get_compile_time_arg_val(25);  // 1: this kernel zeroes the core's state [0, RINGS)
constexpr uint32_t WORDS_OFF = get_compile_time_arg_val(26);
constexpr uint32_t LUT_OFF = get_compile_time_arg_val(27);
constexpr uint32_t LUT_BYTES = get_compile_time_arg_val(28);
#ifdef REC_FUSED
constexpr uint32_t cb_out = get_compile_time_arg_val(29);
constexpr uint32_t NT_ = get_compile_time_arg_val(30);
constexpr auto o_args = TensorAccessorArgs<31>();
#endif

void kernel_main() {
    size_t a = 0;
    const uint32_t area = get_arg_val<uint32_t>(a++);  // the arena (same address on every core and chip)
    const uint16_t mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t me = get_arg_val<uint32_t>(a++), lnk = get_arg_val<uint32_t>(a++);
    const uint32_t bar_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gate_x = get_arg_val<uint32_t>(a++), gate_y = get_arg_val<uint32_t>(a++);
    const uint32_t call = rec::next_call(bar_addr, BAR_SLOT);
    if constexpr (ZERO) {
        rec::zero_l1(area, RINGS);
        rec::count_in(gate_x, gate_y, bar_addr);
    } else {
        rec::wait_released(bar_addr, call);  // (the accumulator's reader on this core zeroes it)
    }
    uint32_t n_inj = 0, n_ret = 0;
    {  // (the device plan: this port's counts at its id)
        const uint32_t plan = get_arg_val<uint32_t>(a++), pid = get_arg_val<uint32_t>(a++);
        const uint32_t sc = (area + MISC + 832 + 63) & ~63u;
        const InterleavedAddrGen<true> pg = {.bank_base_address = plan, .page_size = 4096};
        noc_async_read(get_noc_addr(0, pg), sc, 64);  // n_inj[8], n_ret[8]
        noc_async_read_barrier();
        n_inj = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[pid];
        n_ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc)[8 + pid];
    }
    uint16_t chip_id[N];
    for (uint32_t i = 0; i < N; ++i) {
        chip_id[i] = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    }
#ifdef REC_ACC_LUT
    // (per-chip roles: a home chip's accumulator cells differ - its table entry home * A + o, x | y << 8, copied here)
    const uint32_t lut_dram = get_arg_val<uint32_t>(a++), lut_l1 = area + LUT_OFF;
    {
        const InterleavedAddrGen<true> lg = {.bank_base_address = lut_dram, .page_size = LUT_BYTES};
        noc_async_read(get_noc_addr(0, lg), lut_l1, LUT_BYTES);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint16_t* acc_lut = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lut_l1);
#define ACC_X(h, o) (acc_lut[(h) * A + (o)] & 0xFFu)
#define ACC_Y(h, o) (acc_lut[(h) * A + (o)] >> 8)
#else
    uint32_t ax[A], ay[A];
    for (uint32_t i = 0; i < A; ++i) {
        ax[i] = get_arg_val<uint32_t>(a++);
        ay[i] = get_arg_val<uint32_t>(a++);
    }
#define ACC_X(h, o) ax[o]
#define ACC_Y(h, o) ay[o]
#endif
    // per source chip s: the port on s that feeds my chip on my link (where s's credits go); 0 if not mine to return
    uint32_t sx[N], sy[N];
    for (uint32_t i = 0; i < N; ++i) {
        sx[i] = get_arg_val<uint32_t>(a++);
        sy[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t hc_addr = get_arg_val<uint32_t>(a++), hc_page = get_arg_val<uint32_t>(a++);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    // headers per (destination chip, rotation), routed once; a send only sets the NoC command
    auto hdr_at = [&](uint32_t chip, uint32_t rot) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(area + HDRS + (chip * NH + rot) * HSZ);
    };
    // headers, cached in DRAM: a 2D route costs ~1.7 us (decode_route_to_buffer) - N x NH of them were ~90 us of every
    // call's start. The first call routes one header per destination, copies it to the other rotations (same route: a
    // send rewrites only the NoC command) and stores the block with a key in this port's DRAM page; later calls read
    // the block back
    {
        constexpr uint32_t HBLK = N * NH * HSZ;
        const InterleavedAddrGen<true> hcg = {.bank_base_address = hc_addr, .page_size = HBLK + 64};
        volatile tt_l1_ptr uint32_t* kw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC) + 224;
        const uint32_t key = 0x48C00000u ^ (me << 8) ^ (lnk << 16) ^ N;
        noc_async_read(get_noc_addr(hc_page, hcg, HBLK), reinterpret_cast<uint32_t>(kw), 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (kw[0] == key) {
            noc_async_read(get_noc_addr(hc_page, hcg), area + HDRS, HBLK);
            noc_async_read_barrier();
        } else {
            for (uint32_t i = 0; i < N; ++i) {
                if (i == me) {
                    continue;
                }
                auto* h0 = hdr_at(i, 0);
                (void)fabric_set_unicast_route(h0, chip_id[i], mesh);
                const volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(h0);
                for (uint32_t r = 1; r < NH; ++r) {
                    volatile tt_l1_ptr uint32_t* hr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_at(i, r));
                    for (uint32_t w = 0; w < HW; ++w) {
                        hr[w] = s0[w];
                    }
                }
            }
            noc_async_write(area + HDRS, get_noc_addr(hc_page, hcg), HBLK);
            kw[0] = key;
            noc_async_write(reinterpret_cast<uint32_t>(kw), get_noc_addr(hc_page, hcg, HBLK), 64);
            noc_async_write_barrier();
        }
    }
    const uint32_t cstage = area + HDRS + N * NH * HSZ;  // credit snapshots, one per rotation
    if constexpr (ZERO) {
        rec::wait_released(bar_addr, call);
    }
    conn.open();
#ifdef REC_FUSED
    const uint32_t hW = get_arg_val<uint32_t>(a++), hdid = get_arg_val<uint32_t>(a++), hQ = get_arg_val<uint32_t>(a++);
    const uint32_t hwords = area + WORDS_OFF, o_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_tok = get_arg_val<uint32_t>(a++), acc_me = get_arg_val<uint32_t>(a++);
    uint32_t hwx[32], hwy[32];
    for (uint32_t w = 0; w < hW; ++w) {
        hwx[w] = get_arg_val<uint32_t>(a++);
        hwy[w] = get_arg_val<uint32_t>(a++);
    }
    const auto ob = TensorAccessor(o_args, o_addr, row_bytes);
    volatile tt_l1_ptr uint32_t* hland = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC) + 128;
    const uint32_t hbase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC)[101];
    uint32_t hgranted = 0, hpushed = 0, wr_i = 0;
    // the hand-off receiver's step (as rf_port_reader): grant ring slots, push rows every writer has landed
    auto hand_off = [&]() {
        if (hpushed >= n_inj) {
            return;
        }
        const uint32_t want = hpushed + hQ < n_inj ? hpushed + hQ : n_inj;
        uint32_t g = hgranted;
        while (g < want && cb_pages_reservable_at_back(cb_inj, g - hpushed + 1)) {
            ++g;
        }
        if (g > hgranted && (g - hgranted >= 8 || g == want)) {
            for (uint32_t w = 0; w < hW; ++w) {
                noc_semaphore_inc(get_noc_addr(hwx[w], hwy[w], hwords + 4 * hdid), g - hgranted);
            }
            hgranted = g;
        }
        uint32_t m = 0xFFFFFFFF;
        for (uint32_t w = 0; w < hW; ++w) {
            const uint32_t v = hland[w] - hbase;
            m = v < m ? v : m;
        }
        while (hpushed < m && hpushed < hgranted) {
            cb_push_back(cb_inj, 1);
            ++hpushed;
        }
    };
    // the accumulator writer's step (as rf_acc_writer): one finished token row to the output
    auto write_out = [&]() {
        if (wr_i < n_tok && cb_pages_available_at_front(cb_out, NT_)) {
            noc_async_write(get_read_ptr(cb_out), ob.get_noc_addr(wr_i * A + acc_me), row_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, NT_);
            ++wr_i;
        }
    };
    auto fused_more = [&]() { return hpushed < n_inj || wr_i < n_tok; };
#else
    auto hand_off = [&]() {};
    auto write_out = [&]() {};
    auto fused_more = [&]() { return false; };
#endif
    {
        const uint32_t n_bt = get_arg_val<uint32_t>(a++);
        for (uint32_t i = 0; i < n_bt; ++i) {
            const uint32_t ch = get_arg_val<uint32_t>(a++), bx = get_arg_val<uint32_t>(a++),
                           by = get_arg_val<uint32_t>(a++);
            auto* h = hdr_at(ch, 0);
            h->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(bx, by, bar_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(h), sizeof(PACKET_HEADER_TYPE));
        }
        volatile tt_l1_ptr uint32_t* bw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bar_addr);
        const uint32_t tgt = (bw[1] + 1) * (N - 1);
        do {
            invalidate_l1_cache();
        } while (static_cast<int32_t>(bw[0] - tgt) < 0);
        bw[1] = bw[1] + 1;
    }

    volatile tt_l1_ptr uint32_t* cred = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + CRED);
    volatile tt_l1_ptr uint32_t* used = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + USED);
    volatile tt_l1_ptr uint32_t* ret = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + RET);
    volatile tt_l1_ptr uint32_t* misc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + MISC);
    volatile tt_l1_ptr uint32_t* rtot = misc + MRTOT;    // rows consumed per source chip (accumulators increment)
    volatile tt_l1_ptr uint32_t* rdone = misc + MRDONE;  // rows covered by the last snapshot sent per source chip
    const uint32_t ret0 = misc[1];
    uint32_t returned = ret0;
    // the landing pools' rings this call (layout words: tag << 24 | base << 12 | depth, tag = this call's 1..255): a
    // home's accumulators carve their pools by this call's demand and publish each source's ring to the home port that
    // returns that source's credits (LAYO), which forwards them to the source's port (LAYR). A row waits for its ring's
    // layout; its slot is base + (rows sent on the ring this call) % depth. Rings are empty at a call's start (every
    // call drains), so the cumulative credit counts still hold: room = depth + cred - used.
    volatile tt_l1_ptr uint32_t* lay_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + LAYR);
    volatile tt_l1_ptr uint32_t* lay_o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + LAYO);
    volatile tt_l1_ptr uint32_t* u0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area + U0);
    const uint32_t calls_ = misc[104] + 1;
    misc[104] = calls_;
    const uint32_t tag = (calls_ - 1) % 255 + 1;
    uint32_t lay_pend = 0;  // (sources whose layouts this port still forwards: bit s)
    for (uint32_t ci = 0; ci < N * AP; ++ci) {
        u0[ci] = used[ci];
    }
    for (uint32_t s = 0; s < N; ++s) {
        if (sx[s]) {
            lay_pend |= 1u << s;
        }
    }
    uint32_t inj_done = 0, sent = 0, pend_inj = 0, unflushed = 0;

    auto flush = [&]() {
        noc_async_writes_flushed();
        for (; pend_inj; --pend_inj) {
            cb_pop_front(cb_inj, 1);
        }
        unflushed = 0;
    };
    // credits: a snapshot of the cumulative consumed counts of source s's rings (one per accumulator) written over
    // s's credit array for my chip: one small packet returns any number of rows (idle sends: at least CRED_IDLE rows,
    // or rows waiting CRED_AGE cycles - fewer, fuller snapshots)
    uint32_t tfirst[N];
    for (uint32_t s = 0; s < N; ++s) {
        tfirst[s] = 0;
    }
    auto give_back = [&](bool all) {
        const uint32_t now = get_timestamp_32b();
        for (uint32_t s = 0; s < N; ++s) {
            if (sx[s] == 0) {
                continue;
            }
            const uint32_t pend = rtot[s] - rdone[s];
            if (pend == 0) {
                continue;
            }
            if (!all && pend < CRED_THR) {
                continue;
            }
            if constexpr (CRED_IDLE > 1) {
                if (all && pend < CRED_IDLE) {
                    if (tfirst[s] == 0) {
                        tfirst[s] = now | 1;
                    }
                    if (now - tfirst[s] < CRED_AGE) {
                        continue;
                    }
                }
                tfirst[s] = 0;
            }
            if (unflushed >= NH) {
                flush();
            }
            const uint32_t rot = sent % NH;
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cstage + rot * 4 * AP);
            uint32_t cov = 0;
            for (uint32_t o = 0; o < A; ++o) {
                const uint32_t v = ret[s * AP + o];
                st[o] = v;
                cov += v;
            }
            auto* h = hdr_at(s, rot);
            h->to_noc_unicast_write(
                NocUnicastCommandHeader{get_noc_addr(sx[s], sy[s], area + CRED + 4 * me * AP)}, 4 * AP);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(cstage + rot * 4 * AP, 4 * AP, reinterpret_cast<uint32_t>(h));
            ++sent;
            ++unflushed;
            returned += cov - rdone[s];
            rdone[s] = cov;
        }
    };
    // forward each source's ring layouts once every accumulator that returns its credits here has published them
    auto send_layouts = [&]() {
        for (uint32_t s = 0; s < N && lay_pend; ++s) {
            if (!(lay_pend >> s & 1)) {
                continue;
            }
            bool ok = true;
            for (uint32_t o = 0; o < A && ok; ++o) {
                if ((o + s + me) % L == lnk) {
                    ok = (lay_o[s * AP + o] >> 24) == tag;
                }
            }
            if (!ok) {
                continue;
            }
            if (unflushed >= NH) {
                flush();
            }
            const uint32_t rot = sent % NH;
            volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cstage + rot * 4 * AP);
            for (uint32_t o = 0; o < A; ++o) {
                st[o] = lay_o[s * AP + o];
            }
            auto* h = hdr_at(s, rot);
            h->to_noc_unicast_write(
                NocUnicastCommandHeader{get_noc_addr(sx[s], sy[s], area + LAYR + 4 * me * AP)}, 4 * AP);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(cstage + rot * 4 * AP, 4 * AP, reinterpret_cast<uint32_t>(h));
            ++sent;
            ++unflushed;
            lay_pend &= ~(1u << s);
        }
    };
    auto any_ret = [&]() {
        uint32_t t = 0;
        for (uint32_t s = 0; s < N; ++s) {
            t += rtot[s];
        }
        return t;
    };
    // one row (one packet with its trailer, the header word at LASTPB) from L1 src to its home's ring; false: not yet
    // (the ring's layout has not arrived, or the ring is full)
    auto send_row = [&](uint32_t src) -> bool {
        const uint32_t hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + LASTPB)[0];
        const uint32_t home = hdr & 0xFF, t = (hdr >> 8) & 0xFFFF, o = t % A, ci = home * AP + o;
        const uint32_t u = used[ci], lw = lay_r[ci], dep = lw & 0xFFF;
        if ((lw >> 24) != tag) {
            return false;
        }
        if (dep + cred[ci] - u < 1) {
            return false;
        }
        const uint32_t dst = area + RINGS + (((lw >> 12) & 0xFFF) + (u - u0[ci]) % dep) * SLOT;  // (my ring)
        constexpr uint32_t pb = row_bytes + 32;
        auto* h = hdr_at(home, sent % NH);
        h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(ACC_X(home, o), ACC_Y(home, o), dst)}, pb);
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, pb, reinterpret_cast<uint32_t>(h));
        ++sent;
        ++unflushed;
        used[ci] = u + 1;
        return true;
    };

    while (inj_done < n_inj || returned - ret0 < n_ret || fused_more() || lay_pend) {
        bool did = false;
        invalidate_l1_cache();
        hand_off();
        write_out();
        const uint32_t tot = any_ret();
        const bool pending = tot != returned;
        if (pending && unflushed < NH) {
            give_back(false);
        }
        if (lay_pend) {
            send_layouts();
        }
        while (unflushed < NH && inj_done < n_inj && cb_pages_available_at_front(cb_inj, pend_inj + 1)) {
            uint32_t src = get_read_ptr(cb_inj) + pend_inj * ISLOT;
            const uint32_t ilim = get_local_cb_interface(cb_inj).fifo_limit;
            if (src >= ilim) {
                src -= get_local_cb_interface(cb_inj).fifo_size;
            }
            if (!send_row(src)) {
                break;  // (its ring's layout not here yet, or the ring full)
            }
            ++pend_inj;
            ++inj_done;
            did = true;
        }
        if (unflushed >= NH || (!did && unflushed)) {
            flush();
            did = true;
        }
        if (!did && pending) {
            give_back(true);
        }
    }
    flush();
    misc[1] = returned;
#ifdef REC_FUSED
    misc[101] = hbase + n_inj;
    noc_async_write_barrier();
#endif
    // drain: every row this port injected is consumed (its credit snapshot landed), so nothing of this call reaches
    // this core after the program ends (the arena is per call: its L1 is another op's after this)
    for (bool open_ = true; open_;) {
        invalidate_l1_cache();
        open_ = false;
        for (uint32_t ci = 0; ci < N * AP; ++ci) {
            open_ |= cred[ci] != used[ci];
        }
    }
    conn.close();
    noc_async_atomic_barrier();
}
