// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused head's kernel contract: for every kernel the program launches (one program a chip), where it runs, its
// compile-time args (index enums below), the defines the host must set (and the host model's value behind each), its
// runtime / common runtime arg layout. Host-side only (the program factory includes it); the kernels read the same
// indices literally. The host model (head_program.*) assembles the x all-gather's, the router's, the routing
// exchange's and the shared expert's pieces (XGather3, RouterStream, RoutingStream, SeEngine3; + the kernel_lib mcast
// attachments).
//
// Kernel files (all under ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/):
//   dataflow/xg_reader.cpp       port cores        RISCV_0, NOC1    x all-gather reader
//   dataflow/xg_sender.cpp       port cores        RISCV_1, NOC0    x all-gather sender (+ LStream, RA_SEND)
//   dataflow/xg_watcher.cpp      watcher core      RISCV_1, NOC1    x all-gather watcher (+ RA_SEND under W_RA0)
//   dataflow/ra_y_send.cpp       watcher core      RISCV_0, default routing exchange's Y send (first Y direction only)
//   dataflow/se_brisc.cpp        matmul cores      RISCV_0, NOC0    shared expert BRISC
//   dataflow/se_ncrisc.cpp       matmul cores      RISCV_1, NOC1    shared expert NCRISC
//   compute/se_compute.cpp       matmul cores      compute          shared expert compute (one kernel a column group)
//   dataflow/router_ncrisc.cpp   router cores      RISCV_1, NOC0    router NCRISC (+ router rows, ra::plan)
//   compute/router_compute.cpp   router cores      compute          router compute (+ router rows, rw_ route words)
//   dataflow/router_brisc.cpp    router cores      RISCV_0, NOC1    router BRISC (+ router rows, ra::plan)
//
// Shared pieces are headers; the planner's and the route words' parameters are the RAP_* / RW_* defines; the GROW
// macro is se_defs.hpp; addresses are common runtime args (se_defs.hpp, router_addr.hpp), never defines.
//
// Define values: plain tokens (numbers, with a `u` suffix where marked `u` below; true / false); the list
// defines (RW_*_INIT) are comma-separated without spaces. Every define below is structural (no buffer address), so
// the program hash may include them.

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::moe_block::fused_head::kernels {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/";

// ---------------------------------------------------------------------------------------------------------------------
// x_gather3 port kernels (xg_reader, xg_sender): compile-time args (XGather3.pieces `ct`); the reader appends x's
// TensorAccessorArgs at XG_CT_COUNT (= 14).
enum XgCt : uint32_t {
    XG_CT_KB = 0,  // e.KB, the engine's K step
    XG_CT_MT,      // e.MT
    XG_CT_SPS,     // self.sps, K steps a shard
    XG_CT_RP,      // self.rp, tiles a packet
    XG_CT_CR,      // e.CR = GY MT
    XG_CT_NCH,     // e.NCH
    XG_CT_NV,      // self.nv, links
    XG_CT_GROUP,   // self.group, packets a group
    XG_CT_INC,     // INC_EVERY = 8
    XG_CT_GX,      // e.GX
    XG_CT_GY,      // e.GY
    XG_CT_S0,      // e.S0
    XG_CT_S1,      // e.S1
    XG_CT_G,       // self.G, TP
    XG_CT_COUNT,   // reader: x TensorAccessorArgs from here
};
// Defines:
//   reader:  XG_HS (persist_l1.handoff_bytes(): 16u, 4u with persist_l1_small), LA_CHL (e.LATP * 2048 if e.NCH > 1
//            else 0, `u`), RS_NPASS (rs.plan.npass if rs.plan.pm else 1, `u`)
//   sender:  XG_HS, LS_FIRST (int(router.prio_first)), LA_CHL, RS_NPASS, and with the routing exchange the routing
//            defines (RoutingStream.defines: RA_NC_ len(router.cores), RA_NP_ router.RB, RA_N_ N, RA_M_ M, RA_EPC_ EPC,
//            RA_ME_ r C + c, RA_HBP_ header_bytes(EPC, M), RA_BB_ _al16(2 EPC), RA_BBP_ _al16(2 (EPC + 1)),
//            RA_MAXP_ fabric max payload // 64 * 64, RA_NF_ 16)
//   watcher: XG_HS; with a second Y direction (RoutingStream.watcher_tail): the routing defines + W_RA0 (the index
//            of the RASend block in its args = len of the watcher's own list)
// Runtime args (per port core; XGather3.pieces):
//   reader:  [x_addr, xown (arena + off["xown"]), v (link), arrival (sem addr), n, up_ipc] + entries (q, part, ub,
//            npk) x n + tables(coord) [G ranks, fam0 S0, fam1 S1, (nown, b1) x GX, col NoC x GX, row NoC y GY]
//            + [xloaded (sem addr), GX GY] + rs_block(rs, d, v) + la_block(rs, la, d, v)
//   sender:  [xown, v, arrival, expect_in, ready, fence] + (with a peer on this direction / link:) [peer has opposite
//            ring, rc.x, rc.y, pc.x, pc.y, peer mesh id, peer chip id, peer watcher x, y, watch counter, n]
//            + entries (q, part, npk) x n + tables(peer) + rs_block + router.port_block + la_block + lg_block
//            + routing.send_block + routing.fwd_block + fabric connection args (setup_fabric_connection);
//            (no peer:) [0] x 10 + [0] + tables(coord) + the same blocks (all empty)
//   watcher: [NCH, XREADY sem id, G - 1 (0 with xdram), nc engine cores, ncnt] + (counter addr, final) x ncnt
//            + (x, y) x nc + per rank (nreq, (counter addr, entry index, entries a chunk) x nreq)
//            [+ at W_RA0: RoutingStream._y_args of the second Y key: send_block + [remote] + [hello, peer chip,
//            peer mesh] + fabric connection args + [hellos] (unused here)]
// No common runtime args.

// ---------------------------------------------------------------------------------------------------------------------
// routing_stream Y sender (ra_y_send): no compile-time args. Defines: the routing defines (above). Runtime args
// (RoutingStream._y_args): send_block(coord, key) (RASend: [nd, nr, self, buffer, PCAP, slots addr, c0, c1, planner
// core 0 x, y, calls sem, ready sem, mesh id, fslots addr, flags] + per destination next_hop (5 words)) + [remote]
// + (remote:) [yhello addr, peer chip id, peer mesh id] + fabric connection args + [hellos: how many Y hellos land on
// this chip a call - HeadModel::y_hellos_in: 1 on a ring / R 2; a Y line: 0 at the near end, 2 at the row before the
// far end]. No common runtime args.

// ---------------------------------------------------------------------------------------------------------------------
// se_engine3 matmul-core kernels. BRISC compile-time args (`bct`), then x's TensorAccessorArgs at SE_B_CT_COUNT
// (xa_ct), then the router x's at SE_RTR_BCT0.
enum SeBriscCt : uint32_t {
    SE_B_CT_GX = 0,
    SE_B_CT_MT,
    SE_B_CT_KB,
    SE_B_CT_NSG,
    SE_B_CT_NPG,
    SE_B_CT_NCH,
    SE_B_CT_POMAX,
    SE_B_CT_S0,
    SE_B_CT_S1,
    SE_B_CT_XREADY,  // semaphore id XREADY = 0
    SE_B_CT_COUNT,   // x TensorAccessorArgs from here
};
// NCRISC compile-time args (`nct`), then w, y TensorAccessorArgs, xbuf's at SE_PREP_CT0, the router x and W_r^T at
// SE_RTR_CT0.
enum SeNcriscCt : uint32_t {
    SE_N_CT_GY = 0,
    SE_N_CT_MT,
    SE_N_CT_KB,
    SE_N_CT_NSG,
    SE_N_CT_NPG,
    SE_N_CT_NCH,
    SE_N_CT_KBD,
    SE_N_CT_NSD,
    SE_N_CT_NPD,
    SE_N_CT_NWS,  // 3
    SE_N_CT_WSLOT,
    SE_N_CT_WTB,    // W tile bytes: bf8 1088, bf4 576, bf16 2048
    SE_N_CT_WV,     // semaphore id 1
    SE_N_CT_WFREE,  // semaphore id 1 + NWS
    SE_N_CT_COUNT,  // w TensorAccessorArgs from here
};
// Compute compile-time args (one kernel a column group (wg, wd) of SeEngine3.groups).
enum SeComputeCt : uint32_t {
    SE_C_CT_MT = 0,
    SE_C_CT_KB,
    SE_C_CT_NPG,
    SE_C_CT_NCH,
    SE_C_CT_POMAX,
    SE_C_CT_KBD,
    SE_C_CT_NSD,
    SE_C_CT_NPD,
    SE_C_CT_WSLOT,
    SE_C_CT_G,    // len(so): the K shards
    SE_C_CT_SPS,  // NSG / len(so)
    SE_C_CT_GX,
    SE_C_CT_NC,  // len(common_c): the common args' count (SE_NC)
    SE_C_CT_COUNT,
};
// Defines (all three kernels; SeEngine3.pieces `nd`):
//   SE_HSL (h slots), SE_ACCP (acc_pages, `u`), SE_DT (dest_tiles() = 8);
//   SE_GROW_OWN_CR = CR / shards, SE_GROW_OWN_RT = rows / 32 / shards - only when NCH > 1 and shards > 1
//     (se_defs.hpp GROW);
//   SE_WRING (if WRING); SE_PREP_LAST (NCH > 1 and not xfree); SE_XRING (XR, `u`, if xring); SE_XFREE 21 (if xfree);
//   the down + reduce-scatter: SE_NENT (rs.G), SE_HOLD 6, SE_FINB 7, SE_ARRF 8, SE_ARRB 9, SE_FINC 10, SE_OUTF 11,
//   SE_PSYNC 12,
//     SE_FFREE 20, SE_FIN_ROWS (if _fin_rows()), SE_ZF / SE_ZB (rs.zone_tiles), SE_RS_PM (rs.npass,
//     `u`, if rs.pm);
//   with a latent: SE_LAT 13, SE_NGU, SE_LWMAX, SE_LATP;
//   the router logits (on the matmul cores): SE_RX 17, SE_RW 18, SE_RO 19, SE_KTO, SE_ETC, SE_MTR (MT NCH),
//     SE_RG, SE_RD 3, XACK_ID 128, WACK_ID 144, RXR_ID 0,
//     RWR_ID 16, SE_RDONE (if rtr_overlay), RXS_ID 32, SE_KC, SE_RTRX_L1 + XRL_ID 96 (if not xring),
//     SE_RTR_BCT0 (len(bct + xa_ct)), SE_RTR_CT0 (NCRISC accessor offset);
//     (SE_EW, the ewords address, is a common runtime arg, below)
//   the dispatch prep: SE_PU 14, SE_PO 15, SE_PREP_FULL, SE_PREP_BCT, SE_PREP_ROW (xbuf aligned page size),
//     SE_PREP_CT0, SE_LRDY 16 (with a latent);
//   compute only, first: SE_GLU (glu code: silu 1, situ 2, clamped_silu 3, swiglu_oai 4) and SE_GLU_BETA,
//     SE_GLU_LBETA, SE_GLU_LIMIT, SE_GLU_ALPHA (float literals `...f`, se_engine3.glu_defines).
// Runtime args: BRISC / NCRISC per core the lists of SeEngine3.pieces (`bl`, `nl`), with the span every core of the
// chip shares moved to the common args by _common_span (b_span = (7, 12 + GX + len(sor) + len(tail)), n_span =
// (5 + len(rtr_a) + len(wg[c]) + len(wd[c]), 7 + GY + len(sor) + len(tail))): per core [args before the span] +
// [args after it]; common args [span start, span length n, the n shared words] (or [0, 0] and the full lists per
// core), then at common index 2 + n: the ewords address (SE_EW).
// Compute: common args = wg + wd + h0 + [HT] + the RS / prep args every core of the group shares; per core the rest
// ([0] if none) - SE_NC = len(common).
constexpr uint32_t kSeEwordsCommonArgAfterSpan = 2;  // SE_EW = common[2 + common[1]]

// ---------------------------------------------------------------------------------------------------------------------
// router_stream router-core kernels: compile-time args `ct`, then per kernel the accessors:
//   NCRISC: ct + b, ix TensorAccessorArgs (at 4) + (routing) router_out, slots accessors (ra::plan's RAP_CT0 =
//           their offset); compute: ct; BRISC: ct + out accessor (at 4) + (routing) router_out, slots (RAP_CT0)
//           + (route words) the route tensor's (RT_CT0).
enum RouterCt : uint32_t {
    RT_CT_KT = 0,  // H / TP / 32
    RT_CT_ET,      // expert tiles (E / 32)
    RT_CT_G,
    RT_CT_K,
    RT_CT_COUNT,
};
// Common runtime args (router_addr.hpp) - the same on all three kernels; a padded chunk (RT_VALID, NCRISC and BRISC
// only) appends its scalars and the chip's place in main's rotated layout:
enum RouterCommonArg : uint32_t {
    RT_CRTA_ARENA_BASE = 0,  // RouterStream.base (the router cores' arena range)
    RT_CRTA_ENG_SEM = 1,     // RouterStream.arr[1] (the engine-done semaphore: RT_ENG)
    RT_CRTA_COUNT,
    RT_CRTA_START = 2,       // RT_VALID: actual_start's address (one uint32, DRAM)
    RT_CRTA_END = 3,         // actual_end's address
    RT_CRTA_SP_ROW = 4,      // my SP row r
    RT_CRTA_SP_ROWS = 5,     // SP rows R
    RT_CRTA_ROW_TOKENS = 6,  // tokens an SP row (S C; token loops / splits: every call's, S C splits loops)
    RT_CRTA_FIRST = 7,       // my first token in the row (p S; token loops: the call's first row g S C on)
    RT_CRTA_S = 8,           // S (a call's)
    RT_CRTA_SUB = 9,         // a call's rows of the row (S C)
    RT_CRTA_LOOPS = 10,      // token loops a split (the planner counts every loop's real tokens of its split)
    RT_CRTA_SPLIT0 = 11,     // the split's first token of mine in the row (split loops S C + p S)
    RT_CRTA_VALID_COUNT,
};
// Defines:
//   NCRISC and compute (`eng`): RT_MSYNC_OFF (off["msync"], `u`; RT_MSYNC = arena base + it), RT_FOLD (fold, `u`);
//     the router's partials always go to compute in a fixed order (deterministic);
//   compute: RT_SQRTSOFTPLUS + INP_FLOAT32 1 (score_func sqrtsoftplus); RT_ROUTE + the
//     route-word defines RW_R, RW_C, RW_ME (r C + c), RW_K, RW_RW (route_words(N): 1, 2 past 16 chips), RW_E
//     (ET 32 / fold), RW_WRAPX / RW_WRAPY (true / false: torus and the axis > 2), RW_XRING_INIT / RW_YRING_INIT
//     (ring_table(C / R, wrap) comma lists), RW_CHIPS_INIT (e // EPC for e < E, comma list);
//   BRISC: RT_FOLD; RT_ROUTE, RT_CT0 (len(ct + acc(out) + ra accessors)), RT_PAGE (the
//     route tensor's aligned page size), RT_RW;
//   NCRISC and BRISC, a padded chunk (fused_head's actual_start / actual_end): RT_VALID (common args 2..11);
//   NCRISC and BRISC, an earlier token loop of its split (MoEBlockConfig.token_loops): RAP_SV0 - the planner plans no
//     token (RAP_S 32, its regions past the router's bytes), the exchange runs empty (the chips' barrier between
//     loops); a forwarding port's scratch then a ring of RA_FWD_RING slots (every loop's program);
//   NCRISC and BRISC with the routing exchange (router_stream._ra_src): RA_PLAN, the routing defines (above) and
//     ra::plan's RAP_* (RoutingStream.kernel_subs: RAP_S, RAP_K, RAP_EPC, RAP_R, RAP_C, RAP_ME, RAP_A, RAP_L,
//     RAP_P (nps[coord]), RAP_TPP 32, RAP_M, RAP_HBP, RAP_BB, RAP_BBP, RAP_PCAP, RAP_NPORT, RAP_ROWS_CAP,
//     RAP_ENT_CAP, RAP_WX, RAP_WY (ties always by parity), RAP_CT0 -
//     NCRISC len(ct + b, ix accessors), BRISC len(ct + out accessor)).
// Runtime args (RouterStream.pieces): NCRISC [b, ix addrs] + [bb RB + j % RB for bb < G] + [G - 1, arr[0]]
// + _arrival_order(p) + tail [words, best, out (arena offsets + base), scale bits] + [sv] + (routing) _ra_args2;
// compute [1, the fin op (p << 8 | 2)] + [G - 1 if bb == p else 0 for bb < G] + tail + [sv, base + off["route"],
// 32 (j % RB)]; BRISC [out addr | partner NoC, jr] + [scale bits, out, words] + [best, route L1, route addr, 32 jr, sv]
// + (routing) _ra_args.
// kernel_lib mcast families (named compile-time args "<name>_ct_offset" / "<name>_rt_offset", main's Mcast::attach):
//   se BRISC: "x0" (rotating over fam0's columns, or a fixed sender when one column), "x1" (fam1), "h" (rotating) -
//     NOC0, handshake, one receiver group a row (GX); semaphore ids from NSEM_OWN = 2 + 2 NWS on.

}  // namespace ttnn::operations::experimental::moe_block::fused_head::kernels
