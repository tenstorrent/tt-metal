// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the fused prep→scan chunk_gdn op: ONE program, two disjoint core sets.
// Per head h, NP PRODUCER cores run {unchanged prep reader, unchanged prep compute, fused writer}
// and NV RECEIVER cores run {fused-receiver reader variant, unchanged scan compute, unchanged scan
// writer}. Receiver (h, v) is exactly the phased scan's V-block core — it carries the state slice
// S[:, v*Vtl : (v+1)*Vtl] and produces that V-slice of o — fed over the NoC instead of from DRAM.
// Each producer's writer hands the 7 computed intermediates of chunk c straight into the head's NV
// receivers' CBs: the six V-independent tensors as multicasts to the head's 1xNV row rectangle, v_beta
// as NV per-receiver slice writes — zero DRAM intermediates.
//
// Geometry (design D3/D9; mirrors gdnopt/fused_geometry.py::placement, the host-side oracle):
//   receivers  row-major from row 0: head h at row h / HPR, columns (h % HPR)*NV .. +NV-1, with
//              HPR = grid.x / NV heads per row  =>  feasible iff BH <= HPR * grid.y
//   producers  the remaining cores enumerated row-major from row 0 (the stranded columns of the
//              receiver rows first), the first BH*NP of them; producer p serves head p / NP as its
//              j = p % NP -th producer and owns that head's chunks c = j, j+NP, ...
//
// Handshake (design D5/D6): receiver (h, v) reserves its 7 slots for chunk c, resets its VALID word,
// then atomically increments credit[h] on the producer that owns chunk c. That producer sends only
// at credit[h] == NV, resets the word, writes, waits for the write ACKS (a flush proves departure
// only), then multicasts VALID to the rectangle. The credit words are BH plain L1 words in the last
// tile of the u/mask CB, which is declared on the UNION of both core sets so it has one address on
// every core; because dispatch re-initializes only Semaphore objects per launch, each producer zeroes
// its words at start and bumps the `init` semaphore on its receivers, which wait for all NP before
// their first credit.
//
// Hand-off CB addressing: the 7 hand-off CBs are declared on the UNION of producer+receiver cores,
// so they get identical base addresses on both sides. The receiver reserves/pushes each shared CB
// exactly once per GLOBAL chunk c, so its slot for chunk c is base + (c % nbuf)*slot_bytes. v_beta's
// CB is producer-sized (cv*nbuf tiles) while a receiver reserves only Ct*Vtl per chunk, so its ring
// has NV*nbuf slots and its slot for chunk c is base + ((c*Ct*Vtl) mod (cv*nbuf))*tile_bytes — the
// writer computes both from the global chunk index (writer_chunk_gdn_fused.cpp). After the F1
// scan-side CB renumber (scan v_beta 17->14, dl 11->22, v_new 22->11) the seven hand-off indices
// coincide with prep's output indices, so the same physical CB is prep's output AND scan's input:
//   v_beta=14  kd=18  q_decay=19  intra=20  k_dec_t=24  dl=22  t_inv=13
//
// Bit-exactness: the compute kernels and the math header are byte-identical to the phased path and
// the seven intermediates are packed at the same CB boundaries in fp32, so fused == phased bit for
// bit; any difference is plumbing.

#include "chunk_gdn_fused.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

// CB index plan — kept in sync with the prep/scan compute + dataflow kernels (post-renumber).
// Uniquely named (fcb) so it does not ODR-clash with the phased factory's pcb:: under unity builds.
namespace fcb {
// The 7 hand-off CBs: prep OUTPUT index == scan INPUT index (that identity is the whole design).
constexpr uint32_t Tinv = tt::CBIndex::c_13;    // t_inv
constexpr uint32_t vbeta = tt::CBIndex::c_14;   // v_beta
constexpr uint32_t kd = tt::CBIndex::c_18;      // kd (prep's cb_w)
constexpr uint32_t qdecay = tt::CBIndex::c_19;  // q_decay
constexpr uint32_t intra = tt::CBIndex::c_20;   // intra
constexpr uint32_t dl = tt::CBIndex::c_22;      // dl (1 tile; prep aliases its cb_vnew slot)
constexpr uint32_t kdec_t = tt::CBIndex::c_24;  // k_dec_t
// Producer-only (prep) CBs — same indices/sizes/formats as the phased prep factory.
constexpr uint32_t q = tt::CBIndex::c_0;
constexpr uint32_t k = tt::CBIndex::c_1;
constexpr uint32_t v = tt::CBIndex::c_2;
constexpr uint32_t g = tt::CBIndex::c_3;
constexpr uint32_t beta = tt::CBIndex::c_4;
constexpr uint32_t eye = tt::CBIndex::c_5;
constexpr uint32_t tril = tt::CBIndex::c_6;
constexpr uint32_t ones = tt::CBIndex::c_7;
constexpr uint32_t decay = tt::CBIndex::c_9;
constexpr uint32_t decay_exp = tt::CBIndex::c_10;
constexpr uint32_t decayfac = tt::CBIndex::c_11;
constexpr uint32_t lmask = tt::CBIndex::c_12;
constexpr uint32_t kbeta = tt::CBIndex::c_15;
// u: the prep's mask holder (3 tiles, pushed once, never popped) PLUS one trailing tile that holds the
// BH producer-side credit words. Declared on the UNION so producers and receivers agree on its address
// (receivers never touch the CB's data; they only compute the credit-word address from its base).
constexpr uint32_t u = tt::CBIndex::c_17;
constexpr uint32_t scr2 = tt::CBIndex::c_29;
constexpr uint32_t scr3 = tt::CBIndex::c_30;
// Shared-index CBs (producer and receiver both declare them, on their own disjoint core sets;
// sizes may differ per side). Post-renumber scan indices: vnew moved 22 -> 11.
constexpr uint32_t S = tt::CBIndex::c_8;
constexpr uint32_t vnew = tt::CBIndex::c_11;  // scan-only (receiver); producer's c_11 is decayfac
constexpr uint32_t out = tt::CBIndex::c_16;
constexpr uint32_t s2 = tt::CBIndex::c_21;
constexpr uint32_t ointer = tt::CBIndex::c_23;
constexpr uint32_t supd = tt::CBIndex::c_25;
constexpr uint32_t stmp = tt::CBIndex::c_26;
constexpr uint32_t final_s = tt::CBIndex::c_27;
constexpr uint32_t scr1 = tt::CBIndex::c_28;
constexpr uint32_t s3 = tt::CBIndex::c_31;
}  // namespace fcb

namespace {
ComputeConfigDescriptor fused_compute_cfg() {
    return ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
}
}  // namespace

tt::tt_metal::ProgramDescriptor ChunkGdnFusedProgramFactory::create_descriptor(
    const ChunkGdnFusedParams& attrs, const ChunkGdnFusedInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t Ct = attrs.chunk_size / TILE_HEIGHT;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;  // full V (tiles): the producer's v_beta width
    const uint32_t has_s0 = attrs.has_initial_state ? 1u : 0u;

    const uint32_t NP = attrs.np;  // producers per head (the op host clamps it to NC)
    const uint32_t NV = attrs.nv;  // receivers per head (validated: divides Vt, rectangles fit)
    TT_FATAL(NV >= 1 && Vt % NV == 0, "chunk_gdn_fused: nv={} must divide Vt={}", NV, Vt);
    const uint32_t Vtl = Vt / NV;  // per-receiver V-slice width (tiles)

    // Producer-side (full-V) tile counts — the phased prep factory's.
    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt, kc = Kt * Ct;
    const uint32_t scr = std::max({cc, ck, cv, kv, kc});
    // Receiver-side (V-sliced) tile counts — the phased scan factory's at Vt = Vtl.
    const uint32_t cvl = Ct * Vtl, kvl = Kt * Vtl;
    const uint32_t scr_l = std::max({cc, ck, cvl, kvl, kc});

    const tt::DataFormat df_qkv = tt::DataFormat::Float16_b;  // bf16 q/k/v (prep inputs)
    const uint32_t tile_f32 = tt::tile_size(tt::DataFormat::Float32);

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t n_cores = grid.x * grid.y;
    TT_FATAL(NV <= grid.x, "chunk_gdn_fused: nv={} exceeds the grid width {}", NV, grid.x);
    const uint32_t HPR = grid.x / NV;  // heads per receiver row (placement 0)
    const uint32_t R = BH * NV;        // receiver cores
    const uint32_t P = BH * NP;        // producer cores
    TT_FATAL(R + P <= n_cores, "chunk_gdn_fused: R+P = {}+{} cores needed, grid has {}", R, P, n_cores);

    // ---- Placement (design D9; mode 0 mirrors gdnopt/fused_geometry.py::placement) ----
    std::vector<CoreCoord> rcv_cores(R);  // index h*NV + v
    std::vector<CoreCoord> prod_cores;    // index p = h*NP + j
    prod_cores.reserve(P);
    if (attrs.placement == 0) {
        TT_FATAL(
            BH <= HPR * grid.y,
            "chunk_gdn_fused: BH={} 1x{} receiver rectangles do not fit a {}x{} grid ({} per row)",
            BH,
            NV,
            grid.x,
            grid.y,
            HPR);
        std::vector<bool> is_rcv(n_cores, false);
        for (uint32_t h = 0; h < BH; h++) {
            const uint32_t y0 = h / HPR;
            const uint32_t x0 = (h % HPR) * NV;
            for (uint32_t v = 0; v < NV; v++) {
                rcv_cores[h * NV + v] = CoreCoord{x0 + v, y0};
                is_rcv[y0 * grid.x + x0 + v] = true;
            }
        }
        for (uint32_t y = 0; y < grid.y && prod_cores.size() < P; y++) {
            for (uint32_t x = 0; x < grid.x && prod_cores.size() < P; x++) {
                if (!is_rcv[y * grid.x + x]) {
                    prod_cores.push_back(CoreCoord{x, y});
                }
            }
        }
    } else {
        // Row-local: head h < grid.y owns row h — receivers at columns 0..NV-1, producers at NV..L-1.
        // NOC_1 routes -x then -y, so every producer's writes travel west inside the head's own row and
        // never share a link with another head. Heads h >= grid.y live in the leftover columns [L, W)
        // as vertical blocks: an rw x rh receiver rectangle on top, the producers row-major below it —
        // their traffic is confined to the block's columns (short -x legs, then -y within the block).
        const uint32_t L = NV + NP;
        TT_FATAL(L <= grid.x, "chunk_gdn_fused: row-local placement needs NV+NP={} <= grid.x={}", L, grid.x);
        // k heads per row, each in its own column segment [i*L, (i+1)*L): the segments' -x legs are
        // disjoint, so heads sharing a row still share no link.
        const uint32_t k_per_row = grid.x / L;
        const uint32_t n_row_heads = std::min<uint32_t>(BH, k_per_row * grid.y);
        for (uint32_t h = 0; h < n_row_heads; h++) {
            const uint32_t row = h / k_per_row;
            const uint32_t xs = (h % k_per_row) * L;
            for (uint32_t v = 0; v < NV; v++) {
                rcv_cores[h * NV + v] = CoreCoord{xs + v, row};
            }
            for (uint32_t j = 0; j < NP; j++) {
                prod_cores.push_back(CoreCoord{xs + NV + j, row});
            }
        }
        if (BH > n_row_heads) {
            const uint32_t rem = BH - n_row_heads;
            const uint32_t wl = grid.x - k_per_row * L;
            TT_FATAL(wl >= 1, "chunk_gdn_fused: row-local placement: no leftover columns for {} extra heads", rem);
            const uint32_t rw = std::min<uint32_t>(NV, wl);
            TT_FATAL(
                NV % rw == 0,
                "chunk_gdn_fused: row-local placement: NV={} not a multiple of the leftover width {}",
                NV,
                rw);
            const uint32_t rh = NV / rw;
            const uint32_t block_h = rh + (NP + wl - 1) / wl;
            TT_FATAL(
                rem * block_h <= grid.y,
                "chunk_gdn_fused: row-local placement: {} leftover heads need {} rows, grid has {}",
                rem,
                rem * block_h,
                grid.y);
            for (uint32_t kk = 0; kk < rem; kk++) {
                const uint32_t h = n_row_heads + kk;
                const uint32_t y_base = kk * block_h;
                const uint32_t xl = k_per_row * L;  // first leftover column
                for (uint32_t v = 0; v < NV; v++) {
                    rcv_cores[h * NV + v] = CoreCoord{xl + (v % rw), y_base + v / rw};
                }
                for (uint32_t j = 0; j < NP; j++) {
                    prod_cores.push_back(CoreCoord{xl + (j % wl), y_base + rh + j / wl});
                }
            }
        }
    }
    TT_FATAL(prod_cores.size() == P, "chunk_gdn_fused: placement produced {} producers, need {}", prod_cores.size(), P);

    std::set<CoreRange> rcv_crs, prod_crs, union_crs;
    for (const auto& c : rcv_cores) {
        rcv_crs.insert(CoreRange{c, c});
        union_crs.insert(CoreRange{c, c});
    }
    for (const auto& c : prod_cores) {
        prod_crs.insert(CoreRange{c, c});
        union_crs.insert(CoreRange{c, c});
    }
    const CoreRangeSet rcv_set{rcv_crs};
    const CoreRangeSet prod_set{prod_crs};
    const CoreRangeSet union_set{union_crs};

    ProgramDescriptor desc;
    auto add_cb = [&](const CoreRangeSet& on,
                      uint32_t idx,
                      uint32_t n_tiles,
                      uint32_t nbuf = 1,
                      tt::DataFormat fmt = tt::DataFormat::Float32) {
        const uint32_t ts = tt::tile_size(fmt);
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = on,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    // (1) The 7 hand-off CBs FIRST, on the UNION core set => same base address on producer and every
    // receiver (the slot-addressing precondition). fp32, PRODUCER sizes, in the receiver's reserve order
    // (v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv). Double-buffered: the producer runs one chunk
    // ahead of the receivers' consumption. The writer's explicit destination slots (global c % nbuf, and
    // the v_beta ring) are computed against THIS depth, so kHandoffNbuf travels to the writer as a CT arg.
    const uint32_t kHandoffNbuf = attrs.nbuf;
    add_cb(union_set, fcb::vbeta, cv, kHandoffNbuf);
    add_cb(union_set, fcb::kd, ck, kHandoffNbuf);
    add_cb(union_set, fcb::qdecay, ck, kHandoffNbuf);
    add_cb(union_set, fcb::intra, cc, kHandoffNbuf);
    add_cb(union_set, fcb::kdec_t, kc, kHandoffNbuf);
    // dl is 1 tile. (The phased prep factory sized this index cv as cb_vnew for monolithic layout
    // parity, but the prep kernel only ever uses 1 tile of it, as cb_dl.)
    add_cb(union_set, fcb::dl, 1, kHandoffNbuf);
    add_cb(union_set, fcb::Tinv, cc, kHandoffNbuf);
    // (1b) The u/mask CB, ALSO on the union: 3 mask tiles (prep reads them once) + 1 credit tile whose
    // BH x nbuf leading words are the producer-side credit counters credit[h][slot].
    // Union-declared so the receivers can address a producer's credit word from their own CB base.
    const uint32_t u_tiles = std::max<uint32_t>(cv, 3) + 1;
    const uint32_t credit_off_bytes = (u_tiles - 1) * tile_f32;
    add_cb(union_set, fcb::u, u_tiles);

    // (2) The remaining 24 prep CBs on the PRODUCER cores only — same sizes/formats as the phased
    // prep factory (which mirrors the monolithic op's layout). The absolute L1 layout necessarily
    // shifts (the hand-off CBs above allocate first), which prior measurement showed to be
    // perf-neutral for prep; the math is layout-independent.
    // Producer input CBs are double-buffered: the producer has no DRAM writes, so its reader
    // prefetching item i+1's ~32KB while compute works item i directly shortens the per-chunk
    // critical path. (The phased prep keeps nbuf=1 — there this prefetch measured harmful in the
    // write-bound regime; here the producer is math/latency-bound.) +32KB L1 on producer cores.
    add_cb(prod_set, fcb::q, ck, 2, df_qkv);
    add_cb(prod_set, fcb::k, ck, 2, df_qkv);
    add_cb(prod_set, fcb::v, cv, 2, df_qkv);
    add_cb(prod_set, fcb::g, Ct, 2);
    add_cb(prod_set, fcb::beta, Ct, 2);
    add_cb(prod_set, fcb::eye, cc);
    add_cb(prod_set, fcb::tril, cc);
    add_cb(prod_set, fcb::ones, cc);
    add_cb(prod_set, fcb::S, kv, 2);
    add_cb(prod_set, fcb::decay, Ct);
    add_cb(prod_set, fcb::decay_exp, Ct);
    add_cb(prod_set, fcb::decayfac, Ct);
    add_cb(prod_set, fcb::lmask, cc);
    add_cb(prod_set, fcb::kbeta, ck);
    add_cb(prod_set, fcb::out, cv, 2, df_qkv);
    add_cb(prod_set, fcb::s2, kv, 2);
    add_cb(prod_set, fcb::ointer, cv);
    add_cb(prod_set, fcb::supd, kv);
    add_cb(prod_set, fcb::stmp, kv);
    add_cb(prod_set, fcb::final_s, kv);
    add_cb(prod_set, fcb::scr1, scr);
    add_cb(prod_set, fcb::scr2, scr);
    add_cb(prod_set, fcb::scr3, scr);
    add_cb(prod_set, fcb::s3, kv, 2);

    // (3) The remaining 10 scan CBs on the RECEIVER cores only, at the per-receiver V-slice width
    // Vtl (exactly the phased scan factory's sizes at Vt = Vtl) and the post-renumber indices
    // (vnew = 11). o is fp32 (see compute_output_specs).
    add_cb(rcv_set, fcb::S, kvl);
    add_cb(rcv_set, fcb::vnew, cvl);
    add_cb(rcv_set, fcb::out, cvl, 2, tt::DataFormat::Float32);
    add_cb(rcv_set, fcb::s2, kvl);
    add_cb(rcv_set, fcb::ointer, cvl);
    add_cb(rcv_set, fcb::supd, kvl);
    add_cb(rcv_set, fcb::stmp, kvl);
    add_cb(rcv_set, fcb::final_s, kvl);
    add_cb(rcv_set, fcb::scr1, scr_l);
    add_cb(rcv_set, fcb::s3, kvl);

    // Handshake semaphores, declared on the UNION so each id resolves to the same L1 address on
    // producer and receiver. Ids reach both kernels as trailing compile-time args.
    //   id 0 = ready  — legacy single counter; superseded by the credit words (kept so the shared
    //                   scan reader's trailing-arg layout is uniform across its variants)
    //   id 1 = init   — producer -> receivers: "my credit words are zeroed"; receivers wait for NP
    //   ids 2 .. 2+nbuf-1 = valid[slot] — producer -> receivers: "chunk c (slot c % nbuf) is in your
    //                   CBs". One flag per hand-off slot lets a receiver keep nbuf-1 hand-offs in
    //                   flight. Consecutive ids => consecutive L1 words, so the kernels
    //                   address slot s as id (sem_valid_id + s). Program cap is 16 semaphores: nbuf <= 8.
    constexpr uint32_t sem_ready_id = 0;
    constexpr uint32_t sem_init_id = 1;
    constexpr uint32_t sem_valid_id = 2;
    TT_FATAL(sem_valid_id + kHandoffNbuf <= 16, "chunk_gdn_fused: nbuf {} needs too many semaphores", kHandoffNbuf);
    for (uint32_t id = 0; id < sem_valid_id + kHandoffNbuf; id++) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = union_set, .initial_value = 0});
    }

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";

    // ---- Producer-side CT args: byte-identical to the phased PREP factory's ----
    const std::vector<uint32_t> ct_prep = {Ct, Kt, Vt};

    std::vector<uint32_t> prep_reader_ct = ct_prep;
    TensorAccessorArgs(*in.q.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.eye_c.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.tril_c.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.ones_c.buffer()).append_to(prep_reader_ct);
    TensorAccessorArgs(*in.masks_c.buffer()).append_to(prep_reader_ct);
    // OPT-A: trailing compile args after all TensorAccessorArgs — 1 => read that tensor flat token-major.
    prep_reader_ct.push_back(attrs.v_flat ? 1u : 0u);
    prep_reader_ct.push_back(attrs.qk_flat ? 1u : 0u);

    auto f32_bits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    std::vector<uint32_t> prep_compute_ct = ct_prep;
    prep_compute_ct.push_back(attrs.qk_norm ? 1u : 0u);
    prep_compute_ct.push_back(f32_bits(attrs.scale));
    prep_compute_ct.push_back(f32_bits(1e-6f));

    // Fused writer: plain scalars, no accessors (it writes no DRAM at all).
    const std::vector<uint32_t> fused_writer_ct = {
        Ct,
        Kt,
        Vt,
        sem_valid_id,
        sem_init_id,
        kHandoffNbuf,
        NV,
        Vtl,
        fcb::u,
        credit_off_bytes,
        attrs.unicast ? 1u : 0u,
        attrs.posted ? 1u : 0u};

    // ---- Receiver-side CT args: the phased SCAN layout at the V-slice width, with Vt_full for strides ----
    const std::vector<uint32_t> ct_scan = {Ct, Kt, Vtl, has_s0, Vt};

    // Fused-receiver reader: s0 is its ONLY DRAM tensor (chain of one accessor, starting at CT
    // index 5), then the semaphore ids and the credit-word location as trailing args.
    std::vector<uint32_t> receiver_ct = ct_scan;
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(receiver_ct);
    receiver_ct.push_back(sem_ready_id);
    receiver_ct.push_back(sem_valid_id);
    receiver_ct.push_back(sem_init_id);
    receiver_ct.push_back(fcb::u);
    receiver_ct.push_back(credit_off_bytes);
    receiver_ct.push_back(kHandoffNbuf);

    std::vector<uint32_t> scan_writer_ct = ct_scan;
    TensorAccessorArgs(*outputs[0].buffer()).append_to(scan_writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(scan_writer_ct);

    // ---- Kernels. Push order FIXED (part of the program-cache identity):
    // prep_reader, prep_compute, fused_writer, fused_receiver_reader, scan_compute, scan_writer.
    KernelDescriptor prep_reader;
    prep_reader.kernel_source = kdir + "dataflow/reader_chunk_gdn_prep.cpp";
    prep_reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    prep_reader.core_ranges = prod_set;
    prep_reader.compile_time_args = prep_reader_ct;
    prep_reader.config = ReaderConfigDescriptor{};
    prep_reader.runtime_args.reserve(P);

    KernelDescriptor prep_compute;
    prep_compute.kernel_source = kdir + "compute/chunk_gdn_prep.cpp";
    prep_compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    prep_compute.core_ranges = prod_set;
    prep_compute.compile_time_args = prep_compute_ct;
    prep_compute.config = fused_compute_cfg();
    // Fused-only perf: hoisted WY-path reconfigs (see chunk_gdn_math.hpp kGdnHoistReconfig).
    prep_compute.defines = {{"GDN_HOIST_RECONFIG", "1"}};
    prep_compute.runtime_args.reserve(P);

    // The fused writer runs on the WriterConfigDescriptor's RISC/NoC (BRISC / NOC_1 on Blackhole).
    // Its multicast rectangles must be given in that NoC's own order: NOC_1 multicasts from the
    // bottom-right to the top-left, so the (start, end) pair is swapped there — the same idiom the
    // device layer applies in Device::get_noc_multicast_encoding. Coordinates themselves are virtual
    // and identical on both NoCs.
    const bool writer_on_noc1 = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch()) == NOC::NOC_1;

    KernelDescriptor fused_writer;
    fused_writer.kernel_source = kdir + "dataflow/writer_chunk_gdn_fused.cpp";
    fused_writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    fused_writer.core_ranges = prod_set;
    fused_writer.compile_time_args = fused_writer_ct;
    fused_writer.config = WriterConfigDescriptor{};
    fused_writer.runtime_args.reserve(P);

    KernelDescriptor receiver_reader;
    receiver_reader.kernel_source = kdir + "dataflow/reader_chunk_gdn_scan.cpp";
    receiver_reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_reader.core_ranges = rcv_set;
    receiver_reader.compile_time_args = receiver_ct;
    receiver_reader.defines = {{"GDN_FUSED_RECEIVER", "1"}};
    receiver_reader.config = ReaderConfigDescriptor{};
    receiver_reader.runtime_args.reserve(R);

    KernelDescriptor scan_compute;
    scan_compute.kernel_source = kdir + "compute/chunk_gdn_scan.cpp";
    scan_compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    scan_compute.core_ranges = rcv_set;
    scan_compute.compile_time_args = ct_scan;
    scan_compute.config = fused_compute_cfg();
    scan_compute.runtime_args.reserve(R);

    KernelDescriptor scan_writer;
    scan_writer.kernel_source = kdir + "dataflow/writer_chunk_gdn_scan.cpp";
    scan_writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    scan_writer.core_ranges = rcv_set;
    scan_writer.compile_time_args = scan_writer_ct;
    scan_writer.config = WriterConfigDescriptor{};
    scan_writer.runtime_args.reserve(R);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* g_buf = in.g.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* eye_buf = in.eye_c.buffer();
    auto* tril_buf = in.tril_c.buffer();
    auto* ones_buf = in.ones_c.buffer();
    auto* masks_buf = in.masks_c.buffer();
    auto* s0_buf = in.initial_state.has_value() ? in.initial_state->buffer() : nullptr;
    auto* o_buf = outputs[0].buffer();
    auto* fs_buf = outputs[1].buffer();

    for (uint32_t h = 0; h < BH; h++) {
        // Virtual worker coords of this head's NV receivers (a rectangle: 1xNV row in placement 0,
        // rw x rh block for the leftover heads of placement 1) and NP producers.
        std::vector<CoreCoord> rv(NV), pv(NP);
        for (uint32_t v = 0; v < NV; v++) {
            rv[v] = device->worker_core_from_logical_core(rcv_cores[h * NV + v]);
        }
        for (uint32_t j = 0; j < NP; j++) {
            pv[j] = device->worker_core_from_logical_core(prod_cores[h * NP + j]);
        }
        // Multicast rectangle = the receivers' bounding box. Density is checked in LOGICAL coords (no
        // other worker of the program inside); the virtual box may additionally span non-worker
        // columns (Blackhole's virtual grid skips the DRAM/ethernet columns: logical 7 -> virtual 10),
        // which the multicast tolerates — the row-major 1xNV row rectangles crossed that gap all along.
        CoreCoord l_tl = rcv_cores[h * NV], l_br = rcv_cores[h * NV];
        for (uint32_t v = 0; v < NV; v++) {
            const CoreCoord& c = rcv_cores[h * NV + v];
            l_tl = CoreCoord{std::min(l_tl.x, c.x), std::min(l_tl.y, c.y)};
            l_br = CoreCoord{std::max(l_br.x, c.x), std::max(l_br.y, c.y)};
        }
        TT_FATAL(
            (l_br.x - l_tl.x + 1) * (l_br.y - l_tl.y + 1) == NV,
            "chunk_gdn_fused: head {} receivers do not form a dense {}-core rectangle in logical coords",
            h,
            NV);
        const CoreCoord m_tl = device->worker_core_from_logical_core(l_tl);
        const CoreCoord m_br = device->worker_core_from_logical_core(l_br);
        const CoreCoord& m_start = writer_on_noc1 ? m_br : m_tl;  // NOC_1: bottom-right -> top-left
        const CoreCoord& m_end = writer_on_noc1 ? m_tl : m_br;

        // Producer j of head h owns the interleaved chunks c = j, j+NP, ... — as flat work-items
        // wi = h*NC + c that is start h*NC + j with stride NP (trailing reader arg). The op host
        // clamps NP <= NC, so every producer owns at least one chunk.
        for (uint32_t j = 0; j < NP; j++) {
            const CoreCoord& pc = prod_cores[h * NP + j];
            const uint32_t cnt = (NC - j + NP - 1) / NP;
            prep_reader.emplace_runtime_args(
                pc,
                {h * NC + j,
                 cnt,
                 q_buf,
                 k_buf,
                 v_buf,
                 g_buf,
                 beta_buf,
                 eye_buf,
                 tril_buf,
                 ones_buf,
                 masks_buf,
                 NC,
                 attrs.HV,
                 attrs.Hk,
                 NP});
            prep_compute.emplace_runtime_args(pc, {cnt});
            std::vector<std::variant<uint32_t, Buffer*>> w_args = {
                NC,
                NP,
                j,
                h,
                BH,
                static_cast<uint32_t>(m_start.x),
                static_cast<uint32_t>(m_start.y),
                static_cast<uint32_t>(m_end.x),
                static_cast<uint32_t>(m_end.y)};
            for (uint32_t v = 0; v < NV; v++) {
                w_args.push_back(static_cast<uint32_t>(rv[v].x));
                w_args.push_back(static_cast<uint32_t>(rv[v].y));
            }
            fused_writer.emplace_runtime_args(pc, w_args);
        }

        // Receiver (h, v): its V-slice index, s0 from DRAM, then NP and N_INIT (= NP: every producer
        // of this head bumps `init` once) and the producers' coords for the rotating credit.
        for (uint32_t v = 0; v < NV; v++) {
            const CoreCoord& rc = rcv_cores[h * NV + v];
            std::vector<std::variant<uint32_t, Buffer*>> r_args = {h, v, NC, s0_buf, NP, NP};
            for (uint32_t j = 0; j < NP; j++) {
                r_args.push_back(static_cast<uint32_t>(pv[j].x));
                r_args.push_back(static_cast<uint32_t>(pv[j].y));
            }
            receiver_reader.emplace_runtime_args(rc, r_args);
            scan_compute.emplace_runtime_args(rc, {NC});
            scan_writer.emplace_runtime_args(rc, {h, v, NC, o_buf, fs_buf});
        }
    }

    desc.kernels.push_back(std::move(prep_reader));
    desc.kernels.push_back(std::move(prep_compute));
    desc.kernels.push_back(std::move(fused_writer));
    desc.kernels.push_back(std::move(receiver_reader));
    desc.kernels.push_back(std::move(scan_compute));
    desc.kernels.push_back(std::move(scan_writer));
    return desc;
}

}  // namespace ttnn::prim
