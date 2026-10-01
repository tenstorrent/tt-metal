// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the fused prep→scan chunk_gdn op: ONE program, two disjoint core sets.
// PRODUCER cores (NP per head, or one pool of P serving every head) run {unchanged prep reader, unchanged
// prep compute, fused writer}; per head h, NV RECEIVER cores run {fused-receiver reader variant, unchanged
// scan compute, unchanged scan writer}. Receiver (h, v) is exactly the phased scan's V-block core — it carries the
// state slice S[:, v*Vtl : (v+1)*Vtl] and produces that V-slice of o — fed over the NoC instead of from DRAM. Each
// producer's writer hands the 7 computed intermediates of chunk c straight into the head's NV receivers' CBs: the six
// V-independent tensors as multicasts to the head's 1xNV row rectangle, v_beta as NV per-receiver slice writes — zero
// DRAM intermediates.
//
// Geometry (computed by fused_placement() in chunk_gdn_device_operation.cpp, which the host-side
// geometry tests check per grid). Placement 0 (row-major):
//   receivers  row-major from row 0: head h at row h / HPR, columns (h % HPR)*NV .. +NV-1, with
//              HPR = grid.x / NV heads per row  =>  feasible iff BH <= HPR * grid.y
//   producers  the remaining cores enumerated row-major from row 0 (the stranded columns of the
//              receiver rows first), the first BH*NP of them; producer p serves head p / NP as its
//              j = p % NP -th producer and owns that head's chunks c = j, j+NP, ...
// Placement 1 (row-local, the default whenever it fits): each head's NV receivers and NP producers
// share one row segment, producers east of receivers, so NOC_1's -x-then-y routes of different heads
// never share a link; heads that do not fit the rows go to the leftover columns as vertical blocks.
// Placement 2 (producer pool, attrs.np = the pool size P): the row-local map of NPH home producers per
// head for the largest NPH with BH*NPH <= P, plus P - BH*NPH EXTRA producers on the remaining cores.
// Which producer computes chunk c of head h, and each producer's item list, is the one formula of
// kernels/dataflow/chunk_gdn_fused_map.hpp, evaluated here and in the three dataflow kernels: the home
// producers take their head's chunks round-robin, the extras take the share num/den of every head's
// chunks chunk-major. Placements 0/1 are its NX = 0 case.
//
// Handshake: receiver (h, v) reserves its 7 slots for chunk c, resets slot (c % nbuf)'s VALID flag,
// then atomically increments credit[h][c % nbuf] on the producer that owns chunk c. That producer sends
// only at credit[h][c % nbuf] == NV, resets the word, writes, waits for the write ACKS (a flush proves
// departure only), then sets VALID[c % nbuf] on the receivers. A receiver keeps nbuf-1 hand-offs in
// flight; the per-slot flags keep their VALIDs apart. The credit words are BH x nbuf plain L1 words in
// the last tile of the u/mask CB, which is declared on the UNION of both core sets so it has one address
// on every core; because dispatch re-initializes only Semaphore objects per launch, each producer zeroes
// its words at start and bumps the `init` semaphore on the receivers of every head it serves, which
// wait for all their distinct producers before their first credit.
//
// Hand-off CB addressing: the 7 hand-off CBs are declared on the UNION of producer+receiver cores,
// so they get identical base addresses on both sides. The receiver reserves/pushes each shared CB
// exactly once per GLOBAL chunk c, so its slot for chunk c is base + (c % nbuf)*slot_bytes. v_beta's
// CB is producer-sized (cv*nbuf tiles) while a receiver reserves only Ct*Vtl per chunk, so its ring
// has NV*nbuf slots and its slot for chunk c is base + ((c*Ct*Vtl) mod (cv*nbuf))*tile_bytes — the
// writer computes both from the global chunk index (writer_chunk_gdn_fused.cpp). After the F1
// scan-side CB renumber (scan v_beta 17->14, dl 11->22, v_new 22->11) the seven hand-off indices
// coincide with prep's output indices, so the same physical CB is prep's output AND scan's input:
//   v_beta=14  nkd=18  q_decay=19  intra=20  k_dec_t=24  dl=22  t_inv=13
//
// Bit-exactness: the compute kernels and the math header are byte-identical to the phased path and
// the seven intermediates are packed at the same CB boundaries in fp32, so fused == phased bit for
// bit; any difference is plumbing.

#include "chunk_gdn_device_operation.hpp"
#include "chunk_gdn_compute_config.hpp"
#include "kernels/dataflow/chunk_gdn_fused_map.hpp"

#include <algorithm>
#include <bit>
#include <numeric>
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

// Kickoff staggering (cycles at the 1.35 GHz core clock), keeping chunk 0's reads out of one burst: a producer
// whose first chunk is c waits c * kProducerKickoffStaggerCycles (chunk c is needed c receiver steps after
// chunk 0); receivers issue their first credits, then hold the initial-state read back by
// kReceiverKickoffWaitCycles.
constexpr uint32_t kProducerKickoffStaggerCycles = 4050;  // ~3 us
constexpr uint32_t kReceiverKickoffWaitCycles = 5400;     // ~4 us

// CB index plan — kept in sync with the prep/scan compute + dataflow kernels (post-renumber).
// Uniquely named (fcb) so it does not ODR-clash with the phased factory's pcb:: under unity builds.
namespace fcb {
// The 7 hand-off CBs: prep OUTPUT index == scan INPUT index (that identity is the whole design).
constexpr uint32_t Tinv = tt::CBIndex::c_13;    // t_inv
constexpr uint32_t vbeta = tt::CBIndex::c_14;   // v_beta
constexpr uint32_t nkd = tt::CBIndex::c_18;     // nkd (prep's cb_w)
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
// BH x nbuf producer-side credit words. Declared on the UNION so producers and receivers agree on its address
// (receivers never touch the CB's data; they only compute the credit-word address from its base).
constexpr uint32_t u = tt::CBIndex::c_17;
constexpr uint32_t scr2 = tt::CBIndex::c_29;
constexpr uint32_t scr3 = tt::CBIndex::c_30;
// Shared-index CBs (producer and receiver both declare them, on their own disjoint core sets;
// sizes may differ per side). Post-renumber scan indices: vnew moved 22 -> 11.
constexpr uint32_t S = tt::CBIndex::c_8;
constexpr uint32_t vnew = tt::CBIndex::c_11;  // scan-only (receiver); producer's c_11 is decayfac
constexpr uint32_t out = tt::CBIndex::c_16;   // receiver; the producer's c_16 is kdec
constexpr uint32_t kdec = tt::CBIndex::c_16;  // producer-only: k_dec before its transpose
constexpr uint32_t s2 = tt::CBIndex::c_21;
constexpr uint32_t ointer = tt::CBIndex::c_23;
constexpr uint32_t supd = tt::CBIndex::c_25;
constexpr uint32_t stmp = tt::CBIndex::c_26;
constexpr uint32_t final_s = tt::CBIndex::c_27;
constexpr uint32_t scr1 = tt::CBIndex::c_28;
constexpr uint32_t s3 = tt::CBIndex::c_31;
}  // namespace fcb

tt::tt_metal::ProgramDescriptor ChunkGdnFusedProgramFactory::create_descriptor(
    const ChunkGdnParams& attrs, const ChunkGdnInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t Ct = attrs.chunk_size / TILE_HEIGHT;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;  // full V (tiles): the producer's v_beta width

    const uint32_t NV = attrs.nv;  // receivers per head (validated: divides Vt, rectangles fit)
    TT_FATAL(NV >= 1 && Vt % NV == 0, "chunk_gdn_fused: nv={} must divide Vt={}", NV, Vt);
    const uint32_t Vtl = Vt / NV;  // per-receiver V-slice width (tiles)

    // Producer-side (full-V) tile counts — the phased prep factory's.
    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kc = Kt * Ct;
    // Prep scratch sizes, as in the phased prep factory (see the comment there). Each scratch CB receives blocks of
    // one size, equal to its capacity: scr1 decay_row (Ct), kdec k_dec (ck), scr2 / ointer single tiles, scr3 cc.
    const uint32_t scr1_tiles = Ct, scr2_tiles = 1, scr3_tiles = cc, kdec_tiles = ck, qk_tiles = ck, one_tile = 1;
    // Receiver-side (V-sliced) tile counts — the phased scan factory's at Vt = Vtl.
    const uint32_t cvl = Ct * Vtl, kvl = Kt * Vtl;

    const tt::DataFormat df_qkv = tt::DataFormat::Float16_b;  // bf16 q/k/v (prep inputs)
    const uint32_t tile_f32 = tt::tile_size(tt::DataFormat::Float32);

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    // Placement is a pure function of (grid, BH, NV, np, placement), shared with the
    // nanobind geometry oracle so the host-side tests assert the core map this factory uses.
    const FusedPlacement layout = fused_placement(grid.x, grid.y, BH, NV, attrs.np, attrs.placement);
    const std::vector<CoreCoord>& rcv_cores = layout.receivers;
    const std::vector<CoreCoord>& prod_cores = layout.producers;
    const uint32_t R = BH * NV;                                   // receiver cores
    const uint32_t P = static_cast<uint32_t>(prod_cores.size());  // producer cores
    // The producer map (chunk_gdn_fused_map.hpp): home producers per head, extras, the extras' share.
    const GdnFusedMap map{
        BH, NC, layout.home_per_head, P - BH * layout.home_per_head, attrs.pool_extra_num, attrs.pool_extra_den};

    // CoreRangeSet(Span<const CoreCoord>) merges the per-core coordinates into rectangles.
    const CoreRangeSet rcv_set(rcv_cores);
    const CoreRangeSet prod_set(prod_cores);
    const CoreRangeSet union_set = rcv_set.merge(prod_set);

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
    // (v_beta, nkd, q_decay, intra, k_dec_t, dl, t_inv). Double-buffered: the producer runs one chunk
    // ahead of the receivers' consumption. The writer's explicit destination slots (global c % nbuf, and
    // the v_beta ring) are computed against THIS depth, so kHandoffNbuf travels to the writer as a CT arg.
    const uint32_t kHandoffNbuf = attrs.nbuf;
    add_cb(union_set, fcb::vbeta, cv, kHandoffNbuf);
    add_cb(union_set, fcb::nkd, ck, kHandoffNbuf);
    add_cb(union_set, fcb::qdecay, ck, kHandoffNbuf);
    add_cb(union_set, fcb::intra, cc, kHandoffNbuf);
    add_cb(union_set, fcb::kdec_t, kc, kHandoffNbuf);
    // dl*I is 1 tile. (The phased prep factory sized this index cv as cb_vnew for monolithic layout
    // parity, but the prep kernel only ever uses 1 tile of it, as cb_dl.)
    add_cb(union_set, fcb::dl, 1, kHandoffNbuf);
    add_cb(union_set, fcb::Tinv, cc, kHandoffNbuf);
    // (1b) The u/mask CB, ALSO on the union: 3 mask tiles (prep reads them once) + 1 credit tile whose
    // BH x nbuf leading words are the producer-side credit counters credit[h][slot].
    // Union-declared so the receivers can address a producer's credit word from their own CB base.
    const uint32_t u_tiles = 3 + 1;
    const uint32_t credit_off_bytes = (u_tiles - 1) * tile_f32;
    add_cb(union_set, fcb::u, u_tiles);

    // (2) The remaining prep CBs on the PRODUCER cores only — the phased prep factory's sizes (scratch
    // sized to prep's use, not the mono op's state). The absolute L1 layout shifts against the phased
    // prep (the hand-off CBs above allocate first), which measurement showed to be perf-neutral; the
    // math is layout-independent.
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
    add_cb(prod_set, fcb::S, one_tile);  // invert_block scratch A
    add_cb(prod_set, fcb::decay, Ct);
    add_cb(prod_set, fcb::decay_exp, Ct);
    add_cb(prod_set, fcb::decayfac, Ct + 1);  // + the dl = exp(g_sum) column tile
    add_cb(prod_set, fcb::lmask, cc);
    add_cb(prod_set, fcb::kbeta, ck);
    add_cb(prod_set, fcb::kdec, kdec_tiles);
    add_cb(prod_set, fcb::s2, one_tile);      // invert_block scratch C
    add_cb(prod_set, fcb::ointer, one_tile);  // invert_block's tmpN; Ct == 2: the off-diagonal inverse block
    add_cb(prod_set, fcb::supd, qk_tiles);
    add_cb(prod_set, fcb::stmp, qk_tiles);
    add_cb(prod_set, fcb::final_s, one_tile);  // invert_block scratch B
    add_cb(prod_set, fcb::scr1, scr1_tiles);
    add_cb(prod_set, fcb::scr2, scr2_tiles);
    add_cb(prod_set, fcb::scr3, scr3_tiles);
    add_cb(prod_set, fcb::s3, one_tile);  // invert_block scratch D

    // (3) The remaining 10 scan CBs on the RECEIVER cores only, at the per-receiver V-slice width
    // Vtl (exactly the phased scan factory's sizes at Vt = Vtl) and the post-renumber indices
    // (vnew = 11). o is fp32 (see compute_output_specs).
    add_cb(rcv_set, fcb::S, kvl);
    add_cb(rcv_set, fcb::vnew, cvl);
    add_cb(rcv_set, fcb::out, cvl, 2, tt::DataFormat::Float32);
    add_cb(rcv_set, fcb::s2, kvl);
    add_cb(rcv_set, fcb::ointer, cvl);
    add_cb(rcv_set, fcb::final_s, kvl);
    add_cb(rcv_set, fcb::eye, 1);  // the reader-written identity tile (scan_step's I @ v_beta)
    add_cb(rcv_set, fcb::s3, kvl);

    // Handshake semaphores, declared on the UNION so each id resolves to the same L1 address on
    // producer and receiver. Ids reach both kernels as trailing compile-time args.
    //   id 0 = ready  — legacy single counter; superseded by the credit words (kept so the shared
    //                   scan reader's trailing-arg layout is uniform across its variants)
    //   id 1 = init   — producer -> receivers: "my credit words are zeroed"; a receiver waits for N_INIT,
    //                   the distinct producers serving its head
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

    std::vector<uint32_t> prep_compute_ct = ct_prep;
    prep_compute_ct.push_back(attrs.qk_norm ? 1u : 0u);
    prep_compute_ct.push_back(std::bit_cast<uint32_t>(attrs.scale));
    prep_compute_ct.push_back(std::bit_cast<uint32_t>(1e-6f));

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
    const std::vector<uint32_t> ct_scan = {Ct, Kt, Vtl, Vt};

    // Fused-receiver reader: s0 is its ONLY DRAM tensor (chain of one accessor, starting at CT
    // index 5), then the semaphore ids and the credit-word location as trailing args.
    std::vector<uint32_t> receiver_ct = ct_scan;
    TensorAccessorArgs(*in.initial_state.buffer()).append_to(receiver_ct);
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
    KernelDescriptor prep_reader{
        .kernel_source = kdir + "dataflow/reader_chunk_gdn_prep.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = prod_set,
        .compile_time_args = prep_reader_ct,
        // The fused producer variant: items from the shared producer map instead of a strided slice.
        .defines = {{"GDN_FUSED_PRODUCER", "1"}},
        // Size-optimised like the phased prep reader (see chunk_gdn_phased_program_factory.cpp).
        .opt_level = KernelBuildOptLevel::Os,
        .config = ReaderConfigDescriptor{},
    };
    prep_reader.runtime_args.reserve(P);

    KernelDescriptor prep_compute{
        .kernel_source = kdir + "compute/chunk_gdn_prep.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = prod_set,
        .compile_time_args = prep_compute_ct,
        // Fused-only perf: hoisted WY-path reconfigs (see chunk_gdn_math.hpp kGdnHoistReconfig).
        .defines = gdn_prep_defines(attrs.tinv, true /*hoist_reconfig*/),
        .config = gdn_compute_config(attrs.compute_kernel_config),
    };
    prep_compute.runtime_args.reserve(P);

    // The fused writer runs on the WriterConfigDescriptor's RISC/NoC (BRISC / NOC_1 on Blackhole).
    // Its multicast rectangles must be given in that NoC's own order: NOC_1 multicasts from the
    // bottom-right to the top-left, so the (start, end) pair is swapped there — the same idiom the
    // device layer applies in Device::get_noc_multicast_encoding. Coordinates themselves are virtual
    // and identical on both NoCs.
    const bool writer_on_noc1 = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch()) == NOC::NOC_1;

    KernelDescriptor fused_writer{
        .kernel_source = kdir + "dataflow/writer_chunk_gdn_fused.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = prod_set,
        .compile_time_args = fused_writer_ct,
        .config = WriterConfigDescriptor{},
    };
    fused_writer.runtime_args.reserve(P);

    KernelDescriptor receiver_reader{
        .kernel_source = kdir + "dataflow/reader_chunk_gdn_scan.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = rcv_set,
        .compile_time_args = receiver_ct,
        .defines = {{"GDN_FUSED_RECEIVER", "1"}},
        .config = ReaderConfigDescriptor{},
    };
    receiver_reader.runtime_args.reserve(R);

    KernelDescriptor scan_compute{
        .kernel_source = kdir + "compute/chunk_gdn_scan.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = rcv_set,
        .compile_time_args = ct_scan,
        .config = gdn_compute_config(attrs.compute_kernel_config),
    };
    scan_compute.runtime_args.reserve(R);

    KernelDescriptor scan_writer{
        .kernel_source = kdir + "dataflow/writer_chunk_gdn_scan.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = rcv_set,
        .compile_time_args = scan_writer_ct,
        .config = WriterConfigDescriptor{},
    };
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
    auto* s0_buf = in.initial_state.buffer();
    auto* o_buf = outputs[0].buffer();
    auto* fs_buf = outputs[1].buffer();

    // Virtual worker coords, packed x | y << 8 (Blackhole's virtual grid is below 256 on both axes).
    auto pack_xy = [&](const CoreCoord& logical) {
        const CoreCoord c = device->worker_core_from_logical_core(logical);
        TT_FATAL(c.x < 256 && c.y < 256, "chunk_gdn_fused: virtual core ({}, {}) does not pack into a byte", c.x, c.y);
        return static_cast<uint32_t>(c.x) | (static_cast<uint32_t>(c.y) << 8);
    };

    // Common runtime args. Receiver reader: the P producers' coords, two per word (producer p in word p/2,
    // high half when p is odd). Fused writer: per head, the multicast rectangle word (start x | start y << 8
    // | end x << 16 | end y << 24, ordered for the writer's NoC) then the NV receivers' coords, two per word.
    const uint32_t head_words = 1 + (NV + 1) / 2;
    std::vector<uint32_t> producer_table((P + 1) / 2, 0);
    std::vector<uint32_t> head_table(BH * head_words, 0);
    for (uint32_t p = 0; p < P; p++) {
        producer_table[p / 2] |= pack_xy(prod_cores[p]) << (16 * (p % 2));
    }

    // Each producer's items from the shared map: count, first chunk (kickoff stagger), the heads it serves
    // (init bumps), and per head the distinct producers serving it (the receivers' N_INIT).
    const uint32_t mask_words = (BH + 31) / 32;
    std::vector<uint32_t> n_items(P), c_first(P, 0), head_mask(P * mask_words, 0), n_init(BH, 0);
    for (uint32_t p = 0; p < P; p++) {
        n_items[p] = gdn_fused_item_count(map, p);
        for (uint32_t n = 0; n < n_items[p]; n++) {
            const GdnFusedItem it = gdn_fused_item(map, p, n);
            TT_FATAL(
                it.h < BH && it.c < NC && gdn_fused_owner(map, it.h, it.c) == p,
                "chunk_gdn_fused: producer map inconsistent at producer {} item {} (head {}, chunk {})",
                p,
                n,
                it.h,
                it.c);
            if (n == 0) {
                c_first[p] = it.c;
            }
            uint32_t& mask = head_mask[p * mask_words + it.h / 32];
            if (((mask >> (it.h % 32)) & 1u) == 0) {
                mask |= 1u << (it.h % 32);
                n_init[it.h]++;
            }
        }
    }
    TT_FATAL(
        std::accumulate(n_items.begin(), n_items.end(), 0u) == BH * NC,
        "chunk_gdn_fused: the producer map does not cover the {} items exactly once",
        BH * NC);

    for (uint32_t h = 0; h < BH; h++) {
        // Multicast rectangle = the receivers' bounding box (a 1xNV row in placement 0, an rw x rh block
        // for the leftover heads of placements 1/2). Density is checked in LOGICAL coords (no
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
        const CoreCoord& l_start = writer_on_noc1 ? l_br : l_tl;  // NOC_1: bottom-right -> top-left
        const CoreCoord& l_end = writer_on_noc1 ? l_tl : l_br;
        head_table[h * head_words] = pack_xy(l_start) | (pack_xy(l_end) << 16);
        for (uint32_t v = 0; v < NV; v++) {
            head_table[h * head_words + 1 + v / 2] |= pack_xy(rcv_cores[h * NV + v]) << (16 * (v % 2));
        }

        // Receiver (h, v): its V-slice index, s0 from DRAM, N_INIT (the distinct producers serving this
        // head, each bumps `init` once), the kickoff hold and the map; the producers' coords are common.
        for (uint32_t v = 0; v < NV; v++) {
            const CoreCoord& rc = rcv_cores[h * NV + v];
            receiver_reader.emplace_runtime_args(
                rc, {h, v, NC, s0_buf, n_init[h], kReceiverKickoffWaitCycles, BH, map.NPH, map.NX, map.num, map.den});
            scan_compute.emplace_runtime_args(rc, {NC});
            scan_writer.emplace_runtime_args(rc, {h, v, NC, o_buf, fs_buf});
        }
    }
    receiver_reader.common_runtime_args = producer_table;

    // Producer p: its map index and item count (the reader and writer walk the same list), the map, and for
    // the writer the heads it serves; the receivers' coords are common.
    for (uint32_t p = 0; p < P; p++) {
        const CoreCoord& pc = prod_cores[p];
        prep_reader.emplace_runtime_args(
            pc, {p,        n_items[p], q_buf,     k_buf,
                 v_buf,    g_buf,      beta_buf,  eye_buf,
                 tril_buf, ones_buf,   masks_buf, NC,
                 attrs.HV, attrs.Hk,   BH,        c_first[p] * kProducerKickoffStaggerCycles,
                 map.NPH,  map.NX,     map.num,   map.den});
        prep_compute.emplace_runtime_args(pc, {n_items[p]});
        std::vector<std::variant<uint32_t, Buffer*>> w_args = {
            p, n_items[p], BH, NC, map.NPH, map.NX, map.num, map.den};
        for (uint32_t w = 0; w < mask_words; w++) {
            w_args.push_back(head_mask[p * mask_words + w]);
        }
        fused_writer.emplace_runtime_args(pc, w_args);
    }
    fused_writer.common_runtime_args = head_table;

    desc.kernels.push_back(std::move(prep_reader));
    desc.kernels.push_back(std::move(prep_compute));
    desc.kernels.push_back(std::move(fused_writer));
    desc.kernels.push_back(std::move(receiver_reader));
    desc.kernels.push_back(std::move(scan_compute));
    desc.kernels.push_back(std::move(scan_writer));
    return desc;
}

}  // namespace ttnn::prim
