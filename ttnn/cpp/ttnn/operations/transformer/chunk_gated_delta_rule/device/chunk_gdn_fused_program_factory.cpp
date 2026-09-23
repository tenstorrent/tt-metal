// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the fused prep→scan chunk_gdn op: ONE program, two disjoint core sets.
// Per head h, NP PRODUCER cores run {unchanged prep reader, unchanged prep compute, NEW fused
// writer} and one RECEIVER core runs {NEW fused-receiver reader variant, unchanged scan compute,
// unchanged scan writer}. Each producer's writer NoC-writes the 7 computed intermediates
// (v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv) directly into the receiver's CBs via the
// shipped ready/valid handshake — zero DRAM intermediates. NP >= 1 producers/head (attrs.np,
// F3a: producer p owns chunks c = p, p+NP, ... round-robin; prep is chunk-independent, so the
// split is free of cross-chunk state), NV=1 (full V) per receiver.
//
// F3a in-order protocol (P1): the RECEIVER drives ordering — per chunk c it reserves the
// hand-off slots, then credits SEM_READY on producer (c % NP) only. At most one hand-off is in
// flight per receiver, so the single SEM_VALID flag never mixes chunks across producers.
//
// v_beta note (F1 deviates from the design plan's F1 line): the plan recommends Option R
// (scan-side recompute of v_beta from a DRAM v-slice + mcast beta). F1 ships Option U instead —
// the producer computes v_beta as today and SENDS it like the other six tensors — because it
// keeps both compute kernels byte-identical to the phased path (the bit-exactness gate needs
// that). Option R is deferred to F2.
//
// Hand-off CB addressing: the 7 hand-off CBs are declared FIRST, on the UNION of producer+
// receiver cores, so they get identical base addresses on both sides. The receiver reserves/
// pushes each CB exactly once per GLOBAL chunk c, so its reserved slot for chunk c is
// base + (c % nbuf)*slot_bytes; the writer computes that destination explicitly from the global
// chunk index (F3a). At NP=1 this degenerates to the F2 lockstep lemma (the producer's own read
// pointer names the same address); at NP>1 the producer's local slot index diverges from the
// receiver's — sourcing from read_ptr stays correct, but the DESTINATION must come from the
// global c, which is exactly what writer_chunk_gdn_fused.cpp does. After the F1 scan-side CB
// renumber (scan v_beta 17->14, dl 11->22, v_new 22->11) the seven hand-off indices coincide
// with prep's output indices, so the same physical CB is prep's output AND scan's input:
//   v_beta=14  kd=18  q_decay=19  intra=20  k_dec_t=24  dl=22  t_inv=13

#include "chunk_gdn_fused.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
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
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;  // NV=1: per-core Vt == Vt_full everywhere
    const uint32_t has_s0 = attrs.has_initial_state ? 1u : 0u;

    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt, kc = Kt * Ct;
    const uint32_t scr = std::max({cc, ck, cv, kv, kc});

    const tt::DataFormat df_qkv = tt::DataFormat::Float16_b;  // bf16 q/k/v (prep inputs)

    const uint32_t NP = attrs.np;  // producers per head (F3a); the op host clamps it to NC
    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        BH * (1 + NP) <= grid.x * grid.y,
        "chunk_gdn_fused: BH*(1+NP) ({}) cores needed, grid has {}",
        BH * (1 + NP),
        grid.x * grid.y);

    // Placement: receivers occupy row-major grid slots 0..BH-1, producers slots BH..BH+BH*NP-1;
    // producer j of head h sits at slot BH + h*NP + j and pairs with receiver h. (NV=1: 1x1
    // NoC rectangles everywhere, so the geometry is orientation-neutral and packing is free.)
    auto slot_core = [&](uint32_t slot) { return CoreCoord{slot % grid.x, slot / grid.x}; };
    std::vector<CoreCoord> rcv_cores(BH), prod_cores(BH * NP);
    std::set<CoreRange> rcv_crs, prod_crs, union_crs;
    for (uint32_t i = 0; i < BH; i++) {
        rcv_cores[i] = slot_core(i);
        rcv_crs.insert(CoreRange{rcv_cores[i], rcv_cores[i]});
        union_crs.insert(CoreRange{rcv_cores[i], rcv_cores[i]});
    }
    for (uint32_t i = 0; i < BH * NP; i++) {
        prod_cores[i] = slot_core(BH + i);
        prod_crs.insert(CoreRange{prod_cores[i], prod_cores[i]});
        union_crs.insert(CoreRange{prod_cores[i], prod_cores[i]});
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

    // (1) The 7 hand-off CBs FIRST, on the UNION core set: declared first => same base address on
    // producer and receiver (the slot-addressing precondition). fp32, in the receiver's reserve
    // order (v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv). Double-buffered (F2): the producer
    // runs one chunk ahead of the receiver's consumption instead of alternating with it. The
    // writer's explicit destination slot (global c % nbuf) must be computed against THIS depth,
    // so kHandoffNbuf also travels to the writer as a compile-time arg — one source of truth.
    constexpr uint32_t kHandoffNbuf = 2;
    add_cb(union_set, fcb::vbeta, cv, kHandoffNbuf);
    add_cb(union_set, fcb::kd, ck, kHandoffNbuf);
    add_cb(union_set, fcb::qdecay, ck, kHandoffNbuf);
    add_cb(union_set, fcb::intra, cc, kHandoffNbuf);
    add_cb(union_set, fcb::kdec_t, kc, kHandoffNbuf);
    // dl is 1 tile. (The phased prep factory sized this index cv as cb_vnew for monolithic layout
    // parity, but the prep kernel only ever uses 1 tile of it, as cb_dl.)
    add_cb(union_set, fcb::dl, 1, kHandoffNbuf);
    add_cb(union_set, fcb::Tinv, cc, kHandoffNbuf);

    // (2) The remaining 25 prep CBs on the PRODUCER cores only — same sizes/formats as the phased
    // prep factory (which mirrors the monolithic op's layout). The absolute L1 layout necessarily
    // shifts (the hand-off CBs above allocate first), which prior measurement showed to be
    // perf-neutral for prep; the math is layout-independent.
    // Producer input CBs are double-buffered (F2): the producer has no DRAM writes, so its reader
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
    add_cb(prod_set, fcb::u, cv);
    add_cb(prod_set, fcb::s2, kv, 2);
    add_cb(prod_set, fcb::ointer, cv);
    add_cb(prod_set, fcb::supd, kv);
    add_cb(prod_set, fcb::stmp, kv);
    add_cb(prod_set, fcb::final_s, kv);
    add_cb(prod_set, fcb::scr1, scr);
    add_cb(prod_set, fcb::scr2, scr);
    add_cb(prod_set, fcb::scr3, scr);
    add_cb(prod_set, fcb::s3, kv, 2);

    // (3) The remaining 10 scan CBs on the RECEIVER cores only, with Vtl = Vt_full (NV=1) and the
    // post-renumber indices (vnew = 11). o is fp32 (see compute_output_specs).
    add_cb(rcv_set, fcb::S, kv);
    add_cb(rcv_set, fcb::vnew, cv);
    add_cb(rcv_set, fcb::out, cv, 2, tt::DataFormat::Float32);
    add_cb(rcv_set, fcb::s2, kv);
    add_cb(rcv_set, fcb::ointer, cv);
    add_cb(rcv_set, fcb::supd, kv);
    add_cb(rcv_set, fcb::stmp, kv);
    add_cb(rcv_set, fcb::final_s, kv);
    add_cb(rcv_set, fcb::scr1, scr);
    add_cb(rcv_set, fcb::s3, kv);

    // Handshake semaphores, declared on the UNION so each id resolves to the same L1 address on
    // producer and receiver (the producer multicasts the VALID flag into the receiver's copy;
    // the receiver atomically incs the producer's ready copy). Ids reach both kernels as trailing
    // compile-time args — no kernel-side mirror constants to keep in sync.
    //   id 0 = ready (receiver -> producer: "my hand-off CB slots for this chunk are reserved")
    //   id 1 = valid (producer -> receiver: "this chunk's 7 tensors are in your CBs")
    constexpr uint32_t sem_ready_id = 0;
    constexpr uint32_t sem_valid_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_ready_id, .core_type = tt::CoreType::WORKER, .core_ranges = union_set, .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_valid_id, .core_type = tt::CoreType::WORKER, .core_ranges = union_set, .initial_value = 0});

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

    // Fused writer: 6 plain scalars, no accessors (it writes no DRAM at all). kHandoffNbuf feeds
    // the writer's explicit destination-slot arithmetic (global c % nbuf).
    const std::vector<uint32_t> fused_writer_ct = {Ct, Kt, Vt, sem_ready_id, sem_valid_id, kHandoffNbuf};

    // ---- Receiver-side CT args: the phased SCAN layout with per-core Vt == Vt_full ----
    const std::vector<uint32_t> ct_scan = {Ct, Kt, Vt, has_s0, Vt};

    // Fused-receiver reader: s0 is its ONLY DRAM tensor (chain of one accessor, starting at CT
    // index 5), then the semaphore ids at s0_a.next_compile_time_args_offset() and +1 — the same
    // trailing-args idiom the mcast scan reader variants use.
    std::vector<uint32_t> receiver_ct = ct_scan;
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(receiver_ct);
    receiver_ct.push_back(sem_ready_id);
    receiver_ct.push_back(sem_valid_id);

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
    prep_reader.runtime_args.reserve(BH * NP);

    KernelDescriptor prep_compute;
    prep_compute.kernel_source = kdir + "compute/chunk_gdn_prep.cpp";
    prep_compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    prep_compute.core_ranges = prod_set;
    prep_compute.compile_time_args = prep_compute_ct;
    prep_compute.config = fused_compute_cfg();
    // Fused-only perf: hoisted WY-path reconfigs (see chunk_gdn_math.hpp kGdnHoistReconfig).
    prep_compute.defines = {{"GDN_HOIST_RECONFIG", "1"}};
    prep_compute.runtime_args.reserve(BH * NP);

    KernelDescriptor fused_writer;
    fused_writer.kernel_source = kdir + "dataflow/writer_chunk_gdn_fused.cpp";
    fused_writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    fused_writer.core_ranges = prod_set;
    fused_writer.compile_time_args = fused_writer_ct;
    fused_writer.config = WriterConfigDescriptor{};
    fused_writer.runtime_args.reserve(BH * NP);

    KernelDescriptor receiver_reader;
    receiver_reader.kernel_source = kdir + "dataflow/reader_chunk_gdn_scan.cpp";
    receiver_reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_reader.core_ranges = rcv_set;
    receiver_reader.compile_time_args = receiver_ct;
    receiver_reader.defines = {{"GDN_FUSED_RECEIVER", "1"}};
    receiver_reader.config = ReaderConfigDescriptor{};
    receiver_reader.runtime_args.reserve(BH);

    KernelDescriptor scan_compute;
    scan_compute.kernel_source = kdir + "compute/chunk_gdn_scan.cpp";
    scan_compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    scan_compute.core_ranges = rcv_set;
    scan_compute.compile_time_args = ct_scan;
    scan_compute.config = fused_compute_cfg();
    scan_compute.runtime_args.reserve(BH);

    KernelDescriptor scan_writer;
    scan_writer.kernel_source = kdir + "dataflow/writer_chunk_gdn_scan.cpp";
    scan_writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    scan_writer.core_ranges = rcv_set;
    scan_writer.compile_time_args = scan_writer_ct;
    scan_writer.config = WriterConfigDescriptor{};
    scan_writer.runtime_args.reserve(BH);

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
        const CoreCoord& rc = rcv_cores[h];
        // Virtual worker coords for the cross-core NoC transactions. Every rectangle here is 1x1
        // (a single peer), which is orientation-neutral — no NOC_0/NOC_1 coordinate swap is
        // needed on either kernel regardless of which NoC the risc runs on.
        const CoreCoord rv = device->worker_core_from_logical_core(rc);

        // Receiver RT prefix: head h, vb=0 (full V), s0 from DRAM; then NP and the producers'
        // coords (appended in the producer loop below) for the rotating ready credit.
        std::vector<std::variant<uint32_t, Buffer*>> rcv_args = {h, 0u, NC, s0_buf, NP};

        for (uint32_t j = 0; j < NP; j++) {
            const CoreCoord& pc = prod_cores[h * NP + j];
            const CoreCoord pv = device->worker_core_from_logical_core(pc);
            rcv_args.push_back(static_cast<uint32_t>(pv.x));
            rcv_args.push_back(static_cast<uint32_t>(pv.y));

            // Producer j of head h owns the interleaved chunks c = j, j+NP, ... — as flat
            // work-items wi = h*NC + c that is start h*NC + j with stride NP (trailing reader
            // arg). The op host clamps NP <= NC, so every producer owns at least one chunk.
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
            fused_writer.emplace_runtime_args(
                pc, {NC, NP, j, static_cast<uint32_t>(rv.x), static_cast<uint32_t>(rv.y)});
        }

        receiver_reader.emplace_runtime_args(rc, rcv_args);
        scan_compute.emplace_runtime_args(rc, {NC});
        scan_writer.emplace_runtime_args(rc, {h, 0u, NC, o_buf, fs_buf});
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
