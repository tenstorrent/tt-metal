// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factories for the phase-split chunk_gdn ops.
//   PREP: the state-independent per-chunk math. Every (head, chunk) pair is an independent
//         work-item (no cross-chunk dependency), so we fan the full BH*NC set of work-items
//         out across the ENTIRE compute grid — mirroring FLA's WY-prep Triton kernels, whose
//         launch grid spans (chunk-tile index, batch*head). This is the perf payoff of the
//         phase split: prep runs chunk-parallel across dozens of cores instead of 1 core/head.
//   SCAN: one Tensix core per head walks chunks sequentially carrying state S [K,V]
//         (inherently sequential — the recurrence forbids chunk-parallelism here).

#include "chunk_gdn_phased.hpp"

#include <algorithm>
#include <cstdlib>
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

// CB index plan — kept in sync with the prep/scan compute + dataflow kernels.
// Uniquely named (pcb) so it does not ODR-clash with the sibling factory's pcb:: under unity builds.
namespace pcb {
constexpr uint32_t q = tt::CBIndex::c_0;
constexpr uint32_t k = tt::CBIndex::c_1;
constexpr uint32_t v = tt::CBIndex::c_2;
constexpr uint32_t g = tt::CBIndex::c_3;
constexpr uint32_t beta = tt::CBIndex::c_4;
constexpr uint32_t eye = tt::CBIndex::c_5;
constexpr uint32_t tril = tt::CBIndex::c_6;
constexpr uint32_t ones = tt::CBIndex::c_7;
constexpr uint32_t S = tt::CBIndex::c_8;
constexpr uint32_t decay = tt::CBIndex::c_9;
constexpr uint32_t decay_exp = tt::CBIndex::c_10;
constexpr uint32_t decayfac = tt::CBIndex::c_11;  // prep; reused as cb_dl in scan
constexpr uint32_t lmask = tt::CBIndex::c_12;
constexpr uint32_t Tinv = tt::CBIndex::c_13;
constexpr uint32_t vbeta = tt::CBIndex::c_14;
constexpr uint32_t kbeta = tt::CBIndex::c_15;
constexpr uint32_t out = tt::CBIndex::c_16;
constexpr uint32_t u = tt::CBIndex::c_17;
constexpr uint32_t w = tt::CBIndex::c_18;
constexpr uint32_t qdecay = tt::CBIndex::c_19;
constexpr uint32_t intra = tt::CBIndex::c_20;
constexpr uint32_t s2 = tt::CBIndex::c_21;
constexpr uint32_t vnew = tt::CBIndex::c_22;
constexpr uint32_t ointer = tt::CBIndex::c_23;
constexpr uint32_t kdec_t = tt::CBIndex::c_24;
constexpr uint32_t supd = tt::CBIndex::c_25;
constexpr uint32_t stmp = tt::CBIndex::c_26;
constexpr uint32_t final_s = tt::CBIndex::c_27;
constexpr uint32_t scr1 = tt::CBIndex::c_28;
constexpr uint32_t scr2 = tt::CBIndex::c_29;
constexpr uint32_t scr3 = tt::CBIndex::c_30;
constexpr uint32_t s3 = tt::CBIndex::c_31;
constexpr uint32_t dl = decayfac;  // scan reads dl into this slot
}  // namespace pcb

namespace {

ComputeConfigDescriptor compute_cfg(tt::ARCH arch, const DeviceComputeKernelConfig& config) {
    const auto args = get_compute_kernel_config_args(arch, config);
    return ComputeConfigDescriptor{
        .math_fidelity = std::get<0>(args),
        .fp32_dest_acc_en = std::get<2>(args),
        .math_approx_mode = std::get<1>(args)};
}

// Chunk-parallel work distribution for PREP: split `total` independent (head, chunk) work-items
// as evenly as possible across the compute grid. Work-item wi in [0,total) maps directly to the
// flat DRAM tile index used by every prep tensor ([BH, NC, ...] => wi = h*NC + c), so a core that
// is assigned a contiguous range [wi_start, wi_start+wi_count) reads/writes exactly those tiles —
// the DRAM result is byte-identical no matter how items are partitioned. The first `rem` cores get
// one extra item (ceil), the rest get floor; every used core gets >= 1 (since P <= total).
struct PrepWorkDist {
    std::vector<CoreCoord> cores;  // used cores, in assignment order
    std::vector<uint32_t> wi_start;
    std::vector<uint32_t> wi_count;
    CoreRangeSet core_set;  // all used cores (kernel placement + CB alloc)
};

// `core_cap` bounds how many cores we may use (defaults to the whole grid). Perf A/B only:
// setting core_cap=BH reproduces the old "1 core/head, NC chunks serial" layout exactly (each
// head's NC work-items are contiguous, wi in [h*NC, h*NC+NC)), using the identical kernels.
PrepWorkDist distribute_prep(CoreCoord grid, uint32_t total, uint32_t core_cap) {
    const uint32_t max_cores = std::min<uint32_t>(grid.x * grid.y, core_cap);
    const uint32_t P = std::min(total, max_cores);
    TT_FATAL(P > 0, "prep work distribution needs >= 1 work-item (total={})", total);
    const uint32_t base = total / P;
    const uint32_t rem = total % P;  // first `rem` cores get base+1

    PrepWorkDist d;
    d.cores.reserve(P);
    d.wi_start.reserve(P);
    d.wi_count.reserve(P);
    std::set<CoreRange> crs;
    uint32_t off = 0;
    for (uint32_t i = 0; i < P; i++) {
        const CoreCoord core{i % grid.x, i / grid.x};  // row-major over the grid
        const uint32_t cnt = base + (i < rem ? 1u : 0u);
        d.cores.push_back(core);
        d.wi_start.push_back(off);
        d.wi_count.push_back(cnt);
        crs.insert(CoreRange{core, core});
        off += cnt;
    }
    d.core_set = CoreRangeSet{crs};
    return d;
}

// The scan factorizes over V, but splitting V duplicates every V-independent tensor read and matmul setup.
// For Kimi KDA (Vt=4), one full-V core per head is 36% faster than two V-shards. Keep value
// splitting only as an explicit A/B knob until a larger-V crossover is measured.
struct ScanWorkDist {
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> head;  // head index per core
    std::vector<uint32_t> vblk;  // v-block index per core
    uint32_t Vtl = 1;            // per-core v-block width (tiles) = Vt / NV
    uint32_t NV = 1;             // v-blocks per head
    CoreRangeSet core_set;
};

ScanWorkDist distribute_scan(CoreCoord grid, uint32_t BH, uint32_t Vt, bool vector_gate) {
    const uint32_t ncores = grid.x * grid.y;
    TT_FATAL(BH <= ncores, "num_heads {} exceeds compute cores {}", BH, ncores);
    // The measured KDA crossover favors finest-grain V splitting at <=8 local heads and complete
    // V blocks at >=16. Scalar GDN retains its established full-V mapping.
    const char* split_env = std::getenv("QWEN_GDN_SCAN_VALUE_SPLIT");
    const bool value_split = split_env ? split_env[0] == '1' : (vector_gate && BH <= 8);
    uint32_t NV = 1;
    if (value_split) {
        for (uint32_t cand = Vt; cand >= 1; cand--) {  // cand==1 always satisfies
            if (Vt % cand == 0 && BH * cand <= ncores) {
                NV = cand;
                break;
            }
        }
    }
    ScanWorkDist d;
    d.NV = NV;
    d.Vtl = Vt / NV;
    const uint32_t total = BH * NV;
    d.cores.reserve(total);
    d.head.reserve(total);
    d.vblk.reserve(total);
    std::set<CoreRange> crs;
    for (uint32_t i = 0; i < total; i++) {
        const CoreCoord core{i % grid.x, i / grid.x};  // row-major over the grid
        d.cores.push_back(core);
        d.head.push_back(i / NV);  // heads' v-blocks grouped contiguously
        d.vblk.push_back(i % NV);
        crs.insert(CoreRange{core, core});
    }
    d.core_set = CoreRangeSet{crs};
    return d;
}

}  // namespace

uint32_t chunk_gdn_prep_cb_size_bytes(
    uint32_t chunk_size,
    uint32_t key_dim,
    uint32_t val_dim,
    bool vector_gate,
    DataType gate_dtype,
    uint32_t output_bf16_mask) {
    const uint32_t Ct = chunk_size / TILE_HEIGHT;
    const uint32_t Kt = key_dim / TILE_WIDTH;
    const uint32_t Vt = val_dim / TILE_WIDTH;
    const uint32_t cc = Ct * Ct;
    const uint32_t ck = Ct * Kt;
    const uint32_t cv = Ct * Vt;
    const uint32_t kv = Kt * Vt;
    const uint32_t kc = Kt * Ct;
    const uint32_t scr = std::max({cc, ck, cv, kv, kc});
    const auto output_format = [&](uint32_t index) {
        return (output_bf16_mask & (1u << index)) ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;
    };
    uint32_t bytes = 0;
    const auto add = [&](uint32_t tiles, uint32_t buffers = 1, tt::DataFormat format = tt::DataFormat::Float32) {
        bytes += tiles * buffers * tt::tile_size(format);
    };
    constexpr auto bf16 = tt::DataFormat::Float16_b;
    add(ck, 1, bf16);  // q
    add(ck, 1, bf16);  // k
    add(cv, 1, bf16);  // v
    add(vector_gate ? ck : Ct, 1, tt::tt_metal::datatype_to_dataformat_converter(gate_dtype));
    add(Ct);                                                        // beta
    add(cc);                                                        // eye
    add(cc);                                                        // tril
    add(cc);                                                        // ones
    add(kv, 2);                                                     // S
    add(vector_gate ? ck : Ct);                                     // decay
    add(vector_gate ? ck : Ct);                                     // decay_exp
    add(vector_gate ? ck : Ct);                                     // decayfac
    add(cc);                                                        // lmask
    add(cc, 1, output_format(6));                                   // Tinv
    add(cv, 1, output_format(0));                                   // vbeta
    add(ck);                                                        // kbeta
    add(cv, 2, bf16);                                               // out
    add(std::max(cv, 3u));                                          // u
    add(ck, 1, output_format(1));                                   // w
    add(ck, 1, output_format(2));                                   // qdecay
    add(cc, 1, output_format(3));                                   // intra
    add(kv, 2);                                                     // s2
    add(vector_gate ? std::max(cv, Kt) : cv, 1, output_format(5));  // vnew / dl
    add(cv);                                                        // ointer
    add(kc, 1, output_format(4));                                   // kdec_t
    add(kv);                                                        // supd
    add(kv);                                                        // stmp
    add(kv);                                                        // final_s
    add(scr);                                                       // scr1
    add(scr);                                                       // scr2
    add(scr);                                                       // scr3
    add(kv, 2);                                                     // s3
    return bytes;
}

// ---------------------------------------------------------------------------
// PREP
// ---------------------------------------------------------------------------
tt::tt_metal::ProgramDescriptor ChunkGdnPrepProgramFactory::create_descriptor(
    const ChunkGdnPrepParams& attrs, const ChunkGdnPrepInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t Ct = attrs.chunk_size / TILE_HEIGHT;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;

    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt, kc = Kt * Ct;
    uint32_t scr = std::max({cc, ck, cv, kv, kc});

    const tt::DataFormat df_io = tt::DataFormat::Float16_b;  // bf16 q/k/v

    auto* device = in.q.device();
    // Fan the BH*NC independent (head, chunk) prep work-items across the whole grid.
    // QWEN_GDN_PREP_SERIAL=1 caps to BH cores (old 1-core/head layout) for perf A/B only.
    const uint32_t total_work = BH * NC;
    const char* serial_env = std::getenv("QWEN_GDN_PREP_SERIAL");
    const uint32_t core_cap = (serial_env && serial_env[0] == '1') ? BH : ~0u;
    auto dist = distribute_prep(device->compute_with_storage_grid_size(), total_work, core_cap);
    const CoreRangeSet& cores = dist.core_set;
    const uint32_t n_used = static_cast<uint32_t>(dist.cores.size());

    uint32_t cb_size_bytes = 0;
    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf = 1, tt::DataFormat fmt = tt::DataFormat::Float32) {
        const uint32_t ts = tt::tile_size(fmt);
        const uint32_t total_size = n_tiles * nbuf * ts;
        cb_size_bytes += total_size;
        desc.cbs.push_back(CBDescriptor{
            .total_size = total_size,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    // Allocate the full CB set in the SAME order/sizes as the monolithic op, so the prep phase's
    // L1 layout is byte-identical (the Horner's matmul L1 access pattern is layout-sensitive).
    add_cb(pcb::q, ck, 1, df_io);
    add_cb(pcb::k, ck, 1, df_io);
    add_cb(pcb::v, cv, 1, df_io);
    add_cb(pcb::g, attrs.vector_gate ? ck : Ct, 1, tt::tt_metal::datatype_to_dataformat_converter(in.g.dtype()));
    add_cb(pcb::beta, Ct);
    add_cb(pcb::eye, cc);
    add_cb(pcb::tril, cc);
    add_cb(pcb::ones, cc);
    add_cb(pcb::S, kv, 2);
    add_cb(pcb::decay, attrs.vector_gate ? ck : Ct);
    add_cb(pcb::decay_exp, attrs.vector_gate ? ck : Ct);
    add_cb(pcb::decayfac, attrs.vector_gate ? ck : Ct);
    add_cb(pcb::lmask, cc);
    const auto output_df = [&](uint32_t index) {
        return tt::tt_metal::datatype_to_dataformat_converter(outputs[index].dtype());
    };
    add_cb(pcb::Tinv, cc, 1, output_df(6));
    add_cb(pcb::vbeta, cv, 1, output_df(0));
    add_cb(pcb::kbeta, ck);
    add_cb(pcb::out, cv, 2, df_io);
    add_cb(pcb::u, std::max(cv, 3u));  // startup pacing tiles; then unused scratch
    add_cb(pcb::w, ck, 1, output_df(1));
    add_cb(pcb::qdecay, ck, 1, output_df(2));
    add_cb(pcb::intra, cc, 1, output_df(3));
    add_cb(pcb::s2, kv, 2);
    add_cb(pcb::vnew, attrs.vector_gate ? std::max(cv, Kt) : cv, 1, output_df(5));  // aliased as cb_dl
    add_cb(pcb::ointer, cv);
    add_cb(pcb::kdec_t, kc, 1, output_df(4));
    add_cb(pcb::supd, kv);
    add_cb(pcb::stmp, kv);
    add_cb(pcb::final_s, kv);
    add_cb(pcb::scr1, scr);

    add_cb(pcb::scr2, scr);
    add_cb(pcb::scr3, scr);
    add_cb(pcb::s3, kv, 2);
    TT_FATAL(
        cb_size_bytes == chunk_gdn_prep_cb_size_bytes(
                             attrs.chunk_size,
                             attrs.key_dim,
                             attrs.val_dim,
                             attrs.vector_gate,
                             in.g.dtype(),
                             attrs.output_bf16_mask),
        "KDA prep CB size estimator is out of sync with the program factory");

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";
    const std::vector<uint32_t> ct_args = {Ct, Kt, Vt};

    std::vector<uint32_t> reader_ct = ct_args;
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.eye_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tril_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.ones_c.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.masks_c.buffer()).append_to(reader_ct);
    // OPT-A: trailing compile args after all TensorAccessorArgs — 1 => read that tensor flat token-major.
    reader_ct.push_back(attrs.v_flat ? 1u : 0u);
    reader_ct.push_back(attrs.qk_flat ? 1u : 0u);
    reader_ct.push_back(attrs.g_flat ? 1u : 0u);

    std::vector<uint32_t> writer_ct = ct_args;
    for (auto& t : outputs) {
        TensorAccessorArgs(*t.buffer()).append_to(writer_ct);
    }

    KernelDescriptor reader;
    reader.kernel_source =
        kdir + (attrs.vector_gate ? "dataflow/reader_chunk_kda_prep.cpp" : "dataflow/reader_chunk_gdn_prep.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(n_used);

    KernelDescriptor writer;
    writer.kernel_source =
        kdir + (attrs.vector_gate ? "dataflow/writer_chunk_kda_prep.cpp" : "dataflow/writer_chunk_gdn_prep.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_used);

    KernelDescriptor compute;
    compute.kernel_source = kdir + (attrs.vector_gate ? "compute/chunk_kda_prep.cpp" : "compute/chunk_gdn_prep.cpp");
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    // Compute gets extra args for the in-kernel q/k L2-norm (OPT-B): QK_NORM flag, and scale/eps as
    // fp32 bit patterns. QK_NORM is only enabled for Ct==1 (attrs.qk_norm gated on chunk_size==32).
    auto f32_bits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    std::vector<uint32_t> compute_ct = ct_args;
    compute_ct.push_back(attrs.qk_norm ? 1u : 0u);
    compute_ct.push_back(f32_bits(attrs.scale));
    compute_ct.push_back(f32_bits(1e-6f));
    compute.compile_time_args = compute_ct;
    compute.config = compute_cfg(device->arch(), attrs.compute_kernel_config);
    compute.runtime_args.reserve(n_used);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* g_buf = in.g.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* eye_buf = in.eye_c.buffer();
    auto* tril_buf = in.tril_c.buffer();
    auto* ones_buf = in.ones_c.buffer();
    auto* masks_buf = in.masks_c.buffer();
    auto* vb_buf = outputs[0].buffer();    // v_beta
    auto* kd_buf = outputs[1].buffer();    // kd = k_beta*decay_exp
    auto* qd_buf = outputs[2].buffer();    // q_decay
    auto* it_buf = outputs[3].buffer();    // intra
    auto* kdec_buf = outputs[4].buffer();  // k_dec_t
    auto* dl_buf = outputs[5].buffer();    // dl
    auto* ti_buf = outputs[6].buffer();    // t_inv

    // Each used core processes its contiguous slice [wi_start, wi_start+wi_count) of the BH*NC
    // work-items. wi is the flat DRAM tile index (h*NC + c), so the kernels need no h/c at all.
    for (uint32_t i = 0; i < n_used; i++) {
        const auto& core = dist.cores[i];
        const uint32_t wi_start = dist.wi_start[i];
        const uint32_t wi_count = dist.wi_count[i];
        // Trailing runtime args NC, HV, Hk are consumed by the reader's flat branches.
        reader.emplace_runtime_args(
            core,
            {wi_start,
             wi_count,
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
             attrs.Hk});
        writer.emplace_runtime_args(
            core, {wi_start, wi_count, vb_buf, kd_buf, qd_buf, it_buf, kdec_buf, dl_buf, ti_buf});
        compute.emplace_runtime_args(core, {wi_count});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

// ---------------------------------------------------------------------------
// SCAN
// ---------------------------------------------------------------------------
tt::tt_metal::ProgramDescriptor ChunkGdnScanProgramFactory::create_descriptor(
    const ChunkGdnScanParams& attrs, const ChunkGdnScanInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t NC = attrs.num_chunks;
    const uint32_t Ct = attrs.chunk_size / TILE_HEIGHT;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt_full = attrs.val_dim / TILE_WIDTH;
    const uint32_t initial_state_mode = attrs.identity_initial_state ? 2u : (attrs.has_initial_state ? 0u : 1u);

    // o output is fp32 (matches the scan op's compute_output_specs; a bf16 o degraded full-model
    // quality and was removed). cb_out format must match, else the writer strides wrong.
    const tt::DataFormat df_io = tt::DataFormat::Float32;

    auto* device = in.v_beta.device();
    // Value-parallel fan-out: each core runs one (head, v-block) sequential scan.
    auto sdist =
        distribute_scan(device->compute_with_storage_grid_size(), BH, Vt_full, attrs.vector_gate && !attrs.state_only);
    const CoreRangeSet& cores = sdist.core_set;
    const uint32_t Vt = sdist.Vtl;  // per-core V-block width; CBs/compute use this
    const uint32_t n_used = static_cast<uint32_t>(sdist.cores.size());

    std::vector<CoreCoord> rms_cores;
    std::set<CoreRange> rms_ranges;
    if (attrs.fused_rms) {
        const auto grid = device->compute_with_storage_grid_size();
        for (uint32_t i = 0; i < grid.x * grid.y; ++i) {
            const CoreCoord core{i % grid.x, i / grid.x};
            if (std::find(sdist.cores.begin(), sdist.cores.end(), core) == sdist.cores.end()) {
                rms_cores.push_back(core);
                rms_ranges.insert(CoreRange{core, core});
            }
        }
        TT_FATAL(!rms_cores.empty(), "fused RMS requires non-scan consumer cores");
    }
    const CoreRangeSet rms_core_set{rms_ranges};
    std::set<CoreRange> pipeline_ranges = rms_ranges;
    for (const auto& core : sdist.cores) {
        pipeline_ranges.insert(CoreRange{core, core});
    }
    const CoreRangeSet pipeline_core_set{pipeline_ranges};

    // V-independent tensors (kd, q_decay, intra, k_dec_t, T_inv, dl) are read in FULL; only the
    // V-dependent CBs (v_beta/state/out/scratch) shrink to the per-core V-block width Vt(=Vtl).
    const uint32_t cc = Ct * Ct, ck = Ct * Kt, cv = Ct * Vt, kv = Kt * Vt, kc = Kt * Ct;
    uint32_t scr = std::max({cc, ck, cv, kv, kc});

    ProgramDescriptor desc;
    if (attrs.fused_rms) {
        const uint32_t consumer_count = static_cast<uint32_t>(rms_cores.size());
        const uint32_t max_items_per_consumer = tt::div_up(BH * NC, consumer_count);
        const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float32);
        desc.cbs.push_back(CBDescriptor{
            .total_size = max_items_per_consumer * Vt_full * tile_size,
            .core_ranges = pipeline_core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = 0, .data_format = tt::DataFormat::Float32, .page_size = tile_size}}}});
    }
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf = 1, tt::DataFormat fmt = tt::DataFormat::Float32) {
        const uint32_t ts = tt::tile_size(fmt);
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    // The dual-summary specialization reads only state-update tensors and repurposes
    // the unused output-side CBs as a second state ping-pong. Normal scan allocation is unchanged.
    const auto input_df = [](const Tensor& tensor) {
        return tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    };
    add_cb(pcb::u, cv, 1, input_df(in.v_beta));
    add_cb(pcb::w, ck, 1, input_df(in.kd));
    add_cb(pcb::kdec_t, kc, 1, input_df(in.k_dec_t));
    add_cb(pcb::dl, attrs.vector_gate ? Kt : 1, 1, input_df(in.dl));
    add_cb(pcb::Tinv, cc, 1, input_df(in.t_inv));
    add_cb(pcb::S, kv);
    add_cb(pcb::s2, kv);
    add_cb(pcb::s3, kv);
    add_cb(pcb::final_s, kv);
    add_cb(pcb::vnew, cv);
    add_cb(pcb::supd, kv);
    add_cb(pcb::stmp, kv);
    add_cb(pcb::scr1, scr);
    if (attrs.summary_pair) {
        add_cb(pcb::qdecay, kv);         // identity-seeded second initial state
        add_cb(pcb::intra, kv);          // second-state ping
        add_cb(pcb::ointer, kv);         // second-state pong
        add_cb(pcb::out, kv, 1, df_io);  // second final state before subtraction
    } else {
        add_cb(pcb::qdecay, ck, 1, input_df(in.q_decay));
        add_cb(pcb::intra, cc, 1, input_df(in.intra));
        add_cb(pcb::ointer, cv);
        add_cb(pcb::out, cv, 2, df_io);
    }

    if (attrs.fused_rms) {
        auto add_rms_cb = [&](uint32_t idx, uint32_t tiles, tt::DataFormat format, uint32_t buffers = 1) {
            const uint32_t tile_size = tt::tile_size(format);
            desc.cbs.push_back(CBDescriptor{
                .total_size = tiles * buffers * tile_size,
                .core_ranges = rms_core_set,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(idx), .data_format = format, .page_size = tile_size}}}});
        };
        add_rms_cb(1, Vt_full, tt::DataFormat::Float16_b);
        add_rms_cb(2, Vt_full, tt::DataFormat::Float16_b);
        add_rms_cb(3, Vt_full, tt::DataFormat::Float32);
        add_rms_cb(4, 1, tt::DataFormat::Float32);
        add_rms_cb(5, 1, tt::DataFormat::Float32);
        add_rms_cb(6, Vt_full, tt::DataFormat::Float32);
        add_rms_cb(7, Vt_full, tt::DataFormat::Float32, 2);
        add_rms_cb(8, 1, tt::DataFormat::Float32);
    }

    const uint32_t ready_semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    if (attrs.fused_rms) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = ready_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = rms_core_set,
            .initial_value = 0});
    }

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";
    // ct arg 2 = per-core Vt(=Vtl); arg 4 = Vt_full (full V in tiles) for the readers'/writer's
    // V-slice row stride. Compute reads only args 0..2 (Ct, Kt, Vt) so the extra arg is harmless.
    const std::vector<uint32_t> ct_args = {
        Ct, Kt, Vt, initial_state_mode, Vt_full, attrs.state_only ? 1u : 0u, attrs.summary_pair ? 1u : 0u};

    std::vector<uint32_t> reader_ct = ct_args;
    TensorAccessorArgs(*in.v_beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.kd.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.q_decay.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.intra.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k_dec_t.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.dl.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.t_inv.buffer()).append_to(reader_ct);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);
    TensorAccessorArgs(in.identity_tile.has_value() ? in.identity_tile->buffer() : nullptr).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = ct_args;
    if (!attrs.fused_rms) {
        TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    }
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source =
        kdir + (attrs.vector_gate ? "dataflow/reader_chunk_kda_scan.cpp" : "dataflow/reader_chunk_gdn_scan.cpp");
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(n_used);

    KernelDescriptor writer;
    writer.kernel_source =
        kdir + (attrs.fused_rms ? "dataflow/writer_chunk_gdn_scan_rms.cpp" : "dataflow/writer_chunk_gdn_scan.cpp");
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(n_used);

    KernelDescriptor compute;
    compute.kernel_source = kdir + (attrs.vector_gate ? "compute/chunk_kda_scan.cpp" : "compute/chunk_gdn_scan.cpp");
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = ct_args;
    compute.config = compute_cfg(device->arch(), attrs.compute_kernel_config);
    compute.runtime_args.reserve(n_used);

    auto* vb_buf = in.v_beta.buffer();
    auto* kd_buf = in.kd.buffer();
    auto* qd_buf = in.q_decay.buffer();
    auto* it_buf = in.intra.buffer();
    auto* kdec_buf = in.k_dec_t.buffer();
    auto* dl_buf = in.dl.buffer();
    auto* ti_buf = in.t_inv.buffer();
    auto* s0_buf = in.initial_state.has_value() ? in.initial_state->buffer() : nullptr;
    auto* identity_buf = in.identity_tile.has_value() ? in.identity_tile->buffer() : nullptr;
    auto* o_buf = outputs[0].buffer();
    auto* fs_buf = outputs[1].buffer();

    for (uint32_t i = 0; i < n_used; i++) {
        const auto& core = sdist.cores[i];
        const uint32_t h = sdist.head[i];
        const uint32_t vb = sdist.vblk[i];
        reader.emplace_runtime_args(
            core, {h, vb, NC, vb_buf, kd_buf, qd_buf, it_buf, kdec_buf, dl_buf, ti_buf, s0_buf, identity_buf});
        if (attrs.fused_rms) {
            std::vector<std::variant<uint32_t, Buffer*>> writer_args{
                h, vb, NC, fs_buf, static_cast<uint32_t>(rms_cores.size()), ready_semaphore_id};
            for (const auto& rms_core : rms_cores) {
                const auto physical = device->worker_core_from_logical_core(rms_core);
                writer_args.emplace_back(static_cast<uint32_t>(physical.x));
                writer_args.emplace_back(static_cast<uint32_t>(physical.y));
            }
            writer.emplace_runtime_args(core, std::move(writer_args));
        } else {
            writer.emplace_runtime_args(core, {h, vb, NC, o_buf, fs_buf});
        }
        compute.emplace_runtime_args(core, {NC});
    }

    if (attrs.fused_rms) {
        uint32_t eps_bits = 0;
        uint32_t inv_v_bits = 0;
        const float inv_v = 1.0f / static_cast<float>(attrs.val_dim);
        std::memcpy(&eps_bits, &attrs.rms_epsilon, sizeof(float));
        std::memcpy(&inv_v_bits, &inv_v, sizeof(float));
        const uint32_t total = attrs.BH * NC;
        const uint32_t consumer_count = static_cast<uint32_t>(rms_cores.size());
        const uint32_t producer_count = Vt_full / Vt;

        std::vector<uint32_t> rms_reader_ct{Vt_full, attrs.num_heads, NC};
        TensorAccessorArgs(*in.rms_gate->buffer()).append_to(rms_reader_ct);
        TensorAccessorArgs(*in.rms_weight->buffer()).append_to(rms_reader_ct);
        std::vector<uint32_t> rms_writer_ct{Vt_full, attrs.num_heads, NC};
        TensorAccessorArgs(*outputs[2].buffer()).append_to(rms_writer_ct);

        KernelDescriptor rms_reader;
        rms_reader.kernel_source = kdir + "dataflow/reader_kda_gated_rms_stream.cpp";
        rms_reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        rms_reader.core_ranges = rms_core_set;
        rms_reader.compile_time_args = rms_reader_ct;
        rms_reader.config = ReaderConfigDescriptor{};

        KernelDescriptor rms_writer;
        rms_writer.kernel_source = kdir + "dataflow/writer_kda_gated_rms_stream.cpp";
        rms_writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        rms_writer.core_ranges = rms_core_set;
        rms_writer.compile_time_args = rms_writer_ct;
        rms_writer.config = WriterConfigDescriptor{};

        KernelDescriptor rms_compute;
        rms_compute.kernel_source = kdir + "compute/kda_gated_rms.cpp";
        rms_compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
        rms_compute.core_ranges = rms_core_set;
        rms_compute.compile_time_args = {Vt_full, eps_bits, inv_v_bits};
        rms_compute.config = compute_cfg(device->arch(), attrs.compute_kernel_config);

        auto* gate_buf = in.rms_gate->buffer();
        auto* weight_buf = in.rms_weight->buffer();
        auto* rms_out_buf = outputs[2].buffer();
        for (uint32_t i = 0; i < consumer_count; ++i) {
            const uint32_t count = i < total ? tt::div_up(total - i, consumer_count) : 0;
            const auto& core = rms_cores[i];
            rms_reader.emplace_runtime_args(
                core, {i, count, gate_buf, weight_buf, consumer_count, producer_count, ready_semaphore_id});
            rms_writer.emplace_runtime_args(core, {i, count, rms_out_buf, consumer_count});
            rms_compute.emplace_runtime_args(core, {count});
        }
        desc.kernels.push_back(std::move(rms_reader));
        desc.kernels.push_back(std::move(rms_writer));
        desc.kernels.push_back(std::move(rms_compute));
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

// ---------------------------------------------------------------------------
// KDA GROUPED AFFINE PREFIX
// ---------------------------------------------------------------------------
tt::tt_metal::ProgramDescriptor KdaAffinePrefixProgramFactory::create_descriptor(
    const KdaAffinePrefixParams& attrs, const KdaAffinePrefixInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    const uint32_t G = attrs.groups_per_head;
    const uint32_t group_heads = attrs.BH * G;
    const uint32_t kk = Kt * Kt;
    const uint32_t kv = Kt * Vt;

    auto* device = in.transform_a.device();
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(group_heads <= grid.x * grid.y, "affine prefix requires one worker per group");
    auto dist = distribute_prep(grid, group_heads, group_heads);
    const auto& cores = dist.core_set;

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t index, uint32_t tiles) {
        const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float32);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * tile_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = tt::DataFormat::Float32,
                .page_size = tile_size}}}});
    };
    add_cb(0, kk);   // input A
    add_cb(1, kv);   // input B
    add_cb(2, kk);   // prefix A ping
    add_cb(3, kv);   // prefix B ping
    add_cb(4, kk);   // prefix A pong
    add_cb(5, kv);   // prefix B pong
    add_cb(6, kk);   // receiver-owned inbound A
    add_cb(7, kv);   // receiver-owned inbound B
    add_cb(8, kv);   // initial state
    add_cb(9, kv);   // group entry state
    add_cb(10, kv);  // matmul scratch
    add_cb(11, 1);   // dataflow-to-compute stage token

    constexpr uint32_t ready_semaphore_id = 0;
    constexpr uint32_t arrival_semaphore_id = 1;
    constexpr uint32_t release_semaphore_id = 2;
    for (uint32_t id : {ready_semaphore_id, arrival_semaphore_id, release_semaphore_id}) {
        desc.semaphores.push_back(
            SemaphoreDescriptor{.id = id, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
    }

    auto* state_buffer = in.initial_state.has_value() ? in.initial_state->buffer() : in.transform_b.buffer();
    auto* output_a_buffer = outputs[0].buffer();
    auto* output_b_buffer = attrs.compose_only ? outputs[1].buffer() : outputs[0].buffer();
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";
    std::vector<uint32_t> dataflow_ct = {Kt, Vt, attrs.BH, G, attrs.compose_only};
    TensorAccessorArgs(*in.transform_a.buffer()).append_to(dataflow_ct);
    TensorAccessorArgs(*in.transform_b.buffer()).append_to(dataflow_ct);
    TensorAccessorArgs(*state_buffer).append_to(dataflow_ct);
    TensorAccessorArgs(*output_a_buffer).append_to(dataflow_ct);
    TensorAccessorArgs(*output_b_buffer).append_to(dataflow_ct);

    KernelDescriptor dataflow;
    dataflow.kernel_source = kdir + "dataflow/reader_writer_kda_affine_prefix.cpp";
    dataflow.source_type = KernelDescriptor::SourceType::FILE_PATH;
    dataflow.core_ranges = cores;
    dataflow.compile_time_args = dataflow_ct;
    dataflow.config = ReaderConfigDescriptor{};
    dataflow.runtime_args.reserve(group_heads);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/kda_affine_prefix.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {Kt, Vt, G, attrs.compose_only};
    compute.config = compute_cfg(device->arch(), attrs.compute_kernel_config);
    compute.runtime_args.reserve(group_heads);

    auto* a_buffer = in.transform_a.buffer();
    auto* b_buffer = in.transform_b.buffer();
    const auto coordinator = device->worker_core_from_logical_core(dist.cores[0]);
    for (uint32_t flat = 0; flat < group_heads; flat++) {
        const auto& core = dist.cores[flat];
        const uint32_t group = flat % G;
        KernelDescriptor::RTArgList args;
        args.reserve(13 + 2 * group_heads);
        args.push_back(flat);
        args.push_back(group);
        args.push_back(group_heads);
        args.push_back(a_buffer);
        args.push_back(b_buffer);
        args.push_back(state_buffer);
        args.push_back(output_a_buffer);
        args.push_back(output_b_buffer);
        args.push_back(ready_semaphore_id);
        args.push_back(arrival_semaphore_id);
        args.push_back(release_semaphore_id);
        args.push_back(coordinator.x);
        args.push_back(coordinator.y);
        for (const auto& worker : dist.cores) {
            const auto physical = device->worker_core_from_logical_core(worker);
            args.push_back(physical.x);
            args.push_back(physical.y);
        }
        dataflow.emplace_runtime_args(core, std::move(args));
        compute.emplace_runtime_args(core, {group});
    }

    desc.kernels.push_back(std::move(dataflow));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

tt::tt_metal::ProgramDescriptor KdaGatedRmsProgramFactory::create_descriptor(
    const KdaGatedRmsParams& attrs, const KdaGatedRmsInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t Mt = attrs.sequence / TILE_HEIGHT;
    const uint32_t Vt = attrs.value_dim / TILE_WIDTH;
    const uint32_t total = attrs.batch * attrs.num_heads * Mt;
    // Use the fewest workers that preserve the all-core maximum items/worker.
    const auto grid = in.input.device()->compute_with_storage_grid_size();
    const uint32_t max_items_per_core = tt::div_up(total, grid.x * grid.y);
    const uint32_t rms_core_limit = tt::div_up(total, max_items_per_core);
    auto dist = distribute_prep(in.input.device()->compute_with_storage_grid_size(), total, rms_core_limit);
    const auto& cores = dist.core_set;

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t tiles, tt::DataFormat format, uint32_t buffers = 1) {
        const uint32_t tile_size = tt::tile_size(format);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * buffers * tile_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = format, .page_size = tile_size}}}});
    };
    add_cb(0, Vt, tt::DataFormat::Float32);
    add_cb(1, Vt, tt::DataFormat::Float16_b);
    add_cb(2, Vt, tt::DataFormat::Float16_b);
    add_cb(3, Vt, tt::DataFormat::Float32);
    add_cb(4, 1, tt::DataFormat::Float32);
    add_cb(5, 1, tt::DataFormat::Float32);
    add_cb(6, Vt, tt::DataFormat::Float32);
    add_cb(7, Vt, tt::DataFormat::Float32, 2);
    add_cb(8, 1, tt::DataFormat::Float32);

    uint32_t eps_bits = 0;
    uint32_t inv_v_bits = 0;
    const float inv_v = 1.0f / static_cast<float>(attrs.value_dim);
    std::memcpy(&eps_bits, &attrs.epsilon, sizeof(float));
    std::memcpy(&inv_v_bits, &inv_v, sizeof(float));
    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";

    std::vector<uint32_t> reader_ct = {Vt, attrs.num_heads, Mt};
    TensorAccessorArgs(*in.input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.gate.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.weight.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {Vt, attrs.num_heads, Mt};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_kda_gated_rms.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_kda_gated_rms.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/kda_gated_rms.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {Vt, eps_bits, inv_v_bits};
    compute.config = compute_cfg(in.input.device()->arch(), attrs.compute_kernel_config);

    auto* input_buffer = in.input.buffer();
    auto* gate_buffer = in.gate.buffer();
    auto* weight_buffer = in.weight.buffer();
    auto* output_buffer = outputs[0].buffer();
    for (uint32_t i = 0; i < dist.cores.size(); i++) {
        const auto& core = dist.cores[i];
        reader.emplace_runtime_args(
            core, {dist.wi_start[i], dist.wi_count[i], input_buffer, gate_buffer, weight_buffer});
        writer.emplace_runtime_args(core, {dist.wi_start[i], dist.wi_count[i], output_buffer});
        compute.emplace_runtime_args(core, {dist.wi_count[i]});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

tt::tt_metal::ProgramDescriptor KdaCausalConvProgramFactory::create_descriptor(
    const KdaCausalConvParams& attrs, const KdaCausalConvInputs& in, std::vector<Tensor>& outputs) {
    constexpr uint32_t act_rm_cb = tt::CBIndex::c_0;
    constexpr uint32_t act_tile_cb = tt::CBIndex::c_1;
    constexpr uint32_t weights_cb = tt::CBIndex::c_2;
    constexpr uint32_t partial_a_cb = tt::CBIndex::c_3;
    constexpr uint32_t partial_b_cb = tt::CBIndex::c_4;
    constexpr uint32_t output_cb = tt::CBIndex::c_5;
    const uint32_t Mt = attrs.sequence / TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t channels = attrs.q_width + attrs.k_width + attrs.v_width;
    const uint32_t row_bytes = channels * sizeof(uint16_t);
    // Preserve the single-block TP8 case. Wider local head shards use smaller blocks so the nine CBs coexist in L1.
    uint32_t block_ct = Ct <= 48 ? Ct : 24u;
    while (Ct % block_ct != 0) {
        --block_ct;
    }
    const uint32_t num_blocks = Ct / block_ct;
    auto dist = distribute_prep(in.input.device()->compute_with_storage_grid_size(), Mt * num_blocks, ~0u);
    const auto& cores = dist.core_set;

    ProgramDescriptor desc;
    auto add_tile_cb = [&](uint32_t idx, uint32_t tiles) {
        const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float16_b);
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * tile_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = tile_size}}}});
    };
    add_tile_cb(act_rm_cb, block_ct);
    add_tile_cb(act_tile_cb, block_ct);
    add_tile_cb(weights_cb, 4 * block_ct);
    add_tile_cb(partial_a_cb, block_ct);
    add_tile_cb(partial_b_cb, block_ct);
    add_tile_cb(output_cb, block_ct);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/";
    std::vector<uint32_t> reader_ct = {block_ct, channels, row_bytes, num_blocks};
    TensorAccessorArgs(*in.input.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.state.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap0.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap1.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap2.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.tap3.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {Qt, Kt, Vt, block_ct, num_blocks};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[2].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_kda_causal_conv1d.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_kda_causal_conv1d.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/kda_causal_conv1d.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {block_ct, num_blocks};
    compute.config = compute_cfg(in.input.device()->arch(), attrs.compute_kernel_config);

    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        reader.emplace_runtime_args(
            core,
            {dist.wi_start[i],
             dist.wi_count[i],
             in.input.buffer(),
             in.state.buffer(),
             in.tap0.buffer(),
             in.tap1.buffer(),
             in.tap2.buffer(),
             in.tap3.buffer()});
        writer.emplace_runtime_args(
            core, {dist.wi_start[i], dist.wi_count[i], outputs[0].buffer(), outputs[1].buffer(), outputs[2].buffer()});
        compute.emplace_runtime_args(core, {dist.wi_count[i]});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
