// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory for the fused T=1 decode gated delta rule op (Device 2.0
// descriptor style, mirroring chunk_gated_delta_rule).
//
// Parallelism: one Tensix core per (B*H) head. The T=1 inputs [B,1,H,*] are
// TILE tensors whose flat 2D view is [B*H rows, D cols] — 32 heads share each
// row of TILE pages, so the reader GATHERS head bh's row out of the shared
// pages (face-layout addressing) into private row-0 tiles. o is returned
// ROW_MAJOR so head bh owns DRAM page bh (its [V] stick) exclusively and the
// writer uses only full-page writes (sub-page writes do not land). The state
// [B,H,K,V] pages are head-aligned (K multiple of 32) and move as full tiles.

#include "decode_gated_delta_rule_program_factory.hpp"

#include <bit>
#include <cmath>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

// CB index plan (io = input dtype, scratch fp32). Kept in sync with kernels.
namespace cb {
constexpr uint32_t q = tt::CBIndex::c_0;      // [1,Kt] io  gathered q row
constexpr uint32_t k = tt::CBIndex::c_1;      // [1,Kt] io  gathered k row
constexpr uint32_t v = tt::CBIndex::c_2;      // [1,Vt] io  gathered v row
constexpr uint32_t g = tt::CBIndex::c_3;      // 1 tile io  scalar g_h at [0,0]
constexpr uint32_t beta = tt::CBIndex::c_4;   // 1 tile io  scalar beta_h at [0,0]
constexpr uint32_t state = tt::CBIndex::c_5;  // [Kt,Vt] io input state (or zeros)
constexpr uint32_t ones = tt::CBIndex::c_6;   // 1 tile fp32 all-ones (rowsum contraction)
constexpr uint32_t qsq = tt::CBIndex::c_7;    // [1,Kt] fp32 q*q
constexpr uint32_t ksq = tt::CBIndex::c_8;    // [1,Kt] fp32 k*k
constexpr uint32_t sc = tt::CBIndex::c_9;     // 1 tile fp32 rowsum of squares
constexpr uint32_t sc2 = tt::CBIndex::c_28;   // 1 tile fp32 q-chain inv-rms factor
constexpr uint32_t sc3 = tt::CBIndex::c_29;   // 1 tile fp32 k-chain inv-rms factor
// (each norm chain gets its OWN factor CB: reusing one ring made the second
// chain's bcast read the first chain's factor - ttsim gdn_decode_simdiag3)
constexpr uint32_t qn = tt::CBIndex::c_10;     // [1,Kt] fp32 normalized (scaled) q
constexpr uint32_t kn = tt::CBIndex::c_11;     // [1,Kt] fp32 normalized k
constexpr uint32_t kcol = tt::CBIndex::c_12;   // [Kt,1] fp32 transpose(kn)
constexpr uint32_t gexp = tt::CBIndex::c_13;   // 1 tile fp32 exp(g_h)
constexpr uint32_t sdec = tt::CBIndex::c_14;   // [Kt,Vt] fp32 decayed state h*exp(g)
constexpr uint32_t vread = tt::CBIndex::c_15;  // [1,Vt] fp32 k@h
constexpr uint32_t delta = tt::CBIndex::c_16;  // 2x[1,Vt] fp32 v-v_read, then *beta (ping-pong)
constexpr uint32_t outer = tt::CBIndex::c_17;  // [Kt,Vt] fp32 kcol @ delta
constexpr uint32_t sout = tt::CBIndex::c_18;   // [Kt,Vt] io new state
constexpr uint32_t out = tt::CBIndex::c_19;    // [1,Vt] io  o = q@h
// fp32 mirrors of the io inputs + fp32 new state: every math operand is fp32
// (mixed bf16-srcA x fp32-srcB pairs corrupt the fp32 side; chunk-scan pattern).
constexpr uint32_t qf = tt::CBIndex::c_20;       // [1,Kt] fp32 q
constexpr uint32_t kf = tt::CBIndex::c_21;       // [1,Kt] fp32 k
constexpr uint32_t vf = tt::CBIndex::c_22;       // [1,Vt] fp32 v
constexpr uint32_t gf = tt::CBIndex::c_23;       // 1 tile fp32 g_h
constexpr uint32_t betaf = tt::CBIndex::c_24;    // 1 tile fp32 beta_h
constexpr uint32_t sf = tt::CBIndex::c_25;       // [Kt,Vt] fp32 input state
constexpr uint32_t snew = tt::CBIndex::c_26;     // [Kt,Vt] fp32 new state
constexpr uint32_t scratch = tt::CBIndex::c_27;  // max(Kt,Vt) io staging pages (full-page DMA)
}  // namespace cb

tt::tt_metal::ProgramDescriptor DecodeGatedDeltaRuleProgramFactory::create_descriptor(
    const DecodeGatedDeltaRuleParams& attrs, const DecodeGatedDeltaRuleInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t BH = attrs.BH;
    const uint32_t Kt = attrs.K / TILE_WIDTH;
    const uint32_t Vt = attrs.V / TILE_WIDTH;
    const uint32_t kv = Kt * Vt;
    const uint32_t has_s0 = attrs.has_initial_state ? 1u : 0u;

    // q/k/v/o (and state) carry the caller's dtype; all math scratch is fp32.
    const tt::DataFormat df_io =
        (in.q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    // python l2_norm_ttnn: rms_norm(x, eps/K) * K**-0.5 == x / sqrt(sumsq + eps)
    constexpr float kEps = 1e-6f;
    const uint32_t eps_bits = std::bit_cast<uint32_t>(kEps);
    const uint32_t scale_bits = std::bit_cast<uint32_t>(attrs.scale);

    auto* device = in.q.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = grid.y;
    // BH can exceed the grid (decode BH = B*H, e.g. 128*24 = 3072 on a 110-core
    // die): each active core processes a contiguous chunk of head-instances
    // [c*per_core, min((c+1)*per_core, BH)). For BH <= ncores, per_core == 1 and
    // this reduces to the original one-instance-per-core mapping (core c <-> c).
    const uint32_t ncores = grid.x * grid.y;
    const uint32_t per_core = (BH + ncores - 1) / ncores;
    std::vector<CoreCoord> active_cores;
    std::set<CoreRange> core_set;
    for (uint32_t c = 0; c < ncores && c * per_core < BH; c++) {
        active_cores.push_back(CoreCoord{c / grid_y, c % grid_y});
        core_set.insert(CoreRange{active_cores.back(), active_cores.back()});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t n_tiles, uint32_t nbuf, tt::DataFormat fmt) {
        const uint32_t ts = tt::tile_size(fmt);
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_tiles * nbuf * ts,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = ts}}}});
    };

    add_cb(cb::q, Kt, 1, df_io);
    add_cb(cb::k, Kt, 1, df_io);
    add_cb(cb::v, Vt, 1, df_io);
    add_cb(cb::g, 1, 1, df_io);
    add_cb(cb::beta, 1, 1, df_io);
    add_cb(cb::state, kv, 1, df_io);
    add_cb(cb::ones, 1, 1, tt::DataFormat::Float32);
    add_cb(cb::qsq, Kt, 1, tt::DataFormat::Float32);
    add_cb(cb::ksq, Kt, 1, tt::DataFormat::Float32);
    add_cb(cb::sc, 1, 1, tt::DataFormat::Float32);   // rowsum of squares
    add_cb(cb::sc2, 1, 1, tt::DataFormat::Float32);  // q-chain inv-rms factor
    add_cb(cb::sc3, 1, 1, tt::DataFormat::Float32);  // k-chain inv-rms factor
    add_cb(cb::qn, Kt, 1, tt::DataFormat::Float32);
    add_cb(cb::kn, Kt, 1, tt::DataFormat::Float32);
    add_cb(cb::kcol, Kt, 1, tt::DataFormat::Float32);
    add_cb(cb::gexp, 1, 1, tt::DataFormat::Float32);
    add_cb(cb::sdec, kv, 1, tt::DataFormat::Float32);
    add_cb(cb::vread, Vt, 1, tt::DataFormat::Float32);
    add_cb(cb::delta, Vt, 2, tt::DataFormat::Float32);  // 2x pages: in-place *beta
    add_cb(cb::outer, kv, 1, tt::DataFormat::Float32);
    add_cb(cb::sout, kv, 1, df_io);
    add_cb(cb::out, Vt, 2, df_io);
    add_cb(cb::qf, Kt, 1, tt::DataFormat::Float32);
    add_cb(cb::kf, Kt, 1, tt::DataFormat::Float32);
    add_cb(cb::vf, Vt, 1, tt::DataFormat::Float32);
    add_cb(cb::gf, 1, 1, tt::DataFormat::Float32);
    add_cb(cb::betaf, 1, 1, tt::DataFormat::Float32);
    add_cb(cb::sf, kv, 1, tt::DataFormat::Float32);
    add_cb(cb::snew, kv, 1, tt::DataFormat::Float32);
    add_cb(cb::scratch, 2, 1, df_io);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule/device/kernels/";
    const std::vector<uint32_t> ct_args = {Kt, Vt, has_s0, eps_bits, scale_bits};

    // Reader compile args: {Kt,Vt,has_s0,eps,scale,H} + TensorAccessorArgs per
    // input (in order). H decomposes bh = b*H + h for the [B,1,H] scalar gather
    // (beta/g flat 2D is [B,H]: head (b,h) at row b, col h).
    std::vector<uint32_t> reader_ct = {Kt, Vt, has_s0, eps_bits, scale_bits, attrs.H};
    TensorAccessorArgs(*in.q.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.k.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.v.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.beta.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.g.buffer()).append_to(reader_ct);
    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {Kt, Vt};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);  // o
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);  // new state

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_decode_gated_delta_rule.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(BH);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_decode_gated_delta_rule.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(BH);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/decode_gated_delta_rule.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = ct_args;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    compute.runtime_args.reserve(BH);

    auto* q_buf = in.q.buffer();
    auto* k_buf = in.k.buffer();
    auto* v_buf = in.v.buffer();
    auto* beta_buf = in.beta.buffer();
    auto* g_buf = in.g.buffer();
    auto* s0_buf = in.initial_state.has_value() ? in.initial_state->buffer() : nullptr;
    auto* o_buf = outputs[0].buffer();
    auto* s1_buf = outputs[1].buffer();

    for (uint32_t c = 0; c < active_cores.size(); c++) {
        const auto& core = active_cores[c];
        const uint32_t start = c * per_core;
        const uint32_t n_inst = std::min(per_core, BH - start);
        // Runtime args carry the instance RANGE [start, start+n_inst); each
        // kernel derives bh = start + i in its per-instance loop.
        reader.emplace_runtime_args(core, {start, n_inst, q_buf, k_buf, v_buf, beta_buf, g_buf, s0_buf});
        // o is ROW_MAJOR: pass its stick page size (page bh == head bh's row).
        writer.emplace_runtime_args(core, {start, n_inst, o_buf, static_cast<uint32_t>(o_buf->page_size()), s1_buf});
        compute.emplace_runtime_args(core, {n_inst});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
