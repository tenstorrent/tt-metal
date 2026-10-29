// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// One core per (user, value head). Reader: ctrl page, constants, the head's 12 window tiles (win[par]) + 12 raw
// projection tiles + 12 tap tiles, z, a|b (one tile, or two when 2*Nv > 32 and the head's a and b sit in different
// tiles: Nv = 24 at TP = 2), the one-hot selector tiles and the 64 KiB initial state (last). Compute:
// per chunk the window rebuild (2 one-hot matmuls), the 4 shifted operands (4 one-hot matmuls) and the 4-tap conv +
// SiLU in gdn_decode_step_conv's op order; then the row-batched pre (l2norms, mask, kt, gates), T L1-resident
// delta-rule steps (state -> hnew per token), the row-batched gated RMSNorm. Writer: window rows to win[1-par]
// (HOLD: copy-through via a bounce), ring block t*BH + bh per token (HOLD: skipped), the user's T output rows.
//
// fast_mode (GDN_FAST): the T beta/decay gate pairs are computed in software fp32 on the reader (as
// reader_gdn_decode_step_conv_fast.cpp does per user) and pushed as ready scalar tiles `beta_t[t]` / `dec[t]`; the
// compute's one-hot gate gathers + SFPU sigmoid/softplus/exp (g1, a_s, b_s, csel, rsel) are compiled out, and the
// state add packs hnew from the same DST tile as hn (gdn_decode_step_conv_fast.cpp's add_tiles_n_dual).
// fp32_state (GDN_FP32_STATE, fast_mode only): the decay multiply and the state add run on the SFPU in exact fp32
// (gdn_decode_step_conv_fast.cpp's decay_state_sfpu / add_state_sfpu): state_in, dec, outer and the two extra
// KV-tile DFBs hd32 (decayed state) and hn32 (next token's state) are UnpackToDest.
// value_blocks VB > 1 (GDN_VSPLIT): core (u, h, s) owns state column tiles [s*Vs, (s+1)*Vs), Vs = Vt / VB, of the head
// (tt-lab's GDN head split). A state column's recurrence never reads another column, so each core runs the T tokens
// on its KVs = Kt*Vs state tiles with the unchanged per-tile arithmetic (the conv / l2norm / gate prefix is computed
// in full on every block, only the v chunks of the block's columns are convolved). The last token's pre-norm output
// tiles go to `acc_mine`; the writer copies them (exact bytes) into `acc_x` at tile offset s*Vs on every block of the
// head (ready / valid semaphore handshake, unicast only), so the compute runs the gated RMSNorm on the full Vt row as
// before and emits its Vs output tiles. The window rows are written by block 0 only. Bit-identical to VB = 1.
#include "ttnn/operations/experimental/kda/gdn_spec_step/device/gdn_spec_step_program_factory.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {
uint32_t float_bits(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, sizeof(float));
    return b;
}

struct Dfb {
    std::string name;
    uint32_t entries;
    uint32_t entry_bytes;
    tt::DataFormat fmt;
};

constexpr const char* kDir = "ttnn/cpp/ttnn/operations/experimental/kda/gdn_spec_step/device/kernels/";
constexpr uint32_t kHoldSentinel = 0xFFFFFFFFu;
constexpr uint32_t kMaxValueBlocks = 4;  // the writer's peer-coordinate runtime args

// Cores per (u, h): the largest divisor of Vt <= requested (<= kMaxValueBlocks) whose BH * blocks cores fit the grid.
uint32_t effective_value_blocks(uint32_t requested, uint32_t Vt, uint32_t BH, uint32_t num_cores) {
    for (uint32_t vb = std::min({requested, Vt, kMaxValueBlocks}); vb > 1; --vb) {
        if (Vt % vb == 0 && BH * vb <= num_cores) {
            return vb;
        }
    }
    return 1;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts GdnSpecStepProgramFactory::create_program_artifacts(
    const GdnSpecStepParams& a, const GdnSpecStepInputs& in, Tensor& output_tensor) {
    const auto& qkv = in.qkvzab.mesh_tensor();
    const auto& win_a = in.win_a.mesh_tensor();
    const auto& win_b = in.win_b.mesh_tensor();
    const auto& ring = in.ring.mesh_tensor();
    const auto& ctrl = in.ctrl.mesh_tensor();
    const auto& taps = in.taps.mesh_tensor();
    const auto& dtb = in.dt_bias.mesh_tensor();
    const auto& nea = in.neg_exp_A.mesh_tensor();
    const auto& weight = in.weight.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    const auto& device = qkv.device();

    const uint32_t Kt = a.key_dim / TILE_WIDTH;
    const uint32_t Vt = a.value_dim / TILE_WIDTH;
    const uint32_t Ch = 2 * Kt + Vt;  // [q|k|v] chunks (tiles) of one value head
    const uint32_t Nv = a.num_value_heads;
    const uint32_t Nk = a.num_key_heads;
    const uint32_t T = a.T;
    const uint32_t B = a.B;
    const uint32_t M = a.max_users ? a.max_users : B;  // ring/ctrl layout width
    const uint32_t K = a.conv_kernel;
    const uint32_t Lw = K - 1 + T;
    const uint32_t BH = B * Nv;
    const uint32_t C = 2 * Nk * a.key_dim + Nv * a.value_dim;
    const uint32_t Ct = C / TILE_WIDTH;  // tiles per window / tap row
    // a|b gate columns [qkvz_dim, qkvz_dim + 2*Nv): one tile when 2*Nv <= 32 (TP = 4: Nv = 12); two tiles at Nv = 24
    // (TP = 2), where a[h] (column h) and b[h] (column Nv + h) of heads h >= 8 sit in different tiles -> the reader
    // reads b's tile into a second ab_in entry (b_idx = 1) and the compute gathers b's row from it
    const uint32_t AB2 = (2 * Nv > TILE_WIDTH) ? 1u : 0u;
    // gdn_decode_step's fast_mode / fp32_state arithmetic (see GdnSpecStepParams)
    const bool FAST = a.fast_mode;
    const bool FP32 = a.fp32_state && a.fast_mode;
    // kernel_opts (bit-identical restructurings, fast_mode + fp32_state only; the device op masks them otherwise)
    const bool FUSED_DECAY = FP32 && (a.kernel_opts & kGdnSpecOptFusedDecay) != 0;
    // GATE_SCALARS: the writer computes the T gate pairs into one small `gsc` entry and the compute fills its scalar
    // tiles in DST (no reader gate math, no 2*T broadcast-tile fills on a data-movement RISC)
    const bool GSC = FP32 && (a.kernel_opts & kGdnSpecOptGateScalars) != 0;
    // EARLY_SRC: the reader pushes src_in + the selectors before building the taps / weight row / masks
    const bool EARLY_SRC = FP32 && (a.kernel_opts & kGdnSpecOptEarlySrc) != 0;
    // HOIST_INV: the token-independent rsqrt(q stats + eps) * scale of the per-token qn once, before the token loop
    const bool HOIST_INV = FP32 && (a.kernel_opts & kGdnSpecOptHoistInv) != 0;

    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t num_cores_avail = grid.x * grid.y;
    TT_FATAL(BH <= num_cores_avail, "gdn_spec_step: {} (user, head) items exceed {} cores", BH, num_cores_avail);
    // value blocks (column split of each head's state): VB cores per (u, h), Vs value tiles each
    const uint32_t VB = effective_value_blocks(a.value_blocks, Vt, BH, num_cores_avail);
    const uint32_t Vs = Vt / VB;
    const uint32_t KVs = Kt * Vs;
    const uint32_t Chs = 2 * Kt + Vs;  // chunks a block convolves: q, k and its own v chunks
    const bool VSPLIT = VB > 1;
    // QKSHARE (kernel_opts bit 4, VB > 1): block s convolves only the q/k chunks [s*Qk, (s+1)*Qk), Qk = 2*Kt / VB (one
    // contiguous range of q or of k), plus its own v chunks; the head's blocks swap their q/k conv tiles (exact fp32
    // bytes) over the NoC into every block's qc / kc, now writer-produced DFBs. Each block reads, rebuilds, convolves
    // and writes back (window) only its own ChA = Qk + Vs chunks.
    const uint32_t Qk = (VSPLIT && (2 * Kt) % VB == 0) ? (2 * Kt) / VB : 0u;
    const bool QKS = FP32 && Qk != 0 && Kt % Qk == 0 && (a.kernel_opts & kGdnSpecOptQkShare) != 0;
    const uint32_t ChA = QKS ? Qk + Vs : Ch;   // chunks a block reads and rebuilds (src_in / taps / wc / wout)
    const uint32_t ChC = QKS ? Qk + Vs : Chs;  // chunks a block convolves (shift / cv)
    // A3_HALF (bit 5): the conv's SFPU tail on the tile half that holds the user's rows only (kernel-side decision)
    const bool A3_HALF = FP32 && (a.kernel_opts & kGdnSpecOptA3Half) != 0;
    // GSC_SPLIT (bit 6, GATE_SCALARS + VB > 1): the head's blocks split the T gate computations and swap the 64 B
    // per-token slots
    const bool GSC_SPLIT = GSC && VSPLIT && (a.kernel_opts & kGdnSpecOptGscSplit) != 0;
    // VS_TAIL (bit 7, VB > 1): each block normalizes / gates only its own Vs output tiles
    const bool VS_TAIL = FP32 && VSPLIT && (a.kernel_opts & kGdnSpecOptVsTail) != 0;
    // one (u, h, s) per core, item = (u*Nv + h)*VB + s: a head's blocks are consecutive cores (VB = 1: the original
    // one (u, h) per core, item = u*Nv + h)
    auto dist = kda_factory_detail::distribute_prep(grid, BH * VB, BH * VB);
    const auto& cores = dist.core_set;

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName QKV{"qkv"};
    const m2::TensorParamName WIN_A{"win_a"};
    const m2::TensorParamName WIN_B{"win_b"};
    const m2::TensorParamName RING{"ring"};
    const m2::TensorParamName CTRL{"ctrl"};
    const m2::TensorParamName TAPS{"taps"};
    const m2::TensorParamName DTB{"dtb"};
    const m2::TensorParamName NEA{"nea"};
    const m2::TensorParamName WEIGHT{"weight"};
    const m2::TensorParamName OUT{"out"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto u32 = tt::DataFormat::UInt32;
    const auto out_fmt = datatype_to_dataformat_converter(a.output_dtype);
    const uint32_t tmp_tiles = std::max(Kt, Vt);
    const uint32_t ctrl_bytes = static_cast<uint32_t>(in.ctrl.buffer()->aligned_page_size());
    const uint32_t t32 = tt::tile_size(fp32), t16 = tt::tile_size(bf16), tout = tt::tile_size(out_fmt);

    // ---- dataflow buffers {name, entries, entry bytes, format}
    std::vector<Dfb> reader_out = {
        {"src_in", 2 * ChA, t16, bf16},  // [win tiles 0..Ch-1 of user u (rows 0..Lw-1) | raw qkv tiles 0..Ch-1]
        {"taps", ChA, t16, bf16},        // tile c: tap j in row j (rows >= K zero)
        {"z_in", Vt, t16, bf16},
        {"sel", 3 + K, t16, bf16},  // WO_W, WO_N (window rebuild), SH_j (shifted conv operands), P (placement)
        {"mask_T", 1, t16, bf16},   // 1.0 at (u*T + t, 0) for t < T (row-batched l2norm mask)
        {"e_t", T, t16, bf16},      // tile t: 1.0 at (u*T + t, 0)
        {"w_in", Vt, t16, bf16},    // norm weight in row 0
        {"scaler", 1, t32, fp32},
        {"eps_l2", 1, t16, bf16},
        {"eps_norm", 1, t16, bf16}};
    if (!GSC) {
        // dtb_s / nea_s: all-equal fp32 scalar tiles (plain), or the reader's staging tiles for the gate scalars
        // (FAST: reserved, never pushed; the compute does not wait on them). GSC: the writer reads the scalars.
        reader_out.push_back({"dtb_s", 1, t32, fp32});
        reader_out.push_back({"nea_s", 1, t32, fp32});
    }
    std::vector<Dfb> reader_local = {{"ctrl_r", 1, ctrl_bytes, u32}};
    if (FAST && !GSC) {  // GSC: nothing gate-related on the reader (see the writer's staging DFBs and gsc below)
        // the a|b tile(s) are only staging for the reader's gate math; beta_t / dec arrive from the reader
        reader_local.push_back({"ab_in", 1 + AB2, t16, bf16});
        reader_out.push_back({"beta_t", T, t32, fp32});  // tile t: sigmoid(b[u*T + t, h]) (all-equal)
        reader_out.push_back({"dec", T, t32, fp32});     // tile t: exp(nea[h] * softplus(a[u*T + t, h] + dtb[h]))
    } else if (!FAST) {
        reader_out.push_back({"ab_in", 1 + AB2, t16, bf16});  // [a's tile | b's tile when it differs (AB2)]
        reader_out.push_back(
            {"rsel", T, t16, bf16});  // tile t: 1.0 at column u*T + t in every row (gate row extractor)
        reader_out.push_back({"csel", 2, t16, bf16});  // tile 0: row h all ones; tile 1: row Nv + h (gate column
                                                       // extractors)
    }
    reader_out.push_back({"state_in", KVs, t32, fp32});
    // per-block sizes (Vs, KVs, Chs) equal the full-head ones (Vt, KV, Ch) at VB = 1
    std::vector<Dfb> compute_local = {
        {"wc", ChA, t16, bf16},        {"shift", ChC * K, t16, bf16}, {"cv", ChC, t32, fp32},
        {"tmp", tmp_tiles, t32, fp32}, {"stats", 1, t32, fp32},       {"scratch", 1, t32, fp32},
        {"inv", 1, t32, fp32},         {"qc", Kt, t32, fp32},         {"kc", Kt, t32, fp32},
        {"vc", Vs, t32, fp32},         {"qn", Kt, t32, fp32},         {"kn", Kt, t32, fp32},
        {"vm", Vs, t32, fp32},         {"kt", Kt, t32, fp32},         {"eb", 1, t32, fp32},
        {"hd", KVs, t32, fp32},        {"vread", Vs, t32, fp32},      {"delta", Vs, t32, fp32},
        {"outer", KVs, t32, fp32},     {"hn", KVs, t32, fp32},        {"gq", Vt, t32, fp32},
        {"acc_a", Vs, t32, fp32},      {"acc_b", Vs, t32, fp32},      {"on", Vt, t32, fp32},
        {"zs", Vt, t32, fp32}};
    if (!FAST) {
        for (const auto& d : std::vector<Dfb>{
                 {"g1", 1 + AB2, t32, fp32},
                 {"a_s", 1, t32, fp32},
                 {"b_s", 1, t32, fp32},
                 {"beta_t", T, t32, fp32},
                 {"dec", T, t32, fp32}}) {
            compute_local.push_back(d);
        }
    }
    if (FP32) {
        // exact-fp32 copies for the SFPU state update: hd32 = the decayed state (SFPU add operand), hn32 = the new
        // state as the next token's decay operand (hn itself is the FPU matmul operand of o = qn @ S)
        compute_local.push_back({"hd32", KVs, t32, fp32});
        if (!FUSED_DECAY) {
            // FUSED_DECAY: the next token's decay reads the new state from DST in the state add's acquire
            compute_local.push_back({"hn32", KVs, t32, fp32});
        }
    }
    if (GSC) {
        // beta_t[t] is filled in DST from the writer's bits and packed here (one token at a time); dec stays in DST
        compute_local.push_back({"beta_t", 1, t32, fp32});
    }
    if (HOIST_INV) {
        compute_local.push_back({"qsc", 1, t32, fp32});  // rsqrt(q stats + eps) * scale, live through the token loop
    }
    if (VSPLIT) {
        // the token loop's Vs-tile intermediate (vm - vread): `tmp` (max(Kt, Vt) tiles) takes only Kt / Vt blocks, as
        // in the VB = 1 kernel, since an indexed block must not straddle a DFB's ring end (see the compute header)
        compute_local.push_back({"tmpv", Vs, t32, fp32});
    }
    std::vector<Dfb> compute_out = {
        {"hnew", a.hnew_depth * KVs, t32, fp32}, {"out", Vs, tout, out_fmt}, {"wout", ChA, t16, bf16}};
    std::vector<Dfb> writer_local = {{"ctrl_w", 1, ctrl_bytes, u32}, {"bounce", Ch, t16, bf16}};
    // VSPLIT: acc_mine = this block's last-token pre-norm output tiles (compute -> writer); acc_x = the head's full
    // pre-norm row, every block's tiles at offset s*Vs (writer -> compute, filled over the NoC by all VB blocks)
    std::vector<Dfb> writer_out;
    if (VSPLIT) {
        compute_out.push_back({"acc_mine", Vs, t32, fp32});
        writer_out.push_back({"acc_x", Vt, t32, fp32});
    }
    if (QKS) {
        // qc / kc arrive from the writer (every block's q/k conv tiles); this block's Qk tiles go out through qkm
        compute_local.erase(
            std::remove_if(
                compute_local.begin(),
                compute_local.end(),
                [](const Dfb& d) { return d.name == "qc" || d.name == "kc"; }),
            compute_local.end());
        writer_out.push_back({"qc", Kt, t32, fp32});
        writer_out.push_back({"kc", Kt, t32, fp32});
        compute_out.push_back({"qkm", Qk, t32, fp32});
    }
    // gsc: beta[0..T) | dec[0..T) words; GSC_SPLIT: one 64 B slot per token (beta, dec in its first two words)
    const uint32_t gsc_bytes =
        GSC_SPLIT ? T * 64u : std::max<uint32_t>(64u, ((2 * T * 4 + 63) / 64) * 64);
    if (GSC) {
        // writer staging for its gate reads (never pushed): the a|b tile(s), tile 0 of dt_bias and of -exp(A)
        writer_local.push_back({"ab_w", 1 + AB2, t16, bf16});
        writer_local.push_back({"dtb_ws", 1, t32, fp32});
        writer_local.push_back({"nea_ws", 1, t32, fp32});
        writer_out.push_back({"gsc", 1, gsc_bytes, u32});
        if (GSC_SPLIT) {
            writer_local.push_back({"gss", 1, gsc_bytes, u32});  // this block's token slots, sent to every block
        }
    }

    m2::Group<m2::DataflowBufferSpec> dfbs;
    std::vector<Dfb> compute_bound;
    for (const auto* v : {&reader_out, &reader_local, &compute_local, &compute_out, &writer_local, &writer_out}) {
        for (const auto& d : *v) {
            if (v != &reader_local && v != &writer_local) {
                compute_bound.push_back(d);
            }
            dfbs.push_back(m2::DataflowBufferSpec{
                .unique_id = m2::DFBSpecName{d.name},
                .entry_size = d.entry_bytes,
                .num_entries = d.entries,
                .data_format_metadata = d.fmt,
            });
        }
    }
    using EP = m2::DFBEndpointType;
    auto bind = [](const std::string& name, EP type) { return m2::DFBBinding{m2::DFBSpecName{name}, name, type}; };

    const uint32_t z_tile0 = (2 * Nk * Kt * TILE_WIDTH + Nv * Vt * TILE_WIDTH) / TILE_WIDTH;
    const uint32_t ab_page = a.qkvz_dim / TILE_WIDTH;
    const uint32_t w_tiles = static_cast<uint32_t>(in.qkvzab.padded_shape()[-1]) / TILE_WIDTH;  // tiles per qkv row
    const uint32_t out_w_tiles = Nv * Vt;                                                       // tiles per out row

    // ---- reader
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kDir) + "dataflow/reader_gdn_spec_step.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::Os},  // code size (kernel config buffer)
        .runtime_arg_schema = {.runtime_arg_names = {"u", "h", "b_idx", "s"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    if (FAST) {
        reader.compiler_options.defines.emplace("GDN_FAST", "1");
    }
    if (GSC) {
        reader.compiler_options.defines.emplace("GDN_GATE_SCALARS", "1");
    }
    if (EARLY_SRC) {
        reader.compiler_options.defines.emplace("GDN_EARLY_SRC", "1");
    }
    if (QKS) {
        reader.compiler_options.defines.emplace("GDN_QKSHARE", "1");
    }
    for (const auto& d : reader_out) {
        reader.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
    }
    for (const auto& d : reader_local) {
        reader.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
        reader.dfb_bindings.push_back(bind(d.name, EP::CONSUMER));
    }
    reader.tensor_bindings = {
        m2::TensorBinding{QKV, "qkv"},
        m2::TensorBinding{WIN_A, "win_a"},
        m2::TensorBinding{WIN_B, "win_b"},
        m2::TensorBinding{RING, "ring"},
        m2::TensorBinding{CTRL, "ctrl"},
        m2::TensorBinding{TAPS, "taps"},
        m2::TensorBinding{DTB, "dtb"},
        m2::TensorBinding{NEA, "nea"},
        m2::TensorBinding{WEIGHT, "weight"}};
    reader.compile_time_args = {
        {"Kt", Kt},
        {"Vt", Vt},
        {"Nk", Nk},
        {"Nv", Nv},
        {"T", T},
        {"B", B},
        {"M", M},
        {"K", K},
        {"Lw", Lw},
        {"Ct", Ct},
        {"z_tile0", z_tile0},
        {"ab_page", ab_page},
        {"w_tiles", w_tiles},
        {"dtb_fp32", in.dt_bias.dtype() == DataType::FLOAT32 ? 1u : 0u},
        {"nea_fp32", in.neg_exp_A.dtype() == DataType::FLOAT32 ? 1u : 0u},
        {"l2_eps_bits", float_bits(a.l2_epsilon)},
        {"norm_eps_bits", float_bits(a.norm_epsilon)},
        {"ctrl_bytes", ctrl_bytes},
        {"hold_sentinel", kHoldSentinel},
        // the ab_in entry count (1 + AB2) the reader reserves/pushes: the factory's value, the same contract compute
        // has, so the reader's DFB accounting cannot drift from the ab_in / g1 sizing above
        {"AB2", AB2},
        {"VB", VB}};

    // ---- writer
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kDir) + "dataflow/writer_gdn_spec_step.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::Os},  // code size (kernel config buffer)
        .dfb_bindings = {bind("hnew", EP::CONSUMER), bind("out", EP::CONSUMER), bind("wout", EP::CONSUMER)},
        .tensor_bindings =
            {m2::TensorBinding{RING, "ring_out"},
             m2::TensorBinding{OUT, "out"},
             m2::TensorBinding{WIN_A, "win_a_w"},
             m2::TensorBinding{WIN_B, "win_b_w"},
             m2::TensorBinding{CTRL, "ctrl_w"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"u",
                  "h",
                  "s",
                  "peer_x0",
                  "peer_y0",
                  "peer_x1",
                  "peer_y1",
                  "peer_x2",
                  "peer_y2",
                  "peer_x3",
                  "peer_y3"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    for (const auto& d : writer_out) {
        writer.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
    }
    if (GSC) {
        // the gate inputs; their geometry goes in as defines so the writer's named-argument signature is unchanged
        writer.tensor_bindings.push_back(m2::TensorBinding{QKV, "qkv_w"});
        writer.tensor_bindings.push_back(m2::TensorBinding{DTB, "dtb_w"});
        writer.tensor_bindings.push_back(m2::TensorBinding{NEA, "nea_w"});
        writer.compiler_options.defines.emplace("GDN_GATE_SCALARS", "1");
        writer.compiler_options.defines.emplace("GDN_AB_PAGE", std::to_string(ab_page));
        writer.compiler_options.defines.emplace("GDN_W_TILES", std::to_string(w_tiles));
        writer.compiler_options.defines.emplace(
            "GDN_DTB_FP32", in.dt_bias.dtype() == DataType::FLOAT32 ? "1" : "0");
        writer.compiler_options.defines.emplace(
            "GDN_NEA_FP32", in.neg_exp_A.dtype() == DataType::FLOAT32 ? "1" : "0");
    }
    const m2::SemaphoreSpecName READY{"ready"};
    const m2::SemaphoreSpecName VALID{"valid"};
    const m2::SemaphoreSpecName QKSEM{"qkv"};
    const m2::SemaphoreSpecName GSSEM{"gsv"};
    if (VSPLIT) {
        writer.dfb_bindings.push_back(bind("acc_mine", EP::CONSUMER));
        writer.semaphore_bindings.push_back(m2::SemaphoreBinding{READY, "ready"});
        writer.semaphore_bindings.push_back(m2::SemaphoreBinding{VALID, "valid"});
        writer.compiler_options.defines.emplace("GDN_VSPLIT", "1");
    }
    if (QKS) {
        writer.dfb_bindings.push_back(bind("qkm", EP::CONSUMER));
        writer.semaphore_bindings.push_back(m2::SemaphoreBinding{QKSEM, "qkv"});
        writer.compiler_options.defines.emplace("GDN_QKSHARE", "1");
    }
    if (GSC_SPLIT) {
        writer.semaphore_bindings.push_back(m2::SemaphoreBinding{GSSEM, "gsv"});
        writer.compiler_options.defines.emplace("GDN_GSC_SPLIT", "1");
    }
    for (const auto& d : writer_local) {
        writer.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
        writer.dfb_bindings.push_back(bind(d.name, EP::CONSUMER));
    }
    writer.compile_time_args = {
        {"Kt", Kt},
        {"Vt", Vt},
        {"Nk", Nk},
        {"Nv", Nv},
        {"T", T},
        {"B", B},
        {"M", M},
        {"Lw", Lw},
        {"Ct", Ct},
        {"out_w_tiles", out_w_tiles},
        {"ctrl_bytes", ctrl_bytes},
        {"hold_sentinel", kHoldSentinel},
        {"VB", VB}};

    // ---- compute
    auto compute_hw = ttnn::to_compute_hardware_config(a.compute_kernel_config);
    auto& unpack_modes = compute_hw.unpack_modes;
    // fp32_state: the state update stays in full fp32 on the SFPU. These DFBs are only read through copy_tile into
    // DST (UnpackToDest = exact fp32, no SrcA truncation); every other fp32 DFB feeds the FPU (UnpackToSrc).
    for (const auto& d : compute_bound) {
        if (d.fmt == fp32) {
            const bool to_dest = FP32 && (d.name == "state_in" || d.name == "dec" || d.name == "hd32" ||
                                          d.name == "hn32" || d.name == "outer");
            unpack_modes[m2::DFBSpecName{d.name}] = to_dest ? UnpackMode::UnpackToDest : UnpackMode::UnpackToSrc;
        }
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(kDir) + "compute/gdn_spec_step.cpp",
        // O3: at O2 GCC stops constant-folding the LLK addr-mod structs in the T = 8 body (impossible asm constraint,
        // also with the phased conv block). The five kernel binaries must fit the 69 KB kernel config buffer
        // (program.cpp finalize_kernel_bins: all five packed binaries + RT args + DFB configs), so the dataflow kernels
        // are built at Os and the compute loops are not unrolled (#pragma GCC unroll 1). Helpers stay inlined with
        // constant DFB ids: non-inlined generic wrappers made the binaries ~6 KB LARGER (runtime id lookups).
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"T", T},
             {"K", K},
             {"AB2", AB2},
             {"scale_bits", float_bits(a.scale)},
             {"inv_dv_bits", float_bits(1.0f / static_cast<float>(a.value_dim))},
             {"VB", VB}},
        .runtime_arg_schema = {.runtime_arg_names = {"u", "b_idx", "s"}},
        .hw_config = std::move(compute_hw),
    };
    if (FAST) {
        // the gate DFBs (g1, a_s, b_s, csel, rsel) only exist in the plain mode; the kernel guards their references
        compute.compiler_options.defines.emplace("GDN_FAST", "1");
    }
    if (FP32) {
        // the hd32 / hn32 DFBs only exist then; the kernel guards their references with #if GDN_FP32_STATE
        compute.compiler_options.defines.emplace("GDN_FP32_STATE", "1");
    }
    if (VSPLIT) {
        // the acc_mine / acc_x DFBs only exist then; the kernel guards their references with #if GDN_VSPLIT
        compute.compiler_options.defines.emplace("GDN_VSPLIT", "1");
    }
    if (FUSED_DECAY) {
        // no hn32 DFB then; the kernel guards its references with #if !GDN_FUSED_DECAY
        compute.compiler_options.defines.emplace("GDN_FUSED_DECAY", "1");
    }
    if (GSC) {
        // no dec DFB then (and beta_t is compute-local); the kernel guards their references with GDN_GATE_SCALARS
        compute.compiler_options.defines.emplace("GDN_GATE_SCALARS", "1");
    }
    if (HOIST_INV) {
        compute.compiler_options.defines.emplace("GDN_HOIST_INV", "1");
    }
    if (QKS) {
        // qc / kc are writer-produced then and qkm exists; the kernel guards both with #if GDN_QKSHARE
        compute.compiler_options.defines.emplace("GDN_QKSHARE", "1");
    }
    if (A3_HALF) {
        compute.compiler_options.defines.emplace("GDN_A3_HALF", "1");
    }
    if (GSC_SPLIT) {
        compute.compiler_options.defines.emplace("GDN_GSC_SPLIT", "1");  // gsc slot layout
    }
    if (VS_TAIL) {
        compute.compiler_options.defines.emplace("GDN_VS_TAIL", "1");
    }
    for (const auto& d : reader_out) {
        compute.dfb_bindings.push_back(bind(d.name, EP::CONSUMER));
    }
    for (const auto& d : compute_local) {
        compute.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
        compute.dfb_bindings.push_back(bind(d.name, EP::CONSUMER));
    }
    for (const auto& d : compute_out) {
        compute.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
    }
    for (const auto& d : writer_out) {
        compute.dfb_bindings.push_back(bind(d.name, EP::CONSUMER));
    }

    // ---- runtime args: core i handles item (bh, s) = (i / VB, i % VB), (u, h) = (bh / Nv, bh % Nv); b_idx = 1 when
    //      head h's b gate column lies in a different tile than its a column (only possible with AB2). The writer gets
    //      the NoC coordinates of all VB blocks of its head (itself included; unused slots 0).
    TT_FATAL(dist.cores.size() == BH * VB, "gdn_spec_step: expected one (u, h, s) item per core");
    m2::KernelRunArgs reader_args{.kernel = READER};
    m2::KernelRunArgs writer_args{.kernel = WRITER};
    m2::KernelRunArgs compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        const uint32_t item = dist.wi_start[i];
        const uint32_t bh = item / VB;
        const uint32_t s = item % VB;
        const uint32_t u = bh / Nv;
        const uint32_t h = bh % Nv;
        const uint32_t b_idx = (AB2 != 0 && ((Nv + h) / TILE_WIDTH) != (h / TILE_WIDTH)) ? 1u : 0u;
        uint32_t px[kMaxValueBlocks] = {0, 0, 0, 0};
        uint32_t py[kMaxValueBlocks] = {0, 0, 0, 0};
        if (VSPLIT) {
            for (uint32_t p = 0; p < VB; ++p) {
                const auto peer = device.worker_core_from_logical_core(dist.cores[bh * VB + p]);
                px[p] = static_cast<uint32_t>(peer.x);
                py[p] = static_cast<uint32_t>(peer.y);
            }
        }
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values, core, {{"u", u}, {"h", h}, {"b_idx", b_idx}, {"s", s}});
        m2::AddRuntimeArgsForNode(
            writer_args.runtime_arg_values,
            core,
            {{"u", u},
             {"h", h},
             {"s", s},
             {"peer_x0", px[0]},
             {"peer_y0", py[0]},
             {"peer_x1", px[1]},
             {"peer_y1", py[1]},
             {"peer_x2", px[2]},
             {"peer_y2", py[2]},
             {"peer_x3", px[3]},
             {"peer_y3", py[3]}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"u", u}, {"b_idx", b_idx}, {"s", s}});
    }

    // ---- tensor parameters (ring and the window pair are read by the reader and written in place by the writer)
    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = QKV, .spec = qkv.tensor_spec()},
        m2::TensorParameter{.unique_id = WIN_A, .spec = win_a.tensor_spec()},
        m2::TensorParameter{.unique_id = WIN_B, .spec = win_b.tensor_spec()},
        m2::TensorParameter{.unique_id = RING, .spec = ring.tensor_spec()},
        m2::TensorParameter{.unique_id = CTRL, .spec = ctrl.tensor_spec()},
        m2::TensorParameter{.unique_id = TAPS, .spec = taps.tensor_spec()},
        m2::TensorParameter{.unique_id = DTB, .spec = dtb.tensor_spec()},
        m2::TensorParameter{.unique_id = NEA, .spec = nea.tensor_spec()},
        m2::TensorParameter{.unique_id = WEIGHT, .spec = weight.tensor_spec()},
        m2::TensorParameter{.unique_id = OUT, .spec = output.tensor_spec()}};
    m2::ProgramRunArgs run_args;
    run_args.tensor_args = {
        {QKV, qkv},
        {WIN_A, win_a},
        {WIN_B, win_b},
        {RING, ring},
        {CTRL, ctrl},
        {TAPS, taps},
        {DTB, dtb},
        {NEA, nea},
        {WEIGHT, weight},
        {OUT, output}};

    m2::ProgramSpec spec{
        .name = "gdn_spec_step",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = cores}},
    };
    if (VSPLIT) {
        // dispatch re-initializes both to 0 on every launch; only the writer binds them (remote increments + polls)
        spec.semaphores = {
            m2::SemaphoreSpec{.unique_id = READY, .target_nodes = cores},
            m2::SemaphoreSpec{.unique_id = VALID, .target_nodes = cores}};
        if (QKS) {
            spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = QKSEM, .target_nodes = cores});
        }
        if (GSC_SPLIT) {
            spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = GSSEM, .target_nodes = cores});
        }
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.kernel_run_args.push_back(std::move(compute_args));
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
