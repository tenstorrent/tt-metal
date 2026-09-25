// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// One core per (user, value head). Reader: ctrl page, constants, the head's 12 window tiles (win[par]) + 12 raw
// projection tiles + 12 tap tiles, z, a|b (one tile, or two when 2*Nv > 32 and the head's a and b sit in different
// tiles: Nv = 24 at TP = 2), the one-hot selector tiles and the 64 KiB initial state (last). Compute:
// per chunk the window rebuild (2 one-hot matmuls), the 4 shifted operands (4 one-hot matmuls) and the 4-tap conv +
// SiLU in gdn_decode_step_conv's op order; then the row-batched pre (l2norms, mask, kt, gates), T L1-resident
// delta-rule steps (state -> hnew per token), the row-batched gated RMSNorm. Writer: window rows to win[1-par]
// (HOLD: copy-through via a bounce), ring block t*BH + bh per token (HOLD: skipped), the user's T output rows.
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
    const auto arch = device.arch();

    const uint32_t Kt = a.key_dim / TILE_WIDTH;
    const uint32_t Vt = a.value_dim / TILE_WIDTH;
    const uint32_t KV = Kt * Vt;
    const uint32_t Ch = 2 * Kt + Vt;  // [q|k|v] chunks (tiles) of one value head
    const uint32_t Nv = a.num_value_heads;
    const uint32_t Nk = a.num_key_heads;
    const uint32_t T = a.T;
    const uint32_t B = a.B;
    const uint32_t K = a.conv_kernel;
    const uint32_t Lw = K - 1 + T;
    const uint32_t BH = B * Nv;
    const uint32_t C = 2 * Nk * a.key_dim + Nv * a.value_dim;
    const uint32_t Ct = C / TILE_WIDTH;  // tiles per window / tap row
    // a|b gate columns [qkvz_dim, qkvz_dim + 2*Nv): one tile when 2*Nv <= 32 (TP = 4: Nv = 12); two tiles at Nv = 24
    // (TP = 2), where a[h] (column h) and b[h] (column Nv + h) of heads h >= 8 sit in different tiles -> the reader
    // reads b's tile into a second ab_in entry (b_idx = 1) and the compute gathers b's row from it
    const uint32_t AB2 = (2 * Nv > TILE_WIDTH) ? 1u : 0u;

    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t num_cores_avail = grid.x * grid.y;
    TT_FATAL(BH <= num_cores_avail, "gdn_spec_step: {} (user, head) items exceed {} cores", BH, num_cores_avail);
    auto dist = kda_factory_detail::distribute_prep(grid, BH, BH);  // one (u, h) per core, item = u*Nv + h
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
    const std::vector<Dfb> reader_out = {
        {"src_in", 2 * Ch, t16, bf16},  // [win tiles 0..Ch-1 of user u (rows 0..Lw-1) | raw qkv tiles 0..Ch-1]
        {"taps", Ch, t16, bf16},        // tile c: tap j in row j (rows >= K zero)
        {"z_in", Vt, t16, bf16},
        {"ab_in", 1 + AB2, t16, bf16},  // [a's tile | b's tile when it differs (AB2)]
        {"sel", 3 + K, t16, bf16},      // WO_W, WO_N (window rebuild), SH_j (shifted conv operands), P (placement)
        {"mask_T", 1, t16, bf16},       // 1.0 at (u*T + t, 0) for t < T (row-batched l2norm mask)
        {"e_t", T, t16, bf16},          // tile t: 1.0 at (u*T + t, 0)
        {"rsel", T, t16, bf16},         // tile t: 1.0 at column u*T + t in every row (gate row extractor)
        {"csel", 2, t16, bf16},         // tile 0: row h all ones; tile 1: row Nv + h all ones (gate column extractors)
        {"w_in", Vt, t16, bf16},        // norm weight in row 0
        {"scaler", 1, t32, fp32},
        {"eps_l2", 1, t16, bf16},
        {"eps_norm", 1, t16, bf16},
        {"dtb_s", 1, t32, fp32},
        {"nea_s", 1, t32, fp32},
        {"state_in", KV, t32, fp32}};
    const std::vector<Dfb> reader_local = {{"ctrl_r", 1, ctrl_bytes, u32}};
    const std::vector<Dfb> compute_local = {
        {"wc", Ch, t16, bf16},         {"shift", Ch * K, t16, bf16}, {"cv", Ch, t32, fp32},
        {"tmp", tmp_tiles, t32, fp32}, {"stats", 1, t32, fp32},      {"scratch", 1, t32, fp32},
        {"inv", 1, t32, fp32},         {"qc", Kt, t32, fp32},        {"kc", Kt, t32, fp32},
        {"vc", Vt, t32, fp32},         {"qn", Kt, t32, fp32},        {"kn", Kt, t32, fp32},
        {"vm", Vt, t32, fp32},         {"kt", Kt, t32, fp32},        {"g1", 1 + AB2, t32, fp32},
        {"a_s", 1, t32, fp32},         {"b_s", 1, t32, fp32},        {"beta_t", T, t32, fp32},
        {"dec", T, t32, fp32},         {"eb", 1, t32, fp32},         {"hd", KV, t32, fp32},
        {"vread", Vt, t32, fp32},      {"delta", Vt, t32, fp32},     {"outer", KV, t32, fp32},
        {"hn", KV, t32, fp32},         {"gq", Vt, t32, fp32},        {"acc_a", Vt, t32, fp32},
        {"acc_b", Vt, t32, fp32},      {"on", Vt, t32, fp32},        {"zs", Vt, t32, fp32}};
    const std::vector<Dfb> compute_out = {
        {"hnew", a.hnew_depth * KV, t32, fp32}, {"out", Vt, tout, out_fmt}, {"wout", Ch, t16, bf16}};
    const std::vector<Dfb> writer_local = {{"ctrl_w", 1, ctrl_bytes, u32}, {"bounce", Ch, t16, bf16}};

    m2::Group<m2::DataflowBufferSpec> dfbs;
    std::vector<Dfb> compute_bound;
    for (const auto* v : {&reader_out, &reader_local, &compute_local, &compute_out, &writer_local}) {
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
        .runtime_arg_schema = {.runtime_arg_names = {"u", "h", "b_idx"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
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
        {"AB2", AB2}};

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
        .runtime_arg_schema = {.runtime_arg_names = {"u", "h"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
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
        {"Lw", Lw},
        {"Ct", Ct},
        {"out_w_tiles", out_w_tiles},
        {"ctrl_bytes", ctrl_bytes},
        {"hold_sentinel", kHoldSentinel}};

    // ---- compute
    auto compute_hw = ttnn::to_compute_hardware_config(arch, a.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& d : compute_bound) {
        if (d.fmt == fp32) {
            unpack_modes[m2::DFBSpecName{d.name}] = UnpackMode::UnpackToSrc;
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
             {"inv_dv_bits", float_bits(1.0f / static_cast<float>(a.value_dim))}},
        .runtime_arg_schema = {.runtime_arg_names = {"u", "b_idx"}},
        .hw_config = std::move(compute_hw),
    };
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

    // ---- runtime args: core i handles item (u, h) = (i / Nv, i % Nv); b_idx = 1 when head h's b gate column lies in
    //      a different tile than its a column (only possible with AB2)
    m2::KernelRunArgs reader_args{.kernel = READER};
    m2::KernelRunArgs writer_args{.kernel = WRITER};
    m2::KernelRunArgs compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        const uint32_t item = dist.wi_start[i];
        const uint32_t u = item / Nv;
        const uint32_t h = item % Nv;
        const uint32_t b_idx = (AB2 != 0 && ((Nv + h) / TILE_WIDTH) != (h / TILE_WIDTH)) ? 1u : 0u;
        m2::AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {{"u", u}, {"h", h}, {"b_idx", b_idx}});
        m2::AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"u", u}, {"h", h}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"u", u}, {"b_idx", b_idx}});
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
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.kernel_run_args.push_back(std::move(compute_args));
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
