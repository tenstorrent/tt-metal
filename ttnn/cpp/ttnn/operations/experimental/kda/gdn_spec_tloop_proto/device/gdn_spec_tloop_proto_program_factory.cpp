// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// M0a scratch: one core per (user, value head); reader loads the head's whole 32-row q/k/v/z tiles, the a|b tile,
// one-hot selector tiles and the 64 KiB state once; compute runs the pre-processing (row-batched or per token), the
// T-step recurrence with the state resident in L1, and the post-processing; writer streams ring block (t*BH + bh)
// per token from a double-buffered `hnew` and writes the user's T output rows once.
#include "ttnn/operations/experimental/kda/gdn_spec_tloop_proto/device/gdn_spec_tloop_proto_program_factory.hpp"

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
    uint32_t tiles;
    tt::DataFormat fmt;
};

constexpr const char* kDir = "ttnn/cpp/ttnn/operations/experimental/kda/gdn_spec_tloop_proto/device/kernels/";
}  // namespace

ttnn::device_operation::ProgramArtifacts GdnSpecTloopProtoProgramFactory::create_program_artifacts(
    const GdnSpecTloopProtoParams& a, const GdnSpecTloopProtoInputs& in, Tensor& output_tensor) {
    const auto& qkv = in.qkv.mesh_tensor();
    const auto& dtb = in.dt_bias.mesh_tensor();
    const auto& nea = in.neg_exp_A.mesh_tensor();
    const auto& ring = in.ring.mesh_tensor();
    const auto& weight = in.weight.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    const auto& device = qkv.device();
    const auto arch = device.arch();

    const uint32_t Kt = a.key_dim / TILE_WIDTH;
    const uint32_t Vt = a.value_dim / TILE_WIDTH;
    const uint32_t KV = Kt * Vt;
    const uint32_t Nv = a.num_value_heads;
    const uint32_t Nk = a.num_key_heads;
    const uint32_t T = a.T;
    const uint32_t B = a.B;
    const uint32_t BH = B * Nv;

    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t num_cores_avail = grid.x * grid.y;
    TT_FATAL(BH <= num_cores_avail, "gdn_spec_tloop_proto: {} (user, head) items exceed {} cores", BH, num_cores_avail);
    auto dist = kda_factory_detail::distribute_prep(grid, BH, BH);  // one (u, h) per core, item = u*Nv + h
    const auto& cores = dist.core_set;

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName QKV{"qkv"};
    const m2::TensorParamName DTB{"dtb"};
    const m2::TensorParamName NEA{"nea"};
    const m2::TensorParamName RING{"ring"};
    const m2::TensorParamName WEIGHT{"weight"};
    const m2::TensorParamName OUT{"out"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto out_fmt = datatype_to_dataformat_converter(a.output_dtype);
    const uint32_t tmp_tiles = std::max(Kt, Vt);

    // ---- dataflow buffers {name, tiles, format}
    const std::vector<Dfb> reader_out = {
        {"q_in", Kt, bf16},
        {"k_in", Kt, bf16},
        {"v_in", Vt, bf16},
        {"z_in", Vt, bf16},
        {"ab_in", 1, bf16},
        {"mask_T", 1, bf16},  // 1.0 at (u*T + t, 0) for t < T (row-batched l2norm mask)
        {"e_t", T, bf16},     // tile t: 1.0 at (u*T + t, 0)
        {"rsel", T, bf16},    // tile t: 1.0 at column u*T + t in every row (row extractor)
        {"csel", 2, bf16},    // tile 0: row h all ones; tile 1: row Nv + h all ones (column extractors)
        {"state_in", KV, fp32},
        {"w_in", Vt, bf16},
        {"scaler", 1, fp32},
        {"eps_l2", 1, bf16},
        {"eps_norm", 1, bf16},
        {"dtb_s", 1, fp32},
        {"nea_s", 1, fp32}};
    const std::vector<Dfb> compute_local = {
        {"tmp", tmp_tiles, fp32}, {"stats", 1, fp32},  {"scratch", 1, fp32}, {"inv", 1, fp32},    {"qn", Kt, fp32},
        {"kn", Kt, fp32},         {"vm", Vt, fp32},    {"kt", Kt, fp32},     {"g1", 1, fp32},     {"a_s", 1, fp32},
        {"b_s", 1, fp32},         {"beta_t", T, fp32}, {"dec", T, fp32},     {"eb", 1, fp32},     {"hd", KV, fp32},
        {"vread", Vt, fp32},      {"delta", Vt, fp32}, {"outer", KV, fp32},  {"hn_a", KV, fp32},  {"hn_b", KV, fp32},
        {"o", Vt, fp32},          {"on", Vt, fp32},    {"gq", Vt, fp32},     {"acc_a", Vt, fp32}, {"acc_b", Vt, fp32},
        {"zs", Vt, fp32}};
    // hnew depth: 2 states (default) or 4 (opt_flags bit4) -- the write-overlap probe at 96 cores
    const uint32_t hnew_depth = (a.opt_flags & 16u) ? 4u : 2u;
    const std::vector<Dfb> compute_out = {{"hnew", hnew_depth * KV, fp32}, {"out", Vt, out_fmt}};

    m2::Group<m2::DataflowBufferSpec> dfbs;
    std::vector<Dfb> all;
    for (const auto* v : {&reader_out, &compute_local, &compute_out}) {
        for (const auto& d : *v) {
            all.push_back(d);
            dfbs.push_back(m2::DataflowBufferSpec{
                .unique_id = m2::DFBSpecName{d.name},
                .entry_size = tt::tile_size(d.fmt),
                .num_entries = d.tiles,
                .data_format_metadata = d.fmt,
            });
        }
    }
    using EP = m2::DFBEndpointType;
    auto bind = [](const std::string& name, EP type) { return m2::DFBBinding{m2::DFBSpecName{name}, name, type}; };

    const uint32_t z_tile0 = (2 * Nk * Kt * TILE_WIDTH + Nv * Vt * TILE_WIDTH) / TILE_WIDTH;
    const uint32_t ab_page = a.qkvz_dim / TILE_WIDTH;
    const uint32_t w_tiles = static_cast<uint32_t>(in.qkv.padded_shape()[-1]) / TILE_WIDTH;  // tiles per qkv tile row
    const uint32_t out_w_tiles = Nv * Vt;                                                    // tiles per out tile row

    // ---- reader
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kDir) + "dataflow/reader_gdn_spec_tloop.cpp",
        .runtime_arg_schema = {.runtime_arg_names = {"u", "h"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    for (const auto& d : reader_out) {
        reader.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
    }
    reader.tensor_bindings = {
        m2::TensorBinding{QKV, "qkv"},
        m2::TensorBinding{DTB, "dtb"},
        m2::TensorBinding{NEA, "nea"},
        m2::TensorBinding{RING, "ring"},
        m2::TensorBinding{WEIGHT, "weight"}};
    reader.compile_time_args = {
        {"Kt", Kt},
        {"Vt", Vt},
        {"Nk", Nk},
        {"Nv", Nv},
        {"T", T},
        {"B", B},
        {"z_tile0", z_tile0},
        {"ab_page", ab_page},
        {"s0_slot", a.s0_slot},
        {"dtb_fp32", in.dt_bias.dtype() == DataType::FLOAT32 ? 1u : 0u},
        {"nea_fp32", in.neg_exp_A.dtype() == DataType::FLOAT32 ? 1u : 0u},
        {"l2_eps_bits", float_bits(a.l2_epsilon)},
        {"norm_eps_bits", float_bits(a.norm_epsilon)},
        {"row_batched", a.row_batched ? 1u : 0u},
        {"w_tiles", w_tiles},
        {"opt_flags", a.opt_flags}};

    // ---- writer
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kDir) + "dataflow/writer_gdn_spec_tloop.cpp",
        .dfb_bindings = {bind("hnew", EP::CONSUMER), bind("out", EP::CONSUMER)},
        .tensor_bindings = {m2::TensorBinding{RING, "ring_out"}, m2::TensorBinding{OUT, "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"u", "h"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
    writer.compile_time_args = {
        {"Kt", Kt},
        {"Vt", Vt},
        {"Nv", Nv},
        {"T", T},
        {"B", B},
        {"write_ring", a.write_ring ? 1u : 0u},
        {"out_w_tiles", out_w_tiles},
        {"opt_flags", a.opt_flags}};

    // ---- compute
    auto compute_hw = ttnn::to_compute_hardware_config(arch, a.compute_kernel_config);
    auto& unpack_modes = m2::unpack_modes(compute_hw);
    for (const auto& d : all) {
        if (d.fmt == fp32) {
            unpack_modes[m2::DFBSpecName{d.name}] = UnpackMode::UnpackToSrc;
        }
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(kDir) + "compute/gdn_spec_tloop.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"T", T},
             {"row_batched", a.row_batched ? 1u : 0u},
             {"opt_flags", a.opt_flags},
             {"scale_bits", float_bits(a.scale)},
             {"inv_dv_bits", float_bits(1.0f / static_cast<float>(a.value_dim))}},
        .runtime_arg_schema = {.runtime_arg_names = {"u"}},
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

    // ---- runtime args: core i handles item (u, h) = (i / Nv, i % Nv)
    m2::KernelRunArgs reader_args{.kernel = READER};
    m2::KernelRunArgs writer_args{.kernel = WRITER};
    m2::KernelRunArgs compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        const uint32_t item = dist.wi_start[i];
        const uint32_t u = item / Nv;
        const uint32_t h = item % Nv;
        m2::AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {{"u", u}, {"h", h}});
        m2::AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"u", u}, {"h", h}});
        m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"u", u}});
    }

    // ---- tensor parameters (ring is read by the reader and written in place by the writer)
    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = QKV, .spec = qkv.tensor_spec()},
        m2::TensorParameter{.unique_id = DTB, .spec = dtb.tensor_spec()},
        m2::TensorParameter{.unique_id = NEA, .spec = nea.tensor_spec()},
        m2::TensorParameter{.unique_id = RING, .spec = ring.tensor_spec()},
        m2::TensorParameter{.unique_id = WEIGHT, .spec = weight.tensor_spec()},
        m2::TensorParameter{.unique_id = OUT, .spec = output.tensor_spec()}};
    m2::ProgramRunArgs run_args;
    run_args.tensor_args = {{QKV, qkv}, {DTB, dtb}, {NEA, nea}, {RING, ring}, {WEIGHT, weight}, {OUT, output}};

    m2::ProgramSpec spec{
        .name = "gdn_spec_tloop_proto",
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
