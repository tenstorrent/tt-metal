// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/kda/gdn_decode_step/device/gdn_decode_step_program_factory.hpp"

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
    uint32_t tiles;
    tt::DataFormat fmt;
};

constexpr const char* kDir = "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/";
constexpr uint32_t kMaxValueBlocks = 4;  // the vsplit writer's peer-coordinate runtime args

// Cores per (head, user group): the largest divisor of Vt <= requested (<= kMaxValueBlocks) whose items * blocks cores
// fit the grid (gdn_spec_step's effective_value_blocks).
uint32_t effective_value_blocks(uint32_t requested, uint32_t Vt, uint32_t items, uint32_t num_cores) {
    for (uint32_t vb = std::min({requested, Vt, kMaxValueBlocks}); vb > 1; --vb) {
        if (Vt % vb == 0 && items * vb <= num_cores) {
            return vb;
        }
    }
    return 1;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts GdnDecodeStepProgramFactory::create_program_artifacts(
    const GdnDecodeStepParams& a, const GdnDecodeStepInputs& in, Tensor& output_tensor) {
    const auto& qkv = in.qkv.mesh_tensor();
    const auto& beta = in.beta.mesh_tensor();
    const auto& g = in.g.mesh_tensor();
    const auto& state = in.state.mesh_tensor();
    const auto& weight = in.weight.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    const auto& device = qkv.device();
    const bool fused = a.fuse_conv;

    const uint32_t Kt = a.key_dim / TILE_WIDTH;
    const uint32_t Vt = a.value_dim / TILE_WIDTH;
    const uint32_t Ct = 2 * Kt + Vt;
    const uint32_t Nv = a.num_value_heads;
    const uint32_t Nk = a.num_key_heads;

    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t num_cores_avail = grid.x * grid.y;
    // work items: plain variant = one value head per core (B = 1); fused variant = (head, user group) with the smallest
    // even group size (1 for B = 1) that fits the grid -> B=32: 12 heads x 8 groups of 4 users on 96 cores.
    const uint32_t B = fused ? static_cast<uint32_t>(in.qkv.logical_shape()[-2]) : 1u;
    uint32_t gs = (B == 1) ? 1u : 2u;
    while (Nv * ((B + gs - 1) / gs) > num_cores_avail) {
        gs += 2;
    }
    const uint32_t ugroups = (B + gs - 1) / gs;
    const uint32_t num_items = Nv * ugroups;
    TT_FATAL(
        num_items <= num_cores_avail, "gdn_decode_step: {} work items exceed {} cores", num_items, num_cores_avail);
    // value blocks (fused + fast_mode only; GDN_VSPLIT kernels): VB cores per (head, user group), each owning state /
    // output column tiles [s*Vs, (s+1)*Vs). item = (head*ugroups + g)*VB + s: a head-group's blocks are consecutive
    // cores. VB = 1: Vs = Vt, KVs = KV and every DFB / kernel / arg below is the original program's.
    const uint32_t VB =
        (fused && a.fast_mode) ? effective_value_blocks(a.value_blocks, Vt, num_items, num_cores_avail) : 1u;
    const bool VSPLIT = VB > 1;
    const uint32_t Vs = Vt / VB;
    const uint32_t KVs = Kt * Vs;
    auto dist = kda_factory_detail::distribute_prep(grid, num_items * VB, num_items * VB);  // one item per core
    const auto& cores = dist.core_set;

    const m2::KernelSpecName READER{"reader"};
    const m2::KernelSpecName WRITER{"writer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName QKV{"qkv"};
    const m2::TensorParamName BETA{"beta"};
    const m2::TensorParamName G{"g"};
    const m2::TensorParamName STATE{"state"};
    const m2::TensorParamName WEIGHT{"weight"};
    const m2::TensorParamName OUT{"out"};
    const m2::TensorParamName HIST{"hist"};
    const m2::TensorParamName TAPS{"taps"};

    const auto fp32 = tt::DataFormat::Float32;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto in_fmt = datatype_to_dataformat_converter(in.qkv.dtype());
    const auto out_fmt = datatype_to_dataformat_converter(a.output_dtype);
    const uint32_t tmp_tiles = std::max(Kt, Vt);

    // ---- dataflow buffers: {name, tiles, format}; producer/consumer roles listed per kernel below
    std::vector<Dfb> reader_out;  // produced by reader, consumed by compute
    if (fused) {
        reader_out = {
            {"hist", 3, bf16},
            {"taps", 4, bf16},
            {"cur", 1, bf16},
            {"sel", Ct, bf16},
            {"z_in", Vt, in_fmt},
            {"a_s", 1, fp32},
            {"b_s", 1, fp32},
            {"dtb_s", 1, fp32},
            {"nea_s", 1, fp32}};
    } else {
        reader_out = {
            {"q_in", Kt, in_fmt}, {"k_in", Kt, in_fmt}, {"v_in", Vt, in_fmt}, {"beta_s", 1, fp32}, {"g_s", 1, fp32}};
    }
    const std::vector<Dfb> common_in = {
        {"state_in", KVs, fp32},
        {"w_in", Vt, bf16},
        {"scaler", 1, fp32},
        {"eps_l2", 1, bf16},
        {"eps_norm", 1, bf16},
        {"mask", 1, bf16}};
    for (const auto& d : common_in) {
        reader_out.push_back(d);
    }
    std::vector<Dfb> compute_local = {
        {"tmp", tmp_tiles, fp32},
        {"stats", 1, fp32},
        {"scratch", 1, fp32},
        {"inv", 1, fp32},
        {"qn", Kt, fp32},
        {"kn", Kt, fp32},
        {"vm", Vs, fp32},
        {"dec", 1, fp32},
        {"hd", KVs, fp32},
        {"vread", Vs, fp32},
        {"delta", Vs, fp32},
        {"kt", Kt, fp32},
        {"outer", KVs, fp32},
        {"hn", KVs, fp32},
        {"o", Vs, fp32},
        {"on", Vt, fp32}};
    // fp32_state (fused + fast only): exact-fp32 state decay/add on the SFPU (see GdnDecodeStepParams::fp32_state)
    const bool fp32_state = a.fp32_state && fused && a.fast_mode;
    if (fused) {
        for (const auto& d : std::vector<Dfb>{
                 {"conv_p", 1, fp32},
                 {"out_acc", Vs, fp32},
                 {"acc2", Vs, fp32},
                 {"qc", Kt, fp32},
                 {"kc", Kt, fp32},
                 {"vc", Vs, fp32},
                 {"beta_t", 1, fp32},
                 {"zs", Vt, fp32}}) {
            compute_local.push_back(d);
        }
        if (fp32_state) {
            // exact-fp32 copy of hd for the SFPU state add (KV tiles = 64 KB)
            compute_local.push_back({"hd32", KVs, fp32});
        }
    }
    std::vector<Dfb> compute_out = {{"hnew", KVs, fp32}, {"out", Vs, out_fmt}};
    std::vector<Dfb> writer_local;
    // VSPLIT: acc_mine = this block's pre-norm output tiles of every user (compute -> writer); acc_x = the head's full
    // pre-norm rows, user ui's tiles at ui*Vt, block s's at + s*Vs (writer -> compute, filled over the NoC by all VB
    // blocks); wshift = the shifted history of every user, read by the reader (NOC_0) instead of the writer, so the
    // writer (NOC_1) only writes. Block 0 alone writes the history, after the exchange (every block has read it).
    std::vector<Dfb> writer_out;
    std::vector<Dfb> reader_to_writer;
    if (VSPLIT) {
        compute_out.push_back({"acc_mine", gs * Vs, fp32});
        writer_out.push_back({"acc_x", gs * Vt, fp32});
        reader_to_writer.push_back({"wshift", gs * 4, bf16});
    } else if (fused) {
        writer_local = {{"wshift", 4, bf16}};
    }

    m2::Group<m2::DataflowBufferSpec> dfbs;
    std::vector<Dfb> all;
    const std::vector<const std::vector<Dfb>*> groups = {
        &reader_out, &compute_local, &compute_out, &writer_local, &writer_out, &reader_to_writer};
    for (const auto* v : groups) {
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

    // ---- reader
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kDir) +
                  (fused ? (a.fast_mode ? (VSPLIT ? "dataflow/reader_gdn_decode_step_conv_fast_vsplit.cpp"
                                                  : "dataflow/reader_gdn_decode_step_conv_fast.cpp")
                                        : "dataflow/reader_gdn_decode_step_conv.cpp")
                         : "dataflow/reader_gdn_decode_step.cpp"),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 (VSPLIT ? std::vector<std::string>{"head", "u0", "nu", "s"}
                         : (fused ? std::vector<std::string>{"head", "u0", "nu"}
                                  : std::vector<std::string>{"wi_start", "wi_count"}))},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    for (const auto& d : reader_out) {
        reader.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
    }
    for (const auto& d : reader_to_writer) {
        reader.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
    }
    reader.tensor_bindings = {
        m2::TensorBinding{QKV, "qkv"},
        m2::TensorBinding{BETA, fused ? "dtb" : "beta"},
        m2::TensorBinding{G, fused ? "nea" : "g"},
        m2::TensorBinding{STATE, "state"},
        m2::TensorBinding{WEIGHT, "weight"}};
    if (fused) {
        reader.tensor_bindings.push_back(m2::TensorBinding{HIST, "hist"});
        reader.tensor_bindings.push_back(m2::TensorBinding{TAPS, "taps"});
        reader.compile_time_args = {
            {"Kt", Kt},
            {"Vt", Vt},
            {"Nk", Nk},
            {"Nv", Nv},
            {"z_tile0", (2 * Nk * Kt * TILE_WIDTH + Nv * Vt * TILE_WIDTH) / TILE_WIDTH},
            {"ab_page", a.qkvz_dim / TILE_WIDTH},
            {"dtb_fp32", in.beta.dtype() == DataType::FLOAT32 ? 1u : 0u},
            {"nea_fp32", in.g.dtype() == DataType::FLOAT32 ? 1u : 0u},
            {"l2_eps_bits", float_bits(a.l2_epsilon)},
            {"norm_eps_bits", float_bits(a.norm_epsilon)}};
        if (VSPLIT) {
            reader.compile_time_args.emplace("VB", VB);
        }
    } else {
        reader.compile_time_args = {
            {"Kt", Kt},
            {"Vt", Vt},
            {"Nk", Nk},
            {"Nv", Nv},
            {"beta_fp32", in.beta.dtype() == DataType::FLOAT32 ? 1u : 0u},
            {"g_fp32", in.g.dtype() == DataType::FLOAT32 ? 1u : 0u},
            {"l2_eps_bits", float_bits(a.l2_epsilon)},
            {"norm_eps_bits", float_bits(a.norm_epsilon)}};
    }

    // ---- writer
    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kDir) + (VSPLIT ? "dataflow/writer_gdn_decode_step_conv_vsplit.cpp"
                                              : (fused ? "dataflow/writer_gdn_decode_step_conv.cpp"
                                                       : "dataflow/writer_gdn_decode_step.cpp")),
        .dfb_bindings = {bind("hnew", EP::CONSUMER), bind("out", EP::CONSUMER)},
        .tensor_bindings = {m2::TensorBinding{STATE, "state_out"}, m2::TensorBinding{OUT, "out"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 (VSPLIT ? std::vector<std::string>{"head",
                                                    "u0",
                                                    "nu",
                                                    "s",
                                                    "peer_x0",
                                                    "peer_y0",
                                                    "peer_x1",
                                                    "peer_y1",
                                                    "peer_x2",
                                                    "peer_y2",
                                                    "peer_x3",
                                                    "peer_y3"}
                         : (fused ? std::vector<std::string>{"head", "u0", "nu"}
                                  : std::vector<std::string>{"wi_start", "wi_count"}))},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    const m2::SemaphoreSpecName READY{"ready"};
    const m2::SemaphoreSpecName VALID{"valid"};
    if (VSPLIT) {
        // write-only on NOC_1: state, output rows, block 0's history (wshift from the reader), the acc exchange
        writer.dfb_bindings.push_back(bind("acc_mine", EP::CONSUMER));
        for (const auto& d : writer_out) {
            writer.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
        }
        for (const auto& d : reader_to_writer) {
            writer.dfb_bindings.push_back(bind(d.name, EP::CONSUMER));
        }
        writer.semaphore_bindings.push_back(m2::SemaphoreBinding{READY, "ready"});
        writer.semaphore_bindings.push_back(m2::SemaphoreBinding{VALID, "valid"});
        writer.tensor_bindings.push_back(m2::TensorBinding{HIST, "hist_w"});
        writer.compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"Nv", Nv}, {"VB", VB}};
    } else if (fused) {
        for (const auto& d : writer_local) {
            writer.dfb_bindings.push_back(bind(d.name, EP::PRODUCER));
            writer.dfb_bindings.push_back(bind(d.name, EP::CONSUMER));
        }
        writer.tensor_bindings.push_back(m2::TensorBinding{QKV, "qkv_w"});
        writer.tensor_bindings.push_back(m2::TensorBinding{HIST, "hist_w"});
        writer.compile_time_args = {{"Kt", Kt}, {"Vt", Vt}, {"Nk", Nk}, {"Nv", Nv}};
    } else {
        writer.compile_time_args = {{"Kt", Kt}, {"Vt", Vt}};
    }

    // ---- compute
    auto compute_hw = ttnn::to_compute_hardware_config(a.compute_kernel_config);
    auto& unpack_modes = compute_hw.unpack_modes;
    // fp32_state: keep the recurrent-state update (decay multiply, state add) in full fp32 on the SFPU. These DFBs
    // are only read through copy_tile into DST (UnpackToDest = exact fp32, no SrcA truncation).
    for (const auto& d : all) {
        if (d.fmt == fp32) {
            const bool to_dest =
                fp32_state && (d.name == "state_in" || d.name == "b_s" || d.name == "hd32" || d.name == "outer");
            unpack_modes[m2::DFBSpecName{d.name}] = to_dest ? UnpackMode::UnpackToDest : UnpackMode::UnpackToSrc;
        }
    }
    m2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(kDir) +
                  (fused ? (a.fast_mode ? (VSPLIT ? "compute/gdn_decode_step_conv_fast_vsplit.cpp"
                                                  : "compute/gdn_decode_step_conv_fast.cpp")
                                        : "compute/gdn_decode_step_conv.cpp")
                         : "compute/gdn_decode_step.cpp"),
        .compiler_options = {.opt_level = fused ? KernelBuildOptLevel::O2 : KernelBuildOptLevel::O3},
        .compile_time_args =
            {{"Kt", Kt},
             {"Vt", Vt},
             {"scale_bits", float_bits(a.scale)},
             {"inv_dv_bits", float_bits(1.0f / static_cast<float>(a.value_dim))}},
        .runtime_arg_schema =
            {.runtime_arg_names = (fused ? std::vector<std::string>{"nu"} : std::vector<std::string>{"wi_count"})},
        .hw_config = std::move(compute_hw),
    };
    if (fp32_state) {
        // the hd32 DFB only exists then; the kernel guards its references with #if GDN_FP32_STATE
        compute.compiler_options.defines.emplace("GDN_FP32_STATE", "1");
    }
    if (VSPLIT) {
        compute.compile_time_args.emplace("VB", VB);
        compute.runtime_arg_schema.runtime_arg_names.push_back("s");
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

    // ---- runtime args
    TT_FATAL(
        !VSPLIT || dist.cores.size() == num_items * VB,
        "gdn_decode_step: expected one (head, group, block) per core");
    m2::KernelRunArgs reader_args{.kernel = READER};
    m2::KernelRunArgs writer_args{.kernel = WRITER};
    m2::KernelRunArgs compute_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        if (VSPLIT) {
            const uint32_t item = dist.wi_start[i];  // one (head, group, block) per core
            const uint32_t hg = item / VB;
            const uint32_t s = item % VB;
            const uint32_t head = hg / ugroups;
            const uint32_t g = hg % ugroups;
            const uint32_t u0 = g * gs;
            const uint32_t nu = std::min(gs, B - u0);
            uint32_t px[kMaxValueBlocks] = {0, 0, 0, 0};
            uint32_t py[kMaxValueBlocks] = {0, 0, 0, 0};
            for (uint32_t p = 0; p < VB; ++p) {
                const auto peer = device.worker_core_from_logical_core(dist.cores[hg * VB + p]);
                px[p] = static_cast<uint32_t>(peer.x);
                py[p] = static_cast<uint32_t>(peer.y);
            }
            m2::AddRuntimeArgsForNode(
                reader_args.runtime_arg_values, core, {{"head", head}, {"u0", u0}, {"nu", nu}, {"s", s}});
            m2::AddRuntimeArgsForNode(
                writer_args.runtime_arg_values,
                core,
                {{"head", head},
                 {"u0", u0},
                 {"nu", nu},
                 {"s", s},
                 {"peer_x0", px[0]},
                 {"peer_y0", py[0]},
                 {"peer_x1", px[1]},
                 {"peer_y1", py[1]},
                 {"peer_x2", px[2]},
                 {"peer_y2", py[2]},
                 {"peer_x3", px[3]},
                 {"peer_y3", py[3]}});
            m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"nu", nu}, {"s", s}});
        } else if (fused) {
            const uint32_t item = dist.wi_start[i];  // one item per core
            const uint32_t head = item / ugroups;
            const uint32_t g = item % ugroups;
            const uint32_t u0 = g * gs;
            const uint32_t nu = std::min(gs, B - u0);
            m2::AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {{"head", head}, {"u0", u0}, {"nu", nu}});
            m2::AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"head", head}, {"u0", u0}, {"nu", nu}});
            m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"nu", nu}});
        } else {
            m2::AddRuntimeArgsForNode(
                reader_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
            m2::AddRuntimeArgsForNode(
                writer_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
            m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, core, {{"wi_count", dist.wi_count[i]}});
        }
    }

    // ---- tensor parameters
    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = QKV, .spec = qkv.tensor_spec()},
        m2::TensorParameter{.unique_id = BETA, .spec = beta.tensor_spec()},
        m2::TensorParameter{.unique_id = G, .spec = g.tensor_spec()},
        m2::TensorParameter{.unique_id = STATE, .spec = state.tensor_spec()},
        m2::TensorParameter{.unique_id = WEIGHT, .spec = weight.tensor_spec()},
        m2::TensorParameter{.unique_id = OUT, .spec = output.tensor_spec()}};
    m2::ProgramRunArgs run_args;
    run_args.tensor_args = {{QKV, qkv}, {BETA, beta}, {G, g}, {STATE, state}, {WEIGHT, weight}, {OUT, output}};
    if (fused) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = HIST, .spec = in.conv_hist->mesh_tensor().tensor_spec()});
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = TAPS, .spec = in.conv_taps->mesh_tensor().tensor_spec()});
        run_args.tensor_args.emplace(HIST, in.conv_hist->mesh_tensor());
        run_args.tensor_args.emplace(TAPS, in.conv_taps->mesh_tensor());
    }

    m2::ProgramSpec spec{
        .name = fused ? "gdn_decode_step_conv" : "gdn_decode_step",
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
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.kernel_run_args.push_back(std::move(compute_args));
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
