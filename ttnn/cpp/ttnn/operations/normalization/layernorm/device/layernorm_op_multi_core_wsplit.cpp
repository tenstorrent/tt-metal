// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Width-split interleaved RMSNorm (LayerNormDefaultProgramConfig.width_split = S in {2, 3}).
//
// The interleaved multi-core RMSNorm puts one tile row on one core, so a [M, K] input runs on M / 32 cores and each
// core walks all K / 32 tiles of its row (compute-bound for the 1024 x 2048 prefill norms: 32 cores, 64 + 64 tiles).
// Here each tile row is split across S cores (widths as even as possible, e.g. 22 / 21 / 21 of 64 tiles). Every core
// runs the streaming layernorm.cpp pipeline over its own tiles (a from the reader on one NoC, b from the writer on the
// other, h and the output drained by the writer), except that its partial mean of squares is exchanged with the row's
// other S - 1 cores (one tile over the NoC plus a semaphore per core, reader kernel) and the S partials are summed
// before eps / rsqrt. One tile row per core: M / 32 * S cores of the device grid.

#include <bit>
#include <optional>
#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

namespace m2 = tt::tt_metal::experimental;

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE_WIDE{"compute_wide"};      // cores with the wider width part
const m2::KernelSpecName COMPUTE_NARROW{"compute_narrow"};  // cores with the narrower one

const m2::DFBSpecName IN{"in"};
const m2::DFBSpecName INB{"inb"};
const m2::DFBSpecName SCALER{"scaler"};
const m2::DFBSpecName EPS{"eps"};
const m2::DFBSpecName GAMMA{"gamma"};
const m2::DFBSpecName OUT{"out"};    // the second half of the output blocks (to the writer)
const m2::DFBSpecName OUT2{"out2"};  // the first half (to the reader), so the output writes use both NoCs
const m2::DFBSpecName XMM2{"xmm2"};
const m2::DFBSpecName EX2PE{"ex2pe"};
const m2::DFBSpecName FUSION{"fusion"};
const m2::DFBSpecName XMM{"xmm"};
const m2::DFBSpecName H_OUT{"h_out"};
const m2::DFBSpecName PART{"part"};  // this core's partial mean of squares (compute -> reader)
const m2::DFBSpecName RECV{"recv"};  // the row's S partials (reader -> compute), written by the S cores

const m2::TensorParamName INPUT{"input"};
const m2::TensorParamName RESIDUAL{"residual"};
const m2::TensorParamName GAMMA_T{"weight"};
const m2::TensorParamName OUTPUT{"output"};
const m2::TensorParamName RESIDUAL_OUT_T{"residual_output"};

const m2::SemaphoreSpecName PARTIAL_READY{"partial_ready"};

void bind_dfb(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name, m2::DFBEndpointType role) {
    kernel.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = dfb,
        .accessor_name = std::move(accessor_name),
        .endpoint_type = role,
    });
}

void bind_self_loop(m2::KernelSpec& kernel, const m2::DFBSpecName& dfb, std::string accessor_name) {
    bind_dfb(kernel, dfb, accessor_name, m2::DFBEndpointType::PRODUCER);
    bind_dfb(kernel, dfb, std::move(accessor_name), m2::DFBEndpointType::CONSUMER);
}

void bind_tensor(m2::KernelSpec& kernel, const m2::TensorParamName& tensor, std::string accessor_name) {
    kernel.tensor_bindings.push_back(m2::TensorBinding{
        .tensor_parameter_name = tensor,
        .accessor_name = std::move(accessor_name),
    });
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts LayerNormWidthSplitProgramFactory::create_program_artifacts(
    const LayerNormParams& operation_attributes,
    const LayerNormInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<CoreRangeSet>& core_range_set) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& residual_output = tensor_args.residual_output;
    auto& output = tensor_return_value;
    const bool fuse_pre_add = b.has_value();
    const bool residual_out = residual_output.has_value();
    const auto& pc = std::get<LayerNormDefaultProgramConfig>(operation_attributes.program_config);
    const uint32_t S = pc.width_split;
    const float eps = operation_attributes.eps;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    IDevice* device = a.device();

    [[maybe_unused]] auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t W = a.logical_shape()[-1];
    const uint32_t Kt = a.padded_shape()[-1] / tile_width;
    const uint32_t Mt = a.physical_volume() / a.padded_shape()[-1] / tile_height;
    const uint32_t block_size = fp32_dest_acc_en ? 4 : 8;

    // width parts: the first rem parts are one tile wider
    const uint32_t base_wt = Kt / S;
    const uint32_t rem = Kt % S;
    const uint32_t wt_max = base_wt + (rem > 0 ? 1 : 0);
    const uint32_t wt_up = tt::round_up(wt_max, block_size);

    // cores: index i = p * Mt + r (width part p of tile row r), laid out column-major over the grid, so each width
    // part (and so each compute-kernel group) covers a few rectangles and its kernels / config are multicast
    // (per-core ranges made the dispatch ~80 us slower)
    CoreCoord grid = device->compute_with_storage_grid_size();
    if (core_range_set.has_value()) {
        const auto bb = core_range_set.value().bounding_box();
        grid = CoreCoord{bb.end_coord.x - bb.start_coord.x + 1, bb.end_coord.y - bb.start_coord.y + 1};
    }
    const uint32_t num_cores = Mt * S;
    TT_FATAL(num_cores <= grid.x * grid.y, "width_split: {} cores needed, grid has {}", num_cores, grid.x * grid.y);
    const CoreCoord grid_start =
        core_range_set.has_value() ? core_range_set.value().bounding_box().start_coord : CoreCoord{0, 0};
    auto core_of = [&](uint32_t i) { return CoreCoord{grid_start.x + i / grid.y, grid_start.y + i % grid.y}; };

    std::vector<CoreRange> wide_ranges, narrow_ranges, all_ranges;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t p = i / Mt;
        const CoreCoord c = core_of(i);
        ((rem == 0 || p < rem) ? wide_ranges : narrow_ranges).emplace_back(c, c);
        all_ranges.emplace_back(c, c);
    }
    const CoreRangeSet all_cores = CoreRangeSet(all_ranges).merge_ranges();
    const CoreRangeSet wide_cores = CoreRangeSet(wide_ranges).merge_ranges();
    const bool has_narrow = !narrow_ranges.empty();
    const CoreRangeSet narrow_cores = has_narrow ? CoreRangeSet(narrow_ranges).merge_ranges() : CoreRangeSet();

    // data formats
    const tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const tt::DataFormat interm_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const tt::DataFormat gamma_df = tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype());
    const tt::DataFormat inb_df =
        fuse_pre_add ? tt::tt_metal::datatype_to_dataformat_converter(b.value().dtype()) : tt::DataFormat::Invalid;
    const tt::DataFormat h_df = residual_out
                                    ? tt::tt_metal::datatype_to_dataformat_converter(residual_output.value().dtype())
                                    : tt::DataFormat::Invalid;
    // as layernorm_op_multi_core.cpp: with a residual output, x = a + b is kept in h's dtype
    const tt::DataFormat xmm_df = residual_out ? h_df : interm_df;
    const uint32_t interm_ts = tt::tile_size(interm_df);
    const uint32_t bf16_ts = tt::tile_size(tt::DataFormat::Float16_b);

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{.name = "layernorm_width_split"};
    auto add_dfb = [&spec](const m2::DFBSpecName& id, uint32_t n, uint32_t entry, tt::DataFormat df) {
        spec.dataflow_buffers.push_back(
            m2::DataflowBufferSpec{.unique_id = id, .entry_size = entry, .num_entries = n, .data_format_metadata = df});
    };
    add_dfb(IN, fuse_pre_add ? 2 * block_size : wt_up, tt::tile_size(in_df), in_df);
    add_dfb(OUT, 2 * block_size, tt::tile_size(out_df), out_df);
    const uint32_t reader_out_blocks = (tt::div_up(wt_max, block_size) + 1) / 2;
    add_dfb(OUT2, reader_out_blocks * block_size, tt::tile_size(out_df), out_df);
    add_dfb(SCALER, 2, bf16_ts, tt::DataFormat::Float16_b);
    add_dfb(EPS, 2, bf16_ts, tt::DataFormat::Float16_b);
    add_dfb(XMM2, wt_up, interm_ts, interm_df);
    add_dfb(EX2PE, 2, interm_ts, interm_df);
    add_dfb(FUSION, 2 * block_size, interm_ts, interm_df);
    add_dfb(GAMMA, wt_up, tt::tile_size(gamma_df), gamma_df);
    add_dfb(PART, 1, interm_ts, interm_df);
    add_dfb(RECV, S, interm_ts, interm_df);
    if (fuse_pre_add) {
        add_dfb(XMM, wt_up, tt::tile_size(xmm_df), xmm_df);
        // the writer reads all of b before it drains h, so b and h each hold the whole row (no deadlock)
        add_dfb(INB, wt_up, tt::tile_size(inb_df), inb_df);
    }
    if (residual_out) {
        add_dfb(H_OUT, wt_up, tt::tile_size(h_df), h_df);
    }

    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = GAMMA_T, .spec = gamma.value().tensor_spec()});
    if (fuse_pre_add) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = RESIDUAL, .spec = b.value().tensor_spec()});
    }
    if (residual_out) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = RESIDUAL_OUT_T, .spec = residual_output.value().tensor_spec()});
    }
    spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = PARTIAL_READY, .target_nodes = all_cores});

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    const std::string kdir = "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/";
    m2::KernelSpec reader{
        .unique_id = READER,
        .source = kdir + "dataflow/reader_ln_wsplit.cpp",
        .compile_time_args = {{"block_size", block_size}, {"W", W}, {"nsplit", S}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"Wt",
                  "reader_start",
                  "eps",
                  "my_slot",
                  "gamma_start",
                  "peer_x0",
                  "peer_x1",
                  "peer_x2",
                  "peer_y0",
                  "peer_y1",
                  "peer_y2"}},
        .hw_config = create_reader_datamovement_config(device->arch()),
    };
    bind_dfb(reader, IN, "in", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, EPS, "eps", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, SCALER, "scaler", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, PART, "part", m2::DFBEndpointType::CONSUMER);
    bind_dfb(reader, RECV, "recv", m2::DFBEndpointType::PRODUCER);
    bind_tensor(reader, INPUT, "src");
    bind_dfb(reader, GAMMA, "gamma", m2::DFBEndpointType::PRODUCER);
    bind_tensor(reader, GAMMA_T, "gamma");
    // the first half of the output blocks leave through the reader (after the exchange)
    bind_dfb(reader, OUT2, "out2", m2::DFBEndpointType::CONSUMER);
    bind_tensor(reader, OUTPUT, "dst");

    reader.semaphore_bindings.push_back(
        m2::SemaphoreBinding{.semaphore_spec_name = PARTIAL_READY, .accessor_name = "partial_ready"});

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir + "dataflow/writer_ln_wsplit.cpp",
        .compile_time_args = {{"block_size", block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"Wt", "writer_start"}},
        .hw_config = create_writer_datamovement_config(device->arch()),
    };
    bind_dfb(writer, OUT, "out", m2::DFBEndpointType::CONSUMER);
    bind_tensor(writer, OUTPUT, "dst");

    // h = a + b (after b and gamma), then the second half of the output blocks
    if (residual_out) {
        writer.compiler_options.defines.emplace("RESIDUAL_OUT", "1");
        bind_dfb(writer, H_OUT, "h_out", m2::DFBEndpointType::CONSUMER);
        bind_tensor(writer, RESIDUAL_OUT_T, "dst_h");
    }

    if (fuse_pre_add) {
        writer.compiler_options.defines.emplace("FUSE_PRE_ADD", "1");
        bind_dfb(writer, INB, "inb", m2::DFBEndpointType::PRODUCER);
        bind_tensor(writer, RESIDUAL, "src_b");
    }

    const bool float32_reduction = fp32_dest_acc_en && !pc.legacy_reduction;
    auto make_compute = [&](const m2::KernelSpecName& name, uint32_t wt) {
        m2::KernelSpec compute{
            .unique_id = name,
            .source = kdir + "compute/layernorm_wsplit.cpp",
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .compile_time_args =
                {{"Wt", wt},
                 {"block_size", block_size},
                 {"do_gamma", 1u},
                 {"do_beta", 0u},
                 {"fp32_dest_acc_en", static_cast<uint32_t>(fp32_dest_acc_en)},
                 {"W", W},
                 {"tile_width", tile_width},
                 {"float32_reduction", static_cast<uint32_t>(float32_reduction)},
                 {"legacy_rsqrt", static_cast<uint32_t>(pc.legacy_rsqrt)},
                 {"nsplit", S}},
            .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
            .hw_config = to_compute_hardware_config(device->arch(), compute_kernel_config),
        };
        compute.compiler_options.defines.emplace("RMSNORM", "1");
        compute.compiler_options.defines.emplace("FUSE_GAMMA", "1");
        if (fuse_pre_add) {
            compute.compiler_options.defines.emplace("FUSE_PRE_ADD", "1");
            bind_dfb(compute, INB, "inb", m2::DFBEndpointType::CONSUMER);
            bind_self_loop(compute, XMM, "xmm");
        }
        if (residual_out) {
            compute.compiler_options.defines.emplace("RESIDUAL_OUT", "1");
            bind_dfb(compute, H_OUT, "h_out", m2::DFBEndpointType::PRODUCER);
        }
        bind_dfb(compute, IN, "in", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, OUT, "out", m2::DFBEndpointType::PRODUCER);
        bind_dfb(compute, OUT2, "out2", m2::DFBEndpointType::PRODUCER);
        bind_dfb(compute, EPS, "eps", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, SCALER, "scaler", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, GAMMA, "gamma", m2::DFBEndpointType::CONSUMER);
        bind_self_loop(compute, XMM2, "xmm2");
        bind_self_loop(compute, EX2PE, "ex2pe");
        bind_self_loop(compute, FUSION, "fusion");
        bind_dfb(compute, PART, "part", m2::DFBEndpointType::PRODUCER);
        bind_dfb(compute, RECV, "recv", m2::DFBEndpointType::CONSUMER);
        if (fp32_dest_acc_en) {
            auto& modes = m2::unpack_modes(std::get<m2::ComputeHardwareConfig>(compute.hw_config));
            for (const auto& binding : compute.dfb_bindings) {
                if (binding.endpoint_type != m2::DFBEndpointType::CONSUMER) {
                    continue;
                }
                for (const auto& d : spec.dataflow_buffers) {
                    if (d.unique_id == binding.dfb_spec_name && d.data_format_metadata == tt::DataFormat::Float32) {
                        modes.emplace(binding.dfb_spec_name, UnpackMode::UnpackToSrc);
                    }
                }
            }
        }
        return compute;
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime arguments
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_args{.kernel = READER};
    m2::KernelRunArgs writer_args{.kernel = WRITER};
    m2::KernelRunArgs compute_wide_args{.kernel = COMPUTE_WIDE};
    m2::KernelRunArgs compute_narrow_args{.kernel = COMPUTE_NARROW};
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t r = i % Mt;
        const uint32_t p = i / Mt;
        const uint32_t wt = base_wt + (p < rem ? 1 : 0);
        const uint32_t w0 = p * base_wt + std::min(p, rem);
        const CoreCoord core = core_of(i);
        uint32_t px[3] = {0, 0, 0}, py[3] = {0, 0, 0};
        for (uint32_t q = 0; q < S; ++q) {
            const CoreCoord peer = device->worker_core_from_logical_core(core_of(q * Mt + r));
            px[q] = peer.x;
            py[q] = peer.y;
        }
        const uint32_t start = r * Kt + w0;
        m2::AddRuntimeArgsForNode(
            reader_args.runtime_arg_values,
            core,
            {{"Wt", wt},
             {"reader_start", start},
             {"eps", std::bit_cast<uint32_t>(eps)},
             {"my_slot", p},
             {"gamma_start", w0},
             {"peer_x0", px[0]},
             {"peer_x1", px[1]},
             {"peer_x2", px[2]},
             {"peer_y0", py[0]},
             {"peer_y1", py[1]},
             {"peer_y2", py[2]}});
        m2::AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"Wt", wt}, {"writer_start", start}});
        auto& cargs = (rem == 0 || p < rem) ? compute_wide_args : compute_narrow_args;
        m2::AddRuntimeArgsForNode(cargs.runtime_arg_values, core, {{"NCHt", 1u}});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(make_compute(COMPUTE_WIDE, wt_max));
    spec.work_units.push_back(
        m2::WorkUnitSpec{.name = "wide", .kernels = {READER, WRITER, COMPUTE_WIDE}, .target_nodes = wide_cores});
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.kernel_run_args.push_back(std::move(compute_wide_args));
    if (has_narrow) {
        spec.kernels.push_back(make_compute(COMPUTE_NARROW, base_wt));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "narrow", .kernels = {READER, WRITER, COMPUTE_NARROW}, .target_nodes = narrow_cores});
        run_args.kernel_run_args.push_back(std::move(compute_narrow_args));
    }
    run_args.tensor_args.emplace(INPUT, a.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    run_args.tensor_args.emplace(GAMMA_T, gamma.value().mesh_tensor());
    if (fuse_pre_add) {
        run_args.tensor_args.emplace(RESIDUAL, b.value().mesh_tensor());
    }
    if (residual_out) {
        run_args.tensor_args.emplace(RESIDUAL_OUT_T, residual_output.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
