// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core W reduction program factory, ported to the Metal 2.0 host API
// (ProgramSpecFactoryConcept). The framework adapter
// (ProgramSpecMeshWorkloadFactoryAdapter in
// ttnn/api/ttnn/mesh_device_operation_adapter.hpp) handles cache-miss /
// cache-hit dispatch and TensorArg updates.

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/metal2_artifacts.hpp"

#include <bit>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// Ids — kept local to this TU. The four reduction factories live in one Unity
// build target, so anonymous-namespace symbols from each .cpp file are merged;
// prefix with `W_` to avoid collision with the H / HW / Welford factories that
// will follow the same pattern. DFB ids and tensor parameter ids are local to
// each ProgramSpec, so reusing strings across factories is fine (uniqueness is
// per-Program).
constexpr const char* W_READER = "reader";
constexpr const char* W_WRITER = "writer";
constexpr const char* W_COMPUTE_G1 = "compute_g1";
constexpr const char* W_COMPUTE_G2 = "compute_g2";

constexpr const char* W_WU_G1 = "wu_g1";
constexpr const char* W_WU_G2 = "wu_g2";

constexpr const char* INPUT_DFB = "input";
constexpr const char* SCALER_DFB = "scaler";
constexpr const char* OUTPUT_DFB = "output";
constexpr const char* ACC_DFB = "acc";    // negate only
constexpr const char* INEG_DFB = "ineg";  // negate only

constexpr const char* INPUT_TENSOR = "input";
constexpr const char* OUTPUT_TENSOR = "output";

// Helper to convert legacy reduce-defines map to KernelSpec::CompilerOptions::Defines.
m2::KernelSpec::CompilerOptions::Defines defines_from_map(const std::map<std::string, std::string>& src) {
    m2::KernelSpec::CompilerOptions::Defines out;
    out.reserve(src.size());
    for (const auto& [k, v] : src) {
        out.emplace_back(k, v);
    }
    return out;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts ReduceDeviceOperation::ReduceMultiCoreWProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    const tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    const tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_rows = NC * Ht;

    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_per_core_group_1, num_rows_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);
    }
    TT_FATAL(num_cores > 0, "Reduce W requires at least one worker core");
    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
    // For now, defer the negate path to a follow-up — needs reduce_w_neg_metal2.cpp fork.
    TT_FATAL(
        !operation_attributes.negate,
        "Reduce W Metal 2.0 port: negate path not yet implemented in the ported factory (TODO)");

    // For MAX/MIN with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after
    // the reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    // ----- DataflowBufferSpecs -----
    // (Placement is derived from kernel bindings; no node_ranges field.)

    constexpr uint32_t kNumInputEntries = 2;
    constexpr uint32_t kNumScalerEntries = 1;
    constexpr uint32_t kNumOutputEntries = 2;
    constexpr uint32_t kNumNegateEntries = 1;

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = kNumInputEntries,
        .data_format_metadata = src0_cb_data_format,
        .tile_format_metadata = a.tensor_spec().tile(),
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = kNumScalerEntries,
        .data_format_metadata = scaler_cb_data_format,
    });
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = OUTPUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = kNumOutputEntries,
        .data_format_metadata = dst_cb_data_format,
        .tile_format_metadata = output.tensor_spec().tile(),
    });
    if (operation_attributes.negate) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = kNumNegateEntries,
            .data_format_metadata = dst_cb_data_format,
        });
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = kNumNegateEntries,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    // ----- KernelSpecs -----

    // Reader kernel: produces INPUT (from tensor) and SCALER.
    m2::KernelSpec reader{
        .unique_id = W_READER,
        .source = m2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
                                                 "reader_unary_reduce_universal_start_id.cpp"},
        .compiler_options =
            {
                .defines = defines_from_map(reduce_defines),
            },
        .dfb_bindings =
            {
                {.dfb_spec_name = INPUT_DFB,
                 .local_accessor_name = "input",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
                {.dfb_spec_name = SCALER_DFB,
                 .local_accessor_name = "scaler",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"},
            },
        .compile_time_arg_bindings = {{"scaler_bits", scaler_bits}},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles", "start_id"}},
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,  // RISCV_1 -> reader convention
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    // Writer kernel: consumes OUTPUT.
    m2::KernelSpec writer{
        .unique_id = W_WRITER,
        .source = m2::KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                                 "writer_unary_interleaved_start_id.cpp"},
        .compiler_options =
            {
                .defines = defines_from_map(reduce_defines),
            },
        .dfb_bindings =
            {
                {.dfb_spec_name = OUTPUT_DFB,
                 .local_accessor_name = "output",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"num_pages", "start_id"}},
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    // Compute kernel (group 1): consumes INPUT + SCALER, produces OUTPUT.
    // Per-group CTAs (Ht differs) — preserve multiplicity (Anti-pattern: Demoting per-group CTA to RTA).
    const std::string compute_kernel_source =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/") +
        (operation_attributes.negate ? "reduce_w_neg.cpp" : "reduce.cpp");

    auto make_compute = [&](const char* unique_id, uint32_t this_Ht, const CoreRangeSet&) {
        m2::KernelSpec spec{
            .unique_id = unique_id,
            .source = m2::KernelSpec::SourceFilePath{compute_kernel_source},
            .compiler_options =
                {
                    .defines = defines_from_map(reduce_defines),
                },
            .dfb_bindings =
                {
                    {.dfb_spec_name = INPUT_DFB,
                     .local_accessor_name = "input",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                    {.dfb_spec_name = SCALER_DFB,
                     .local_accessor_name = "scaler",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                    {.dfb_spec_name = OUTPUT_DFB,
                     .local_accessor_name = "output",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
                },
            .compile_time_arg_bindings = {{"Ht", this_Ht}, {"Wt", Wt}, {"NC", 1u}},
            .config_spec =
                m2::ComputeConfiguration{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                },
        };
        if (use_post_mul) {
            spec.compile_time_arg_bindings.push_back({"post_mul_scaler_bits", post_mul_scaler_bits});
        }
        // Conditional / optional DFB bindings for negate.
        if (operation_attributes.negate) {
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = ACC_DFB,
                 .local_accessor_name = "acc",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = ACC_DFB,
                 .local_accessor_name = "acc",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = INEG_DFB,
                 .local_accessor_name = "ineg",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER});
            spec.dfb_bindings.push_back(
                {.dfb_spec_name = INEG_DFB,
                 .local_accessor_name = "ineg",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER});
        }
        return spec;
    };

    const bool g2_present = !core_group_2.ranges().empty();
    m2::KernelSpec compute_g1 = make_compute(W_COMPUTE_G1, num_rows_per_core_group_1, core_group_1);
    std::optional<m2::KernelSpec> compute_g2;
    if (g2_present) {
        compute_g2 = make_compute(W_COMPUTE_G2, num_rows_per_core_group_2, core_group_2);
    }

    // ----- WorkUnitSpecs -----
    // One WU per compute group; reader + writer co-located with each.

    std::vector<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .unique_id = W_WU_G1,
        .kernels = {W_READER, W_WRITER, W_COMPUTE_G1},
        .target_nodes = core_group_1,
    });
    if (g2_present) {
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = W_WU_G2,
            .kernels = {W_READER, W_WRITER, W_COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    // ----- ProgramSpec assembly -----

    std::vector<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(std::move(compute_g1));
    if (compute_g2.has_value()) {
        kernels.push_back(std::move(*compute_g2));
    }

    m2::ProgramSpec spec{
        .program_id = "reduce_multi_core_w",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                {.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()},
                {.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    // ----- ProgramRunParams -----

    m2::ProgramRunParams run_params;
    m2::ProgramRunParams::KernelRunParams reader_rp{.kernel_spec_name = W_READER};
    m2::ProgramRunParams::KernelRunParams writer_rp{.kernel_spec_name = W_WRITER};

    // Compute kernels: no RTAs (per-core dimensions are CTAs).
    // Still need a KernelRunParams entry per kernel-spec-with-RTAs; if the schema has no
    // RTAs, we omit the entry per the recipe note ("If the kernel has no RTAs, the
    // run-params entry may be omitted entirely.").

    const uint32_t out_dim_divider = Wt;

    std::vector<CoreCoord> cores;
    if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);

    uint32_t num_tiles_read = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        const uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;

        reader_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args = {{"num_tiles", num_tensor_tiles_per_core}, {"start_id", num_tiles_read}},
        });
        writer_rp.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args =
                {{"num_pages", num_tensor_tiles_per_core / out_dim_divider},
                 {"start_id", num_tiles_read / out_dim_divider}},
        });
        num_tiles_read += num_tensor_tiles_per_core;
    }
    TT_FATAL(
        num_tiles_read == num_rows * Wt,
        "Reduce W assigned {} input tiles, expected {}",
        num_tiles_read,
        num_rows * Wt);

    run_params.kernel_run_params = {std::move(reader_rp), std::move(writer_rp)};

    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = INPUT_TENSOR, .tensor = std::cref(a.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
