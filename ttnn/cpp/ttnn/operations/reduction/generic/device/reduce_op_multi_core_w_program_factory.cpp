// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core W reduction program factory, migrated to the Metal 2.0 host API.
//
// The factory follows ttnn's ProgramFactoryConcept (create + override_runtime_arguments)
// because the device_operation framework does not yet have a first-class adapter for
// ProgramSpec-based factories. The Program returned from MakeProgramFromSpec() is wrapped
// into a CachedProgram alongside ReduceMultiCoreWSharedVariables.
//
// On cache hit, override_runtime_arguments() recomputes the per-node RTAs (only the
// buffer addresses change between executions) and re-applies them via
// SetProgramRunParameters() — that single call replaces the legacy
// GetRuntimeArgs(...) write-back loop.

#include "reduce_op_multi_core_w_program_factory.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "reduce_metal2_factory_helpers.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// KernelSpec / WorkUnitSpec ids — kept local with a `W_` prefix so the symbols
// don't clash with the HW factory's analogues under Unity builds (which merge
// anonymous namespaces from different .cpp files into one TU). DFB ids and the
// MakeDFB / BindDFB / DefinesFromMap helpers are shared and live in
// reduce_metal2_factory_helpers.hpp.
constexpr const char* W_READER_KERNEL = "reduce_w_reader";
constexpr const char* W_WRITER_KERNEL = "reduce_w_writer";
constexpr const char* W_COMPUTE_KERNEL = "reduce_w_compute";
constexpr const char* W_WORK_UNIT = "all_workers";

// Determine the work distribution across cores. Mirrors the legacy factory; lifted
// into a helper because both create() and override_runtime_arguments() consume it
// (create uses it to build the spec; override uses the cached snapshot in
// shared_variables to re-emit RTAs).
struct WWorkDistribution {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

WWorkDistribution ComputeWWorkDistribution(const ReduceParams& attrs, const tt::tt_metal::Tensor& input) {
    using namespace tt::tt_metal;

    const auto& shape = input.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    (void)W;
    (void)NC;

    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t Ht = H / tile_height;

    auto* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_rows = (shape[1] * shape[0]) * Ht;

    WWorkDistribution wd;
    if (attrs.sub_core_grids.has_value()) {
        std::tie(
            wd.num_cores,
            wd.all_cores,
            wd.core_group_1,
            wd.core_group_2,
            wd.num_rows_per_core_group_1,
            wd.num_rows_per_core_group_2) = split_work_to_cores(*attrs.sub_core_grids, num_rows);
        for (const auto& range : wd.all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    wd.cores.emplace_back(x, y);
                }
            }
        }
    } else {
        std::tie(
            wd.num_cores,
            wd.all_cores,
            wd.core_group_1,
            wd.core_group_2,
            wd.num_rows_per_core_group_1,
            wd.num_rows_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size, num_rows);
        wd.cores =
            grid_to_cores(wd.num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    TT_FATAL(wd.num_cores > 0, "Reduce W requires at least one worker core");
    TT_FATAL(
        wd.cores.size() == wd.num_cores,
        "Resolved core list size {} must match split num_cores {}",
        wd.cores.size(),
        wd.num_cores);
    TT_FATAL(num_rows == 0 || !wd.cores.empty(), "Non-zero reduce workload requires non-empty core list");
    return wd;
}

// Build the ProgramRunParams for the reader/writer/compute kernels given the
// distribution snapshot and the (possibly-updated) buffer addresses. Used both at
// initial creation and on cache-hit re-parameterization.
m2::ProgramRunParams BuildRunParams(
    const ReduceMultiCoreWSharedVariables& shared, uint32_t src_addr, uint32_t dst_addr) {
    m2::ProgramRunParams params;

    const uint32_t out_dim_divider = shared.Wt;
    TT_FATAL(out_dim_divider != 0, "Wt cached in shared_variables must be non-zero");

    m2::ProgramRunParams::KernelRunParams reader_params;
    reader_params.kernel_spec_name = W_READER_KERNEL;

    m2::ProgramRunParams::KernelRunParams writer_params;
    writer_params.kernel_spec_name = W_WRITER_KERNEL;

    m2::ProgramRunParams::KernelRunParams compute_params;
    compute_params.kernel_spec_name = W_COMPUTE_KERNEL;

    uint32_t num_tiles_read = 0;
    for (const auto& core : shared.cores) {
        uint32_t num_rows_per_core = 0;
        if (shared.core_group_1.contains(core)) {
            num_rows_per_core = shared.num_rows_per_core_group_1;
        } else if (shared.core_group_2.contains(core)) {
            num_rows_per_core = shared.num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        const uint32_t num_tensor_tiles_per_core = num_rows_per_core * shared.Wt;

        reader_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args =
                {
                    {"src_addr", src_addr},
                    {"num_tiles", num_tensor_tiles_per_core},
                    {"start_id", num_tiles_read},
                },
        });
        writer_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args =
                {
                    {"dst_addr", dst_addr},
                    {"num_pages", num_tensor_tiles_per_core / out_dim_divider},
                    {"start_id", num_tiles_read / out_dim_divider},
                },
        });
        compute_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args = {{"Ht", num_rows_per_core}},
        });

        num_tiles_read += num_tensor_tiles_per_core;
        if (i == num_cores - 1) {
            TT_FATAL(
                num_tiles_read == num_rows * Wt,
                "Reduce W assigned {} input tiles, expected {}",
                num_tiles_read,
                num_rows * Wt);
        }
    }

    params.kernel_run_params = {std::move(reader_params), std::move(writer_params), std::move(compute_params)};
    return params;
}

}  // namespace

ReduceMultiCoreWProgramFactory::cached_program_t ReduceMultiCoreWProgramFactory::create(
    const ReduceParams& operation_attributes,
    const tt::tt_metal::Tensor& tensor_args,
    tt::tt_metal::Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;

    // Sharded inputs aren't supported by the Metal 2.0 reader/writer kernels yet —
    // the kernels rely on InterleavedAddrGenFast because Metal 2.0 ProgramSpec doesn't
    // support the positional CTAs that TensorAccessorArgs<N>() reads from.
    TT_FATAL(
        !a.memory_config().is_sharded(),
        "ReduceMultiCoreWProgramFactory (Metal 2.0): only interleaved input buffers are supported (got memory_config "
        "= {})",
        a.memory_config());
    TT_FATAL(
        !output.memory_config().is_sharded(),
        "ReduceMultiCoreWProgramFactory (Metal 2.0): only interleaved output buffers are supported (got memory_config "
        "= {})",
        output.memory_config());

    const auto& shape = a.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t Wt = W / tile_width;
    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);
    (void)math_approx_mode;
    (void)packer_l1_acc;
    (void)dst_full_sync_en;

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const uint32_t src0_single_tile_size = tile_size(src0_cb_data_format);
    const tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_single_tile_size = tile_size(scaler_cb_data_format);
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tile_size(dst_cb_data_format);

    // Work distribution across cores.
    const WWorkDistribution wd = ComputeWWorkDistribution(operation_attributes, a);

    // For MAX/MIN with non-unity scalar, GMPOOL only respects the scaler's exponent, so the
    // high-level reduce() in reduce_op.cpp has already swapped scaler→1.0 and stashed the user
    // scaler in post_mul_scaler. Here we just forward both values to the kernels; the compute
    // kernel applies post_mul via SFPU mul_unary_tile when REDUCE_POST_MUL is defined.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // ----------------------------------------------------------
    // ProgramSpec construction
    // ----------------------------------------------------------
    const Buffer* src_buffer = a.buffer();
    const Buffer* dst_buffer = output.buffer();
    const uint32_t src_is_dram = src_buffer->is_dram() ? 1u : 0u;
    const uint32_t dst_is_dram = dst_buffer->is_dram() ? 1u : 0u;
    const uint32_t src_aligned_page_size = src_buffer->aligned_page_size();
    const uint32_t dst_aligned_page_size = dst_buffer->aligned_page_size();

    // DFBs
    // NOTE: With Quasar's implicit-sync DFB scheduling, the runtime allocates 2 transaction
    // IDs per side and asserts `capacity % num_txn_ids == 0`. Therefore every DFB capacity
    // must be a multiple of 2, even when only a single tile is logically needed (scaler,
    // accumulator, intermediate-negation). On Gen1 these can stay at 1 (CB capacity), but
    // we standardize on 2 here to keep one factory work for both arches.
    constexpr uint32_t kNumInputEntries = 2;
    constexpr uint32_t kNumScalerEntries = 2;
    constexpr uint32_t kNumOutputEntries = 2;
    constexpr uint32_t kNumScratchEntries = 2;

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(
        MakeDFB(INPUT_DFB, src0_single_tile_size, kNumInputEntries, src0_cb_data_format, a.tensor_spec().tile()));
    dataflow_buffers.push_back(
        MakeDFB(SCALER_DFB, scaler_single_tile_size, kNumScalerEntries, scaler_cb_data_format, a.tensor_spec().tile()));
    dataflow_buffers.push_back(
        MakeDFB(OUTPUT_DFB, dst_single_tile_size, kNumOutputEntries, dst_cb_data_format, output.tensor_spec().tile()));
    if (operation_attributes.negate) {
        dataflow_buffers.push_back(MakeDFB(
            ACC_DFB, dst_single_tile_size, kNumScratchEntries, dst_cb_data_format, output.tensor_spec().tile()));
        dataflow_buffers.push_back(MakeDFB(
            INEG_DFB, dst_single_tile_size, kNumScratchEntries, dst_cb_data_format, output.tensor_spec().tile()));
    }

    // Defines shared by all kernels
    const std::map<std::string, std::string> reduce_defines_map =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    const auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // -- Reader kernel --
    m2::KernelSpec reader;
    reader.unique_id = W_READER_KERNEL;
    reader.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_universal_start_id.cpp"};
    reader.num_threads = 1;
    reader.compile_time_arg_bindings = {
        {"scaler_bits", scaler_bits},
        {"is_dram", src_is_dram},
        {"aligned_page_size", src_aligned_page_size},
    };
    reader.runtime_arguments_schema.named_runtime_args = {"src_addr", "num_tiles", "start_id"};
    reader.compiler_options.defines = reduce_defines;
    // We provide both Gen1 and Gen2 DM configs so the same KernelSpec can target either
    // arch. The kernel-side helper library (`dataflow_kernel_lib::prepare_reduce_scaler`,
    // `experimental::DataflowBuffer`) is arch-agnostic; what isn't yet portable is the
    // address generator (`InterleavedAddrGenFast` + `noc_async_read_tile`) — those are
    // Gen1-only. See the kernel header and METAL2_MIGRATION_NOTES.md (#1, #13).
    reader.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
            },
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{},
    };
    BindDFB(reader, INPUT_DFB, "input", m2::KernelSpec::DFBEndpointType::PRODUCER);
    BindDFB(reader, SCALER_DFB, "scaler", m2::KernelSpec::DFBEndpointType::PRODUCER);

    // -- Writer kernel --
    m2::KernelSpec writer;
    writer.unique_id = W_WRITER_KERNEL;
    writer.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_unary_interleaved.cpp"};
    writer.num_threads = 1;
    writer.compile_time_arg_bindings = {
        {"is_dram", dst_is_dram},
        {"aligned_page_size", dst_aligned_page_size},
    };
    writer.runtime_arguments_schema.named_runtime_args = {"dst_addr", "num_pages", "start_id"};
    writer.compiler_options.defines = reduce_defines;
    writer.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            },
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{},
    };
    BindDFB(writer, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::CONSUMER);

    // -- Compute kernel --
    const std::string compute_kernel_path =
        operation_attributes.negate
            ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp"
            : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp";

    m2::KernelSpec compute;
    compute.unique_id = W_COMPUTE_KERNEL;
    compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
    compute.num_threads = 1;
    compute.compile_time_arg_bindings = {
        {"Wt", Wt},
        {"NC", 1u},
        {"post_mul_scaler_bits", post_mul_scaler_bits},
    };
    compute.runtime_arguments_schema.named_runtime_args = {"Ht"};
    auto compute_defines = reduce_defines;
    if (use_post_mul) {
        compute_defines.emplace_back("REDUCE_POST_MUL", "1");
    }
    compute.compiler_options.defines = std::move(compute_defines);
    compute.config_spec = m2::ComputeConfiguration{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };
    BindDFB(compute, INPUT_DFB, "input", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, SCALER_DFB, "scaler", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::PRODUCER);
    if (operation_attributes.negate) {
        // The acc and ineg DFBs are produced AND consumed by this same kernel. Metal 2.0
        // requires distinct local_accessor_names per binding even when both endpoints are
        // the same kernel; on Gen1 the two accessor ids resolve to the same underlying CB.
        BindDFB(compute, ACC_DFB, "acc_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, ACC_DFB, "acc_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
        BindDFB(compute, INEG_DFB, "ineg_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, INEG_DFB, "ineg_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
    }

    // -- Single work unit covering all worker cores --
    m2::WorkUnitSpec work_unit;
    work_unit.unique_id = W_WORK_UNIT;
    work_unit.kernels = {W_READER_KERNEL, W_WRITER_KERNEL, W_COMPUTE_KERNEL};
    work_unit.target_nodes = wd.all_cores;

    // -- Assemble the spec --
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::reduce_multi_core_w";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.work_units = {std::move(work_unit)};

    // ----------------------------------------------------------
    // Build the Program and apply initial run parameters
    // ----------------------------------------------------------
    Program program = m2::MakeProgramFromSpec(*a.device(), spec);

    shared_variables_t shared{
        .cores = wd.cores,
        .core_group_1 = wd.core_group_1,
        .core_group_2 = wd.core_group_2,
        .num_rows_per_core_group_1 = wd.num_rows_per_core_group_1,
        .num_rows_per_core_group_2 = wd.num_rows_per_core_group_2,
        .Wt = Wt,
    };

    auto run_params = BuildRunParams(shared, src_buffer->address(), dst_buffer->address());
    m2::SetProgramRunParameters(program, run_params);

    return cached_program_t{std::move(program), std::move(shared)};
}

void ReduceMultiCoreWProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ReduceParams& /*operation_attributes*/,
    const tt::tt_metal::Tensor& tensor_args,
    tt::tt_metal::Tensor& tensor_return_value) {
    const auto* src_buffer = tensor_args.buffer();
    const auto* dst_buffer = tensor_return_value.buffer();

    auto run_params = BuildRunParams(cached_program.shared_variables, src_buffer->address(), dst_buffer->address());
    m2::SetProgramRunParameters(cached_program.program, run_params);
}

}  // namespace ttnn::prim
