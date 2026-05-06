// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core H reduction program factory, migrated to the Metal 2.0 host API.
//
// The H factory splits work along W columns: each core processes a contiguous slice
// of (NC * Wt) columns. Per-core Wt varies between two groups produced by
// split_work_to_cores, so the compute kernel binds Wt as a per-node runtime arg —
// a single KernelSpec covers both groups (mirroring the W factory's treatment of Ht).
//
// Sharded inputs are not supported on this path; sharded H reductions still go
// through the Gen1 pipeline upstream. (The Gen1 factory's WIDTH_SHARDED branch
// used a sharded-specific reader and the data_movement writer_unary_sharded.cpp,
// neither of which is wired through Metal 2.0 yet.)

#include "reduce_op_multi_core_h_program_factory.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
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

// KernelSpec / WorkUnitSpec ids — H_-prefixed to avoid Unity-build collisions
// with the W and HW factories. DFB ids and helpers come from the shared header.
constexpr const char* H_READER_KERNEL = "reduce_h_reader";
constexpr const char* H_WRITER_KERNEL = "reduce_h_writer";
constexpr const char* H_COMPUTE_KERNEL = "reduce_h_compute";
constexpr const char* H_WORK_UNIT = "all_workers";

struct HWorkDistribution {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t num_cols_per_core_group_1 = 0;
    uint32_t num_cols_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

HWorkDistribution ComputeHWorkDistribution(const ReduceParams& attrs, const tt::tt_metal::Tensor& input) {
    using namespace tt::tt_metal;

    const auto& shape = input.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t NC = shape[1] * shape[0];

    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t Wt = W / tile_width;

    auto* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cols = NC * Wt;

    HWorkDistribution wd;
    if (attrs.sub_core_grids.has_value()) {
        std::tie(
            wd.num_cores,
            wd.all_cores,
            wd.core_group_1,
            wd.core_group_2,
            wd.num_cols_per_core_group_1,
            wd.num_cols_per_core_group_2) = split_work_to_cores(*attrs.sub_core_grids, num_cols);
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
            wd.num_cols_per_core_group_1,
            wd.num_cols_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size, num_cols);
        wd.cores =
            grid_to_cores(wd.num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    return wd;
}

m2::ProgramRunParams BuildRunParams(
    const ReduceMultiCoreHSharedVariables& shared, uint32_t src_addr, uint32_t dst_addr) {
    m2::ProgramRunParams params;

    m2::ProgramRunParams::KernelRunParams reader_params;
    reader_params.kernel_spec_name = H_READER_KERNEL;

    m2::ProgramRunParams::KernelRunParams writer_params;
    writer_params.kernel_spec_name = H_WRITER_KERNEL;

    m2::ProgramRunParams::KernelRunParams compute_params;
    compute_params.kernel_spec_name = H_COMPUTE_KERNEL;

    const uint32_t Wt = shared.Wt;
    const uint32_t HtWt = shared.HtWt;
    TT_FATAL(Wt != 0, "Wt cached in shared_variables must be non-zero");

    uint32_t num_cols_read = 0;
    for (const auto& core : shared.cores) {
        uint32_t num_cols_per_core = 0;
        if (shared.core_group_1.contains(core)) {
            num_cols_per_core = shared.num_cols_per_core_group_1;
        } else if (shared.core_group_2.contains(core)) {
            num_cols_per_core = shared.num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // The H factory reader expects column-partitioned tile addressing:
        //   col_start_tile_id = (num_cols_read / Wt) * HtWt + (num_cols_read % Wt)
        //   curr_col_in_batch = num_cols_read % Wt
        // This lets the reader walk H rows for each (NC slice, column) pair while
        // the compute kernel sees tiles as (per-core Wt columns × Ht rows × NC=1).
        const uint32_t col_start_tile_id = (num_cols_read / Wt) * HtWt + (num_cols_read % Wt);
        const uint32_t curr_col_in_batch = num_cols_read % Wt;

        reader_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args =
                {
                    {"src_addr", src_addr},
                    {"col_start_tile_id", col_start_tile_id},
                    {"curr_col_in_batch", curr_col_in_batch},
                    {"num_cols", num_cols_per_core},
                },
        });
        writer_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args =
                {
                    {"dst_addr", dst_addr},
                    {"num_pages", num_cols_per_core},
                    {"start_id", num_cols_read},
                },
        });
        compute_params.named_runtime_args.push_back(m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{
            .node = core,
            .args = {{"Wt", num_cols_per_core}},
        });

        num_cols_read += num_cols_per_core;
    }

    params.kernel_run_params = {std::move(reader_params), std::move(writer_params), std::move(compute_params)};
    return params;
}

}  // namespace

ReduceMultiCoreHProgramFactory::cached_program_t ReduceMultiCoreHProgramFactory::create(
    const ReduceParams& operation_attributes,
    const tt::tt_metal::Tensor& tensor_args,
    tt::tt_metal::Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;

    TT_FATAL(
        !a.memory_config().is_sharded(),
        "ReduceMultiCoreHProgramFactory (Metal 2.0): only interleaved input buffers are supported (got memory_config "
        "= {})",
        a.memory_config());
    TT_FATAL(
        !output.memory_config().is_sharded(),
        "ReduceMultiCoreHProgramFactory (Metal 2.0): only interleaved output buffers are supported (got memory_config "
        "= {})",
        output.memory_config());

    const auto& shape = a.padded_shape();
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;
    const uint32_t HtWt = Ht * Wt;
    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
    TT_FATAL(Ht != 0, "Height in tiles (Ht) must be non-zero (H={}, tile_height={})", H, tile_height);

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

    const HWorkDistribution wd = ComputeHWorkDistribution(operation_attributes, a);
    TT_FATAL(wd.num_cores > 0, "Reduce H requires at least one worker core");

    // chunk_size mirrors the kernel's row_chunk = DEST_AUTO_LIMIT.
    const uint32_t chunk_size = ttnn::get_dest_reg_count(operation_attributes.compute_kernel_config);

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    const Buffer* src_buffer = a.buffer();
    const Buffer* dst_buffer = output.buffer();
    const uint32_t src_is_dram = src_buffer->is_dram() ? 1u : 0u;
    const uint32_t dst_is_dram = dst_buffer->is_dram() ? 1u : 0u;
    const uint32_t src_aligned_page_size = src_buffer->aligned_page_size();
    const uint32_t dst_aligned_page_size = dst_buffer->aligned_page_size();

    // ---- DFBs ----
    // Gen1 used 2-tile CBs (chunk_size for negate to absorb the kernel's batched
    // pushes). Quasar's implicit-sync DFB scheduling requires capacity to be a
    // multiple of 2; the negate scratch sizing rounds up to satisfy that.
    const uint32_t num_input_entries = operation_attributes.negate ? std::max<uint32_t>(chunk_size, 2u) : 2u;
    const uint32_t num_output_entries = operation_attributes.negate ? std::max<uint32_t>(chunk_size, 2u) : 2u;
    constexpr uint32_t kNumScalerEntries = 2;

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(
        MakeDFB(INPUT_DFB, src0_single_tile_size, num_input_entries, src0_cb_data_format, a.tensor_spec().tile()));
    dataflow_buffers.push_back(
        MakeDFB(SCALER_DFB, scaler_single_tile_size, kNumScalerEntries, scaler_cb_data_format, a.tensor_spec().tile()));
    dataflow_buffers.push_back(
        MakeDFB(OUTPUT_DFB, dst_single_tile_size, num_output_entries, dst_cb_data_format, output.tensor_spec().tile()));
    if (operation_attributes.negate) {
        // The reduce_h_neg kernel pushes `ntiles` per inner-loop iteration via
        // push_back(ntiles). The DFB / CB FIFO write pointer only wraps when
        // wr_ptr reaches fifo_limit exactly, so it isn't enough for capacity to
        // be a multiple of each individual push size — the cumulative offset
        // across the inner Ht loop must also wrap to 0 at the end of each nc
        // iteration. Per nc the kernel advances wr_ptr by Ht * Wt_per_core, so
        // sizing the buffer at Ht * Wt_per_core makes the trajectory land on
        // fifo_limit exactly. For two core groups, the single-buffer sizing
        // uses Ht * lcm(Wt_g1, Wt_g2) so the same allocation works for both.
        const uint32_t compute_Wt_g1 = wd.num_cols_per_core_group_1;
        const uint32_t compute_Wt_g2 = wd.num_cols_per_core_group_2;
        uint32_t per_nc_advance = 0;
        if (compute_Wt_g2 == 0) {
            per_nc_advance = compute_Wt_g1;
        } else if (compute_Wt_g1 == 0) {
            per_nc_advance = compute_Wt_g2;
        } else {
            per_nc_advance = std::lcm(compute_Wt_g1, compute_Wt_g2);
        }
        TT_FATAL(
            per_nc_advance > 0,
            "Negate H reduce: per-core Wt resolved to 0 (g1={}, g2={})",
            compute_Wt_g1,
            compute_Wt_g2);

        // Compute in uint64_t to avoid uint32_t overflow before the L1 fit check.
        const uint64_t negate_cb_tiles = static_cast<uint64_t>(Ht) * per_nc_advance;

        // L1 fit check: acc and ineg DFBs are each sized at negate_cb_tiles.
        // If the combined allocation would not fit in the available L1 budget,
        // the caller is expected to fall back to external negation — see
        // ttnn::prim::h_reduce_negate_fits_in_l1, which mirrors this calculation.
        const uint64_t per_cb_bytes = negate_cb_tiles * dst_single_tile_size;
        const uint64_t negate_cb_bytes = 2ull * per_cb_bytes;
        auto* device = a.device();
        const auto lowest_address = device->lowest_occupied_compute_l1_address();
        uint64_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
        const uint64_t base_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        TT_FATAL(
            max_l1_space > base_addr,
            "Negate H reduce: L1 base allocator address {} >= lowest occupied address {}; no room for CBs",
            base_addr,
            max_l1_space);
        max_l1_space -= base_addr;
        TT_FATAL(
            negate_cb_bytes <= max_l1_space,
            "Negate H reduce: acc + ineg ({} B for {} tiles) would not fit in available L1 ({} B). "
            "Caller must use h_reduce_negate_fits_in_l1 to choose the external-negate fallback.",
            negate_cb_bytes,
            negate_cb_tiles,
            max_l1_space);
        // num_entries is uint32_t; the L1 fit check above already bounds
        // negate_cb_tiles by the per-core L1 budget (well under 4 Gi tiles),
        // but assert the narrowing explicitly so any future budget change
        // surfaces here instead of silently truncating.
        TT_FATAL(
            negate_cb_tiles <= std::numeric_limits<uint32_t>::max(),
            "Negate H reduce: negate_cb_tiles {} exceeds uint32_t range",
            negate_cb_tiles);
        const uint32_t negate_entries = static_cast<uint32_t>(negate_cb_tiles);

        dataflow_buffers.push_back(MakeDFB(
            ACC_DFB, dst_single_tile_size, negate_entries, dst_cb_data_format, output.tensor_spec().tile()));
        dataflow_buffers.push_back(MakeDFB(
            INEG_DFB, dst_single_tile_size, negate_entries, dst_cb_data_format, output.tensor_spec().tile()));
    }

    std::map<std::string, std::string> reduce_defines_map =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::H);
    // The reader's row_chunk sizing depends on FP32_DEST_ACC / DST_SYNC_FULL config —
    // the kernel reads these via DEST_AUTO_LIMIT. Pass them through as defines so the
    // reader's compile-time row_chunk matches what the negate compute kernel uses.
    reduce_defines_map["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    reduce_defines_map["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
    const auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // ---- Reader ----
    m2::KernelSpec reader;
    reader.unique_id = H_READER_KERNEL;
    reader.source = m2::KernelSpec::SourceFilePath{
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp"};
    reader.num_threads = 1;
    reader.compile_time_arg_bindings = {
        {"Ht", Ht},
        {"Wt", Wt},
        {"HtWt", HtWt},
        {"scaler_bits", scaler_bits},
        {"use_welford", 0u},
        {"is_dram", src_is_dram},
        {"aligned_page_size", src_aligned_page_size},
    };
    reader.runtime_arguments_schema.named_runtime_args = {
        "src_addr", "col_start_tile_id", "curr_col_in_batch", "num_cols"};
    reader.compiler_options.defines = reduce_defines;
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

    // ---- Writer ----
    m2::KernelSpec writer;
    writer.unique_id = H_WRITER_KERNEL;
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

    // ---- Compute ----
    // H factory passes compute_NC = 1 because the reader's column-partitioned
    // addressing already iterates NC slices (a per-core column run can span them).
    // From the kernel's view, it sees Wt_per_core columns × Ht rows × NC=1.
    constexpr uint32_t compute_NC = 1u;
    const std::string compute_kernel_path =
        operation_attributes.negate
            ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h_neg.cpp"
            : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp";

    m2::KernelSpec compute;
    compute.unique_id = H_COMPUTE_KERNEL;
    compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
    compute.num_threads = 1;
    compute.compile_time_arg_bindings = {
        {"Ht", Ht},
        {"NC", compute_NC},
        {"post_mul_scaler_bits", post_mul_scaler_bits},
    };
    compute.runtime_arguments_schema.named_runtime_args = {"Wt"};
    auto compute_defines = reduce_defines;
    if (use_post_mul) {
        compute_defines.emplace_back("REDUCE_POST_MUL", "1");
    }
    compute.compiler_options.defines = std::move(compute_defines);
    compute.config_spec = m2::ComputeConfiguration{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
    };
    BindDFB(compute, INPUT_DFB, "input", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, SCALER_DFB, "scaler", m2::KernelSpec::DFBEndpointType::CONSUMER);
    BindDFB(compute, OUTPUT_DFB, "output", m2::KernelSpec::DFBEndpointType::PRODUCER);
    if (operation_attributes.negate) {
        BindDFB(compute, ACC_DFB, "acc_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, ACC_DFB, "acc_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
        BindDFB(compute, INEG_DFB, "ineg_w", m2::KernelSpec::DFBEndpointType::PRODUCER);
        BindDFB(compute, INEG_DFB, "ineg_r", m2::KernelSpec::DFBEndpointType::CONSUMER);
    }

    // ---- Single work unit ----
    m2::WorkUnitSpec work_unit;
    work_unit.unique_id = H_WORK_UNIT;
    work_unit.kernels = {H_READER_KERNEL, H_WRITER_KERNEL, H_COMPUTE_KERNEL};
    work_unit.target_nodes = wd.all_cores;

    // ---- Assemble + parameterize ----
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::reduce_multi_core_h";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.work_units = {std::move(work_unit)};

    Program program = m2::MakeProgramFromSpec(*a.device(), spec);

    shared_variables_t shared{
        .cores = wd.cores,
        .core_group_1 = wd.core_group_1,
        .core_group_2 = wd.core_group_2,
        .num_cols_per_core_group_1 = wd.num_cols_per_core_group_1,
        .num_cols_per_core_group_2 = wd.num_cols_per_core_group_2,
        .Wt = Wt,
        .Ht = Ht,
        .HtWt = HtWt,
    };

    auto run_params = BuildRunParams(shared, src_buffer->address(), dst_buffer->address());
    m2::SetProgramRunParameters(program, run_params);

    return cached_program_t{std::move(program), std::move(shared)};
}

void ReduceMultiCoreHProgramFactory::override_runtime_arguments(
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
