// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope kernel paths. Names are unique across the untilize_with_unpadding device/ sibling .cpp files
// (the _UWU_MCI suffix) to avoid unity-build collisions. These are op-private Metal 2.0 copies of the
// shared kernels; the legacy kernels are still consumed positionally by the un-migrated variants.
constexpr const char* UWU_MCI_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
    "reader_unary_interleaved_start_id_m2.cpp";
constexpr const char* UWU_MCI_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
    "writer_unary_stick_layout_split_rows_multicore_m2.cpp";
constexpr const char* UWU_MCI_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/compute/untilize_m2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] == 0 ? 0 : a.physical_volume() / input_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;

    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
    }

    bool float32_dtype = input_cb_data_format == tt::DataFormat::Float32 or
                         input_cb_data_format == tt::DataFormat::UInt32 or
                         input_cb_data_format == tt::DataFormat::Int32;

    const m2::DFBSpecName src0_dfb{"cb_id_in0"};
    const m2::DFBSpecName out0_dfb{"cb_id_out0"};

    m2::ProgramSpec spec;
    spec.name = "untilize_with_unpadding_multi_core_interleaved";

    // DFBs: src0 (formerly CB c_0) produced by reader / consumed by compute; out0 (formerly CB c_16)
    // produced by compute / consumed by writer. Both L1-allocated, one full row of tiles deep.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = src0_dfb,
            .entry_size = input_single_tile_size,
            .num_entries = num_tiles_per_row,
            .data_format_metadata = input_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = out0_dfb,
            .entry_size = output_single_tile_size,
            .num_entries = num_tiles_per_row,
            .data_format_metadata = output_cb_data_format,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // Reader on NCRISC (RISCV_1 / NOC1): reads interleaved input tiles into src0.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{UWU_MCI_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(src0_dfb, "cb_id_in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_page_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    // Writer on BRISC: consumes out0, writes unpadded sticks. The (formerly positional) FLOAT32_DTYPE flag
    // and unpadded_X_size become named compile-time args; padded_X_size / start_stick_id / n_block_reps are
    // named runtime args; the variable-length tail of per-block-rep quintuples is carried as runtime varargs.
    // The legacy writer body indexes arg positions dynamically (num_runtime_varargs covers the worst case).
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{UWU_MCI_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(out0_dfb, "cb_id_out0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .compile_time_args =
            {{"float32_dtype", static_cast<uint32_t>(float32_dtype)}, {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "start_stick_id", "n_block_reps"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    auto make_compute_hw = [&]() {
        m2::ComputeHardwareConfig hw{.fp32_dest_acc_en = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            hw.unpack_to_dest_mode.emplace(src0_dfb, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
        }
        return hw;
    };
    auto make_compute = [&](const m2::KernelSpecName& name, uint32_t per_core_block_cnt) {
        return m2::KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path{UWU_MCI_COMPUTE_KERNEL_PATH},
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = {m2::ConsumerOf(src0_dfb, "src_cb_id"), m2::ProducerOf(out0_dfb, "out_cb_id")},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config = make_compute_hw(),
        };
    };

    spec.kernels = {std::move(reader), std::move(writer)};

    // Local DFBs require producer AND consumer to share a WorkUnitSpec. reader→src0→compute→out0→writer is
    // one chain; reader/writer co-locate with each core group's compute kernel (a KernelSpec may belong to
    // multiple WorkUnitSpecs, so the reader/writer specs are reused across both groups).
    const bool has_full = !core_range.ranges().empty();
    if (has_full) {
        spec.kernels.push_back(make_compute(m2::KernelSpecName{"compute_full"}, nblocks_per_core));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "untilize_with_unpadding_interleaved_full",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_full"}},
            .target_nodes = core_range});
    }
    if (has_cliff) {
        spec.kernels.push_back(make_compute(m2::KernelSpecName{"compute_cliff"}, nblocks_per_core_cliff));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "untilize_with_unpadding_interleaved_cliff",
            .kernels =
                {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_cliff"}},
            .target_nodes = core_range_cliff});
    }

    // Per-core run args. Mirror the legacy distribute_work / BlockRep state machine exactly: the writer's
    // block-rep quintuples become its runtime varargs, the reader gets num_tiles / start_page_id.
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;

    const auto& cores = corerange_to_cores(available_grid);

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    // The writer's vararg count differs per core (distinct BlockRep groups vary), so declare a fixed
    // worst-case num_runtime_varargs and zero-pad each core's tail. The kernel only walks the first
    // n_block_reps groups, so trailing padding is never read.
    std::vector<std::vector<uint32_t>> per_core_writer_varargs(ncores);
    std::vector<uint32_t> per_core_n_block_reps(ncores);
    size_t max_writer_varargs = 0;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // start_stick_id for this core is the row_start_id BEFORE this core's assignment is accumulated
        // (the legacy writer captured row_start_id at the top of its per-core arg build).
        uint32_t writer_start_stick_id = row_start_id;

        // Build the writer's block-rep quintuple varargs (n_data, n_mixed, n_pads, times, repeat_count).
        std::vector<uint32_t>& writer_varargs = per_core_writer_varargs[i];

        uint32_t nblocks_per_core_core = 0;

        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core_core += el.block_count();
            row_start_id += el.data_row_count();
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                // push back information for previous elements
                writer_varargs.push_back(ref_el.n_data);
                writer_varargs.push_back(ref_el.n_mixed);
                writer_varargs.push_back(ref_el.n_pads);
                writer_varargs.push_back(ref_el.times);
                writer_varargs.push_back(count_repeated);
                // Set up assignment for this element
                ref_el = el;
                count_repeated = 1;
            }
        }
        writer_varargs.push_back(ref_el.n_data);
        writer_varargs.push_back(ref_el.n_mixed);
        writer_varargs.push_back(ref_el.n_pads);
        writer_varargs.push_back(ref_el.times);
        writer_varargs.push_back(count_repeated);

        // n_block_reps is the total number of block iterations the kernel walks (= assignment.size());
        // the quintuples are an RLE of those iterations, advanced via the per-group repeat_count. (Matches
        // the legacy writer arg layout exactly.)
        per_core_n_block_reps[i] = static_cast<uint32_t>(assignment.size());
        max_writer_varargs = std::max(max_writer_varargs, writer_varargs.size());

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_core;

        // Reader run args
        reader_args.runtime_arg_values.push_back(
            {core, {{"num_tiles", num_tiles_per_core}, {"start_page_id", tile_start_id}}});

        // Writer named run args (varargs registered below, after the worst-case length is known).
        writer_args.runtime_arg_values.push_back(
            {core,
             {{"padded_X_size", padded_row_size_bytes},
              {"start_stick_id", writer_start_stick_id},
              {"n_block_reps", per_core_n_block_reps[i]}}});

        tile_start_id += num_tiles_per_core;
    }

    // Declare the worst-case vararg count on the writer spec and zero-pad each core's varargs to match.
    for (auto& ks : spec.kernels) {
        if (ks.unique_id == m2::KernelSpecName{"writer"}) {
            ks.advanced_options.num_runtime_varargs = static_cast<uint32_t>(max_writer_varargs);
            break;
        }
    }
    for (uint32_t i = 0; i < ncores; ++i) {
        std::vector<uint32_t>& varargs = per_core_writer_varargs[i];
        varargs.resize(max_writer_varargs, 0u);
        writer_args.advanced_options.runtime_varargs.emplace(cores[i], std::move(varargs));
    }

    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(a.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
