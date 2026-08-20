// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_default_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingMultiCoreDefaultFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    tt::DataFormat input_dfb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);
    tt::DataFormat output_dfb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid =
        operation_attributes.sub_core_grids.has_value() ? operation_attributes.sub_core_grids.value() : default_grid;
    uint32_t tile_width = output.tensor_spec().tile().get_width();
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    uint32_t num_blocks = output.physical_volume() / output.padded_shape()[-1] / tile_height;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / tile_width;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t unpadded_row_size_bytes = a.logical_shape()[-1] * a.element_size();    // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // ---------------------------------------------------------------------
    // Program-scope resource names (typed handles → generated dfb:: / tensor:: tokens)
    // ---------------------------------------------------------------------
    const DFBSpecName IN{"in"};    // legacy src0 buffer c_0: row-major staging for tilize
    const DFBSpecName OUT{"out"};  // legacy output buffer c_16: tilized output
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramSpec spec;
    spec.name = "tilize_with_val_padding_multi_core_default";

    // ---------------------------------------------------------------------
    // DataflowBufferSpecs (replaces the legacy c_0 / c_16 buffer descriptors)
    // ---------------------------------------------------------------------
    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN,
            .entry_size = input_single_tile_size,
            .num_entries = num_tiles_per_row,
            .data_format_metadata = input_dfb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUT,
            .entry_size = output_single_tile_size,
            .num_entries = num_tiles_per_row,
            .data_format_metadata = output_dfb_data_format,
        },
    };

    // ---------------------------------------------------------------------
    // Tensor parameters (typed bindings replace the Buffer* RTA slots and the
    // TensorAccessorArgs(...).append_to(...) CTA plumbing)
    // ---------------------------------------------------------------------
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    /** reader
     */
    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    // log2(tile_height * data_format_size_in_bytes)
    uint32_t shift_bits = static_cast<uint32_t>(std::log2(
        a.element_size() *
        tile_height));  // This gives log2 of bytes per tile row, so in the kernel we
                        // can shift right by this to get number of tiles.
                        // ex: bf16/uint16 -> log2(2 * 32) = 6, float32/int32/uint32 -> log2(4 * 32) = 7, etc.
    uint32_t elem_size = a.element_size();
    uint32_t num_pages_in_row = 1;
    uint32_t page_size = a.logical_shape()[-1] * a.element_size();
    uint32_t aligned_page_size = a.buffer()->aligned_page_size();
    uint32_t size_of_valid_data_in_last_page_in_row = a.logical_shape()[-1] * a.element_size();
    if (a.is_sharded() && a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        page_size = a.buffer()->page_size();
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_pages_in_row = tt::div_up(a.logical_shape()[-1], shard_width);
        size_of_valid_data_in_last_page_in_row = unpadded_row_size_bytes - (num_pages_in_row - 1) * page_size;
    }

    // NOTE: `aligned_page_size` is carried over from the legacy CTA list even though the reader kernel
    // does not read it (it held positional slot 5 there, ahead of the accessor args). Dropping it is a
    // cleanup for the op owner, not port work.
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
            "reader_unary_pad_dims_split_rows_multicore.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"tile_row_shift_bits", shift_bits},
             {"unpadded_X_size", unpadded_row_size_bytes},
             {"elem_size", elem_size},
             {"num_pages_in_row", num_pages_in_row},
             {"page_size", page_size},
             {"aligned_page_size", aligned_page_size},
             {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "pad_value", "start_page_id", "n_block_reps"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    /** writer
     */
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    /** compute
     */
    // Legacy ComputeConfigDescriptor set only fp32_dest_acc_en and unpack_to_dest_mode; every other
    // field stayed at its default, which ComputeGen1Config reproduces exactly. The legacy
    // unpack_to_dest_mode vector was Default everywhere except v[c_0] = UnpackToDestFp32 when
    // fp32_llk_acc, i.e. UnpackToDest on the tilize input DFB (Default == UnpackToSrc is expressed by
    // omitting the entry). Both compute KernelSpecs get the same config, as in legacy.
    ComputeGen1Config compute_gen1{.enable_32_bit_dest = fp32_llk_acc};
    if (fp32_llk_acc) {
        compute_gen1.unpack_modes = ComputeUnpackModes{{IN, UnpackMode::UnpackToDest}};
    }
    ComputeHardwareConfig compute_hw{std::move(compute_gen1)};

    // One KernelSpec per legacy compute KernelDescriptor: the per-group block count stays a CTA, so
    // the two groups keep their distinct specialization instead of collapsing onto a runtime arg.
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t per_core_block_cnt) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = "ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp",
            // O3 explicitly: the legacy ComputeConfigDescriptor set no opt_level and so resolved to
            // O3, but Metal 2.0's type-agnostic CompilerOptions defaults to O2. Leaving it unset
            // would drop a level on both compute specs' compile and link.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = IN,
                     .accessor_name = "in",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = OUT,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config = compute_hw,
        };
    };

    /* RUNTIME ARGS */
    // 1D distribution of blocks across cores
    auto core_assignments = ttnn::distribute_work(
        output.logical_shape(),
        output.padded_shape(),
        ncores,
        nblocks_per_core,
        has_cliff,
        nblocks_per_core_cliff,
        tile_height);

    uint32_t tile_start_id = 0;
    uint32_t start_page_id = 0;

    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};

    const auto cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // The block-representation stream is a genuine variable-count, loop-indexed collection (the
        // kernel walks it with a running index bounded by the runtime n_block_reps), so it stays
        // positional: it rides the kernel's runtime varargs rather than being named. Its length
        // differs per core, hence the per-node vararg-count schema below.
        AdvancedKernelRunArgs::Varargs block_reps;
        block_reps.reserve(assignment.size() * 5);

        const uint32_t reader_start_page_id = start_page_id;
        const uint32_t n_block_reps = static_cast<uint32_t>(assignment.size());

        uint32_t nblocks_per_core_local = 0;
        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core_local += el.block_count();
            start_page_id += el.data_row_count() * num_pages_in_row;
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                // push back information for previous elements
                block_reps.push_back(ref_el.n_data);
                block_reps.push_back(ref_el.n_mixed);
                block_reps.push_back(ref_el.n_pads);
                block_reps.push_back(ref_el.times);
                block_reps.push_back(count_repeated);
                // set up assignment for this element
                ref_el = el;
                count_repeated = 1;
            }
        }
        block_reps.push_back(ref_el.n_data);
        block_reps.push_back(ref_el.n_mixed);
        block_reps.push_back(ref_el.n_pads);
        block_reps.push_back(ref_el.times);
        block_reps.push_back(count_repeated);

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_local;

        // reader runtime args — the buffer address now rides the TensorBinding.
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"padded_X_size", padded_row_size_bytes},
             {"pad_value", packed_pad_value},
             {"start_page_id", reader_start_page_id},
             {"n_block_reps", n_block_reps}});
        reader.advanced_options.num_runtime_varargs_per_node[core] = static_cast<uint32_t>(block_reps.size());
        reader_ra.advanced_options.runtime_varargs[core] = std::move(block_reps);

        // writer runtime args
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", tile_start_id}});

        tile_start_id += num_tiles_per_core;
    }

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    // WorkUnitSpecs may not overlap in their target nodes, so the reader and writer — which legacy
    // placed on all_cores — ride along in each compute group's work unit instead of getting one of
    // their own. The union of the two groups is exactly all_cores (see split_blocks_for_tilize), so
    // their placement is unchanged.
    if (!core_range.empty()) {
        spec.kernels.push_back(make_compute(COMPUTE, nblocks_per_core));
        spec.work_units.push_back(
            WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_range});
    }
    if (has_cliff) {
        spec.kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    run_args.kernel_run_args.push_back(std::move(reader_ra));
    run_args.kernel_run_args.push_back(std::move(writer_ra));

    run_args.tensor_args.emplace(INPUT, TensorArgument{a.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
