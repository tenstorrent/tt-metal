// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Spec resource names (prefixed to stay distinct under unity builds — see
// port_patterns.md "Unity-build hygiene for anonymous-namespace symbols").
const DFBSpecName FRNC_IN0{"frnc_in0"};  // legacy c_0 (input)
const DFBSpecName FRNC_IN1{"frnc_in1"};  // legacy c_1 (zero tile)
const DFBSpecName FRNC_OUT{"frnc_out"};  // legacy c_16 (output)
const TensorParamName FRNC_INPUT{"frnc_input"};
const TensorParamName FRNC_OUTPUT{"frnc_output"};
const KernelSpecName FRNC_READER{"frnc_reader"};
const KernelSpecName FRNC_WRITER{"frnc_writer"};
const KernelSpecName FRNC_COMPUTE_G1{"frnc_compute_g1"};
const KernelSpecName FRNC_COMPUTE_G2{"frnc_compute_g2"};

bool is_tensor_divisible_by_shard(const ttnn::Shape& tensor_shape, const ttnn::Shape& shard_shape) {
    // Only compare common (end) dimensions. Any extra front dimensions would be
    // divisible by the implied 1 in the non-existent dimensions in shard_shape.
    // Use negative dimensions to compare the end of both shapes.
    for (int i = 1; i <= shard_shape.size(); i++) {
        if (shard_shape[-i] == 0 || tensor_shape[-i] % shard_shape[-i] != 0) {
            return false;
        }
    }
    return true;
}
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_and_scale_spatial_dims(
    const ttnn::Shape& shape, uint32_t dim) {
    const auto rank = shape.rank();

    TT_FATAL(rank >= 2, "Shape must have at least two dims.");
    uint32_t Wt = shape[-1] / TILE_WIDTH;
    uint32_t Ht = shape[-2] / TILE_HEIGHT;

    uint32_t reduce_dim = shape[dim];
    uint32_t inner_dims_product = 1;
    for (auto i = dim + 1; i < rank - 2; ++i) {
        inner_dims_product *= shape[i];
    }

    uint32_t inner_tile_size = inner_dims_product * Ht * Wt;
    uint32_t reduce_tile_size = reduce_dim * inner_tile_size;

    return {Wt, Ht, inner_tile_size, reduce_tile_size};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts FastReduceNCProgramFactory::create_program_artifacts(
    const FastReduceNCParams& operation_attributes,
    const FastReduceNCInputs& tensor_args,
    Tensor& tensor_return_value) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tensor_args.input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // Input and output CBs may differ when the Sum precision chain requests FP32 packing.
    const auto input_data_format = datatype_to_dataformat_converter(tensor_args.input.dtype());
    const auto input_tile_size = tt::tile_size(input_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);
    const auto cb_1_data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const auto cb_1_tile_size = tt::tile_size(cb_1_data_format);

    const auto& input_shape = tensor_args.input.padded_shape();
    const auto [Wt, Ht, inner_tile_size, reduce_tile_size] =
        extract_and_scale_spatial_dims(input_shape, static_cast<uint32_t>(operation_attributes.dim));
    const auto num_reduce_input_tile = input_shape[operation_attributes.dim];
    const auto num_output_tiles = tensor_return_value.physical_volume() / TILE_HW;
    const bool fp32_dest_acc_en =
        std::get<2>(ttnn::get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config));
    // Choose granularity as the largest factor of num_reduce_input_tile that is less than or equal to 8.
    // Helps with locality and increases work unit for better performance.
    uint32_t input_granularity;
    for (input_granularity = 8; input_granularity > 1; --input_granularity) {
        if (num_reduce_input_tile % input_granularity == 0) {
            break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    const auto num_cores_y = grid.y;

    const uint32_t in0_t = input_granularity * 2;  // input
    const uint32_t in1_t = 1;                      // zero
    const uint32_t out0_t = 2;                     // output
    uint32_t shard_factor = 1;

    // When dim=0, nd sharded, tile sizes are the same, the shards are compatible
    // with the kernel accesses, tensor shape is divisible by shard shape, and
    // number of shards is larger than core count, divide the work by shards.
    uint32_t output_shard_size = 1;
    auto input_tile = tensor_args.input.tensor_spec().tile().get_tile_shape();
    auto output_tile = tensor_return_value.tensor_spec().tile().get_tile_shape();
    bool nd_sharded = tensor_args.input.nd_shard_spec().has_value() && tensor_return_value.nd_shard_spec().has_value();
    bool same_tiles = input_tile[0] == output_tile[0] && input_tile[1] == output_tile[1];
    bool divide_by_shards = false;
    const auto& dspec = *tensor_return_value.buffer()->buffer_distribution_spec();
    if (nd_sharded && same_tiles && operation_attributes.dim == 0) {
        const NdShardSpec& input_nd_shard_spec = tensor_args.input.nd_shard_spec().value();
        const NdShardSpec& output_nd_shard_spec = tensor_return_value.nd_shard_spec().value();
        const Shape& input_shard_shape = input_nd_shard_spec.shard_shape;
        bool compatible_shards =
            input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR &&
            input_nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D &&
            output_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR &&
            output_nd_shard_spec.shard_distribution_strategy == ShardDistributionStrategy::ROUND_ROBIN_1D;
        if (compatible_shards && is_tensor_divisible_by_shard(input_shape, input_shard_shape)) {
            uint32_t num_output_shards = dspec.num_shards();
            output_shard_size = dspec.shard_shape_in_pages().volume();
            bool more_shards_than_cores = num_output_shards > (num_cores_x * num_cores_y);
            if (more_shards_than_cores) {
                divide_by_shards = true;
                shard_factor = output_shard_size;
            }
        }
    }
    bool use_sub_core_grids = operation_attributes.sub_core_grids.has_value() && !divide_by_shards;
    auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] =
            divide_by_shards
                ? dspec.core_groups_tuple()
                : (use_sub_core_grids
                       ? tt::tt_metal::split_work_to_cores(
                             *operation_attributes.sub_core_grids, num_output_tiles, /*row_wise=*/true)
                       : tt::tt_metal::split_work_to_cores(grid, num_output_tiles, /*row_wise=*/true));
    num_cols_per_core_group_1 *= shard_factor;
    num_cols_per_core_group_2 *= shard_factor;

    ////////////////////////////////////////////////////////////////////////////
    //                            ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec;
    spec.name = "fast_reduce_nc";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = FRNC_INPUT, .spec = tensor_args.input.tensor_spec()},
        TensorParameter{.unique_id = FRNC_OUTPUT, .spec = tensor_return_value.tensor_spec()},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // Legacy c_0 (input). Double-buffered by input_granularity.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = FRNC_IN0,
        .entry_size = input_tile_size,
        .num_entries = in0_t,
        .data_format_metadata = input_data_format,
    });
    // Legacy c_1 (zero tile). Single BF16 tile broadcast into the reduction.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = FRNC_IN1,
        .entry_size = cb_1_tile_size,
        .num_entries = in1_t,
        .data_format_metadata = cb_1_data_format,
    });
    // Legacy c_16 (output). Double-buffered.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = FRNC_OUT,
        .entry_size = output_tile_size,
        .num_entries = out0_t,
        .data_format_metadata = output_data_format,
    });
    // Legacy c_24 ("accumulated sum" intermediate) is dropped: the compute kernel
    // accumulates in DST registers and no kernel touched index 24 (dead CB). A
    // bindingless DFB cannot be expressed in Metal 2.0; removal is L1-only, zero
    // behavior change.

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reader_reduce_nc.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/writer_reduce_nc.cpp";

    KernelSpec reader{
        .unique_id = FRNC_READER,
        .source = reader_kernel_file,
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = FRNC_IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = FRNC_IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = FRNC_INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"input_granularity", input_granularity},
             {"shard_factor", shard_factor},
             {"num_cores_to_be_used", num_cores_to_be_used}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_tiles", "id_range_length", "start_id", "dim", "reduce_tile_size", "inner_tile_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = FRNC_WRITER,
        .source = writer_kernel_file,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = FRNC_OUT, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = FRNC_OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = {{"shard_factor", shard_factor}, {"num_cores_to_be_used", num_cores_to_be_used}},
        .runtime_arg_schema = {.runtime_arg_names = {"id_range_length", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp";

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    // Style A: op resolves a TTNN ComputeKernelConfig, so the arch-agnostic helper
    // carries the four knobs (math_fidelity / math_approx_mode / fp32_dest_acc_en /
    // dst_full_sync_en) to their Metal 2.0 equivalents.
    ComputeHardwareConfig compute_hw =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    // Metal 2.0 requires an explicit unpack_modes entry when a compute kernel consumes a
    // Float32 DFB with enable_32_bit_dest = true. Legacy set no unpack_to_dest_mode (Default),
    // which maps to UnpackToSrc. Only the input DFB (c_0) can be Float32; the zero DFB is BF16.
    if (fp32_dest_acc_en && input_data_format == tt::DataFormat::Float32) {
        std::get<ComputeGen1Config>(compute_hw).unpack_modes.emplace(FRNC_IN0, UnpackMode::UnpackToSrc);
    }

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t num_cols_per_core_group) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel_file,
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = FRNC_IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = FRNC_IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = FRNC_OUT, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"num_output_tiles", num_cols_per_core_group},
                 {"num_input_tiles", num_reduce_input_tile},
                 {"input_granularity", input_granularity}},
            .hw_config = compute_hw,
        };
    };

    const bool group_2_present = !core_group_2.ranges().empty();

    spec.kernels = {reader, writer, make_compute(FRNC_COMPUTE_G1, num_cols_per_core_group_1)};
    if (group_2_present) {
        spec.kernels.push_back(make_compute(FRNC_COMPUTE_G2, num_cols_per_core_group_2));
    }

    // Two compute KernelSpecs of the same source over disjoint node sets → one WorkUnitSpec
    // each (reader + writer + that group's compute). Reader / writer belong to both work units;
    // their effective node set is the union (all_cores). Each DFB sees one producer + one
    // consumer per node — a legal 1:1 binding, not the multi-binding flag.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "wu_g1", .kernels = {FRNC_READER, FRNC_WRITER, FRNC_COMPUTE_G1}, .target_nodes = core_group_1});
    if (group_2_present) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "wu_g2", .kernels = {FRNC_READER, FRNC_WRITER, FRNC_COMPUTE_G2}, .target_nodes = core_group_2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Each core is assigned an output work unit in a row wise round robin
    // fashion. For a given core, the first index is i, and all subsequent
    // indices are increments of num_cores_to_be_used. The total number of
    // units is num_tiles_per_group times num_cores_to_be_used.
    // For example, with 130 output tiles to be processed and no shards (shard
    // factor is 1) on an 8x8 grid
    // - the increment is 64
    // - the first 2 cores will have num_tiles_per_core 3 and the rest 2
    // - core x=0,y=0 will process output tiles 0, 64, and 128
    // - core x=1,y=0 will process output tiles 1, 65, and 129
    // - core x=2,y=0 will process output tiles 2 and 66
    // - core x=3,y=0 will process output tiles 3 and 67
    // - etc
    // The first tile that needs to be reduced has the same as the output tile.
    // That is the starting point for the reader, which then processes all
    // subsequent tiles to be reduced. The increment for the input indices is
    // the size of the inner dimensions in tiles (inner_tile_size). The number
    // of tiles to process is the size of the reduce dimension in tiles
    // (reduce_tile_size).
    // The shard factor is used to iterate over shards instead of tiles.
    // It is taken into account in the num_cols_per_core_group variables and
    // the tile_offset is incremented by it for the reader to adjust it's
    // reading pattern.
    std::vector<CoreCoord> ordered_cores;
    ordered_cores.reserve(use_sub_core_grids ? all_cores.num_cores() : num_cores_to_be_used);
    if (use_sub_core_grids) {
        for (const auto& range : all_cores.ranges()) {
            for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    ordered_cores.emplace_back(x, y);
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            ordered_cores.emplace_back(i % num_cores_x, i / num_cores_x);
        }
    }

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = FRNC_READER};
    KernelRunArgs writer_run{.kernel = FRNC_WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = ordered_cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_input_tiles", num_reduce_input_tile},
             {"id_range_length", num_tiles_per_core * num_cores_to_be_used},
             {"start_id", tile_offset},
             {"dim", static_cast<uint32_t>(operation_attributes.dim)},
             {"reduce_tile_size", reduce_tile_size},
             {"inner_tile_size", inner_tile_size}});

        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"id_range_length", num_tiles_per_core * num_cores_to_be_used}, {"start_id", tile_offset}});

        tile_offset += shard_factor;
    }

    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args.emplace(FRNC_INPUT, TensorArgument{tensor_args.input.mesh_tensor()});
    run_args.tensor_args.emplace(FRNC_OUTPUT, TensorArgument{tensor_return_value.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
