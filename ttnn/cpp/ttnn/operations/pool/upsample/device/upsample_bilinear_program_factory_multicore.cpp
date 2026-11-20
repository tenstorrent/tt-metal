// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [Quasar bring-up: bilinear upsample Metal-2.0 port]
//
// This factory was the one legacy ProgramDescriptor holdout in the upsample family: its three
// siblings (nearest_float, multicore_interleaved, multicore_sharded) are already Metal 2.0, and
// select_program_factory returns THIS factory for every mode=="bilinear" call. On Quasar a
// ProgramDescriptor's ReaderConfigDescriptor/WriterConfigDescriptor reaches CreateDataMovementKernel
// and aborts at kernel.hpp:451 ("DataMovementKernel is not supported on Quasar").
//
// Ported in place (not forked behind an arch gate) because the two kernels this factory binds have no
// other consumer, so their Metal-2.0 named-binding vocabulary can replace the positional one outright.
// The only architecture-conditional piece is the compute hardware config, which must be a
// ComputeGen2Config on Quasar and a ComputeGen1Config elsewhere -- ttnn::to_compute_hardware_config
// picks the right alternative from the arch, the same idiom tilize_multi_core_sharded uses.
//
// DFB-vs-Scratchpad (the one bespoke decision, per QUASAR_DM_PORT_SCOPE §4):
//   halo            carries data BETWEEN kernels? No -- it is the borrowed sharded INPUT shard, read
//                   (never written) by both DM kernels through get_read_ptr(). It is not scratch
//                   either: it is a view of a tensor, so it must be a borrowed DFB, with its
//                   producer/consumer pair present only to satisfy the Gen2 single-producer /
//                   single-consumer census. The reader takes the producer face, the writer the
//                   consumer face (exactly as upsample_program_factory_multicore_sharded does for its
//                   own borrowed input shard). No credit traffic flows on it.
//   tilize_reduce_0 filled by the reader DM kernel, drained by the compute kernel -> a true DFB.
//   tilize_reduce_1 filled by the writer DM kernel, drained by the compute kernel -> a true DFB.
//   in_scalar_0/1   filled by the reader/writer DM kernel (the four bilinear weights), drained by the
//                   compute kernel -> true DFBs. NOT scratchpads: the producing and consuming kernels
//                   differ, so a ScratchpadSpec (single-kernel scratch) would be the wrong shape and
//                   would lose the handshake the compute kernel's pop_front relies on.
//   out             borrowed from the sharded OUTPUT and produced by the compute kernel alone. It is
//                   sized to the whole output shard, so the producer never wraps and nothing has to
//                   drain it; the compute kernel self-loops as PRODUCER + CONSUMER to complete the
//                   census, the same way the Quasar pool_generic factory handles its own borrowed
//                   output.
// No buffer here is single-kernel scratch, so no ScratchpadSpec appears.

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
namespace metal2 = tt::tt_metal::experimental;

namespace {

using FixedPoint = int32_t;
constexpr int32_t FIXED_POINT_SHIFT = 16;
constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;

FixedPoint float_to_fixed(float value) { return static_cast<FixedPoint>(value * FIXED_ONE); }

// Metal 2.0 resource names. Shared by create_program_artifacts and override_runtime_arguments.
const metal2::KernelSpecName READER{"bilinear_reader"};
const metal2::KernelSpecName WRITER{"bilinear_writer"};
const metal2::KernelSpecName COMPUTE{"bilinear_compute"};

const metal2::DFBSpecName DFB_HALO{"halo_cb"};
const metal2::DFBSpecName DFB_TILIZE_REDUCE_0{"tilize_reduce_cb_0"};
const metal2::DFBSpecName DFB_TILIZE_REDUCE_1{"tilize_reduce_cb_1"};
const metal2::DFBSpecName DFB_IN_SCALAR_0{"in_scalar_cb_0"};
const metal2::DFBSpecName DFB_IN_SCALAR_1{"in_scalar_cb_1"};
const metal2::DFBSpecName DFB_OUT{"out_cb"};

const metal2::TensorParamName INPUT{"input"};
const metal2::TensorParamName OUTPUT{"output"};

constexpr const char* DM_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp";
constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts UpsampleBilinearProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const ttnn::Tensor& input = input_tensor;
    const ttnn::Tensor& output = output_tensor;
    // This factory only supports integer scale factors
    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Bilinear upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const uint32_t scale_factor_h = static_cast<uint32_t>(operation_attributes.scale_factor_h);
    const uint32_t scale_factor_w = static_cast<uint32_t>(operation_attributes.scale_factor_w);
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    // Use the sliding window config passed from upsample.cpp (contains original input dimensions)
    TT_FATAL(
        operation_attributes.sliding_window_config.has_value(),
        "Bilinear upsample requires sliding_window_config to be provided");
    const ttnn::operations::sliding_window::SlidingWindowConfig sliding_window_config =
        operation_attributes.sliding_window_config.value();

    // Extract original (pre-halo) dimensions from sliding_window_config
    // These are the TRUE dimensions, not the haloed tensor dimensions
    const uint32_t in_batch_size = sliding_window_config.batch_size;
    const uint32_t in_h = sliding_window_config.input_hw.first;
    const uint32_t in_w = sliding_window_config.input_hw.second;
    const uint32_t in_channels = sliding_window_config.channels;

    // Output dimensions
    const Shape& output_shape = output.padded_shape();
    const uint32_t out_w = output_shape[2];

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(in_channels % 32 == 0, "input channels should be divisible by 32");
    // NOTE: input is assumed to have channels last format: {N, H, W, C}, {N, 1, H * W, C}, {1, 1, N * H * W, C}
    // NOTE: Bfp8_b/TILE is not yet supported
    const uint32_t input_stick_nbytes = in_channels * input.element_size();
    const uint32_t output_stick_nbytes = output_shape[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    auto* const device = input.device();

    const std::tuple<MathFidelity, bool, bool, bool, bool> compute_config_tuple =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    const MathFidelity math_fidelity = std::get<0>(compute_config_tuple);
    const bool math_approx_mode = std::get<1>(compute_config_tuple);
    const bool fp32_dest_acc_en = std::get<2>(compute_config_tuple);

    const ShardSpec shard_spec = input.shard_spec().value();
    const CoreRangeSet all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();
    const uint32_t ncores_nhw = ncores;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    uint32_t input_block_size_bytes = input_stick_nbytes;
    input_block_size_bytes =
        std::min(input_block_size_bytes, MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * input.element_size());

    const ttnn::Tensor& halo_in = input;
    const std::array<uint32_t, 2> halo_shard_shape = halo_in.shard_spec().value().shape;

    const std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata_bilinear(sliding_window_config);

    constexpr uint32_t buffering_factor = 2;

    // ---------------------------------------------------------------------------------------
    // Dataflow buffers (was: CBDescriptor list). Sizes are carried over byte for byte.
    // ---------------------------------------------------------------------------------------

    // input data is in a sharded CB -- borrowed view of the haloed input shard
    const uint32_t in_cb_pagesize = input_stick_nbytes;
    const uint32_t in_cb_npages = halo_shard_shape[0];
    const uint32_t in_ntiles_c = tt::div_up(in_channels, tt::constants::TILE_WIDTH);

    metal2::DataflowBufferSpec halo_dfb{
        .unique_id = DFB_HALO,
        .entry_size = in_cb_pagesize,
        .num_entries = in_cb_npages,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT,
    };

    // first intermediate DFB (4 pixels per entry are needed for the intermediate tensor)
    const uint32_t in1_cb_pagesize =
        std::min(tt::constants::TILE_WIDTH * input.element_size() * MAX_TILES_PER_REDUCTION, input_stick_nbytes);
    metal2::DataflowBufferSpec tilize_reduce_dfb_0{
        .unique_id = DFB_TILIZE_REDUCE_0,
        .entry_size = in1_cb_pagesize,
        .num_entries = 4 * buffering_factor,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = FaceGeometry{.face_r_dim = 4, .num_faces = 2},
    };

    // second intermediate DFB
    metal2::DataflowBufferSpec tilize_reduce_dfb_1{
        .unique_id = DFB_TILIZE_REDUCE_1,
        .entry_size = in_cb_pagesize,
        .num_entries = 4 * buffering_factor,
        .data_format_metadata = input_cb_data_format,
        .unpack_face_geometry_metadata = FaceGeometry{.face_r_dim = 4, .num_faces = 2},
    };

    // scalar (bilinear weight) DFBs, one per data-movement kernel
    const uint32_t in_scalar_cb_pagesize = tt::tile_size(input_cb_data_format);
    const uint32_t in_scalar_cb_npages = 1 * buffering_factor;

    metal2::DataflowBufferSpec in_scalar_dfb_0{
        .unique_id = DFB_IN_SCALAR_0,
        .entry_size = in_scalar_cb_pagesize,
        .num_entries = in_scalar_cb_npages,
        .data_format_metadata = input_cb_data_format,
    };
    metal2::DataflowBufferSpec in_scalar_dfb_1{
        .unique_id = DFB_IN_SCALAR_1,
        .entry_size = in_scalar_cb_pagesize,
        .num_entries = in_scalar_cb_npages,
        .data_format_metadata = input_cb_data_format,
    };

    // output sharded DFB with upsampled data -- borrowed view of the output shard
    const uint32_t out_cb_pagesize = tt::constants::TILE_WIDTH * output.element_size();
    const uint32_t out_cb_npages = output.shard_spec().value().shape[0] * in_ntiles_c;

    metal2::DataflowBufferSpec out_dfb{
        .unique_id = DFB_OUT,
        .entry_size = out_cb_pagesize,
        .num_entries = out_cb_npages,
        .data_format_metadata = output_cb_data_format,
        .unpack_face_geometry_metadata = FaceGeometry{.face_r_dim = 1, .num_faces = 2},
        .borrowed_from = OUTPUT,
    };

    log_debug(tt::LogOp, "halo_dfb npages: {}, pagesize: {}", in_cb_npages, in_cb_pagesize);
    log_debug(tt::LogOp, "out_dfb npages: {}, pagesize: {}", out_cb_npages, out_cb_pagesize);
    log_debug(tt::LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(tt::LogOp, "ncores: {}", ncores);

    const float scale_h_inv = 1.0f / static_cast<float>(scale_factor_h);
    const float scale_w_inv = 1.0f / static_cast<float>(scale_factor_w);

    const float y_index = (0.5f * scale_h_inv) + 0.5f;
    const float x_index_compute = (0.5f * scale_w_inv) + 0.5f;

    const FixedPoint scale_h_inv_fixed = float_to_fixed(scale_h_inv);
    const FixedPoint scale_w_inv_fixed = float_to_fixed(scale_w_inv);
    const FixedPoint y_index_fixed = float_to_fixed(y_index);
    const FixedPoint x_index_compute_fixed = float_to_fixed(x_index_compute);

    const uint32_t num_input_width_blocks = static_cast<uint32_t>(
        std::ceil(static_cast<float>(in_channels) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH)));

    // ---------------------------------------------------------------------------------------
    // Kernels. The "writer" runs the same source as the reader; the is_reader named CTA selects
    // which half of the output sticks it produces (as it did in the legacy positional form).
    // ---------------------------------------------------------------------------------------
    auto dm_compile_time_args = [&](uint32_t is_reader) {
        return metal2::KernelSpec::CompileTimeArgs{
            {"stick_nbytes", input_stick_nbytes},
            {"scale_h", scale_factor_h},
            {"scale_w", scale_factor_w},
            {"in_w", in_w},
            {"out_w", out_w},
            {"in_h", in_h},
            {"scale_h_inv_comp", static_cast<uint32_t>(scale_h_inv_fixed)},
            {"scale_w_inv_comp", static_cast<uint32_t>(scale_w_inv_fixed)},
            {"y_starting_coordinate_u32", static_cast<uint32_t>(y_index_fixed)},
            {"x_starting_coordinate_u32", static_cast<uint32_t>(x_index_compute_fixed)},
            {"is_reader", is_reader},
            {"blocks", num_input_width_blocks},
            {"input_block_size_bytes", input_block_size_bytes},
        };
    };

    const metal2::KernelSpec::RuntimeArgSchema dm_rta_schema{
        .runtime_arg_names = {"start_output_idx", "min_input_offset", "output_shard_height"}};

    // The DM kernels issue many sub-stick NOC reads and keep their own reserve_back/push_back
    // accounting; Gen2 implicit-sync credit tracking stalls on that pattern, so opt out (ignored
    // on Gen1).
    metal2::KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path{DM_KERNEL_PATH},
        .dfb_bindings =
            {// Borrowed input shard: producer face, census only (no credit traffic).
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_HALO,
                 .accessor_name = "halo_cb",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_TILIZE_REDUCE_0,
                 .accessor_name = "tilize_reduce_cb",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_IN_SCALAR_0,
                 .accessor_name = "in_scalar_cb",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER}},
        .compile_time_args = dm_compile_time_args(/*is_reader=*/1),
        .runtime_arg_schema = dm_rta_schema,
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    metal2::KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path{DM_KERNEL_PATH},
        .dfb_bindings =
            {// Borrowed input shard: consumer face, census only (no credit traffic).
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_HALO,
                 .accessor_name = "halo_cb",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_TILIZE_REDUCE_1,
                 .accessor_name = "tilize_reduce_cb",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_IN_SCALAR_1,
                 .accessor_name = "in_scalar_cb",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER}},
        .compile_time_args = dm_compile_time_args(/*is_reader=*/0),
        .runtime_arg_schema = dm_rta_schema,
        .hw_config =
            ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    TT_FATAL(fp32_dest_acc_en == false, "fp32_dest_acc_en as true not supported for upsample bilinear");

    constexpr ReduceOpMath reduce_op = ReduceOpMath::SUM;
    constexpr ReduceOpDim reduce_dim = ReduceOpDim::H;
    const std::map<std::string, std::string> reduce_defines_map = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    metal2::KernelSpec::CompilerOptions::Defines compute_defines(reduce_defines_map);

    // Quasar is Gen2 and rejects a ComputeGen1Config at program build; the knobs are the same
    // either way, and to_compute_hardware_config picks the alternative that matches the arch.
    const metal2::ComputeHardwareConfig compute_hw_config = ttnn::to_compute_hardware_config(
        device->arch(),
        ttnn::ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .math_approx_mode = math_approx_mode,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = false});

    metal2::KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .dfb_bindings =
            {metal2::DFBBinding{
                 .dfb_spec_name = DFB_TILIZE_REDUCE_0,
                 .accessor_name = "tilize_reduce_cb_0",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_TILIZE_REDUCE_1,
                 .accessor_name = "tilize_reduce_cb_1",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_IN_SCALAR_0,
                 .accessor_name = "in_scalar_cb_1",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_IN_SCALAR_1,
                 .accessor_name = "in_scalar_cb_2",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             // Borrowed output shard, sized to the whole shard: the compute kernel packs straight
             // into the resident output and nothing drains it, so it self-loops to complete the
             // single-producer/single-consumer census. No kernel-side pop is needed.
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_OUT,
                 .accessor_name = "out_cb",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = DFB_OUT,
                 .accessor_name = "out_cb",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {
                {"in_ntiles_c", in_ntiles_c},
                {"out_ntiles_c", tt::div_up(in_channels, tt::constants::TILE_WIDTH)},
                {"blocks", num_input_width_blocks},
                {"input_block_size_bytes", input_block_size_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"nsticks_per_core_by_nblocks"}},
        .hw_config = compute_hw_config,
    };
    compute.compiler_options.defines = std::move(compute_defines);

    metal2::ProgramSpec spec{
        .name = "upsample_bilinear_multicore",
        .kernels = {reader, writer, compute},
        .dataflow_buffers =
            {halo_dfb, tilize_reduce_dfb_0, tilize_reduce_dfb_1, in_scalar_dfb_0, in_scalar_dfb_1, out_dfb},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = input.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {metal2::WorkUnitSpec{
            .name = "bilinear", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    // ---------------------------------------------------------------------------------------
    // Runtime args: work distribution based on output sticks (unchanged from the legacy loop).
    // ---------------------------------------------------------------------------------------
    const uint32_t total_output_sticks = in_batch_size * output.logical_shape()[1] * output.logical_shape()[2];
    const uint32_t max_out_sticks_per_core = tt::div_up(total_output_sticks, ncores_nhw);

    log_debug(
        tt::LogOp,
        "total_output_sticks: {}, max_out_sticks_per_core: {}",
        total_output_sticks,
        max_out_sticks_per_core);

    const std::vector<CoreCoord> logical_cores = corerange_to_cores(
        shard_spec.grid, shard_spec.num_cores(), shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    metal2::ProgramRunArgs run_args;
    metal2::KernelRunArgs reader_run_args{.kernel = READER};
    metal2::KernelRunArgs writer_run_args{.kernel = WRITER};
    metal2::KernelRunArgs compute_run_args{.kernel = COMPUTE};

    uint32_t start_output_idx = 0;
    uint32_t total_sticks_processed = 0;

    for (const auto& core_coord : logical_cores) {
        // Calculate actual output sticks for this core
        const uint32_t out_sticks_this_core =
            std::min(max_out_sticks_per_core, total_output_sticks - total_sticks_processed);

        if (out_sticks_this_core == 0) {
            // No work for this core
            metal2::AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core_coord,
                {{"start_output_idx", start_output_idx}, {"min_input_offset", 0}, {"output_shard_height", 0}});
            metal2::AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core_coord,
                {{"start_output_idx", start_output_idx}, {"min_input_offset", 0}, {"output_shard_height", 0}});
            metal2::AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core_coord, {{"nsticks_per_core_by_nblocks", 0}});
            continue;
        }

        // Calculate the output range for this core (only after confirming there's work)
        const uint32_t output_index_start = start_output_idx;
        const uint32_t output_index_end =
            std::min(output_index_start + out_sticks_this_core, static_cast<uint32_t>(op_trace_metadata.size())) - 1;

        // Find the minimum input index for this core's output range
        const std::pair<uint32_t, uint32_t> minmax_indices =
            ttnn::operations::sliding_window::find_minmax_trace_indices(
                op_trace_metadata, output_index_start, output_index_end);
        const uint32_t min_trace_idx = minmax_indices.first;
        const uint32_t min_input_offset = op_trace_metadata[min_trace_idx];

        metal2::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core_coord,
            {{"start_output_idx", start_output_idx},
             {"min_input_offset", min_input_offset},
             {"output_shard_height", out_sticks_this_core}});
        metal2::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core_coord,
            {{"start_output_idx", start_output_idx},
             {"min_input_offset", min_input_offset},
             {"output_shard_height", out_sticks_this_core}});
        metal2::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core_coord,
            {{"nsticks_per_core_by_nblocks", out_sticks_this_core}});

        // Next core starts where this core ends
        start_output_idx += out_sticks_this_core;
        total_sticks_processed += out_sticks_this_core;
    }

    run_args.kernel_run_args = {
        std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {INPUT, metal2::TensorArgument{input.mesh_tensor()}},
        {OUTPUT, metal2::TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs UpsampleBilinearProgramFactory::override_runtime_arguments(
    const UpsampleParams& /*operation_attributes*/,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Both borrowed DFBs (the haloed input shard and the output shard) re-resolve their backing
    // address from the tensor args on every cache hit; the per-core runtime args depend only on
    // shapes, which are part of the program cache key, so they do not need re-deriving.
    metal2::ProgramRunArgs params;
    params.tensor_args = {
        {INPUT, metal2::TensorArgument{input_tensor.mesh_tensor()}},
        {OUTPUT, metal2::TensorArgument{output_tensor.mesh_tensor()}},
    };
    return params;
}

}  // namespace ttnn::prim
