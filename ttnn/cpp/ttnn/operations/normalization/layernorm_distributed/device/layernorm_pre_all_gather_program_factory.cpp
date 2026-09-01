// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_device_operation.hpp"
#include "layernorm_distributed_metal2_helpers.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <string>

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace m2 = tt::tt_metal::experimental;
using namespace ttnn::prim::layernorm_distributed_metal2;

// ---- Pre 1D spec names ----
const m2::KernelSpecName PRE1D_READER{"pre1d_reader"};
const m2::KernelSpecName PRE1D_WRITER{"pre1d_writer"};
const m2::KernelSpecName PRE1D_COMPUTE{"pre1d_compute"};

const m2::DFBSpecName PRE1D_INPUT{"pre1d_input"};
const m2::DFBSpecName PRE1D_REDUCE{"pre1d_reduce"};
const m2::DFBSpecName PRE1D_RESIDUAL{"pre1d_residual"};
const m2::DFBSpecName PRE1D_FUSED{"pre1d_fused"};
const m2::DFBSpecName PRE1D_X2{"pre1d_x2"};
const m2::DFBSpecName PRE1D_OUT{"pre1d_out"};

const m2::TensorParamName PRE1D_INPUT_T{"pre1d_input_t"};
const m2::TensorParamName PRE1D_RESIDUAL_T{"pre1d_residual_t"};
const m2::TensorParamName PRE1D_OUTPUT_T{"pre1d_output_t"};

// ---- Pre 2D spec names ----
const m2::KernelSpecName PRE2D_READER{"pre2d_reader"};
const m2::KernelSpecName PRE2D_WRITER{"pre2d_writer"};
const m2::KernelSpecName PRE2D_COMPUTE_MERGE{"pre2d_compute_merge"};
const m2::KernelSpecName PRE2D_COMPUTE_WORKER{"pre2d_compute_worker"};

const m2::DFBSpecName PRE2D_INPUT{"pre2d_input"};
const m2::DFBSpecName PRE2D_REDUCE{"pre2d_reduce"};
const m2::DFBSpecName PRE2D_RESIDUAL{"pre2d_residual"};
const m2::DFBSpecName PRE2D_FUSED{"pre2d_fused"};
const m2::DFBSpecName PRE2D_X2{"pre2d_x2"};
const m2::DFBSpecName PRE2D_X2_MERGE{"pre2d_x2_merge"};
const m2::DFBSpecName PRE2D_PARTIAL_OUT{"pre2d_partial_out"};
const m2::DFBSpecName PRE2D_ZERO{"pre2d_zero"};
const m2::DFBSpecName PRE2D_OUT_FINAL{"pre2d_out_final"};

const m2::SemaphoreSpecName PRE2D_REDUCER{"pre2d_reducer"};

const m2::TensorParamName PRE2D_INPUT_T{"pre2d_input_t"};
const m2::TensorParamName PRE2D_RESIDUAL_T{"pre2d_residual_t"};
const m2::TensorParamName PRE2D_OUTPUT_T{"pre2d_output_t"};

constexpr const char* PRE_READER_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
    "reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp";
constexpr const char* PRE_WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_blocked.cpp";
constexpr const char* PRE2D_READER_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
    "reader_layernorm_preallgather_2d.cpp";
constexpr const char* PRE2D_COMPUTE_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
    "layernorm_pre_allgather_2d.cpp";

}  // namespace

// =============================================================================
// LayerNormPreAllGatherProgramFactory - Normal (non-Welford, non-2D) operation
// =============================================================================

ttnn::device_operation::ProgramArtifacts LayerNormPreAllGatherProgramFactory::create_program_artifacts(
    const LayerNormPreAllGatherParams& operation_attributes,
    const LayerNormPreAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const bool fuse_pre_add = b.has_value();
    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);

    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const bool fp32_dest_acc_en = operation_attributes.compute_kernel_config.fp32_dest_acc_en;
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat scaler_cb_data_format = cb_data_format;
    // Float32 + fp32_dest_acc_en + !fast_and_approximate_mode -> SFPU Accurate; else FPU.
    // Quasar has no SFPU Accurate reduce; fall back to FPU.
    const bool unpack_fp32_active =
        (in_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en &&
         !operation_attributes.fast_and_approximate_mode && device->arch() != tt::ARCH::QUASAR);
    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (fuse_pre_add) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b->dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t scaler_tile_size = tt::tile_size(scaler_cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);

    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;  // reduce scalar
    const uint32_t res_tiles = Wt * double_buffer_constant;    // residual b
    const uint32_t fused_tiles = Wt;                           // a + b

    const uint32_t intermed0_tiles = Wt * double_buffer_constant;  // x^2
    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    TT_FATAL(
        W <= tile_width * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now).",
        W,
        tile_width,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        in0_tiles,
        block_size);
    TT_FATAL(
        intermed0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        intermed0_tiles,
        block_size);

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_group_1: {}", core_group_1.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    log_debug(tt::LogOp, "core_group_2: {}", core_group_2.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    const auto* compute_kernel_file =
        is_rmsnorm ? "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/"
                     "rmsnorm_pre_allgather.cpp"
                   : "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
                     "layernorm_pre_allgather.cpp";

    // The residual buffers exist only on the fused path, and the kernel-side handle for a buffer exists
    // only where the host binds it, so a kernel's source must not contain the text `dfb::res` when
    // there is no residual. That means this define is emitted only when the path is taken, and the
    // kernels gate on `#ifdef`. Emitting it always as "0" or "1" and testing it with `#if` would not
    // work: the text naming the unbound handle would still reach the compiler.
    m2::KernelSpec::CompilerOptions::Defines fuse_defines;
    if (fuse_pre_add) {
        fuse_defines.emplace("FUSE_PRE_ADD", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(make_dfb(PRE1D_INPUT, in0_tiles, in_single_tile_size, in_data_format));
    dfbs.push_back(make_dfb(PRE1D_REDUCE, in1_tiles, scaler_tile_size, scaler_cb_data_format));
    if (fuse_pre_add) {
        // Residual b. Sized in the residual's own data format so a residual with a different dtype
        // than the input is read correctly; add_tiles handles the per-operand format.
        dfbs.push_back(make_dfb(PRE1D_RESIDUAL, res_tiles, inb_single_tile_size, inb_data_format));
        // Fused a + b (the compute kernel writes into this and reads it back downstream)
        dfbs.push_back(make_dfb(PRE1D_FUSED, fused_tiles, single_tile_size, cb_data_format));
    }
    dfbs.push_back(make_dfb(PRE1D_X2, intermed0_tiles, single_tile_size, cb_data_format));
    dfbs.push_back(make_dfb(PRE1D_OUT, out0_tiles, out_single_tile_size, out_data_format));

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader{
        .unique_id = PRE1D_READER,
        .source = PRE_READER_KERNEL,
        .compiler_options = {.defines = fuse_defines},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = PRE1D_INPUT,
                    .accessor_name = "inp",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = PRE1D_REDUCE,
                    .accessor_name = "reduce",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = PRE1D_INPUT_T, .accessor_name = "src"}},
        .compile_time_args = {{"blk", block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt", "Wt", "tile_offset"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    if (fuse_pre_add) {
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = PRE1D_RESIDUAL, .accessor_name = "res", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = PRE1D_RESIDUAL_T, .accessor_name = "res_src"});
    }

    m2::KernelSpec writer{
        .unique_id = PRE1D_WRITER,
        .source = PRE_WRITER_KERNEL,
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = PRE1D_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = PRE1D_OUTPUT_T, .accessor_name = "dst"}},
        .compile_time_args = {{"blk", writer_block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    auto compute_hw = ttnn::to_compute_hardware_config(operation_attributes.compute_kernel_config);
    m2::KernelSpec compute{
        .unique_id = PRE1D_COMPUTE,
        .source = compute_kernel_file,
        .compiler_options = {.defines = fuse_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = PRE1D_INPUT,
                    .accessor_name = "in0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = PRE1D_REDUCE,
                    .accessor_name = "reduce",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = PRE1D_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"Wt", Wt}, {"blk", block_size}, {"unpack_fp32_active", unpack_fp32_active ? 1u : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
        .hw_config = compute_hw,
    };
    // x^2 and the fused a + b are private to the compute kernel: it packs into them and unpacks them
    // back, so it is the buffer's only endpoint on both sides.
    bind_self_loop(compute, PRE1D_X2, "x2");
    if (fuse_pre_add) {
        bind_self_loop(compute, PRE1D_FUSED, "fused");
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = PRE1D_RESIDUAL, .accessor_name = "res", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    auto& compute_cfg = std::get<m2::ComputeHardwareConfig>(compute.hw_config);
    // With the 32-bit Dest register enabled, every Float32 buffer the compute kernel consumes needs an
    // explicit unpack mode. Here each one feeds an FPU op (mul_tiles for x**2, the row reduce for the
    // sums), and the FPU reads its operands out of SrcA/SrcB, so SrcA/B is the mode for all of them.
    // The intermediates are Float16_b whatever the Dest width, so only the inputs can qualify.
    if (compute_cfg.enable_32_bit_dest) {
        auto unpack_operand = [&](const m2::DFBSpecName& dfb) {
            if (unpack_fp32_active) {
                unpack_via_dest(compute_cfg, dfb);
            } else {
                unpack_via_src(compute_cfg, dfb);
            }
        };
        if (in_data_format == tt::DataFormat::Float32) {
            unpack_operand(PRE1D_INPUT);
        }
        if (scaler_cb_data_format == tt::DataFormat::Float32) {
            unpack_via_src(compute_cfg, PRE1D_REDUCE);
        }
        if (cb_data_format == tt::DataFormat::Float32) {
            unpack_operand(PRE1D_X2);
            if (fuse_pre_add) {
                unpack_operand(PRE1D_FUSED);
            }
        }
        if (fuse_pre_add && inb_data_format == tt::DataFormat::Float32) {
            unpack_operand(PRE1D_RESIDUAL);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = PRE1D_INPUT_T, .spec = input_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = PRE1D_OUTPUT_T, .spec = output_mesh.tensor_spec()});
    if (fuse_pre_add) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = PRE1D_RESIDUAL_T, .spec = b->mesh_tensor().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime arguments
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_run{.kernel = PRE1D_READER};
    m2::KernelRunArgs writer_run{.kernel = PRE1D_WRITER};
    m2::KernelRunArgs compute_run{.kernel = PRE1D_COMPUTE};

    uint32_t curr_row = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"NCHt", num_tile_rows_per_core}, {"Wt", Wt}, {"tile_offset", in_tile_offset}});
        m2::AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"NCHt", num_tile_rows_per_core}});
        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_tiles", num_tile_rows_per_core * out0_tiles}, {"tile_offset", out_tile_offset}});

        curr_row += num_tile_rows_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .name = "layernorm_pre_all_gather",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {m2::WorkUnitSpec{
            .name = "main", .kernels = {PRE1D_READER, PRE1D_WRITER, PRE1D_COMPUTE}, .target_nodes = all_cores}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args.emplace(PRE1D_INPUT_T, input_mesh);
    run_args.tensor_args.emplace(PRE1D_OUTPUT_T, output_mesh);
    if (fuse_pre_add) {
        run_args.tensor_args.emplace(PRE1D_RESIDUAL_T, b->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// =============================================================================
// LayerNormPreAllGather2DProgramFactory - 2D core grid operation
// =============================================================================

ttnn::device_operation::ProgramArtifacts LayerNormPreAllGather2DProgramFactory::create_program_artifacts(
    const LayerNormPreAllGatherParams& operation_attributes,
    const LayerNormPreAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const bool fuse_pre_add = b.has_value();
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;

    uint32_t num_tile_rows = NC * Ht;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    IDevice* device = a.device();

    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const bool fp32_dest_acc_en = operation_attributes.compute_kernel_config.fp32_dest_acc_en;
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat scaler_cb_data_format = cb_data_format;
    // Float32 + fp32_dest_acc_en + !fast_and_approximate_mode -> SFPU Accurate; else FPU.
    // Quasar has no SFPU Accurate reduce; fall back to FPU.
    const bool unpack_fp32_active =
        (in_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en &&
         !operation_attributes.fast_and_approximate_mode && device->arch() != tt::ARCH::QUASAR);
    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (fuse_pre_add) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b->dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t scaler_tile_size = tt::tile_size(scaler_cb_data_format);

    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;  // reduce scalar
    const uint32_t res_tiles = Wt * double_buffer_constant;    // residual b
    const uint32_t fused_tiles = Wt;                           // a + b

    const uint32_t intermed0_tiles = Wt * double_buffer_constant;  // x^2
    uint32_t out0_tiles = 1;

    TT_FATAL(
        W <= tile_width * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now).",
        W,
        tile_width,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        in0_tiles,
        block_size);
    TT_FATAL(
        intermed0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        intermed0_tiles,
        block_size);

    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t max_cores_y = grid_size.y;
    uint32_t cores_x = std::min(max_cores_y, num_tile_rows);
    while (num_tile_rows % cores_x != 0 && cores_x > 1) {
        cores_x--;
    }
    uint32_t tiles_per_core_x = num_tile_rows / cores_x;
    uint32_t cores_y = std::min(max_cores_y, Wt);
    while (Wt % cores_y != 0 && cores_y > 1) {
        cores_y--;
    }
    uint32_t tiles_per_core_y = Wt / cores_y;

    CoreRange all_cores_range({0, 0}, {cores_x - 1, cores_y - 1});
    CoreRangeSet all_cores = CoreRangeSet(std::vector{all_cores_range});

    std::vector<CoreRange> merge_core_ranges_vec;
    merge_core_ranges_vec.reserve(cores_x);
    for (uint32_t x = 0; x < cores_x; ++x) {
        CoreCoord merge_core = {x, 0};
        merge_core_ranges_vec.emplace_back(CoreRange(merge_core, merge_core));
    }
    CoreRangeSet merge_cores(std::move(merge_core_ranges_vec));

    // Everything below the merge row. The compute kernel is instantiated separately over this set
    // because only the merge row produces into the final output buffer, and a buffer whose producer
    // covers nodes its consumer does not is rejected: a local buffer needs one producer and one
    // consumer on every node it lives on.
    CoreRangeSet worker_cores;
    if (cores_y > 1) {
        worker_cores = CoreRangeSet(std::vector{CoreRange({0, 1}, {cores_x - 1, cores_y - 1})});
    }
    const bool has_worker_cores = worker_cores.num_cores() > 0;

    m2::KernelSpec::CompilerOptions::Defines fuse_defines;
    if (fuse_pre_add) {
        fuse_defines.emplace("FUSE_PRE_ADD", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(make_dfb(PRE2D_INPUT, in0_tiles, in_single_tile_size, in_data_format));
    dfbs.push_back(make_dfb(PRE2D_REDUCE, in1_tiles, scaler_tile_size, scaler_cb_data_format));
    if (fuse_pre_add) {
        // Residual b. Sized in the residual's own data format so a residual with a different dtype
        // than the input is read correctly; add_tiles handles the per-operand format.
        dfbs.push_back(make_dfb(PRE2D_RESIDUAL, res_tiles, inb_single_tile_size, inb_data_format));
        // Fused a + b (the compute kernel writes into this and reads it back downstream)
        dfbs.push_back(make_dfb(PRE2D_FUSED, fused_tiles, single_tile_size, cb_data_format));
    }
    dfbs.push_back(make_dfb(PRE2D_X2, intermed0_tiles, single_tile_size, cb_data_format));
    // Cross-core merge buffer. Each of the cores_y worker rows writes one partial-stat tile into this
    // buffer on the merge core, and the merge core's compute wait_front/add_tiles/pop_front over
    // cores_y tiles. Size it by the gather count (cores_y), NOT the per-core width
    // (tiles_per_core_y): when cores_y > tiles_per_core_y (i.e. cores_y^2 > Wt, e.g. grid.y=8 with
    // Wt=32 -> cores_y=8, tiles_per_core_y=4) the old sizing was too small and workers with
    // y >= tiles_per_core_y wrote past the allocation, corrupting adjacent SRAM (with a matching
    // over-read on the compute side).
    dfbs.push_back(make_dfb(PRE2D_X2_MERGE, cores_y, single_tile_size, cb_data_format));
    dfbs.push_back(make_dfb(PRE2D_PARTIAL_OUT, out0_tiles, single_tile_size, cb_data_format));
    dfbs.push_back(make_dfb(PRE2D_ZERO, out0_tiles, single_tile_size, cb_data_format));
    dfbs.push_back(make_dfb(PRE2D_OUT_FINAL, out0_tiles, out_single_tile_size, out_data_format));

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader{
        .unique_id = PRE2D_READER,
        .source = PRE2D_READER_KERNEL,
        .compiler_options = {.defines = fuse_defines},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = PRE2D_INPUT,
                    .accessor_name = "inp",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = PRE2D_REDUCE,
                    .accessor_name = "reduce",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = PRE2D_X2_MERGE,
                    .accessor_name = "x2_merge",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = PRE2D_ZERO,
                    .accessor_name = "zero",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = PRE2D_PARTIAL_OUT,
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
            },
        .semaphore_bindings = {m2::SemaphoreBinding{.semaphore_spec_name = PRE2D_REDUCER, .accessor_name = "reducer"}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = PRE2D_INPUT_T, .accessor_name = "src"}},
        .compile_time_args = {{"blk", block_size}, {"num_cores_to_wait", cores_y}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"NCHt", "Wt", "tile_offset", "is_merge_core", "reduce_core_noc_x", "reduce_core_noc_y", "y"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    if (fuse_pre_add) {
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = PRE2D_RESIDUAL, .accessor_name = "res", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = PRE2D_RESIDUAL_T, .accessor_name = "res_src"});
    }

    m2::KernelSpec writer{
        .unique_id = PRE2D_WRITER,
        .source = PRE_WRITER_KERNEL,
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = PRE2D_OUT_FINAL, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = PRE2D_OUTPUT_T, .accessor_name = "dst"}},
        .compile_time_args = {{"blk", writer_block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Two instances of the one compute source, over disjoint node sets: the merge row additionally
    // reduces the column's partials into the final output buffer, so only it binds that buffer. The
    // merge flag is a compile-time define rather than a runtime arg because it selects which buffers
    // are bound, and an unbound handle cannot be named even on a dead branch.
    auto make_compute = [&](const m2::KernelSpecName& unique_id, bool is_merge_core) {
        auto defines = fuse_defines;
        if (is_merge_core) {
            defines.emplace("IS_MERGE_CORE", "1");
        }
        m2::KernelSpec compute{
            .unique_id = unique_id,
            .source = PRE2D_COMPUTE_KERNEL,
            .compiler_options = {.defines = std::move(defines), .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {
                    m2::DFBBinding{
                        .dfb_spec_name = PRE2D_INPUT,
                        .accessor_name = "in0",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER},
                    m2::DFBBinding{
                        .dfb_spec_name = PRE2D_REDUCE,
                        .accessor_name = "reduce",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER},
                    m2::DFBBinding{
                        .dfb_spec_name = PRE2D_X2_MERGE,
                        .accessor_name = "x2_merge",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER},
                    m2::DFBBinding{
                        .dfb_spec_name = PRE2D_ZERO,
                        .accessor_name = "zero",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER},
                    m2::DFBBinding{
                        .dfb_spec_name = PRE2D_PARTIAL_OUT,
                        .accessor_name = "out",
                        .endpoint_type = m2::DFBEndpointType::PRODUCER},
                },
            .compile_time_args =
                {{"NCHt", tiles_per_core_x},
                 {"Wt", tiles_per_core_y},
                 {"blk", block_size},
                 {"num_cores_y", cores_y},
                 {"unpack_fp32_active", unpack_fp32_active ? 1u : 0u}},
            .hw_config = ttnn::to_compute_hardware_config(operation_attributes.compute_kernel_config),
        };
        bind_self_loop(compute, PRE2D_X2, "x2");
        if (fuse_pre_add) {
            bind_self_loop(compute, PRE2D_FUSED, "fused");
            compute.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = PRE2D_RESIDUAL,
                .accessor_name = "res",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (is_merge_core) {
            compute.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = PRE2D_OUT_FINAL,
                .accessor_name = "out_final",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        auto& compute_hw = std::get<m2::ComputeHardwareConfig>(compute.hw_config);
        // Float32 operands use UnpackToDest on the accurate SFPU path and SrcA/SrcB on the FPU path.
        // The reduce scaler and the FPU merge's zero tile are always consumed through SrcA/SrcB.
        if (compute_hw.enable_32_bit_dest) {
            auto unpack_operand = [&](const m2::DFBSpecName& dfb) {
                if (unpack_fp32_active) {
                    unpack_via_dest(compute_hw, dfb);
                } else {
                    unpack_via_src(compute_hw, dfb);
                }
            };
            if (in_data_format == tt::DataFormat::Float32) {
                unpack_operand(PRE2D_INPUT);
            }
            if (scaler_cb_data_format == tt::DataFormat::Float32) {
                unpack_via_src(compute_hw, PRE2D_REDUCE);
            }
            if (cb_data_format == tt::DataFormat::Float32) {
                unpack_operand(PRE2D_X2);
                if (fuse_pre_add) {
                    unpack_operand(PRE2D_FUSED);
                }
                // The merge sums the column's partials on the SFPU when accurate, add_tiles
                // otherwise; dfb::zero is only the FPU path's operand.
                unpack_operand(PRE2D_X2_MERGE);
                unpack_via_src(compute_hw, PRE2D_ZERO);
            }
            if (fuse_pre_add && inb_data_format == tt::DataFormat::Float32) {
                unpack_operand(PRE2D_RESIDUAL);
            }
        }
        return compute;
    };

    m2::Group<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(PRE2D_COMPUTE_MERGE, /*is_merge_core=*/true));
    if (has_worker_cores) {
        kernels.push_back(make_compute(PRE2D_COMPUTE_WORKER, /*is_merge_core=*/false));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = PRE2D_INPUT_T, .spec = input_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = PRE2D_OUTPUT_T, .spec = output_mesh.tensor_spec()});
    if (fuse_pre_add) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = PRE2D_RESIDUAL_T, .spec = b->mesh_tensor().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime arguments
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_run{.kernel = PRE2D_READER};
    m2::KernelRunArgs writer_run{.kernel = PRE2D_WRITER};

    for (uint32_t x = 0; x < cores_x; ++x) {
        for (uint32_t y = 0; y < cores_y; ++y) {
            CoreCoord core = {x, y};
            bool is_merge_core = y == 0;
            const auto merge_core = device->worker_core_from_logical_core({x, 0});

            uint32_t num_tile_rows_per_core = tiles_per_core_x;

            uint32_t in_tile_offset = (x * Wt) + (y * tiles_per_core_y);
            uint32_t out_tile_offset = x * out0_tiles;

            m2::AddRuntimeArgsForNode(
                reader_run.runtime_arg_values,
                core,
                {{"NCHt", tiles_per_core_x},
                 {"Wt", tiles_per_core_y},
                 {"tile_offset", in_tile_offset},
                 {"is_merge_core", static_cast<uint32_t>(is_merge_core)},
                 {"reduce_core_noc_x", static_cast<uint32_t>(merge_core.x)},
                 {"reduce_core_noc_y", static_cast<uint32_t>(merge_core.y)},
                 {"y", y}});
            if (is_merge_core) {
                m2::AddRuntimeArgsForNode(
                    writer_run.runtime_arg_values,
                    core,
                    {{"num_tiles", num_tile_rows_per_core * out0_tiles}, {"tile_offset", out_tile_offset}});
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .name = "merge", .kernels = {PRE2D_READER, PRE2D_WRITER, PRE2D_COMPUTE_MERGE}, .target_nodes = merge_cores});
    if (has_worker_cores) {
        work_units.push_back(m2::WorkUnitSpec{
            .name = "worker", .kernels = {PRE2D_READER, PRE2D_COMPUTE_WORKER}, .target_nodes = worker_cores});
    }

    m2::ProgramSpec spec{
        .name = "layernorm_pre_all_gather_2d",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .semaphores = {m2::SemaphoreSpec{.unique_id = PRE2D_REDUCER, .target_nodes = all_cores}},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args.emplace(PRE2D_INPUT_T, input_mesh);
    run_args.tensor_args.emplace(PRE2D_OUTPUT_T, output_mesh);
    if (fuse_pre_add) {
        run_args.tensor_args.emplace(PRE2D_RESIDUAL_T, b->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
