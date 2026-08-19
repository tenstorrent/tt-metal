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

#include <bit>
#include <numeric>
#include <string>

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
namespace m2 = tt::tt_metal::experimental;
using namespace ttnn::prim::layernorm_distributed_metal2;

const m2::KernelSpecName PREWF_READER{"prewf_reader"};
const m2::KernelSpecName PREWF_WRITER{"prewf_writer"};
const m2::KernelSpecName PREWF_COMPUTE{"prewf_compute"};

const m2::DFBSpecName PREWF_INPUT{"prewf_input"};
const m2::DFBSpecName PREWF_SCRATCH{"prewf_scratch"};
const m2::DFBSpecName PREWF_RECIP{"prewf_recip"};
const m2::DFBSpecName PREWF_RESIDUAL{"prewf_residual"};
const m2::DFBSpecName PREWF_FUSED{"prewf_fused"};
const m2::DFBSpecName PREWF_MEAN_SPILL{"prewf_mean_spill"};
const m2::DFBSpecName PREWF_M2_SPILL{"prewf_m2_spill"};
const m2::DFBSpecName PREWF_OUT{"prewf_out"};

const m2::TensorParamName PREWF_INPUT_T{"prewf_input_t"};
const m2::TensorParamName PREWF_RESIDUAL_T{"prewf_residual_t"};
const m2::TensorParamName PREWF_RECIP_T{"prewf_recip_t"};
const m2::TensorParamName PREWF_OUTPUT_T{"prewf_output_t"};

constexpr const char* PREWF_READER_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
    "reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp";
constexpr const char* PREWF_WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_blocked.cpp";
constexpr const char* PREWF_COMPUTE_KERNEL =
    "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
    "layernorm_pre_allgather_welford.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts LayerNormPreAllGatherWelfordProgramFactory::create_program_artifacts(
    const LayerNormPreAllGatherParams& operation_attributes,
    const LayerNormPreAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const bool fuse_pre_add = b.has_value();
    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const auto& logical_shape = a.logical_shape();
    const auto& padded_shape = a.padded_shape();
    const uint32_t W = logical_shape[-1];
    const uint32_t padded_W = padded_shape[-1], padded_H = padded_shape[-2];
    const uint32_t padded_HW = padded_H * padded_W;
    const uint32_t NC = a.physical_volume() / padded_HW;

    const uint32_t Wt = padded_W / tile_width;
    const uint32_t Ht = padded_H / tile_height;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    TT_FATAL(!is_rmsnorm, "rms_norm is not compatible with welford, please disable welford flag to use rms norm");

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "padded_W: {}", padded_W);
    log_debug(tt::LogOp, "padded_H: {}", padded_H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // The welford kernel does the pre-add and welford passes in blk-sized chunks, spilling the
    // welford accumulator to a small buffer between chunks. Per-chunk overhead (state spill +
    // tile_regs scope switch) amortizes over more tiles when blk is larger; the upper bound is
    // how many tiles fit in DST in a single tile_regs scope (4 in fp32_dest_acc, 8 otherwise).
    // Constrain blk to divide Wt so the reader and compute kernel stay aligned without a
    // partial last block.
    const uint32_t dst_capacity = fp32_dest_acc_en ? 4u : 8u;
    uint32_t block_size = std::gcd(Wt, dst_capacity);
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (fuse_pre_add) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b->dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);

    // Sized for double-buffered block-sized chunks: the welford compute kernel waits on
    // block_size tiles at a time, so the reader must be able to fill that many while the
    // compute side processes the previous batch.
    const uint32_t in0_tiles = block_size * 2;
    const uint32_t res_tiles = block_size * 2;
    // The pre-add and welford passes are interleaved in block_size-sized chunks in the welford
    // kernel: each chunk's tiles are added into the fused buffer and then immediately consumed by
    // welford, with the welford accumulator spilled to the mean / M2 spill buffers between
    // chunks. So the fused buffer only needs to hold block_size * 2 tiles (double-buffered for
    // producer/consumer overlap), not the full Wt row.
    const uint32_t fused_tiles = block_size * 2;
    const uint32_t welford_spill_tiles = 1;

    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

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

    // UnpackToDest routes the unpack to DEST instead of SrcA, preserving FP32 precision.
    // That path uses the math-thread replay buffer, which collides with Welford's recurrence
    // slots; welford_unpack_fp32_active gates welford_init<WelfordInitMode::PreserveStats>()
    // after each transpose_tile to re-record the SFPU replay buffer.
    //
    // On the FUSE path, pre-add uses copy_tile + add_binary_tile (SFPU), not add_tiles, so
    // the input and residual buffers can use UnpackToDest for the copy_tile unpack, and the fused
    // buffer for Welford's transpose_tile read of the post-add result.
    bool welford_unpack_fp32_active = (in_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en);

    TT_FATAL(
        tensor_args.recip_tensor.has_value(),
        "Welford algorithm requires recip_tensor. Use ttnn.create_layer_norm_reciprocals() to create it.");
    const auto& recip_tensor = tensor_args.recip_tensor.value();
    const auto& recip_mesh = recip_tensor.mesh_tensor();
    const uint32_t reciprocal_CB_size_bytes = recip_tensor.buffer()->aligned_size_per_bank();
    constexpr tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;

    // Float32 input on the welford path requires fp32_dest_acc_en=true as a prerequisite for
    // UnpackToDest (set below). UnpackToDest is what bypasses the unpacker's
    // Float32 → TF32 truncation in SrcA; fp32_dest_acc_en provides the 32-bit DEST that
    // UnpackToDest writes into. Without fp32 DEST, UnpackToDest can't be enabled
    // and inputs are silently truncated to TF32 (10 mantissa bits) on the way through SrcA.
    TT_FATAL(
        !(in_data_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "layer_norm_pre_all_gather with Float32 input requires fp32_dest_acc_en=true in the "
        "compute kernel config; otherwise precision is silently lost in the unpacker format "
        "conversion.");

    m2::KernelSpec::CompilerOptions::Defines fuse_defines;
    if (fuse_pre_add) {
        fuse_defines.emplace("FUSE_PRE_ADD", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(make_dfb(PREWF_INPUT, in0_tiles, in_single_tile_size, in_data_format));
    if (fuse_pre_add) {
        // Residual b. Sized in the residual's own data format so a residual with a different
        // dtype than the input is read correctly.
        dfbs.push_back(make_dfb(PREWF_RESIDUAL, res_tiles, inb_single_tile_size, inb_data_format));
        // Fused a + b (the compute kernel writes here, welford consumes from it)
        dfbs.push_back(make_dfb(PREWF_FUSED, fused_tiles, single_tile_size, cb_data_format));
        // Welford mean accumulator spill (one tile, ping-pongs each iteration)
        dfbs.push_back(make_dfb(PREWF_MEAN_SPILL, welford_spill_tiles, single_tile_size, cb_data_format));
        // Welford M2 accumulator spill (one tile, ping-pongs each iteration)
        dfbs.push_back(make_dfb(PREWF_M2_SPILL, welford_spill_tiles, single_tile_size, cb_data_format));
    }
    // Intermediate scratch for the post-Welford transpose round-trip.
    // Used only for the last transpose operation before copying data into the
    // output buffer, which is why its data format is tied to the output format.
    // Anything wider would waste SRAM and gain no precision (the read-back
    // unpack truncates to TF32 unless the output is Float32, in which case
    // UnpackToDest below preserves it).
    dfbs.push_back(make_dfb(PREWF_SCRATCH, in0_tiles, out_single_tile_size, out_data_format));
    dfbs.push_back(make_dfb(PREWF_OUT, in0_tiles, out_single_tile_size, out_data_format));
    // Reciprocal LUT, built on the memory of the caller-supplied reciprocals tensor rather than
    // allocating its own; the compute kernel reads the table straight out of it by base pointer.
    auto recip_dfb = make_dfb(PREWF_RECIP, 1, reciprocal_CB_size_bytes, reciprocal_cb_data_format);
    recip_dfb.borrowed_from = PREWF_RECIP_T;
    dfbs.push_back(std::move(recip_dfb));

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader{
        .unique_id = PREWF_READER,
        .source = PREWF_READER_KERNEL,
        .compiler_options = {.defines = fuse_defines},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = PREWF_INPUT,
                    .accessor_name = "inp",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                // This reader is shared with the 1D factory, where this buffer really is the
                // reduce-scalar buffer it pushes a tile into. Here the same buffer is the compute
                // kernel's post-Welford transpose scratch, which the compute kernel also fills and
                // drains itself; the reader takes the producer role and the compute kernel the
                // consumer role, which is the only endpoint pair a buffer instance can have.
                m2::DFBBinding{
                    .dfb_spec_name = PREWF_SCRATCH,
                    .accessor_name = "reduce",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = PREWF_INPUT_T, .accessor_name = "src"}},
        .compile_time_args = {{"blk", block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt", "Wt", "tile_offset"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    if (fuse_pre_add) {
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = PREWF_RESIDUAL, .accessor_name = "res", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = PREWF_RESIDUAL_T, .accessor_name = "res_src"});
    }

    m2::KernelSpec writer{
        .unique_id = PREWF_WRITER,
        .source = PREWF_WRITER_KERNEL,
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = PREWF_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = PREWF_OUTPUT_T, .accessor_name = "dst"}},
        .compile_time_args = {{"blk", writer_block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Welford uses fp32 accumulation; preserve fp32_dest_acc_en from the compute config.
    m2::KernelSpec compute{
        .unique_id = PREWF_COMPUTE,
        .source = PREWF_COMPUTE_KERNEL,
        .compiler_options = {.defines = fuse_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = PREWF_INPUT,
                    .accessor_name = "in0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = PREWF_SCRATCH,
                    .accessor_name = "scratch",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = PREWF_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"Wt", Wt},
             {"W", W},
             {"blk", block_size},
             {"welford_unpack_fp32_active", welford_unpack_fp32_active ? 1u : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"NCHt"}},
        .hw_config = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config),
    };
    // The reciprocal table has no FIFO traffic at all: the kernel reads it by base pointer. It is
    // that kernel's only endpoint, so it takes both roles.
    bind_self_loop(compute, PREWF_RECIP, "recip");
    if (fuse_pre_add) {
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = PREWF_RESIDUAL, .accessor_name = "res", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        bind_self_loop(compute, PREWF_FUSED, "fused");
        bind_self_loop(compute, PREWF_MEAN_SPILL, "mean_spill");
        bind_self_loop(compute, PREWF_M2_SPILL, "m2_spill");
    }

    auto& compute_gen1 = gen1_compute_config(std::get<m2::ComputeHardwareConfig>(compute.hw_config));
    // When welford_unpack_fp32_active:
    //   !fuse_pre_add -> UnpackToDest on the input only (read by transpose_tile in the Welford loop).
    //   fuse_pre_add  -> UnpackToDest on the input, residual and fused buffers (copy_tile pre-add
    //   unpack + transpose_tile on the post-add result).
    if (welford_unpack_fp32_active) {
        unpack_via_dest(compute_gen1, PREWF_INPUT);
        if (fuse_pre_add) {
            unpack_via_dest(compute_gen1, PREWF_RESIDUAL);
            unpack_via_dest(compute_gen1, PREWF_FUSED);
        }
    }
    // The transpose scratch holds data only for the final transpose operation, so its format
    // mirrors out_data_format. When both that format is FP32 and DEST is in FP32 mode, force
    // UnpackToDest on it too so the read-back doesn't truncate to TF32. For non-FP32 outputs the
    // final pack to the output buffer truncates anyway, so unpacking to FP32 would not be useful.
    if (out_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en) {
        unpack_via_dest(compute_gen1, PREWF_SCRATCH);
    }
    // The Welford spill buffers hold the FP32 accumulator between block iterations and are reloaded
    // into DEST via copy_tile. On the SrcA/B path that round-trip truncates FP32 to TF32 on every
    // block iteration. Force UnpackToDest on them so the FP32 precision survives the spill cycle.
    if (fuse_pre_add && fp32_dest_acc_en) {
        unpack_via_dest(compute_gen1, PREWF_MEAN_SPILL);
        unpack_via_dest(compute_gen1, PREWF_M2_SPILL);
    }
    // The remaining Float32 buffers this kernel consumes take the SrcA/B path. Each needs saying out
    // loud, because with the 32-bit Dest register enabled a Float32 buffer has no implicit default.
    if (fp32_dest_acc_en) {
        // The reciprocal table is never unpacked at all: the kernel reads it through a base pointer.
        // SrcA/B is the inert choice for a buffer no unpacker touches.
        unpack_via_src(compute_gen1, PREWF_RECIP);
        // A narrower input leaves welford_unpack_fp32_active off, which puts the pre-add on the FPU
        // add_tiles path instead of the SFPU copy_tile one. The residual and the fused result are then
        // read through SrcA/B, even though the 32-bit Dest register still makes the fused buffer Float32.
        if (fuse_pre_add && !welford_unpack_fp32_active) {
            unpack_via_src(compute_gen1, PREWF_FUSED);
            if (inb_data_format == tt::DataFormat::Float32) {
                unpack_via_src(compute_gen1, PREWF_RESIDUAL);
            }
        }
    }
    // The input and the transpose scratch need nothing further: each is Float32 only when its own
    // tensor is, and both of those conditions already gave it UnpackToDest above. The spill buffers are
    // likewise covered, on the fuse_pre_add && fp32_dest_acc_en condition that creates them.

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = PREWF_INPUT_T, .spec = input_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = PREWF_OUTPUT_T, .spec = output_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = PREWF_RECIP_T, .spec = recip_mesh.tensor_spec()});
    if (fuse_pre_add) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = PREWF_RESIDUAL_T, .spec = b->mesh_tensor().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime arguments
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_run{.kernel = PREWF_READER};
    m2::KernelRunArgs writer_run{.kernel = PREWF_WRITER};
    m2::KernelRunArgs compute_run{.kernel = PREWF_COMPUTE};

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
        .name = "layernorm_pre_all_gather_welford",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {m2::WorkUnitSpec{
            .name = "main", .kernels = {PREWF_READER, PREWF_WRITER, PREWF_COMPUTE}, .target_nodes = all_cores}},
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args.emplace(PREWF_INPUT_T, input_mesh);
    run_args.tensor_args.emplace(PREWF_OUTPUT_T, output_mesh);
    run_args.tensor_args.emplace(PREWF_RECIP_T, recip_mesh);
    if (fuse_pre_add) {
        run_args.tensor_args.emplace(PREWF_RESIDUAL_T, b->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
