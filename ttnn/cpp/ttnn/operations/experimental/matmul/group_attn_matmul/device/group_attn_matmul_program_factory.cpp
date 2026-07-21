// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
// Program-scope resource names (Metal 2.0 strong-typed spec-name constants).
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};

// DFBs (ex-CBs). Kernel accessors: dfb::in0/in1/in2/intermed0/intermed1/out.
const DFBSpecName IN0{"in0"};              // c_0 in0
const DFBSpecName IN1{"in1"};              // c_1 in1 (mcast target)
const DFBSpecName IN2{"in2"};              // c_2 in1-sharded borrowed (IN1_SHARDED only)
const DFBSpecName INTERMED0{"intermed0"};  // c_3
const DFBSpecName INTERMED1{"intermed1"};  // c_4
const DFBSpecName OUT{"out"};              // c_5 out

// Semaphores (mcast handshake; reader-only). sem::sender_sem / sem::receiver_sem.
const SemaphoreSpecName SENDER_SEM{"sender_sem"};
const SemaphoreSpecName RECEIVER_SEM{"receiver_sem"};

// Tensor parameters. Kernel accessors: tensor::src0 (in0), tensor::src1 (in1), tensor::dst (out).
const TensorParamName T_IN0{"t_in0"};
const TensorParamName T_IN1{"t_in1"};
const TensorParamName T_OUT{"t_out"};
}  // namespace

ttnn::device_operation::ProgramArtifacts GroupAttnMatmulProgramFactory::create_program_artifacts(
    const GroupAttnMatmulParams& operation_attributes,
    const GroupAttnMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    // Metalium memory objects for the io tensors (used for specs, bindings, and per-arg identity).
    const auto& a = tensor_args.input_tensor_a.mesh_tensor();
    const auto& b = tensor_args.input_tensor_b.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::tt_metal::IDevice* device = tensor_args.input_tensor_a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32
                                            ? tt::DataFormat::Float32
                                            : tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    if (in0_data_format == tt::DataFormat::Float32 or in1_data_format == tt::DataFormat::Float32 or
        output_data_format == tt::DataFormat::Float32) {
        TT_FATAL(fp32_dest_acc_en == true, "when inputs/output are in fp32 format, fp32_dest_acc_en must be set");
    }

    TT_FATAL(tensor_return_value.buffer() != nullptr, "Output buffer should be allocated on device!");

    // Load kernels on all device cores (cached program covers shape variation via
    // DFB-size re-application on cache hit).
    CoreCoord device_compute_with_storage_grid = device->compute_with_storage_grid_size();
    CoreRangeSet all_device_cores{
        CoreRange({0, 0}, {device_compute_with_storage_grid.x - 1, device_compute_with_storage_grid.y - 1})};

    // ---- Shape-derived quantities (these vary across cache hits) ----
    const bool transpose_hw_bool = operation_attributes.transpose_hw.value_or(false);
    const uint32_t num_tokens_val = operation_attributes.num_tokens.value_or(0);
    uint32_t Q_HEADS = ashape[1];
    uint32_t KV_HEADS = bshape[1];
    uint32_t Mt = ashape[2] / TILE_HEIGHT;
    uint32_t Kt = ashape[3] / TILE_WIDTH;
    uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2] / TILE_HEIGHT;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val / TILE_HEIGHT : bshape[3] / TILE_WIDTH;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;
    uint32_t in1_KtNt = transpose_hw_bool ? bshape[2] / TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
    uint32_t in1_CKtNt = KV_HEADS * in1_KtNt;

    // ---- Matmul/blocking params ----
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t in1_num_subblocks = 1;
    const uint32_t out_block_w = operation_attributes.out_subblock_w;
    const uint32_t in0_block_w = Kt;
    const uint32_t in0_block_num_tiles = in0_block_w * out_subblock_h;
    const uint32_t in0_subblock_num_tiles = in0_block_num_tiles;
    const uint32_t in1_block_num_tiles_per_kv_heads = in0_block_w * operation_attributes.out_subblock_w;
    const uint32_t in1_block_num_tiles = KV_HEADS * in1_block_num_tiles_per_kv_heads;
    const uint32_t in1_num_blocks = ((Nt - 1) / out_block_w) + 1;
    const uint32_t out_subblock_num_tiles = out_subblock_h * out_block_w;
    const uint32_t intermediate_num_tiles = out_subblock_num_tiles;
    const uint32_t in1_per_core_w = in1_num_subblocks * out_block_w;
    const uint32_t in1_block_w_tile_bytes = operation_attributes.out_subblock_w * in1_single_tile_size;
    const uint32_t ONE_ROW_BFLOAT16_BYTES =
        fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32 ? 128u : 64u;
    const uint32_t bfloat16_row_bytes = ONE_ROW_BFLOAT16_BYTES * out_block_w;

    // ---- Work distribution ----
    uint32_t num_active_cores = std::max(Q_HEADS, TILE_HEIGHT);
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_blocks_per_core_group_1,
         num_output_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(
                operation_attributes.compute_with_storage_grid_size, num_active_cores, operation_attributes.row_major);
    TT_FATAL(
        num_output_blocks_per_core_group_1 == 1 and num_output_blocks_per_core_group_2 == 0,
        "Group attention matmul only supports one q_heads per core. Increase compute grid size to at least have as "
        "many cores as q_heads!");

    // ---- Mcast topology (first 32 cores are senders) ----
    CoreRangeSet mcast_receiver_cores = num_cores_to_corerangeset(
        Q_HEADS, operation_attributes.compute_with_storage_grid_size, operation_attributes.row_major);
    CoreRange mcast_receiver_cores_bounding_box = mcast_receiver_cores.bounding_box();
    uint32_t mcast_num_dests = mcast_receiver_cores.num_cores();
    uint32_t mcast_num_cores = mcast_receiver_cores_bounding_box.size();
    CoreCoord top_left_core = mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = mcast_receiver_cores_bounding_box.end_coord;
    CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    CoreCoord mcast_sender_grid =
        ((CoreRangeSet)num_cores_to_corerangeset(
             TILE_HEIGHT, operation_attributes.compute_with_storage_grid_size, operation_attributes.row_major))
            .bounding_box()
            .grid_size();
    std::vector<uint32_t> in1_mcast_sender_noc_x;
    std::vector<uint32_t> in1_mcast_sender_noc_y;
    in1_mcast_sender_noc_x.reserve(mcast_sender_grid.x);
    in1_mcast_sender_noc_y.reserve(mcast_sender_grid.y);
    for (uint32_t core_idx_x = 0; core_idx_x < mcast_sender_grid.x; ++core_idx_x) {
        in1_mcast_sender_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
    }
    for (uint32_t core_idx_y = 0; core_idx_y < mcast_sender_grid.y; ++core_idx_y) {
        in1_mcast_sender_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
    }

    // Reader NOC direction drives the mcast dest-coord order below. On Gen1 this resolves to NOC_0
    // (== the reader DM-config default the helper produces), so the two stay consistent.
    const bool reader_noc_is_NOC_0 =
        tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch()) == tt::tt_metal::NOC::NOC_0;

    const bool in0_is_sharded = a.is_sharded();
    const bool in1_is_sharded = b.is_sharded();
    const bool output_is_sharded = output.is_sharded();

    // ============================ Metal 2.0 spec construction ============================

    // ---- Semaphores (program-scope; reader-only). Initial value 0 == legacy INVALID. ----
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{.unique_id = SENDER_SEM, .target_nodes = all_device_cores},
        SemaphoreSpec{.unique_id = RECEIVER_SEM, .target_nodes = all_device_cores},
    };

    // ---- Tensor parameters (always declared + supplied; used via TensorBinding on the interleaved
    //      path and via DFB borrowed_from on the sharded path). ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = T_IN0, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = T_IN1, .spec = b.tensor_spec()},
        TensorParameter{.unique_id = T_OUT, .spec = output.tensor_spec()},
    };

    // ---- Dataflow buffers (one per legacy CB; entry_size = page_size, num_entries =
    //      total_size / page_size). Sharded variants borrow their backing tensor's memory. ----
    Group<DataflowBufferSpec> dataflow_buffers;

    {  // c_0 in0
        const uint32_t cb0_num_input_tiles = in0_is_sharded ? (a.shard_spec().value().numel() / TILE_HW) : in0_block_w;
        DataflowBufferSpec dfb{
            .unique_id = IN0,
            .entry_size = in0_single_tile_size,
            .num_entries = cb0_num_input_tiles,
            .data_format_metadata = in0_data_format,
        };
        if (in0_is_sharded) {
            dfb.borrowed_from = T_IN0;
        }
        dataflow_buffers.push_back(dfb);
    }

    dataflow_buffers.push_back(DataflowBufferSpec{
        // c_1 in1
        .unique_id = IN1,
        .entry_size = in1_single_tile_size,
        .num_entries = 2 * in1_block_num_tiles,
        .data_format_metadata = in1_data_format,
    });

    if (in1_is_sharded) {  // c_2 in1-sharded borrowed CB (self-loop; IN1_SHARDED only)
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN2,
            .entry_size = in1_single_tile_size,
            .num_entries = b.shard_spec().value().numel() / TILE_HW,
            .data_format_metadata = in1_data_format,
            .borrowed_from = T_IN1,
        });
    }

    dataflow_buffers.push_back(DataflowBufferSpec{
        // c_3 intermed0
        .unique_id = INTERMED0,
        .entry_size = interm_single_tile_size,
        .num_entries = 2 * intermediate_num_tiles,
        .data_format_metadata = interm_data_format,
    });

    dataflow_buffers.push_back(DataflowBufferSpec{
        // c_4 intermed1
        .unique_id = INTERMED1,
        .entry_size = interm_single_tile_size,
        .num_entries = MtNt,
        .data_format_metadata = interm_data_format,
    });

    {  // c_5 out
        const uint32_t num_output_tiles = output_is_sharded ? (output.shard_spec().value().numel() / TILE_HW) : MtNt;
        DataflowBufferSpec dfb{
            .unique_id = OUT,
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_data_format,
        };
        if (output_is_sharded) {
            dfb.borrowed_from = T_OUT;
        }
        dataflow_buffers.push_back(dfb);
    }

    // ---- Reader KernelSpec ----
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    KernelSpec::CompilerOptions::Defines reader_defines;
    Group<TensorBinding> reader_tensor_bindings;
    if (in1_is_sharded) {
        // c_2 is touched only by the reader (raw get_read_ptr) → self-loop (PRODUCER + CONSUMER).
        reader_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = IN2, .accessor_name = "in2", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = IN2, .accessor_name = "in2", .endpoint_type = DFBEndpointType::CONSUMER});
        reader_defines.emplace("IN1_SHARDED", "1");
    } else {
        // Interleaved: the reader reads in1 through a TensorAccessor.
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = T_IN1, .accessor_name = "src1"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/"
            "reader_mcast_transformer_group_attn_matmul.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{.semaphore_spec_name = SENDER_SEM, .accessor_name = "sender_sem"},
                SemaphoreBinding{.semaphore_spec_name = RECEIVER_SEM, .accessor_name = "receiver_sem"},
            },
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args =
            {
                {"transpose_hw", static_cast<uint32_t>(transpose_hw_bool)},
                {"row_major", static_cast<uint32_t>(operation_attributes.row_major)},
                {"out_subblock_w", operation_attributes.out_subblock_w},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"has_work_for_mcast_kv_heads",
                  "has_work_for_q_heads",
                  "Mt",
                  "Nt",
                  "num_kv_heads",
                  "in1_CKtNt",
                  "in1_CKtNt_mul_32",
                  "blocks",
                  "in1_start_id",
                  "in0_block_w",
                  "out_block_w",
                  "in1_num_subblocks",
                  "in1_num_blocks",
                  "in1_block_num_tiles",
                  "Nt_bytes",
                  "in1_block_w_tile_bytes",
                  "out_last_subblock_w",
                  "in1_last_block_w_tile_read_bytes",
                  "in1_last_block_addr_skip",
                  "in1_mcast_dest_noc_start_x",
                  "in1_mcast_dest_noc_start_y",
                  "in1_mcast_dest_noc_end_x",
                  "in1_mcast_dest_noc_end_y",
                  "in1_mcast_num_dests",
                  "in1_mcast_num_cores",
                  "in1_mcast_grid_size",
                  "in1_mcast_sender_size_bytes",
                  "in1_mcast_sender_id",
                  "in1_mcast_sender_num_x",
                  "in1_mcast_sender_num_y"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options =
            {.num_runtime_varargs =
                 static_cast<uint32_t>(in1_mcast_sender_noc_x.size() + in1_mcast_sender_noc_y.size())},
    };

    // ---- Writer KernelSpec ----
    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = INTERMED0, .accessor_name = "intermed0", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = INTERMED1, .accessor_name = "intermed1", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    KernelSpec::CompilerOptions::Defines writer_defines;
    Group<TensorBinding> writer_tensor_bindings;
    if (in0_is_sharded) {
        writer_defines.emplace("IN0_SHARDED", "1");
    } else {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = T_IN0, .accessor_name = "src0"});
    }
    if (output_is_sharded) {
        writer_defines.emplace("OUT_SHARDED", "1");
    } else {
        writer_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = T_OUT, .accessor_name = "dst"});
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/"
            "writer_transformer_group_attn_matmul.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = std::move(writer_dfb_bindings),
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args =
            {
                {"out_subblock_w", operation_attributes.out_subblock_w},
                {"intermediate_num_tiles", intermediate_num_tiles},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"has_work_for_q_heads",
                  "Mt",
                  "Kt",
                  "Nt",
                  "MtKt",
                  "blocks",
                  "in0_start_id",
                  "out_start_id",
                  "in0_block_w",
                  "in1_num_subblocks",
                  "in1_num_blocks",
                  "out_num_tiles",
                  "bfloat16_row_bytes",
                  "bfloat16_Nt_bytes",
                  "bfloat16_last_row_bytes_read"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute KernelSpec ----
    // Legacy ComputeConfigDescriptor set ONLY math_fidelity + fp32_dest_acc_en, leaving
    // math_approx_mode / dst_full_sync_en / bfp8_pack_precise at their `false` defaults. Mirror that
    // resolved descriptor: ComputeGen1Config's defaults (sfpu_precision_mode=Precise,
    // double_buffer_dest=true, bfp_pack_precision_mode=Approximate) already match, so only the two
    // legacy-set fields are overridden. (Deliberately not using to_compute_hardware_config, which
    // would map the config's math_approx_mode/dst_full_sync_en — knobs this op does not wire through.)
    ComputeGen1Config compute_gen1{
        .fpu_math_fidelity = math_fidelity,
        .enable_32_bit_dest = fp32_dest_acc_en,
    };
    // Metal 2.0 validator requires an explicit unpack_modes entry for each Float32 DFB the compute
    // kernel *consumes* when enable_32_bit_dest is set (legacy defaulted silently; empty
    // unpack_to_dest_mode == all Default == UnpackToSrc). Compute consumes IN0, IN1, INTERMED1.
    if (fp32_dest_acc_en) {
        if (in0_data_format == tt::DataFormat::Float32) {
            compute_gen1.unpack_modes.insert({IN0, UnpackMode::UnpackToSrc});
        }
        if (in1_data_format == tt::DataFormat::Float32) {
            compute_gen1.unpack_modes.insert({IN1, UnpackMode::UnpackToSrc});
        }
        if (interm_data_format == tt::DataFormat::Float32) {
            compute_gen1.unpack_modes.insert({INTERMED1, UnpackMode::UnpackToSrc});
        }
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/"
            "transformer_group_attn_matmul.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = INTERMED0,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INTERMED1,
                    .accessor_name = "intermed1",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {
                {"transpose_hw", static_cast<uint32_t>(transpose_hw_bool)},
                {"out_subblock_w", operation_attributes.out_subblock_w},
                {"out_subblock_num_tiles", out_subblock_num_tiles},
                {"intermediate_num_tiles", intermediate_num_tiles},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"has_work_for_q_heads",
                  "batch",
                  "Mt",
                  "num_kv_heads_skip",
                  "num_kv_heads_remaining",
                  "in0_block_w",
                  "out_subblock_h",
                  "in1_num_subblocks",
                  "in1_num_blocks",
                  "in0_block_num_tiles",
                  "in1_block_num_tiles",
                  "out_num_tiles",
                  "in0_subblock_num_tiles",
                  "in1_per_core_w"}},
        .hw_config = ComputeHardwareConfig{compute_gen1},
    };

    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "group_attn_matmul", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_device_cores},
    };

    ProgramSpec spec{
        .name = "group_attn_matmul",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    // ---- Per-core runtime args (name-first tables via AddRuntimeArgsForNode) ----
    uint32_t Nt_bytes = Nt * in1_single_tile_size;
    uint32_t out_last_subblock_w = Nt % out_block_w == 0 ? out_block_w : Nt % out_block_w;
    uint32_t in1_last_block_w_tile_read_bytes = out_last_subblock_w * in1_single_tile_size;
    uint32_t in1_last_block_addr_skip =
        (operation_attributes.out_subblock_w - out_last_subblock_w) * in1_single_tile_size;
    uint32_t bfloat16_Nt_bytes = ONE_ROW_BFLOAT16_BYTES * Nt;
    uint32_t bfloat16_last_row_bytes_read = ONE_ROW_BFLOAT16_BYTES * out_last_subblock_w;

    CoreRange all_cores_bounding_box = all_cores.bounding_box();
    std::vector<CoreCoord> cores = grid_to_cores_with_noop(
        all_cores_bounding_box.end_coord.x,
        all_cores_bounding_box.end_coord.y,
        device_compute_with_storage_grid.x,
        device_compute_with_storage_grid.y,
        operation_attributes.row_major);
    uint32_t g1_numcores = core_group_1.num_cores();

    // The mcast sender noc-x/-y block is identical on every node (a mcast sender-grid constant);
    // ported as runtime varargs (x-block then y-block). See report: RTA->CRTA opportunity.
    std::vector<uint32_t> mcast_sender_noc_varargs;
    mcast_sender_noc_varargs.reserve(in1_mcast_sender_noc_x.size() + in1_mcast_sender_noc_y.size());
    mcast_sender_noc_varargs.insert(
        mcast_sender_noc_varargs.end(), in1_mcast_sender_noc_x.begin(), in1_mcast_sender_noc_x.end());
    mcast_sender_noc_varargs.insert(
        mcast_sender_noc_varargs.end(), in1_mcast_sender_noc_y.begin(), in1_mcast_sender_noc_y.end());

    KernelRunArgs reader_kra{.kernel = READER};
    KernelRunArgs writer_kra{.kernel = WRITER};
    KernelRunArgs compute_kra{.kernel = COMPUTE};

    uint32_t num_blocks_written = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t num_output_blocks_per_core =
            i < num_cores ? (i < g1_numcores ? num_output_blocks_per_core_group_1 : num_output_blocks_per_core_group_2)
                          : 0;

        const uint32_t kv_heads_id = KV_HEADS == 0 ? 0u : i / std::max<uint32_t>(Q_HEADS / KV_HEADS, 1u);
        const uint32_t has_work_for_q_heads = i < Q_HEADS;
        const uint32_t has_work_for_mcast_kv_heads = i < num_active_cores;
        const uint32_t mcast_num_cores_for_core = mcast_num_cores - static_cast<uint32_t>(i < mcast_num_cores);
        const uint32_t in1_mcast_num_dests =
            Q_HEADS < TILE_HEIGHT ? std::min(mcast_num_cores_for_core, num_active_cores - 1) : mcast_num_dests - 1;

        AddRuntimeArgsForNode(
            reader_kra.runtime_arg_values,
            core,
            {{"has_work_for_mcast_kv_heads", has_work_for_mcast_kv_heads},
             {"has_work_for_q_heads", has_work_for_q_heads},
             {"Mt", Mt},
             {"Nt", Nt},
             {"num_kv_heads", KV_HEADS},
             {"in1_CKtNt", in1_CKtNt},
             {"in1_CKtNt_mul_32", in1_CKtNt * TILE_HEIGHT},
             {"blocks", num_output_blocks_per_core},
             {"in1_start_id", 0u},
             {"in0_block_w", in0_block_w},
             {"out_block_w", out_block_w},
             {"in1_num_subblocks", in1_num_subblocks},
             {"in1_num_blocks", in1_num_blocks},
             {"in1_block_num_tiles", in1_block_num_tiles},
             {"Nt_bytes", Nt_bytes},
             {"in1_block_w_tile_bytes", in1_block_w_tile_bytes},
             {"out_last_subblock_w", out_last_subblock_w},
             {"in1_last_block_w_tile_read_bytes", in1_last_block_w_tile_read_bytes},
             {"in1_last_block_addr_skip", in1_last_block_addr_skip},
             {"in1_mcast_dest_noc_start_x",
              static_cast<uint32_t>(reader_noc_is_NOC_0 ? top_left_core_physical.x : bottom_right_core_physical.x)},
             {"in1_mcast_dest_noc_start_y",
              static_cast<uint32_t>(reader_noc_is_NOC_0 ? top_left_core_physical.y : bottom_right_core_physical.y)},
             {"in1_mcast_dest_noc_end_x",
              static_cast<uint32_t>(reader_noc_is_NOC_0 ? bottom_right_core_physical.x : top_left_core_physical.x)},
             {"in1_mcast_dest_noc_end_y",
              static_cast<uint32_t>(reader_noc_is_NOC_0 ? bottom_right_core_physical.y : top_left_core_physical.y)},
             {"in1_mcast_num_dests", in1_mcast_num_dests},
             {"in1_mcast_num_cores", mcast_num_cores_for_core},
             {"in1_mcast_grid_size", mcast_num_cores},
             {"in1_mcast_sender_size_bytes", in1_block_num_tiles * in1_single_tile_size},
             {"in1_mcast_sender_id", i},
             {"in1_mcast_sender_num_x", static_cast<uint32_t>(in1_mcast_sender_noc_x.size())},
             {"in1_mcast_sender_num_y", static_cast<uint32_t>(in1_mcast_sender_noc_y.size())}});
        reader_kra.advanced_options.runtime_varargs[core] = mcast_sender_noc_varargs;

        AddRuntimeArgsForNode(
            writer_kra.runtime_arg_values,
            core,
            {{"has_work_for_q_heads", has_work_for_q_heads},
             {"Mt", Mt},
             {"Kt", Kt},
             {"Nt", Nt},
             {"MtKt", MtKt},
             {"blocks", num_output_blocks_per_core},
             {"in0_start_id", num_blocks_written * MtKt},
             {"out_start_id", num_blocks_written * MtNt},
             {"in0_block_w", in0_block_w},
             {"in1_num_subblocks", in1_num_subblocks},
             {"in1_num_blocks", in1_num_blocks},
             {"out_num_tiles", MtNt},
             {"bfloat16_row_bytes", bfloat16_row_bytes},
             {"bfloat16_Nt_bytes", bfloat16_Nt_bytes},
             {"bfloat16_last_row_bytes_read", bfloat16_last_row_bytes_read}});

        AddRuntimeArgsForNode(
            compute_kra.runtime_arg_values,
            core,
            {{"has_work_for_q_heads", has_work_for_q_heads},
             {"batch", num_output_blocks_per_core},
             {"Mt", Mt},
             {"num_kv_heads_skip", kv_heads_id * in1_block_num_tiles_per_kv_heads},
             {"num_kv_heads_remaining", (KV_HEADS - kv_heads_id) * in1_block_num_tiles_per_kv_heads},
             {"in0_block_w", in0_block_w},
             {"out_subblock_h", out_subblock_h},
             {"in1_num_subblocks", in1_num_subblocks},
             {"in1_num_blocks", in1_num_blocks},
             {"in0_block_num_tiles", in0_block_num_tiles},
             {"in1_block_num_tiles", in1_block_num_tiles},
             {"out_num_tiles", MtNt},
             {"in0_subblock_num_tiles", in0_subblock_num_tiles},
             {"in1_per_core_w", in1_per_core_w}});

        num_blocks_written += num_output_blocks_per_core;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_kra), std::move(writer_kra), std::move(compute_kra)};
    run_args.tensor_args = {{T_IN0, a}, {T_IN1, b}, {T_OUT, output}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
