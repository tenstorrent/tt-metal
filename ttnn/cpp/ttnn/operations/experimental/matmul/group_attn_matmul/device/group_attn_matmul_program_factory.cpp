// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace group_attn_matmul_metal2 {
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};

const DFBSpecName IN0{"in0"};
const DFBSpecName IN1{"in1"};
const DFBSpecName IN2{"in2"};
const DFBSpecName INTERMED0{"intermed0"};
const DFBSpecName INTERMED1{"intermed1"};
const DFBSpecName OUT{"out"};

const SemaphoreSpecName IN1_MCAST_SENDER_SEM{"in1_mcast_sender"};
const SemaphoreSpecName IN1_MCAST_RECEIVER_SEM{"in1_mcast_receiver"};

const TensorParamName INPUT_A{"input_a"};
const TensorParamName INPUT_B{"input_b"};
const TensorParamName OUTPUT{"output"};
}  // namespace group_attn_matmul_metal2

ttnn::device_operation::ProgramArtifacts GroupAttnMatmulProgramFactory::create_program_artifacts(
    const GroupAttnMatmulParams& operation_attributes,
    const GroupAttnMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace group_attn_matmul_metal2;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    const auto& a_mesh = a.mesh_tensor();
    const auto& b_mesh = b.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::tt_metal::distributed::MeshDevice* device = a.device();

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

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // Kernels, DFBs and semaphores span the whole device compute grid; the per-core flags below
    // steer the cores that have no work.
    CoreCoord device_compute_with_storage_grid = device->compute_with_storage_grid_size();
    CoreRangeSet all_device_cores{
        CoreRange({0, 0}, {device_compute_with_storage_grid.x - 1, device_compute_with_storage_grid.y - 1})};

    // ---- Shape-derived quantities ----
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
    // The sender NOC coordinate tables ride as runtime varargs, x entries first then y entries; the
    // reader indexes them by the two named counts.
    const uint32_t in1_mcast_sender_num_x = mcast_sender_grid.x;
    const uint32_t in1_mcast_sender_num_y = mcast_sender_grid.y;
    std::vector<uint32_t> in1_mcast_sender_noc_varargs;
    in1_mcast_sender_noc_varargs.reserve(in1_mcast_sender_num_x + in1_mcast_sender_num_y);
    for (uint32_t core_idx_x = 0; core_idx_x < in1_mcast_sender_num_x; ++core_idx_x) {
        in1_mcast_sender_noc_varargs.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
    }
    for (uint32_t core_idx_y = 0; core_idx_y < in1_mcast_sender_num_y; ++core_idx_y) {
        in1_mcast_sender_noc_varargs.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
    }

    // ---- Semaphores (on every device core: receivers increment the sender's, senders multicast
    // the receiver's, so both live at one address on every node) ----
    SemaphoreSpec in1_mcast_sender_sem{
        .unique_id = IN1_MCAST_SENDER_SEM,
        .target_nodes = all_device_cores,
    };
    SemaphoreSpec in1_mcast_receiver_sem{
        .unique_id = IN1_MCAST_RECEIVER_SEM,
        .target_nodes = all_device_cores,
    };

    // ---- Dataflow buffers (shape-dependent capacities are safe in the program cache: the default
    // reflection hash keys on the full tensor specs; a sharded operand's DFB borrows its L1 shard) ----
    const bool in0_is_sharded = a.is_sharded();
    const bool in1_is_sharded = b.is_sharded();
    const bool output_is_sharded = output.is_sharded();

    DataflowBufferSpec dfb_in0{
        .unique_id = IN0,
        .entry_size = in0_single_tile_size,
        .num_entries = in0_is_sharded ? (a.shard_spec().value().numel() / TILE_HW) : in0_block_w,
        .data_format_metadata = in0_data_format,
    };
    if (in0_is_sharded) {
        dfb_in0.borrowed_from = INPUT_A;
    }

    DataflowBufferSpec dfb_in1{
        .unique_id = IN1,
        .entry_size = in1_single_tile_size,
        .num_entries = 2 * in1_block_num_tiles,
        .data_format_metadata = in1_data_format,
    };

    // The resident in1 shard, read in place by the reader; exists only when in1 is sharded.
    std::optional<DataflowBufferSpec> dfb_in2;
    if (in1_is_sharded) {
        dfb_in2 = DataflowBufferSpec{
            .unique_id = IN2,
            .entry_size = in1_single_tile_size,
            .num_entries = b.shard_spec().value().numel() / TILE_HW,
            .data_format_metadata = in1_data_format,
            .borrowed_from = INPUT_B,
        };
    }

    DataflowBufferSpec dfb_intermed0{
        .unique_id = INTERMED0,
        .entry_size = interm_single_tile_size,
        .num_entries = 2 * intermediate_num_tiles,
        .data_format_metadata = interm_data_format,
    };

    DataflowBufferSpec dfb_intermed1{
        .unique_id = INTERMED1,
        .entry_size = interm_single_tile_size,
        .num_entries = MtNt,
        .data_format_metadata = interm_data_format,
    };

    DataflowBufferSpec dfb_out{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = output_is_sharded ? (output.shard_spec().value().numel() / TILE_HW) : MtNt,
        .data_format_metadata = output_data_format,
    };
    if (output_is_sharded) {
        dfb_out.borrowed_from = OUTPUT;
    }

    // ---- Compute hardware config ----
    ComputeHardwareConfig compute_hw = to_compute_hardware_config(operation_attributes.compute_kernel_config);
    compute_hw.sfpu_precision_mode = Precision::Precise;
    compute_hw.double_buffer_dest = true;
    if (compute_hw.enable_32_bit_dest) {
        if (in0_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(IN0, UnpackMode::UnpackToSrc);
        }
        if (in1_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(IN1, UnpackMode::UnpackToSrc);
        }
        if (interm_data_format == tt::DataFormat::Float32) {
            compute_hw.unpack_modes.emplace(INTERMED1, UnpackMode::UnpackToSrc);
        }
    }

    // ---- Kernels ----
    // The reader takes the DRAM-read NOC (NOC_0 on every 1xx arch) and the writer the other one; the
    // mcast rectangle below is addressed in the orientation of the NOC the reader actually gets.
    const DataMovementHardwareConfig reader_hw_config = ttnn::create_reader_datamovement_config();
    const DataMovementHardwareConfig writer_hw_config = ttnn::create_writer_datamovement_config();
    const bool reader_noc_is_NOC_0 = reader_hw_config.config_1xx->noc == tt::tt_metal::NOC::NOC_0;

    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (in0_is_sharded) {
        writer_defines.emplace("IN0_SHARDED", "1");
    }
    if (in1_is_sharded) {
        reader_defines.emplace("IN1_SHARDED", "1");
    }
    if (output_is_sharded) {
        writer_defines.emplace("OUT_SHARDED", "1");
    }

    // All three kernels run on every device core, as before; the flags steer the cores without work.
    // On the mcast-only cores (q_heads..31 when q_heads < 32) compute and writer return at once and
    // the reader both pushes and pops in1 to keep its write pointer in step with the mcast grid; it
    // stays in1's producer there (the kernel-side FIFO ops do not depend on the binding's role on WH/BH).
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/"
            "reader_mcast_transformer_group_attn_matmul.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {
                // mcast receive all kv_heads; compute chooses which kv_heads to use for matmul
                DFBBinding{
                    .dfb_spec_name = IN1,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .semaphore_bindings =
            {
                SemaphoreBinding{
                    .semaphore_spec_name = IN1_MCAST_SENDER_SEM,
                    .accessor_name = "in1_mcast_sender",
                },
                SemaphoreBinding{
                    .semaphore_spec_name = IN1_MCAST_RECEIVER_SEM,
                    .accessor_name = "in1_mcast_receiver",
                },
            },
        .compile_time_args =
            {
                {"transpose_hw", static_cast<uint32_t>(transpose_hw_bool)},
                {"row_major", static_cast<uint32_t>(operation_attributes.row_major)},
                {"out_subblock_w", operation_attributes.out_subblock_w},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
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
                     "in1_mcast_sender_num_y"},
            },
        .hw_config = reader_hw_config,
        .advanced_options = {.num_runtime_varargs = in1_mcast_sender_num_x + in1_mcast_sender_num_y},
    };
    if (in1_is_sharded) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = IN2,
            .accessor_name = "in2",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = IN2,
            .accessor_name = "in2",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    } else {
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = INPUT_B,
            .accessor_name = "src1",
        });
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/"
            "writer_transformer_group_attn_matmul.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The writer copies one row out of each untilized subblock (intermed0, from compute)
                // into the row-major tile it assembles for compute (intermed1).
                DFBBinding{
                    .dfb_spec_name = INTERMED0,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED1,
                    .accessor_name = "intermed1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"out_subblock_w", operation_attributes.out_subblock_w},
                {"intermediate_num_tiles", intermediate_num_tiles},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
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
                     "bfloat16_last_row_bytes_read"},
            },
        .hw_config = writer_hw_config,
    };
    if (!in0_is_sharded) {
        writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = INPUT_A,
            .accessor_name = "src0",
        });
    }
    if (!output_is_sharded) {
        writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "dst",
        });
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/"
            "transformer_group_attn_matmul.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED0,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED1,
                    .accessor_name = "intermed1",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"transpose_hw", static_cast<uint32_t>(transpose_hw_bool)},
                {"out_subblock_w", operation_attributes.out_subblock_w},
                {"out_subblock_num_tiles", out_subblock_num_tiles},
                {"intermediate_num_tiles", intermediate_num_tiles},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
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
                     "in1_per_core_w"},
            },
        .hw_config = compute_hw,
    };

    // ---- Per-core runtime args ----
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

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

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
            reader_run_args.runtime_arg_values,
            core,
            {
                {"has_work_for_mcast_kv_heads", has_work_for_mcast_kv_heads},
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
                {"in1_mcast_sender_num_x", in1_mcast_sender_num_x},
                {"in1_mcast_sender_num_y", in1_mcast_sender_num_y},
            });
        reader_run_args.advanced_options.runtime_varargs[core] = in1_mcast_sender_noc_varargs;

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"has_work_for_q_heads", has_work_for_q_heads},
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
                {"bfloat16_last_row_bytes_read", bfloat16_last_row_bytes_read},
            });

        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {
                {"has_work_for_q_heads", has_work_for_q_heads},
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
                {"in1_per_core_w", in1_per_core_w},
            });

        num_blocks_written += num_output_blocks_per_core;
    }

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(std::move(dfb_in0));
    dataflow_buffers.push_back(std::move(dfb_in1));
    if (dfb_in2.has_value()) {
        dataflow_buffers.push_back(std::move(*dfb_in2));
    }
    dataflow_buffers.push_back(std::move(dfb_intermed0));
    dataflow_buffers.push_back(std::move(dfb_intermed1));
    dataflow_buffers.push_back(std::move(dfb_out));

    ProgramSpec spec{
        .name = "group_attn_matmul",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = {std::move(in1_mcast_sender_sem), std::move(in1_mcast_receiver_sem)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT_A, .spec = a.tensor_spec()},
                TensorParameter{.unique_id = INPUT_B, .spec = b.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER, COMPUTE},
                    .target_nodes = all_device_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {
        {INPUT_A, a_mesh},
        {INPUT_B, b_mesh},
        {OUTPUT, output_mesh},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
