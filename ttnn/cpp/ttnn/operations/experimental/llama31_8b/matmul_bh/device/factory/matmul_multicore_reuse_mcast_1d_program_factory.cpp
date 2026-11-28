// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/mcast_1d.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_l1.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/kernel_lib/mcast/host/mcast.hpp"

using namespace tt;

using tt::tt_metal::MeshTensor;

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementHardwareConfig;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim::llama31_8b_matmul_bh::mcast_1d {

static void reserve_multicast_tail(
    Group<KernelSpec>& kernels,
    Group<WorkUnitSpec>& work_units,
    const CoreRangeSet& idle_cores,
    const DFBSpecName& dfb,
    const DataMovementHardwareConfig& dm_config) {
    if (idle_cores.num_cores() == 0) {
        return;
    }
    const KernelSpecName producer{"mcast_tail_producer"};
    const KernelSpecName consumer{"mcast_tail_consumer"};
    kernels.push_back(KernelSpec{
        .unique_id = producer,
        .source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                  "mcast_tail_reservation.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = dfb, .accessor_name = "reserved", .endpoint_type = DFBEndpointType::PRODUCER}},
        .hw_config = dm_config,
    });
    kernels.push_back(KernelSpec{
        .unique_id = consumer,
        .source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/compute/"
                  "mcast_tail_reservation.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = dfb, .accessor_name = "reserved", .endpoint_type = DFBEndpointType::CONSUMER}},
        .hw_config = tt::tt_metal::experimental::ComputeHardwareConfig{},
    });
    work_units.push_back(
        WorkUnitSpec{.name = "multicast_tail_storage", .kernels = {producer, consumer}, .target_nodes = idle_cores});
}

static CoreRangeSet rect_set(const Mcast1DRect* rects, uint32_t n) {
    std::vector<CoreRange> ranges;
    ranges.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        ranges.emplace_back(CoreCoord(rects[i].x0, rects[i].y0), CoreCoord(rects[i].x1, rects[i].y1));
    }
    return CoreRangeSet(ranges);
}

ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const Mcast1DPlan& plan) {
    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const MeshTensor& in0_tensor = a.mesh_tensor();
    const MeshTensor& in1_tensor = b.mesh_tensor();
    const MeshTensor& out_tensor = tensor_return_value.at(0).mesh_tensor();
    TT_FATAL(
        !tensor_args.optional_input_tensors.at(0).has_value() && !operation_attributes.untilize_out &&
            !operation_attributes.transpose_a && !operation_attributes.transpose_b &&
            operation_attributes.bcast_batch.has_value() && operation_attributes.compute_kernel_config.has_value() &&
            !in0_tensor.memory_config().is_sharded() && !out_tensor.memory_config().is_sharded(),
        "llama31_8b matmul_bh: the 1D table rows take an interleaved activation and output, no bias, no transpose");
    const uint32_t bcast_batch = operation_attributes.bcast_batch.value() ? 1u : 0u;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const tt_metal::distributed::MeshDevice& device = in0_tensor.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device.arch(), compute_kernel_config);

    const auto in0_tile = a.tensor_spec().tile();
    const auto in1_tile = b.tensor_spec().tile();
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    const tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(in0_tensor.dtype());
    const tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_tensor.dtype());
    const tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(out_tensor.dtype());
    const tt::DataFormat interm0_data_format = plan.interm0_format == 2   ? tt::DataFormat::Float32
                                               : plan.interm0_format == 1 ? tt::DataFormat::Float16_b
                                                                          : output_data_format;
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t in0_tile_size = tt::align(in0_tile.get_tile_size(in0_data_format), dram_alignment);
    const uint32_t in1_tile_size = tt::align(in1_tile.get_tile_size(in1_data_format), dram_alignment);
    const uint32_t output_tile_size = output_tile.get_tile_size(output_data_format);
    const uint32_t interm0_tile_size = output_tile.get_tile_size(interm0_data_format);
    TT_FATAL(output_tile_size == plan.out_tile_bytes, "llama31_8b matmul_bh: output tile size not in the table row");

    const KernelSpecName IN0_SENDER{"in0_sender"};
    const KernelSpecName IN0_RECEIVER{"in0_receiver"};
    const KernelSpecName IN1_SENDER_WRITER{"in1_sender_writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERM0_DFB{"intermed0"};
    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    if (plan.packer_l1_acc) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device.arch(), plan.num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device.arch(), plan.num_cores, mm_kernel_defines, ttnn::get_throttle_level(compute_kernel_config));
    if (plan.in0_group > 1) {
        mm_kernel_in0_sender_writer_defines["IN0_BLOCK_GROUP"] = std::to_string(plan.in0_group);
    }
    if (plan.in1_coalesced_read) {
        mm_kernel_in1_sender_writer_defines["IN1_COALESCED_READ"] = "1";
    }
    if (plan.out_flush_per_block) {
        mm_kernel_in1_sender_writer_defines["OUT_FLUSH_PER_BLOCK"] = "1";
    }

    const tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device.arch());
    const tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device.arch());

    const bool separate_out_and_interm0 = interm0_data_format != output_data_format;
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0_DFB,
        .entry_size = in0_tile_size,
        .num_entries = plan.in0_dfb_tiles,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN1_DFB,
        .entry_size = in1_tile_size,
        .num_entries = plan.in1_dfb_tiles,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_tile_size,
        .num_entries = plan.out_dfb_tiles,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
        .advanced_options =
            {.alias_with = separate_out_and_interm0 ? Group<DFBSpecName>{} : Group<DFBSpecName>{INTERM0_DFB}},
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INTERM0_DFB,
        .entry_size = interm0_tile_size,
        .num_entries = (separate_out_and_interm0 ? plan.interm0_dfb_tiles * interm0_tile_size
                                                 : plan.out_dfb_tiles * output_tile_size) /
                       interm0_tile_size,
        .data_format_metadata = interm0_data_format,
        .tile_format_metadata = output_tile,
        .advanced_options =
            {.alias_with = separate_out_and_interm0 ? Group<DFBSpecName>{} : Group<DFBSpecName>{OUT_DFB}},
    });

    TT_FATAL(
        dfb_l1_bytes(dataflow_buffers) == l1_bytes_1d(plan, a, b, out_tensor.dtype()),
        "llama31_8b matmul_bh: buffer plan and L1 check disagree");

    const auto in0_sender_hw_config = DataMovementHardwareConfig{
        .config_1xx =
            DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
            },
        .config_2xx =
            DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true,
            },
    };
    const auto in1_sender_hw_config = DataMovementHardwareConfig{
        .config_1xx =
            DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
            },
        .config_2xx =
            DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true,
            },
    };

    Group<KernelSpec> kernels;
    kernels.push_back(KernelSpec{
        .unique_id = IN0_SENDER,
        .source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                  "reader_bmm_tile_layout_in0_sender_padding_metal2.cpp",
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in0_sender_writer_defines)},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"}},
        .compile_time_args =
            {
                {"in0_tensor_stride_w", plan.in0_stride_w},
                {"in0_tensor_stride_h", plan.in0_stride_h},
                {"in0_tensor_next_inner_dim_block_stride", plan.in0_next_inner_dim_block_stride},
                {"in0_tensor_next_h_dim_block_stride", plan.in0_next_h_dim_block_stride},
                {"in0_block_w", plan.in0_reader_block_w},
                {"in0_block_h", plan.in0_block_h},
                {"in0_block_num_tiles", plan.in0_reader_block_num_tiles},
                {"in0_last_ktile_w", 0u},
                {"in0_last_ktile_h", 0u},
                {"shard_width_in_tiles", 0u},
                {"shard_height_in_tiles", 0u},
                {"num_blocks_inner_dim", plan.in0_reader_num_blocks_inner},
                {"num_blocks_w_dim", plan.num_blocks_w_dim},
                {"num_blocks_h_dim", plan.num_blocks_h_dim},
                {"MtKt", plan.MtKt},
                {"in0_B", 1u},
                {"in1_B", 1u},
                {"in0_reuse_in_dfb", 0u},
                {"batchB", 0u},
                {"bcast_A", 1u},
                {"get_batch_from_reader", 0u},
                {"num_active", 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"in0_tensor_start_tile_id", "last_block_h"}},
        .hw_config = in0_sender_hw_config,
    });
    const bool has_in0_receiver_kernel = plan.num_receivers > 0;
    if (has_in0_receiver_kernel) {
        kernels.push_back(KernelSpec{
            .unique_id = IN0_RECEIVER,
            .source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                      "reader_bmm_tile_layout_in0_receiver_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {
                    {"in0_block_num_tiles", plan.in0_reader_block_num_tiles},
                    {"num_blocks_inner_dim", plan.in0_reader_num_blocks_inner},
                    {"num_blocks_w_dim", plan.num_blocks_w_dim},
                    {"num_blocks_h_dim", plan.num_blocks_h_dim},
                    {"batch", 1u},
                    {"get_batch_from_reader", 0u},
                },
            .hw_config = in0_sender_hw_config,
        });
    }
    kernels.push_back(KernelSpec{
        .unique_id = IN1_SENDER_WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                  "reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp",
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in1_sender_writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"},
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"},
            },
        .compile_time_args =
            {
                {"in1_tensor_stride_w", plan.in1_stride_w},
                {"in1_tensor_stride_h", plan.in1_stride_h},
                {"in1_tensor_next_block_stride", plan.in1_next_block_stride},
                {"in1_tensor_next_w_dim_block_stride", plan.in1_next_w_dim_block_stride},
                {"in1_block_w", plan.in1_block_w},
                {"in1_block_h", plan.in1_block_h},
                {"in1_block_num_tiles", plan.in1_block_num_tiles},
                {"num_blocks_inner_dim", plan.num_blocks},
                {"num_blocks_w_dim", plan.num_blocks_w_dim},
                {"num_blocks_h_dim", plan.num_blocks_h_dim},
                {"KtNt", plan.KtNt},
                {"batch", 1u},
                {"out_block_num_tiles", plan.out_block_num_tiles},
                {"bcast_B", bcast_batch},
                {"batchB", 0u},
                {"out_tensor_stride_w", 1u},
                {"out_tensor_stride_h", plan.out_stride_h},
                {"out_tensor_next_subblock_stride_w", plan.out_next_subblock_stride_w},
                {"out_tensor_next_subblock_stride_h", plan.out_next_subblock_stride_h},
                {"out_tensor_next_w_dim_block_stride", plan.out_next_w_dim_block_stride},
                {"out_tensor_next_h_dim_block_stride", plan.out_next_h_dim_block_stride},
                {"out_subblock_w", plan.out_subblock_w},
                {"out_subblock_h", plan.out_subblock_h},
                {"out_subblock_tile_count", plan.out_subblock_tile_count},
                {"MtNt", plan.MtNt},
                {"compact_output", 0u},
                {"num_active", 0u},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"in1_tensor_start_tile_id",
                  "out_tensor_start_tile_id",
                  "last_block_w",
                  "out_num_nonzero_subblocks_h",
                  "out_last_subblock_h",
                  "padded_block_tiles_h_skip",
                  "out_num_nonzero_subblocks_w",
                  "out_last_num_nonzero_subblocks_w",
                  "out_last_subblock_w",
                  "padded_subblock_tiles_addr_skip",
                  "padded_block_tiles_w_skip",
                  "last_num_blocks_w_dim"}},
        .hw_config = in1_sender_hw_config,
    });

    auto compute_hw = ttnn::to_compute_hardware_config(compute_kernel_config);
    compute_hw.double_buffer_dest = true;
    if (fp32_dest_acc_en) {
        for (const auto& [name, fmt] : std::initializer_list<std::pair<DFBSpecName, tt::DataFormat>>{
                 {IN0_DFB, in0_data_format}, {IN1_DFB, in1_data_format}, {INTERM0_DFB, interm0_data_format}}) {
            if (fmt == tt::DataFormat::Float32) {
                compute_hw.unpack_modes.emplace(
                    name,
                    name == INTERM0_DFB ? tt::tt_metal::UnpackMode::UnpackToDest
                                        : tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        }
    }
    kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/compute/"
                  "bmm_large_block_zm_fused_bias_activation_metal2.cpp",
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INTERM0_DFB,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INTERM0_DFB,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .compile_time_args =
            {
                {"in0_block_w", plan.in0_block_w},
                {"in0_num_subblocks", plan.in0_num_subblocks},
                {"in0_block_num_tiles", plan.in0_block_num_tiles},
                {"in0_subblock_num_tiles", plan.in0_subblock_num_tiles},
                {"in1_num_subblocks", plan.in1_num_subblocks},
                {"in1_block_num_tiles", plan.in1_compute_block_num_tiles},
                {"in1_block_w", plan.in1_per_core_w},
                {"num_blocks_inner_dim", plan.num_blocks},
                {"num_blocks_w_dim", plan.num_blocks_w_dim},
                {"num_blocks_h_dim", plan.num_blocks_h_dim},
                {"out_subblock_h", plan.out_subblock_h},
                {"out_subblock_w", plan.out_subblock_w},
                {"out_subblock_num_tiles", plan.out_subblock_num_tiles},
                {"batch", 1u},
                {"out_block_num_tiles", plan.out_block_num_tiles},
                {"untilize_out", 0u},
                {"get_batch_from_reader", 0u},
                {"bias_ntiles", plan.in1_per_core_w},
            },
        .hw_config = compute_hw,
    });

    const CoreCoord start_core{0, 0};
    const CoreRangeSet all_cores = rect_set(plan.all_cores, plan.num_all_cores);
    const CoreRange bbox(CoreCoord(plan.bbox.x0, plan.bbox.y0), CoreCoord(plan.bbox.x1, plan.bbox.y1));
    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "in0_senders_with_work",
        .kernels = {IN0_SENDER, IN1_SENDER_WRITER, COMPUTE},
        .target_nodes = CoreRangeSet(CoreRange(start_core, start_core)),
    });
    if (has_in0_receiver_kernel) {
        work_units.push_back(WorkUnitSpec{
            .name = "in0_receivers",
            .kernels = {IN0_RECEIVER, IN1_SENDER_WRITER, COMPUTE},
            .target_nodes = rect_set(plan.receivers, plan.num_receivers),
        });
    }
    reserve_multicast_tail(kernels, work_units, rect_set(plan.tail, plan.num_tail), IN0_DFB, in0_sender_hw_config);

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN0, .spec = in0_tensor.tensor_spec()},
        TensorParameter{.unique_id = IN1, .spec = in1_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = out_tensor.tensor_spec()},
    };

    KernelRunArgs in0_sender_run_args{.kernel = IN0_SENDER};
    KernelRunArgs in0_receiver_run_args{.kernel = IN0_RECEIVER};
    KernelRunArgs in1_sender_run_args{.kernel = IN1_SENDER_WRITER};
    AddRuntimeArgsForNode(
        in0_sender_run_args.runtime_arg_values,
        start_core,
        {{"in0_tensor_start_tile_id", 0u}, {"last_block_h", plan.in0_last_block_h}});
    for (uint32_t i = 0; i < plan.num_cores; ++i) {
        const Mcast1DCore& c = plan.cores[i];
        const CoreCoord core{c.x, c.y};
        AddRuntimeArgsForNode(
            in1_sender_run_args.runtime_arg_values,
            core,
            {{"in1_tensor_start_tile_id", static_cast<uint32_t>(c.in1_tensor_start_tile_id)},
             {"out_tensor_start_tile_id", static_cast<uint32_t>(c.out_tensor_start_tile_id)},
             {"last_block_w", static_cast<uint32_t>(c.last_block_w)},
             {"out_num_nonzero_subblocks_h", plan.out_num_nonzero_subblocks_h},
             {"out_last_subblock_h", plan.out_last_subblock_h},
             {"padded_block_tiles_h_skip", plan.padded_block_tiles_h_skip},
             {"out_num_nonzero_subblocks_w", plan.out_num_nonzero_subblocks_w},
             {"out_last_num_nonzero_subblocks_w", static_cast<uint32_t>(c.out_last_num_nonzero_subblocks_w)},
             {"out_last_subblock_w", static_cast<uint32_t>(c.out_last_subblock_w)},
             {"padded_subblock_tiles_addr_skip", c.padded_subblock_tiles_addr_skip},
             {"padded_block_tiles_w_skip", static_cast<uint32_t>(c.padded_block_tiles_w_skip)}});
        in1_sender_run_args.runtime_arg_values["last_num_blocks_w_dim"][core] =
            static_cast<uint32_t>(c.last_num_blocks_w_dim);
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(in0_sender_run_args));
    if (has_in0_receiver_kernel) {
        run_args.kernel_run_args.push_back(std::move(in0_receiver_run_args));
    }
    run_args.kernel_run_args.push_back(std::move(in1_sender_run_args));
    run_args.tensor_args = {
        {IN0, in0_tensor},
        {IN1, in1_tensor},
        {OUTPUT, out_tensor},
    };

    ProgramSpec spec{
        .name = "matmul_multi_core_reuse_mcast_1d_in0",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    std::vector<KernelSpecName> mcast_kernels{IN0_SENDER};
    if (has_in0_receiver_kernel) {
        mcast_kernels.push_back(IN0_RECEIVER);
    }
    ttnn::kernel_lib::host::Mcast in0_mcast(
        device,
        {.noc = in0_noc, .handshake_cores = all_cores},
        CoreRangeSet(bbox),
        static_cast<uint32_t>(bbox.size()),
        ttnn::kernel_lib::host::McastExplicitFixedSenderConfig{.senders_per_group = {start_core}});
    in0_mcast.attach(spec, run_args, "in0_mcast", mcast_kernels);
    ttnn::kernel_lib::host::attach_absent_mcast(spec, "in1_mcast", std::array{IN1_SENDER_WRITER});

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::llama31_8b_matmul_bh::mcast_1d
