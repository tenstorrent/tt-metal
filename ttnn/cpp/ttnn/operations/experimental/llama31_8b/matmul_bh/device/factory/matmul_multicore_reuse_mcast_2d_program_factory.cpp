// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/factory/mcast_2d.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_l1.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>
#include <limits>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "tt-metalium/buffer_types.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/kernel_lib/mcast/host/mcast.hpp"

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::UnpackMode;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::AdvancedKernelRunArgs;
using tt::tt_metal::experimental::ComputeHardwareConfig;
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
using tt::tt_metal::experimental::Table;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim::llama31_8b_matmul_bh::mcast_2d {

namespace {

std::map<std::string, std::string> define_map(const Mcast2DDefine* d, uint32_t n) {
    std::map<std::string, std::string> m;
    for (uint32_t i = 0; i < n; i++) {
        m[d[i].key] = d[i].value;
    }
    return m;
}

constexpr std::array<const char*, 12> kOutWriterNames = {
    "out_tensor_start_tile_id",
    "out_num_nonzero_subblocks_h",
    "out_last_num_nonzero_subblocks_h",
    "out_last_subblock_h",
    "padded_block_tiles_h_skip",
    "out_num_nonzero_subblocks_w",
    "out_last_num_nonzero_subblocks_w",
    "out_last_subblock_w",
    "padded_subblock_tiles_addr_skip",
    "padded_block_tiles_w_skip",
    "last_num_blocks_h_dim",
    "last_num_blocks_w_dim"};

constexpr std::array<const char*, 15> kIn1SenderNames = {
    "in1_tensor_start_tile_id",
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
    "last_num_blocks_w_dim",
    "vc",
    "num_dram_shards_to_read",
    "dram_tensor_start_offset"};

}  // namespace

ttnn::device_operation::ProgramArtifacts create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const Mcast2DPlan& plan) {
    using namespace tt;

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& in0_tensor = a.mesh_tensor();
    const auto& in1_tensor = b.mesh_tensor();
    const auto& out_tensor = tensor_return_value.at(0).mesh_tensor();
    TT_FATAL(
        !tensor_args.optional_input_tensors.at(0).has_value() && !operation_attributes.untilize_out &&
            !operation_attributes.transpose_a && !operation_attributes.transpose_b &&
            operation_attributes.bcast_batch.has_value() && operation_attributes.compute_kernel_config.has_value() &&
            !operation_attributes.in0_column_offset.has_value(),
        "llama31_8b matmul_bh: the 2D table rows take two plain inputs");
    const uint32_t bcast_batch = operation_attributes.bcast_batch.value() ? 1u : 0u;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const tt_metal::distributed::MeshDevice& device = in0_tensor.device();
    TT_FATAL(
        device.num_dram_channels() == plan.num_dram_banks,
        "llama31_8b matmul_bh: DRAM bank count not in the table row");

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device.arch(), compute_kernel_config);
    auto compute_hw = ttnn::to_compute_hardware_config(compute_kernel_config);

    const auto in0_tile = a.tensor_spec().tile();
    const auto in1_tile = b.tensor_spec().tile();
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    const tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(in0_tensor.dtype());
    const tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_tensor.dtype());
    const tt::DataFormat swiglu_out_format = tt_metal::datatype_to_dataformat_converter(out_tensor.dtype());
    const tt::DataFormat output_data_format = plan.swiglu ? tt::DataFormat::Float16_b : swiglu_out_format;
    const tt::DataFormat interm0_data_format = plan.interm0_format == 2   ? tt::DataFormat::Float32
                                               : plan.interm0_format == 1 ? tt::DataFormat::Float16_b
                                                                          : output_data_format;
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t in0_tile_size = tt::align(in0_tile.get_tile_size(in0_data_format), dram_alignment);
    const uint32_t in1_tile_size = in1_tile.get_tile_size(in1_data_format);
    const uint32_t output_tile_size = output_tile.get_tile_size(output_data_format);
    const uint32_t interm0_tile_size = output_tile.get_tile_size(interm0_data_format);
    const uint32_t swiglu_tile_size = output_tile.get_tile_size(swiglu_out_format);
    TT_FATAL(
        output_tile_size == plan.out_tile_bytes && in1_tile_size == plan.in1_tile_bytes,
        "llama31_8b matmul_bh: tile sizes not in the table row");

    const KernelSpecName IN0_SENDER{"in0_sender"};
    const KernelSpecName IN1_SENDER_WRITER{"in1_sender_writer"};
    const KernelSpecName IN1_RECEIVER_WRITER{"in1_receiver_writer"};
    const KernelSpecName IN0_RECEIVER{"in0_receiver"};
    const KernelSpecName IN1_RECEIVER_WRITER_OTHER{"in1_receiver_writer_other_noc"};
    const KernelSpecName IN0_RECEIVER_OTHER{"in0_receiver_other_noc"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName OUT_B_DFB{"out_b"};
    const DFBSpecName INTERMED0_B_DFB{"intermed0_b"};
    const DFBSpecName INTERMED0_DFB{"intermed0"};
    const DFBSpecName SWIGLU_OUT_DFB{"swiglu_out"};
    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    auto mm_kernel_defines = define_map(plan.defines_compute, plan.num_defines_compute);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device.arch(), plan.num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device.arch(), plan.num_cores, mm_kernel_defines, ttnn::get_throttle_level(compute_kernel_config));
    const auto in0_sender_defines = define_map(plan.defines_in0_send, plan.num_defines_in0_send);
    const auto in0_receiver_defines = define_map(plan.defines_in0_recv, plan.num_defines_in0_recv);
    const auto in1_sender_defines = define_map(plan.defines_in1_send, plan.num_defines_in1_send);
    const auto in1_receiver_defines = define_map(plan.defines_in1_recv, plan.num_defines_in1_recv);
    const auto in1_receiver_other_defines = define_map(plan.defines_in1_recv_o, plan.num_defines_in1_recv_o);

    const tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device.arch());
    const tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device.arch());
    const tt_metal::NOC in0_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device.arch());
    const tt_metal::NOC in1_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device.arch());
    const auto dm_hw_config = [](tt_metal::DataMovementProcessor proc,
                                 tt_metal::NOC noc) -> DataMovementHardwareConfig {
        return DataMovementHardwareConfig{
            .config_1xx = DataMovementHardwareConfig::DataMovement1XXConfig{.processor = proc, .noc = noc},
            .config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{.disable_dfb_implicit_sync_for_all = true},
        };
    };

    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_tile_size,
        .num_entries = plan.out_entries,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
    };
    DataflowBufferSpec intermed0_dfb_spec{
        .unique_id = INTERMED0_DFB,
        .entry_size = interm0_tile_size,
        .num_entries = plan.share_out_interm ? plan.out_entries : plan.interm0_entries,
        .data_format_metadata = interm0_data_format,
        .tile_format_metadata = output_tile,
    };
    if (plan.share_out_interm) {
        out_dfb_spec.advanced_options.alias_with = Group<DFBSpecName>{INTERMED0_DFB};
        intermed0_dfb_spec.advanced_options.alias_with = Group<DFBSpecName>{OUT_DFB};
    }
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0_DFB,
        .entry_size = in0_tile_size,
        .num_entries = plan.in0_entries,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    });
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN1_DFB,
        .entry_size = in1_tile_size,
        .num_entries = plan.in1_entries,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile,
    });
    if (plan.out_writer_on_in0) {
        DataflowBufferSpec out_b_spec = out_dfb_spec;
        out_b_spec.unique_id = OUT_B_DFB;
        out_b_spec.advanced_options.alias_with = Group<DFBSpecName>{INTERMED0_B_DFB};
        DataflowBufferSpec intermed0_b_spec = intermed0_dfb_spec;
        intermed0_b_spec.unique_id = INTERMED0_B_DFB;
        intermed0_b_spec.advanced_options.alias_with = Group<DFBSpecName>{OUT_B_DFB};
        dataflow_buffers.push_back(std::move(out_dfb_spec));
        dataflow_buffers.push_back(std::move(intermed0_dfb_spec));
        dataflow_buffers.push_back(std::move(out_b_spec));
        dataflow_buffers.push_back(std::move(intermed0_b_spec));
        if (plan.swiglu) {
            dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = SWIGLU_OUT_DFB,
                .entry_size = swiglu_tile_size,
                .num_entries = plan.swiglu_entries,
                .data_format_metadata = swiglu_out_format,
                .tile_format_metadata = output_tile,
            });
        }
    } else {
        dataflow_buffers.push_back(std::move(out_dfb_spec));
        dataflow_buffers.push_back(std::move(intermed0_dfb_spec));
    }

    TT_FATAL(
        dfb_l1_bytes(dataflow_buffers) == l1_bytes_2d(plan, a, b, out_tensor.dtype()),
        "llama31_8b matmul_bh: buffer plan and L1 check disagree");

    KernelRunArgs in0_sender_run_args{.kernel = IN0_SENDER};
    KernelRunArgs in1_sender_writer_run_args{.kernel = IN1_SENDER_WRITER};
    KernelRunArgs in1_receiver_writer_run_args{.kernel = IN1_RECEIVER_WRITER};
    KernelRunArgs in0_receiver_run_args{.kernel = IN0_RECEIVER};
    KernelRunArgs in1_receiver_writer_other_run_args{.kernel = IN1_RECEIVER_WRITER_OTHER};
    KernelRunArgs in0_receiver_other_run_args{.kernel = IN0_RECEIVER_OTHER};
    for (uint32_t i = 0; i < plan.num_cores; i++) {
        const Mcast2DCore& c = plan.cores[i];
        const CoreCoord core{c.x, c.y};
        auto& in0_run_args = c.in0_kind == 0   ? in0_sender_run_args
                             : c.in0_kind == 1 ? in0_receiver_run_args
                                               : in0_receiver_other_run_args;
        if (c.in0_kind == 0) {
            in0_run_args.runtime_arg_values["in0_tensor_start_tile_id"][core] = c.in0_args[0];
            in0_run_args.runtime_arg_values["last_block_h"][core] = c.in0_args[1];
        }
        if (plan.out_writer_on_in0) {
            for (uint32_t k = 0; k < kOutWriterNames.size(); k++) {
                in0_run_args.runtime_arg_values[kOutWriterNames[k]][core] = c.out_args[k];
            }
        }
        if (c.in1_kind == 0) {
            for (uint32_t k = 0; k < kIn1SenderNames.size(); k++) {
                in1_sender_writer_run_args.runtime_arg_values[kIn1SenderNames[k]][core] = c.in1_args[k];
            }
            in1_sender_writer_run_args.advanced_options.runtime_varargs[core] =
                AdvancedKernelRunArgs::Varargs(c.varargs, c.varargs + plan.num_runtime_varargs);
        } else {
            auto& in1_run_args = c.in1_kind == 1 ? in1_receiver_writer_run_args : in1_receiver_writer_other_run_args;
            for (uint32_t k = 0; k < kOutWriterNames.size(); k++) {
                in1_run_args.runtime_arg_values[kOutWriterNames[k]][core] = c.in1_args[k];
            }
        }
    }

    auto add_output_writer = [&](KernelSpec& k) {
        if (!plan.out_writer_on_in0) {
            return;
        }
        if (plan.swiglu) {
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = SWIGLU_OUT_DFB,
                .accessor_name = "swiglu_out",
                .endpoint_type = DFBEndpointType::CONSUMER});
        } else {
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER});
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT_B_DFB, .accessor_name = "out_b", .endpoint_type = DFBEndpointType::CONSUMER});
        }
        k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"});
        for (const auto& [n, v] : std::initializer_list<std::pair<const char*, uint32_t>>{
                 {"out_tensor_stride_w", 1u},
                 {"out_tensor_stride_h", plan.ow_stride_h},
                 {"out_tensor_next_subblock_stride_w", plan.ow_next_subblock_stride_w},
                 {"out_tensor_next_subblock_stride_h", plan.ow_next_subblock_stride_h},
                 {"out_tensor_next_w_dim_block_stride", plan.ow_next_w_dim_block_stride},
                 {"out_tensor_next_h_dim_block_stride", plan.ow_next_h_dim_block_stride},
                 {"out_subblock_w", plan.ow_subblock_w},
                 {"out_subblock_h", plan.out_subblock_h},
                 {"out_subblock_tile_count", plan.ow_subblock_tile_count},
                 {"out_write_pace_cycles", plan.out_write_pace_cycles},
                 {"out_write_pace_barrier", plan.out_write_pace_barrier},
             }) {
            k.compile_time_args.insert({n, v});
        }
        for (const char* n : kOutWriterNames) {
            k.runtime_arg_schema.runtime_arg_names.emplace_back(n);
        }
    };
    auto drop_output_consumer = [&](KernelSpec& k) {
        if (plan.out_writer_on_in0) {
            std::erase_if(k.dfb_bindings, [&](const DFBBinding& d) { return d.dfb_spec_name == OUT_DFB; });
        }
    };

    Group<KernelSpec> kernels;
    {
        KernelSpec in0_sender{
            .unique_id = IN0_SENDER,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_padding_metal2.cpp",
            .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(in0_sender_defines)},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {
                    {"in0_tensor_stride_w", 1u},
                    {"in0_tensor_stride_h", plan.in0_stride_h},
                    {"in0_tensor_next_inner_dim_block_stride", plan.in0_next_inner_dim_block_stride},
                    {"in0_tensor_next_h_dim_block_stride", plan.in0_next_h_dim_block_stride},
                    {"in0_block_w", plan.in0_block_w},
                    {"in0_block_h", plan.out_block_h},
                    {"in0_block_num_tiles", plan.in0_block_num_tiles},
                    {"in0_last_ktile_w", 0u},
                    {"in0_last_ktile_h", 0u},
                    {"shard_width_in_tiles", 0u},
                    {"shard_height_in_tiles", 0u},
                    {"num_blocks_inner_dim", plan.num_blocks},
                    {"num_blocks_w_dim", plan.num_blocks_w_dim},
                    {"num_blocks_h_dim", plan.num_blocks_h_dim},
                    {"MtKt", plan.MtKt},
                    {"in0_B", 1u},
                    {"in1_B", 1u},
                    {"in0_reuse_in_dfb", 0u},
                    {"batchB", 0u},
                    {"sparsity_pagesize", 0u},
                    {"bcast_A", 1u},
                    {"get_batch_from_reader", 0u},
                    {"num_batch_compute", 0u},
                    {"num_active", 0u},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"in0_tensor_start_tile_id", "last_block_h"}},
            .hw_config = dm_hw_config(tt_metal::DataMovementProcessor::RISCV_1, in0_noc),
        };
        add_output_writer(in0_sender);
        in0_sender.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"});
        kernels.push_back(std::move(in0_sender));
    }
    {
        KernelSpec in1_sender_writer{
            .unique_id = IN1_SENDER_WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp",
            .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(in1_sender_defines)},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"},
                    TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"},
                },
            .compile_time_args =
                {
                    {"in1_tensor_stride_w", 1u},
                    {"in1_tensor_stride_h", plan.in1_stride_h},
                    {"in1_tensor_next_block_stride", plan.in1_next_block_stride},
                    {"in1_tensor_next_w_dim_block_stride", plan.in1_next_w_dim_block_stride},
                    {"in1_block_w", plan.out_block_w},
                    {"in1_block_h", plan.in0_block_w},
                    {"in1_block_num_tiles", plan.in1_block_num_tiles},
                    {"num_blocks_inner_dim", plan.num_blocks},
                    {"num_blocks_w_dim", plan.num_blocks_w_dim},
                    {"num_blocks_h_dim", plan.num_blocks_h_dim},
                    {"KtNt", plan.KtNt},
                    {"batch", 1u},
                    {"out_block_num_tiles", plan.out_entries},
                    {"bcast_B", bcast_batch},
                    {"batchB", 0u},
                    {"sparsity_pagesize", 0u},
                    {"out_tensor_stride_w", 1u},
                    {"out_tensor_stride_h", plan.out_stride_h},
                    {"out_tensor_next_subblock_stride_w", plan.out_subblock_w},
                    {"out_tensor_next_subblock_stride_h", plan.out_next_subblock_stride_h},
                    {"out_tensor_next_w_dim_block_stride", plan.out_block_w},
                    {"out_tensor_next_h_dim_block_stride", plan.out_next_h_dim_block_stride},
                    {"out_subblock_w", plan.out_subblock_w},
                    {"out_subblock_h", plan.out_subblock_h},
                    {"out_subblock_tile_count", plan.out_subblock_tile_count},
                    {"MtNt", plan.MtNt},
                    {"compact_output", 0u},
                    {"num_active", 0u},
                    {"in1_dram_block_num_tiles", plan.in1_dram_block_num_tiles},
                    {"in1_block_w_dram_bytes", plan.in1_block_w_dram_bytes},
                },
            .runtime_arg_schema = {},
            .hw_config = dm_hw_config(tt_metal::DataMovementProcessor::RISCV_0, in1_noc),
        };
        for (const char* n : kIn1SenderNames) {
            in1_sender_writer.runtime_arg_schema.runtime_arg_names.emplace_back(n);
        }
        in1_sender_writer.advanced_options.num_runtime_varargs = plan.num_runtime_varargs;
        drop_output_consumer(in1_sender_writer);
        kernels.push_back(std::move(in1_sender_writer));
    }

    auto make_in1_receiver_writer = [&](const KernelSpecName& id,
                                        const std::map<std::string, std::string>& defines,
                                        tt_metal::NOC noc) {
        KernelSpec k{
            .unique_id = id,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp",
            .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(defines)},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
                },
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"}},
            .compile_time_args =
                {
                    {"in1_block_num_tiles", plan.in1_block_num_tiles},
                    {"num_blocks_inner_dim", plan.num_blocks},
                    {"num_blocks_w_dim", plan.num_blocks_w_dim},
                    {"num_blocks_h_dim", plan.num_blocks_h_dim},
                    {"batch", 1u},
                    {"out_block_num_tiles", plan.out_entries},
                    {"out_tensor_stride_w", 1u},
                    {"out_tensor_stride_h", plan.out_stride_h},
                    {"out_tensor_next_subblock_stride_w", plan.out_subblock_w},
                    {"out_tensor_next_subblock_stride_h", plan.out_next_subblock_stride_h},
                    {"out_tensor_next_w_dim_block_stride", plan.out_block_w},
                    {"out_tensor_next_h_dim_block_stride", plan.out_next_h_dim_block_stride},
                    {"out_subblock_w", plan.out_subblock_w},
                    {"out_subblock_h", plan.out_subblock_h},
                    {"out_subblock_tile_count", plan.out_subblock_tile_count},
                    {"MtNt", plan.MtNt},
                },
            .runtime_arg_schema = {},
            .hw_config = dm_hw_config(tt_metal::DataMovementProcessor::RISCV_0, noc),
        };
        for (const char* n : kOutWriterNames) {
            k.runtime_arg_schema.runtime_arg_names.emplace_back(n);
        }
        drop_output_consumer(k);
        return k;
    };
    auto make_in0_receiver = [&](const KernelSpecName& id, tt_metal::NOC noc) {
        KernelSpec k{
            .unique_id = id,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_receiver_metal2.cpp",
            .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(in0_receiver_defines)},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {
                    {"in0_block_num_tiles", plan.in0_receiver_block_num_tiles},
                    {"num_blocks_inner_dim", plan.num_blocks},
                    {"num_blocks_w_dim", plan.num_blocks_w_dim},
                    {"num_blocks_h_dim", plan.num_blocks_h_dim},
                    {"batch", 1u},
                    {"get_batch_from_reader", 0u},
                },
            .runtime_arg_schema = {.runtime_arg_names = {}},
            .hw_config = dm_hw_config(tt_metal::DataMovementProcessor::RISCV_1, noc),
        };
        add_output_writer(k);
        return k;
    };
    if (plan.has_in1_receiver) {
        kernels.push_back(make_in1_receiver_writer(IN1_RECEIVER_WRITER, in1_receiver_defines, in1_noc));
    }
    if (plan.has_in0_receiver) {
        kernels.push_back(make_in0_receiver(IN0_RECEIVER, in0_noc));
    }
    if (plan.has_other_noc) {
        kernels.push_back(
            make_in1_receiver_writer(IN1_RECEIVER_WRITER_OTHER, in1_receiver_other_defines, in1_split_noc));
        kernels.push_back(make_in0_receiver(IN0_RECEIVER_OTHER, in0_split_noc));
    }

    Table<DFBSpecName, UnpackMode> unpack_modes_table;
    if (fp32_dest_acc_en) {
        for (const auto& [name, fmt] : std::initializer_list<std::pair<DFBSpecName, tt::DataFormat>>{
                 {IN0_DFB, in0_data_format}, {IN1_DFB, in1_data_format}, {INTERMED0_DFB, interm0_data_format}}) {
            if (fmt == tt::DataFormat::Float32) {
                unpack_modes_table.insert(
                    {name, name == INTERMED0_DFB ? UnpackMode::UnpackToDest : UnpackMode::UnpackToSrc});
            }
        }
    }
    if (!unpack_modes_table.empty()) {
        compute_hw.unpack_modes = std::move(unpack_modes_table);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b/matmul_bh/device/kernels/compute/"
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
                    .dfb_spec_name = INTERMED0_DFB,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = INTERMED0_DFB,
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
                {"in1_block_num_tiles", plan.compute_in1_block_num_tiles},
                {"in1_block_w", plan.in1_per_core_w},
                {"num_blocks_inner_dim", plan.num_blocks},
                {"num_blocks_w_dim", plan.num_blocks_w_dim},
                {"num_blocks_h_dim", plan.num_blocks_h_dim},
                {"out_subblock_h", plan.out_subblock_h},
                {"out_subblock_w", plan.out_subblock_w},
                {"out_subblock_num_tiles", plan.out_subblock_num_tiles},
                {"batch", 1u},
                {"out_block_num_tiles", plan.out_entries},
                {"untilize_out", 0u},
                {"get_batch_from_reader", 0u},
                {"bias_ntiles", plan.in1_per_core_w},
            },
        .hw_config = std::move(compute_hw),
    };
    if (plan.swiglu) {
        compute.dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER});
        compute.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_B_DFB, .accessor_name = "out_b", .endpoint_type = DFBEndpointType::CONSUMER});
        compute.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = SWIGLU_OUT_DFB,
            .accessor_name = "swiglu_out",
            .endpoint_type = DFBEndpointType::PRODUCER});
    }
    if (plan.out_writer_on_in0) {
        compute.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_B_DFB, .accessor_name = "out_b", .endpoint_type = DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INTERMED0_B_DFB,
            .accessor_name = "intermed0_b",
            .endpoint_type = DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INTERMED0_B_DFB,
            .accessor_name = "intermed0_b",
            .endpoint_type = DFBEndpointType::CONSUMER});
    }
    kernels.push_back(std::move(compute));

    constexpr std::array<std::pair<uint32_t, const char*>, 7> kSortedKernels = {{
        {6, "compute"},
        {3, "in0_receiver"},
        {5, "in0_receiver_other_noc"},
        {0, "in0_sender"},
        {2, "in1_receiver_writer"},
        {4, "in1_receiver_writer_other_noc"},
        {1, "in1_sender_writer"},
    }};
    Group<WorkUnitSpec> work_units;
    for (uint32_t u = 0; u < plan.num_units; u++) {
        const Mcast2DWorkUnit& wu = plan.units[u];
        Group<KernelSpecName> wu_kernels;
        for (const auto& [bit, name] : kSortedKernels) {
            if (wu.kernel_mask & (1u << bit)) {
                wu_kernels.push_back(KernelSpecName{name});
            }
        }
        std::vector<CoreRange> ranges;
        for (uint32_t i = 0; i < wu.num_rects; i++) {
            const Mcast2DRect& r = plan.rects[wu.first_rect + i];
            ranges.emplace_back(CoreCoord(r.x0, r.y0), CoreCoord(r.x1, r.y1));
        }
        work_units.push_back(WorkUnitSpec{
            .name = wu.name,
            .kernels = std::move(wu_kernels),
            .target_nodes = CoreRangeSet(std::move(ranges)).merge_ranges()});
    }

    Group<TensorParameter> tensor_parameters;
    tensor_parameters.push_back(TensorParameter{.unique_id = IN0, .spec = in0_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = IN1, .spec = in1_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = out_tensor.tensor_spec()});

    ProgramSpec spec{
        .name = "matmul_multi_core_reuse_mcast_2d_optimized",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(in0_sender_run_args));
    run_args.kernel_run_args.push_back(std::move(in1_sender_writer_run_args));
    if (plan.has_in1_receiver) {
        run_args.kernel_run_args.push_back(std::move(in1_receiver_writer_run_args));
    }
    if (plan.has_in0_receiver) {
        run_args.kernel_run_args.push_back(std::move(in0_receiver_run_args));
    }
    if (plan.has_other_noc) {
        run_args.kernel_run_args.push_back(std::move(in1_receiver_writer_other_run_args));
        run_args.kernel_run_args.push_back(std::move(in0_receiver_other_run_args));
    }
    run_args.tensor_args = {
        {IN0, in0_tensor},
        {IN1, in1_tensor},
        {OUTPUT, out_tensor},
    };

    using ttnn::kernel_lib::host::Mcast;
    using ttnn::kernel_lib::host::McastCoreOrder;
    using ttnn::kernel_lib::host::McastFixedSenderConfig;
    const CoreRangeSet work_cores(CoreRange(CoreCoord(0, 0), CoreCoord(plan.cores_c - 1, plan.cores_r - 1)));
    {
        Mcast in0_mcast(
            device, {.noc = in0_noc}, work_cores, plan.cores_r, McastFixedSenderConfig{}, McastCoreOrder::ColumnMajor);
        std::vector<KernelSpecName> targets{IN0_SENDER};
        if (plan.has_in0_receiver) {
            targets.push_back(IN0_RECEIVER);
        }
        if (plan.has_other_noc) {
            targets.push_back(IN0_RECEIVER_OTHER);
        }
        in0_mcast.attach(spec, run_args, "in0_mcast", targets);
    }
    Mcast in1_mcast(
        device, {.noc = in1_noc}, work_cores, plan.cores_c, McastFixedSenderConfig{}, McastCoreOrder::RowMajor);
    std::vector<KernelSpecName> in1_targets{IN1_SENDER_WRITER};
    if (plan.has_in1_receiver) {
        in1_targets.push_back(IN1_RECEIVER_WRITER);
    }
    if (plan.has_other_noc) {
        in1_targets.push_back(IN1_RECEIVER_WRITER_OTHER);
    }
    in1_mcast.attach(spec, run_args, "in1_mcast", in1_targets);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::llama31_8b_matmul_bh::mcast_2d
