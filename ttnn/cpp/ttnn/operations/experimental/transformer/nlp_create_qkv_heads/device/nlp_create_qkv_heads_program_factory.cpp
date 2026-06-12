// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "nlp_create_qkv_heads_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ProgramDescriptor NlpCreateHeadsDeviceOperation::Interleaved::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor_q;
    std::optional<const Tensor> input_tensor_kv = tensor_args.input_tensor_kv;
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k_heads = operation_attributes.transpose_k_heads;
    auto& output = tensor_return_value;
    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    const auto& input_shape = input_tensor.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = input_tensor.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    tt_metal::Buffer* in1_buffer = nullptr;
    if (read_from_input_tensor_kv) {
        in1_buffer = input_tensor_kv.value().buffer();
        TT_ASSERT(in1_buffer->size() % single_tile_size == 0);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = input_tensor_kv.value().padded_shape()[3] / TILE_WIDTH;
    }

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    // Output shape for K/V is: [B, num_kv_heads, s, head_dim], shuffled from [B, 1, s, num_kv_heads * head_dim]
    // NOTE: Output h and w dims are identical for Q, K, V, so any arg that is related to these dims for q_* can be
    // shared for K, V
    uint32_t q_out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // tiles along head_dim
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of in0_w_tiles per core
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Tensor& q = std::get<0>(output);
    tt_metal::Tensor& k = std::get<1>(output);
    tt_metal::Tensor& v = std::get<2>(output);

    tt_metal::Buffer* q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer* k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer* v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)q_num_tiles,
        (std::uint32_t)kv_num_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args);
    // Always append placeholder/accessor for in1 to keep offsets stable
    tt::tt_metal::TensorAccessorArgs(read_from_input_tensor_kv ? in1_buffer : nullptr)
        .append_to(reader_compile_time_args);

    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)q_out_h_tiles,
        (std::uint32_t)q_out_w_tiles,
        (std::uint32_t)q_out_HtWt,
        (std::uint32_t)num_q_heads,   // q_out_c
        (std::uint32_t)num_kv_heads,  // kv_out_c
    };
    tt::tt_metal::TensorAccessorArgs(q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;
    if (transpose_k_heads) {
        // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
        // resolves the unpack-dst CB to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
        // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
        const bool fp32_dest_acc_en = input_tensor.dtype() == tt_metal::DataType::FLOAT32;

        std::vector<uint32_t> compute_args_core_group_1 = {num_blocks_per_core_group_1 * kv_num_tiles};
        KernelDescriptor compute_desc_1;
        compute_desc_1.kernel_source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
        compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_1.core_ranges = core_group_1;
        compute_desc_1.compile_time_args = std::move(compute_args_core_group_1);
        compute_desc_1.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};
        desc.kernels.push_back(std::move(compute_desc_1));

        if (core_group_2.num_cores() > 0) {
            std::vector<uint32_t> compute_args_core_group_2 = {num_blocks_per_core_group_2 * kv_num_tiles};
            KernelDescriptor compute_desc_2;
            compute_desc_2.kernel_source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
            compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc_2.core_ranges = core_group_2;
            compute_desc_2.compile_time_args = std::move(compute_args_core_group_2);
            compute_desc_2.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};
            desc.kernels.push_back(std::move(compute_desc_2));
        }

        reader_defines.emplace_back("TRANSPOSE_K_HEADS", "1");
        writer_defines.emplace_back("TRANSPOSE_K_HEADS", "1");
    }
    if (read_from_input_tensor_kv) {
        reader_defines.emplace_back("READ_FROM_INPUT_TENSOR_KV", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "writer_tm_tile_layout_nlp_create_qkv_heads.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    // Create circular buffers
    uint32_t micro_block_size = 1;                 // Num tiles to read/wait for in reader and writer
    uint32_t cb_num_tiles = micro_block_size * 4;  // Quadruple buffer everything

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t cb1_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    uint32_t src1_cb_index = 1;  // cb0 is needed for compute if we want to use generic transpose_wh compute kernel
    uint32_t cb1_num_tiles = cb_num_tiles;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb1_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // If we transpose_k_heads:
    // - reader will write to cb0, instead of cb1
    // - compute will wait on cb0 and write to cb16
    // - writer will wait on cb 16, instead of cb1
    if (transpose_k_heads) {
        uint32_t src0_cb_index = 0;
        uint32_t cb0_num_tiles = cb_num_tiles;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb0_num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(src0_cb_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });

        uint32_t out_cb_index = 16;
        uint32_t out_cb_num_tiles = cb_num_tiles;
        desc.cbs.push_back(CBDescriptor{
            .total_size = out_cb_num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(out_cb_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    }

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        KernelDescriptor::RTArgList reader_rt;
        reader_rt.reserve(5);
        reader_rt.push_back(in0_buffer);
        if (in1_buffer != nullptr) {
            reader_rt.push_back(in1_buffer);
        } else {
            reader_rt.push_back(uint32_t{0});
        }
        reader_rt.push_back(num_blocks_per_core);
        reader_rt.push_back(num_blocks_written * in0_w_tiles);
        reader_rt.push_back(num_blocks_written * in1_w_tiles);
        reader_desc.emplace_runtime_args(core, reader_rt);

        writer_desc.emplace_runtime_args(
            core,
            {
                q_buffer,              // q_tensor_addr
                k_buffer,              // k_tensor_addr
                v_buffer,              // v_tensor_addr
                num_blocks_per_core,   // num_blocks
                q_out_h_dim,           // q_out_h_dim
                q_out_tensor_tile_id,  // q_out_tensor_tile_id
                k_out_tensor_tile_id,  // k_out_tensor_tile_id
                v_out_tensor_tile_id,  // v_out_tensor_tile_id
            });

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Sharded::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Metal 2.0 named resource handles for the sharded nlp_create_qkv_heads ProgramSpec.
    // (Declared as locals — not in a file-scope anon namespace — to avoid unity-build collisions.)
    const DFBSpecName Q_OUT_DFB{"q_out"};
    const DFBSpecName K_OUT_DFB{"k_out"};
    const DFBSpecName V_OUT_DFB{"v_out"};
    const TensorParamName INPUT_Q_TENSOR{"input_q"};
    const TensorParamName INPUT_KV_TENSOR{"input_kv"};
    const TensorParamName Q_OUT_TENSOR{"q_output"};
    const TensorParamName K_OUT_TENSOR{"k_output"};
    const TensorParamName V_OUT_TENSOR{"v_output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    constexpr const char* KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp";

    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto& output = tensor_return_value;
    auto head_dim = operation_attributes.head_dim;
    auto num_q_heads = operation_attributes.num_q_heads;
    auto num_kv_heads = operation_attributes.num_kv_heads;

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    auto q_shard_spec = std::get<0>(output).shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    uint32_t per_core_out_q_heads = num_q_heads / q_cores.num_cores();
    uint32_t per_risc0_out_q_heads = div_up(per_core_out_q_heads, 2);
    uint32_t per_risc1_out_q_heads = per_core_out_q_heads / 2;
    uint32_t per_core_in_q_heads = num_q_heads / input_tensor.shard_spec().value().num_cores();

    // Output DFBs borrow the q/k/v output shard buffers (legacy CBs c_16/c_17/c_18). They are
    // write-only address sources (no real FIFO): the kernel grabs base via get_write_ptr() and
    // does explicit NoC reads into the borrowed L1. Bound as self-loops / producer-consumer
    // pairs purely to satisfy the validator's producer-and-consumer rule. See METAL2_PORT_REPORT.
    DataflowBufferSpec q_out_dfb_spec{
        .unique_id = Q_OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = Q_OUT_TENSOR,
    };

    auto k_shard_spec = std::get<1>(output).shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    DataflowBufferSpec k_out_dfb_spec{
        .unique_id = K_OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = k_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = K_OUT_TENSOR,
    };

    auto v_shard_spec = std::get<0>(output).shard_spec().value();
    auto v_cores = q_shard_spec.grid;
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    DataflowBufferSpec v_out_dfb_spec{
        .unique_id = V_OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = v_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = V_OUT_TENSOR,
    };

    uint32_t per_core_out_kv_heads = num_kv_heads / k_cores.num_cores();
    uint32_t per_core_in_kv_heads =
        num_kv_heads / (read_from_input_tensor_kv ? input_tensor_kv.value().shard_spec().value().num_cores()
                                                  : input_tensor.shard_spec().value().num_cores());

    // Host-computed offsets relative to the source-shard bases. The legacy kernel consumed
    // pre-shifted raw addresses (q_base_addr / q_start_addr / k_base_addr / k_start_addr /
    // v_base_addr / v_start_addr); under Metal 2.0 the source bases are recovered kernel-side from
    // the typed tensor binding(s) via get_bank_base_address() (Case 2 bridge) and the host passes
    // only the offsets, added on the kernel side. The arithmetic itself is unchanged from legacy.
    //
    // K/V source base: input_kv when present, otherwise the Q input shard offset by the Q-head span.
    uint32_t kv_base_offset_reader =
        read_from_input_tensor_kv ? 0u : (per_core_in_q_heads * head_tiles * single_tile_size);
    // V (writer) base sits one KV-head span past the K (reader) base within the same source shard.
    uint32_t kv_base_offset_writer = kv_base_offset_reader + (per_core_in_kv_heads * head_tiles * single_tile_size);

    // Tensor parameters: the q/k/v output tensors back the borrowed DFBs. The Q input tensor
    // (and KV input tensor when present) supply the source-shard bases (Case 2 bridge), recovered
    // kernel-side via get_bank_base_address().
    TensorParameter input_q_param{.unique_id = INPUT_Q_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter q_out_param{.unique_id = Q_OUT_TENSOR, .spec = std::get<0>(output).tensor_spec()};
    TensorParameter k_out_param{.unique_id = K_OUT_TENSOR, .spec = std::get<1>(output).tensor_spec()};
    TensorParameter v_out_param{.unique_id = V_OUT_TENSOR, .spec = std::get<2>(output).tensor_spec()};

    uint32_t num_cores = std::max(q_cores.num_cores(), k_cores.num_cores());

    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    // NoC coordinates of the input shard cores. Identical for every output core, so they are
    // broadcast as Metal 2.0 *common* runtime varargs, laid out [x0..x_{nx-1}, y0..y_{ny-1}].
    // The kernel reads get_common_vararg(x) for x-coords, get_common_vararg(num_x + y) for y-coords.
    std::vector<uint32_t> noc_coords;
    noc_coords.reserve(num_cores_x + num_cores_y);
    for (uint32_t x = 0; x < num_cores_x; ++x) {
        noc_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        noc_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    // Runtime-arg schema shared by reader and writer (same kernel source, bound twice).
    const std::vector<std::string> rt_arg_names = {
        "head_size",
        "num_q_heads",
        "num_q_heads_per_core",
        "remote_q_head_start_idx",
        "start_q_x",
        "start_q_y",
        "q_start_offset",
        "q_offset",
        "read_kv_heads",
        "num_kv_heads",
        "num_kv_heads_per_core",
        "remote_kv_head_start_idx",
        "start_kv_x",
        "start_kv_y",
        "kv_base_offset",
        "kv_start_offset",
        "num_kv_tiles",
        "num_x"};

    // When a separate KV input tensor is present the kernel recovers its base from a second
    // tensor binding; gate the binding and its kernel-side accessor on READ_FROM_INPUT_TENSOR_KV.
    auto make_tensor_bindings = [&]() {
        Group<TensorBinding> bindings = {
            TensorBinding{.tensor_parameter_name = INPUT_Q_TENSOR, .accessor_name = "input_q"}};
        if (read_from_input_tensor_kv) {
            bindings.push_back(TensorBinding{.tensor_parameter_name = INPUT_KV_TENSOR, .accessor_name = "input_kv"});
        }
        return bindings;
    };
    Table<std::string, std::string> kv_defines;
    if (read_from_input_tensor_kv) {
        kv_defines.insert({"READ_FROM_INPUT_TENSOR_KV", "1"});
    }

    // Reader binds q_out (shared with writer) + k_out (reader-private). Writer binds q_out +
    // v_out (writer-private). q_out is written by both kernels on the same nodes, so it is bound
    // reader = PRODUCER / writer = CONSUMER; the reader-private k_out and writer-private v_out are
    // self-looped (PRODUCER + CONSUMER on the single kernel that touches them).
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{KERNEL_PATH},
        .compiler_options = {.defines = kv_defines},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = Q_OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = K_OUT_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = K_OUT_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = make_tensor_bindings(),
        .runtime_arg_schema = {.runtime_arg_names = rt_arg_names},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };
    reader_spec.advanced_options.num_common_runtime_varargs = noc_coords.size();

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{KERNEL_PATH},
        .compiler_options = {.defines = kv_defines},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = Q_OUT_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = V_OUT_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = V_OUT_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = make_tensor_bindings(),
        .runtime_arg_schema = {.runtime_arg_names = rt_arg_names},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };
    writer_spec.advanced_options.num_common_runtime_varargs = noc_coords.size();

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.advanced_options.common_runtime_varargs = noc_coords;
    writer_run.advanced_options.common_runtime_varargs = noc_coords;

    uint32_t remote_q_head_start_idx = 0;
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t q_x = 0, q_y = 0, kv_x = 0, kv_y = 0;
    // Offsets (relative to the source-shard bases) tracking the legacy q_start_addr / k_start_addr /
    // v_start_addr cursors. The kernel adds the recovered accessor base to these.
    uint32_t q_start_offset = 0;
    uint32_t k_start_offset = kv_base_offset_reader;
    uint32_t v_start_offset = kv_base_offset_writer;

    uint32_t remote_q_read = 0;
    uint32_t remote_kv_read = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const NodeCoord core = cores[i];
        bool read_kv_heads = i < k_cores.num_cores();

        // RISC0 (reader) per-node runtime args.
        KernelRunArgs::RuntimeArgValues reader_args{
            {"head_size", head_size},
            {"num_q_heads", per_risc0_out_q_heads},
            {"num_q_heads_per_core", per_core_in_q_heads},
            {"remote_q_head_start_idx", remote_q_head_start_idx},
            {"start_q_x", q_x},
            {"start_q_y", q_y},
            {"q_start_offset", q_start_offset},
            {"q_offset", 0u},
            {"read_kv_heads", static_cast<uint32_t>(read_kv_heads)},
            {"num_kv_heads", per_core_out_kv_heads},
            {"num_kv_heads_per_core", per_core_in_kv_heads},
            {"remote_kv_head_start_idx", remote_kv_head_start_idx},
            {"start_kv_x", kv_x},
            {"start_kv_y", kv_y},
            {"kv_base_offset", kv_base_offset_reader},
            {"kv_start_offset", k_start_offset},
            {"num_kv_tiles", k_num_tiles},
            {"num_x", num_cores_x}};

        remote_q_read += per_risc0_out_q_heads;
        q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
        q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;
        q_start_offset = remote_q_head_start_idx * head_size;

        reader_run.runtime_arg_values.push_back({core, reader_args});

        // RISC1 (writer) per-node runtime args: same layout, advanced for the second half of Q.
        KernelRunArgs::RuntimeArgValues writer_args = reader_args;
        writer_args["num_q_heads"] = per_risc1_out_q_heads;
        writer_args["remote_q_head_start_idx"] = remote_q_head_start_idx;
        writer_args["start_q_x"] = q_x;
        writer_args["start_q_y"] = q_y;
        writer_args["q_start_offset"] = q_start_offset;
        writer_args["q_offset"] = per_risc0_out_q_heads * head_size;

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
            q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
            q_start_offset = remote_q_head_start_idx * head_size;
        }

        if (read_kv_heads) {
            writer_args["kv_base_offset"] = kv_base_offset_writer;
            writer_args["kv_start_offset"] = v_start_offset;
            remote_kv_read += per_core_out_kv_heads;
            kv_y = (remote_kv_read / per_core_in_kv_heads) / num_cores_x;
            kv_x = (remote_kv_read / per_core_in_kv_heads) % num_cores_x;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
            k_start_offset = kv_base_offset_reader + remote_kv_head_start_idx * head_size;
            v_start_offset = kv_base_offset_writer + remote_kv_head_start_idx * head_size;
        }

        writer_run.runtime_arg_values.push_back({core, writer_args});
    }

    WorkUnitSpec wu{
        .name = "nlp_create_qkv_heads_sharded",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = q_cores,
    };

    Group<TensorParameter> tensor_params = {input_q_param, q_out_param, k_out_param, v_out_param};
    if (read_from_input_tensor_kv) {
        tensor_params.push_back(
            TensorParameter{.unique_id = INPUT_KV_TENSOR, .spec = input_tensor_kv.value().tensor_spec()});
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_sharded",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {q_out_dfb_spec, k_out_dfb_spec, v_out_dfb_spec},
        .tensor_parameters = tensor_params,
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_Q_TENSOR, TensorArgument{input_tensor.mesh_tensor()}},
        {Q_OUT_TENSOR, TensorArgument{std::get<0>(output).mesh_tensor()}},
        {K_OUT_TENSOR, TensorArgument{std::get<1>(output).mesh_tensor()}},
        {V_OUT_TENSOR, TensorArgument{std::get<2>(output).mesh_tensor()}}};
    if (read_from_input_tensor_kv) {
        run_args.tensor_args.insert({INPUT_KV_TENSOR, TensorArgument{input_tensor_kv.value().mesh_tensor()}});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::transformer
