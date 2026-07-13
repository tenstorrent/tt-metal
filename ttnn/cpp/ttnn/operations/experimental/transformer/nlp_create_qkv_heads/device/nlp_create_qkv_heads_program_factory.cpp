// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "nlp_create_qkv_heads_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Interleaved::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Metal 2.0 named resource handles for the interleaved nlp_create_qkv_heads ProgramSpec.
    // (Declared as locals — not in a file-scope anon namespace — to avoid unity-build collisions.)
    const DFBSpecName QV_DFB{"qv"};        // legacy cb1: Q/V tiles (and K when not transposed)
    const DFBSpecName IN_K_DFB{"in_k"};    // legacy cb0: K tiles into compute (transpose only)
    const DFBSpecName OUT_K_DFB{"out_k"};  // legacy cb16: transposed K tiles out of compute (transpose only)
    const TensorParamName INPUT_Q_TENSOR{"input_q"};
    const TensorParamName INPUT_KV_TENSOR{"input_kv"};
    const TensorParamName Q_OUT_TENSOR{"q_output"};
    const TensorParamName K_OUT_TENSOR{"k_output"};
    const TensorParamName V_OUT_TENSOR{"v_output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL_G1{"compute_g1"};
    const KernelSpecName COMPUTE_KERNEL_G2{"compute_g2"};
    // Forked Metal 2.0 kernel sources (legacy copies stay for unmigrated sibling ops).
    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_metal2.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "writer_tm_tile_layout_nlp_create_qkv_heads_metal2.cpp";
    constexpr const char* COMPUTE_PATH =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/compute/"
        "transpose_wh_metal2.cpp";

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

    if (read_from_input_tensor_kv) {
        TT_ASSERT(input_tensor_kv.value().buffer()->size() % single_tile_size == 0);
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
    //                      Output tensors
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Tensor& q = std::get<0>(output);
    tt_metal::Tensor& k = std::get<1>(output);
    tt_metal::Tensor& v = std::get<2>(output);

    TT_ASSERT(q.buffer() != nullptr, "Output q buffer should be allocated on device!");
    TT_ASSERT(k.buffer() != nullptr, "Output k buffer should be allocated on device!");
    TT_ASSERT(v.buffer() != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers (legacy CBs)
    ////////////////////////////////////////////////////////////////////////////
    // Create dataflow buffers
    uint32_t micro_block_size = 1;                 // Num tiles to read/wait for in reader and writer
    uint32_t cb_num_tiles = micro_block_size * 4;  // Quadruple buffer everything

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t cb1_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    // QV DFB (legacy cb1): cb0 is needed for compute if we want to use the generic transpose_wh compute kernel.
    DataflowBufferSpec qv_dfb_spec{
        .unique_id = QV_DFB,
        .entry_size = single_tile_size,
        .num_entries = cb_num_tiles,
        .data_format_metadata = cb_data_format,
    };

    // If we transpose_k_heads:
    // - reader will write to cb0 (in_k), instead of cb1 (qv)
    // - compute will wait on cb0 (in_k) and write to cb16 (out_k)
    // - writer will wait on cb16 (out_k), instead of cb1 (qv)
    DataflowBufferSpec in_k_dfb_spec{
        .unique_id = IN_K_DFB,
        .entry_size = single_tile_size,
        .num_entries = cb_num_tiles,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec out_k_dfb_spec{
        .unique_id = OUT_K_DFB,
        .entry_size = single_tile_size,
        .num_entries = cb_num_tiles,
        .data_format_metadata = cb_data_format,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_params = {
        TensorParameter{.unique_id = INPUT_Q_TENSOR, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = Q_OUT_TENSOR, .spec = q.tensor_spec()},
        TensorParameter{.unique_id = K_OUT_TENSOR, .spec = k.tensor_spec()},
        TensorParameter{.unique_id = V_OUT_TENSOR, .spec = v.tensor_spec()}};
    if (read_from_input_tensor_kv) {
        tensor_params.push_back(
            TensorParameter{.unique_id = INPUT_KV_TENSOR, .spec = input_tensor_kv.value().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernel specs
    ////////////////////////////////////////////////////////////////////////////
    // Reader/writer carry TRANSPOSE_K_HEADS (gates the in_k/out_k DFB handle aliases) and the reader
    // additionally carries READ_FROM_INPUT_TENSOR_KV (gates the optional ta::input_kv accessor).
    Table<std::string, std::string> reader_defines;
    Table<std::string, std::string> writer_defines;
    if (transpose_k_heads) {
        reader_defines.insert({"TRANSPOSE_K_HEADS", "1"});
        writer_defines.insert({"TRANSPOSE_K_HEADS", "1"});
    }
    if (read_from_input_tensor_kv) {
        reader_defines.insert({"READ_FROM_INPUT_TENSOR_KV", "1"});
    }

    // Reader binds input_q (+ input_kv when present) and produces QV (and IN_K when transposed).
    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = INPUT_Q_TENSOR, .accessor_name = "input_q"}};
    if (read_from_input_tensor_kv) {
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = INPUT_KV_TENSOR, .accessor_name = "input_kv"});
    }
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{.dfb_spec_name = QV_DFB, .accessor_name = "qv", .endpoint_type = DFBEndpointType::PRODUCER}};
    if (transpose_k_heads) {
        reader_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = IN_K_DFB, .accessor_name = "in_k", .endpoint_type = DFBEndpointType::PRODUCER});
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = {{"q_num_tiles", q_num_tiles}, {"kv_num_tiles", kv_num_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_tensor_tile_id", "in1_tensor_tile_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // Writer binds q/k/v outputs and consumes QV (and OUT_K when transposed).
    Group<DFBBinding> writer_dfb_bindings = {
        DFBBinding{.dfb_spec_name = QV_DFB, .accessor_name = "qv", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (transpose_k_heads) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_K_DFB, .accessor_name = "out_k", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = Q_OUT_TENSOR, .accessor_name = "q_output"},
             TensorBinding{.tensor_parameter_name = K_OUT_TENSOR, .accessor_name = "k_output"},
             TensorBinding{.tensor_parameter_name = V_OUT_TENSOR, .accessor_name = "v_output"}},
        .compile_time_args =
            {{"q_out_h_tiles", q_out_h_tiles},
             {"q_out_w_tiles", q_out_w_tiles},
             {"q_out_HtWt", q_out_HtWt},
             {"q_out_c", num_q_heads},
             {"kv_out_c", num_kv_heads}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_blocks", "q_out_h_dim", "q_out_tensor_tile_id", "k_out_tensor_tile_id", "v_out_tensor_tile_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    Group<KernelSpec> kernels = {reader_spec, writer_spec};

    // Compute (transpose_k_heads only): one KernelSpec per work-split core group, consuming IN_K and
    // producing OUT_K. Per-group CTA (NHtWt) preserved as compile-time multiplicity, not demoted to
    // an RTA. For FLOAT32 input, enable fp32 dest accumulation so the unpack-dst CB resolves to Tf32
    // (10-bit mantissa) instead of Float16_b (7-bit mantissa); Metal 2.0 additionally requires an
    // explicit UnpackToDestFp32 entry for the Float32 IN_K consumer DFB.
    const bool fp32_dest_acc_en = transpose_k_heads && input_tensor.dtype() == tt_metal::DataType::FLOAT32;
    auto make_compute_spec = [&](const KernelSpecName& unique_id, uint32_t num_blocks_per_core_group) {
        ComputeHardwareConfig compute_hw{.fp32_dest_acc_en = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            compute_hw.unpack_to_dest_mode.insert({IN_K_DFB, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = std::filesystem::path{COMPUTE_PATH},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = IN_K_DFB, .accessor_name = "in_k", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_K_DFB, .accessor_name = "out_k", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"NHtWt", num_blocks_per_core_group * kv_num_tiles}},
            .hw_config = compute_hw,
        };
    };

    const bool group_2_present = core_group_2.num_cores() > 0;
    if (transpose_k_heads) {
        kernels.push_back(make_compute_spec(COMPUTE_KERNEL_G1, num_blocks_per_core_group_1));
        if (group_2_present) {
            kernels.push_back(make_compute_spec(COMPUTE_KERNEL_G2, num_blocks_per_core_group_2));
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Per-core runtime args
    ////////////////////////////////////////////////////////////////////////////
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

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

        reader_run.runtime_arg_values.push_back(
            {core,
             {{"num_blocks", num_blocks_per_core},
              {"in0_tensor_tile_id", num_blocks_written * in0_w_tiles},
              {"in1_tensor_tile_id", num_blocks_written * in1_w_tiles}}});

        writer_run.runtime_arg_values.push_back(
            {core,
             {{"num_blocks", num_blocks_per_core},
              {"q_out_h_dim", q_out_h_dim},
              {"q_out_tensor_tile_id", q_out_tensor_tile_id},
              {"k_out_tensor_tile_id", k_out_tensor_tile_id},
              {"v_out_tensor_tile_id", v_out_tensor_tile_id}}});

        num_blocks_written += num_blocks_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    // Reader/writer span all_cores; the compute kernel is split per work-split group, so when
    // transposing each group gets its own WorkUnitSpec (reader + writer + per-group compute). A
    // KernelSpec may appear in multiple WorkUnitSpecs; its effective node set is the union.
    Group<WorkUnitSpec> work_units;
    if (transpose_k_heads) {
        work_units.push_back(WorkUnitSpec{
            .name = "nlp_create_qkv_heads_interleaved_g1",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G1},
            .target_nodes = core_group_1});
        if (group_2_present) {
            work_units.push_back(WorkUnitSpec{
                .name = "nlp_create_qkv_heads_interleaved_g2",
                .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G2},
                .target_nodes = core_group_2});
        }
    } else {
        work_units.push_back(WorkUnitSpec{
            .name = "nlp_create_qkv_heads_interleaved",
            .kernels = {READER_KERNEL, WRITER_KERNEL},
            .target_nodes = all_cores});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble spec + run args
    ////////////////////////////////////////////////////////////////////////////
    Group<DataflowBufferSpec> dataflow_buffers = {qv_dfb_spec};
    if (transpose_k_heads) {
        dataflow_buffers.push_back(in_k_dfb_spec);
        dataflow_buffers.push_back(out_k_dfb_spec);
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_interleaved",
        .kernels = kernels,
        .dataflow_buffers = dataflow_buffers,
        .tensor_parameters = tensor_params,
        .work_units = work_units,
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_Q_TENSOR, TensorArgument{input_tensor.mesh_tensor()}},
        {Q_OUT_TENSOR, TensorArgument{q.mesh_tensor()}},
        {K_OUT_TENSOR, TensorArgument{k.mesh_tensor()}},
        {V_OUT_TENSOR, TensorArgument{v.mesh_tensor()}}};
    if (read_from_input_tensor_kv) {
        run_args.tensor_args.insert({INPUT_KV_TENSOR, TensorArgument{input_tensor_kv.value().mesh_tensor()}});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Sharded::create_program_artifacts(
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
