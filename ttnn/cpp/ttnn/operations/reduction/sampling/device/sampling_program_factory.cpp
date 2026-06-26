// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts SamplingProgramFactory::create_program_artifacts(
    const SamplingParams& operation_attributes, const SamplingInputs& tensor_args, Tensor& output_tensor) {
    // ---- Metal 2.0 named handles (locals for unity-build hygiene) ----
    const DFBSpecName INPUT_VALUES_DFB{"input_values"};
    const DFBSpecName INDEX_DFB{"index"};
    const DFBSpecName SCALER_MAX_DFB{"scaler_max"};
    const DFBSpecName SCALER_SUM_DFB{"scaler_sum"};
    const DFBSpecName TOPK_MASK_DFB{"topk_mask"};
    const DFBSpecName INPUT_TRANSPOSED_DFB{"input_transposed"};
    const DFBSpecName INDEX_TRANSPOSED_DFB{"index_transposed"};
    const DFBSpecName VALUES_DFB{"values"};
    const DFBSpecName LOCAL_VALS_DFB{"local_vals"};
    const DFBSpecName OUTPUT_IND_DFB{"output_ind"};
    const DFBSpecName CUR_MAX_DFB{"cur_max"};
    const DFBSpecName CUR_SUM_DFB{"cur_sum"};
    const DFBSpecName RAND_DFB{"rand"};
    const DFBSpecName FINAL_INDICES_DFB{"final_indices"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const DFBSpecName K_DFB{"k"};
    const DFBSpecName P_DFB{"p"};
    const DFBSpecName TEMP_DFB{"temp"};

    const TensorParamName INPUT_VALUES_TENSOR{"input_values_tensor"};
    const TensorParamName INPUT_INDICES_TENSOR{"input_indices_tensor"};
    const TensorParamName OUTPUT_TENSOR{"output_tensor"};
    const TensorParamName TEMP_TENSOR{"temp_tensor"};
    const TensorParamName K_TENSOR{"k_tensor"};
    const TensorParamName P_TENSOR{"p_tensor"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    const auto& input_values_tensor = tensor_args.input_values.mesh_tensor();
    const auto& input_indices_tensor = tensor_args.input_indices.mesh_tensor();
    const auto& k = tensor_args.k.mesh_tensor();
    const auto& p = tensor_args.p.mesh_tensor();
    const auto& temp = tensor_args.temp.mesh_tensor();

    const auto& seed = operation_attributes.seed;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    uint32_t random_seed = 0;

    auto* device = &input_values_tensor.mutable_device();

    // See legacy notes: WH/BH use the cheaper 16-bit (UInt16) index path with fp32 dest acc off;
    // every other arch (Quasar) uses 32-bit (Int32) indices with fp32 dest acc on.
    const bool use_32bit_index = !(device->arch() == tt::ARCH::WORMHOLE_B0 || device->arch() == tt::ARCH::BLACKHOLE);

    tt::DataFormat input_values_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_values_tensor.dtype());
    tt::DataFormat input_indices_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_indices_tensor.dtype());
    tt::DataFormat index_cb_data_format = use_32bit_index ? tt::DataFormat::Int32 : tt::DataFormat::UInt16;
    tt::DataFormat k_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(k.dtype());
    tt::DataFormat p_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(p.dtype());
    tt::DataFormat temp_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(temp.dtype());

    uint32_t input_values_tile_size = tile_size(input_values_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    const auto& output_mesh = output_tensor.mesh_tensor();

    auto input_shape = input_values_tensor.logical_shape();
    const uint32_t tile_height = input_values_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_values_tensor.tensor_spec().tile().get_width();
    const uint32_t num_users = input_shape[2];
    uint32_t Ht = (num_users + tile_height - 1) / tile_height;  // == 1 for 1..32 users
    uint32_t Wt = input_shape[3] / tile_width;
    auto num_cores = num_users;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);

    if (sub_core_grids.has_value()) {
        core_grid = sub_core_grids.value();
    }
    auto cores = corerange_to_cores(core_grid, num_cores, true);

    // Confine to exactly the running cores (sub_core_grids may be over-provisioned).
    if (core_grid.num_cores() != num_cores) {
        std::vector<CoreRange> active_core_ranges;
        active_core_ranges.reserve(cores.size());
        for (const auto& core : cores) {
            active_core_ranges.emplace_back(core);
        }
        core_grid = CoreRangeSet(std::move(active_core_ranges));
    }

    validate_reduce_op_program_grid(
        "Sampling",
        core_grid,
        compute_with_storage_grid_size,
        sub_core_grids.has_value() ? &sub_core_grids.value() : nullptr,
        true,
        {});

    if (seed.has_value()) {
        random_seed = seed.value();
    }

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // ---- Sizing (mirrors the legacy CBDescriptor total_size / page_size) ----
    tt::DataFormat scalar_df =
        (input_values_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scale_tiles = 1;
    uint32_t scalar_tile_size = tile_size(scalar_df);
    uint32_t num_out_tiles = Ht;
    const uint32_t rand_tile_size = tile_size(tt::DataFormat::Float16_b);

    uint32_t final_indices_rm_unit_size = input_indices_tensor.element_size();  // 4 for int32
    uint32_t aligned_final_indices_rm_unit_size = Wt * tile_width * final_indices_rm_unit_size;

    uint32_t output_unit_size = output_mesh.element_size();
    uint32_t aligned_out0_unit_size = Ht * tile_height * output_unit_size;

    const uint32_t uint32_bytes = 4;
    const uint32_t bf16_bytes = 2;
    uint32_t k_chunk_size = num_cores * uint32_bytes;
    uint32_t p_chunk_size = num_cores * bf16_bytes;
    uint32_t temp_chunk_size = num_cores * bf16_bytes;

    // ---- Tensor parameters ----
    TensorParameter input_values_param{.unique_id = INPUT_VALUES_TENSOR, .spec = input_values_tensor.tensor_spec()};
    TensorParameter input_indices_param{.unique_id = INPUT_INDICES_TENSOR, .spec = input_indices_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};
    TensorParameter temp_param{.unique_id = TEMP_TENSOR, .spec = temp.tensor_spec()};
    TensorParameter k_param{.unique_id = K_TENSOR, .spec = k.tensor_spec()};
    TensorParameter p_param{.unique_id = P_TENSOR, .spec = p.tensor_spec()};

    // ---- Dataflow buffers (one per legacy CB; entry_size = page_size, num_entries = total/page) ----
    DataflowBufferSpec input_values_dfb_spec{
        .unique_id = INPUT_VALUES_DFB,
        .entry_size = input_values_tile_size,
        .num_entries = cb_in_units,
        .data_format_metadata = input_values_cb_data_format};
    DataflowBufferSpec index_dfb_spec{
        .unique_id = INDEX_DFB,
        .entry_size = index_tile_size,
        .num_entries = cb_in_units,
        .data_format_metadata = index_cb_data_format};
    DataflowBufferSpec scaler_max_dfb_spec{
        .unique_id = SCALER_MAX_DFB,
        .entry_size = scalar_tile_size,
        .num_entries = scale_tiles,
        .data_format_metadata = scalar_df};
    DataflowBufferSpec scaler_sum_dfb_spec{
        .unique_id = SCALER_SUM_DFB,
        .entry_size = scalar_tile_size,
        .num_entries = scale_tiles,
        .data_format_metadata = scalar_df};
    DataflowBufferSpec topk_mask_dfb_spec{
        .unique_id = TOPK_MASK_DFB,
        .entry_size = input_values_tile_size,
        .num_entries = cb_in_units,
        .data_format_metadata = input_values_cb_data_format};
    DataflowBufferSpec input_transposed_dfb_spec{
        .unique_id = INPUT_TRANSPOSED_DFB,
        .entry_size = input_values_tile_size,
        .num_entries = Wt,
        .data_format_metadata = input_values_cb_data_format};
    DataflowBufferSpec index_transposed_dfb_spec{
        .unique_id = INDEX_TRANSPOSED_DFB,
        .entry_size = index_tile_size,
        .num_entries = Wt,
        .data_format_metadata = index_cb_data_format};
    DataflowBufferSpec values_dfb_spec{
        .unique_id = VALUES_DFB,
        .entry_size = input_values_tile_size,
        .num_entries = num_cb_unit,
        .data_format_metadata = input_values_cb_data_format};
    DataflowBufferSpec local_vals_dfb_spec{
        .unique_id = LOCAL_VALS_DFB,
        .entry_size = input_values_tile_size,
        .num_entries = num_cb_unit,
        .data_format_metadata = input_values_cb_data_format};
    DataflowBufferSpec output_ind_dfb_spec{
        .unique_id = OUTPUT_IND_DFB,
        .entry_size = index_tile_size,
        .num_entries = num_cb_unit,
        .data_format_metadata = index_cb_data_format};
    DataflowBufferSpec cur_max_dfb_spec{
        .unique_id = CUR_MAX_DFB,
        .entry_size = input_values_tile_size,
        .num_entries = num_out_tiles,
        .data_format_metadata = input_values_cb_data_format};
    DataflowBufferSpec cur_sum_dfb_spec{
        .unique_id = CUR_SUM_DFB,
        .entry_size = input_values_tile_size,
        .num_entries = num_out_tiles,
        .data_format_metadata = input_values_cb_data_format};
    DataflowBufferSpec rand_dfb_spec{
        .unique_id = RAND_DFB,
        .entry_size = rand_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b};
    DataflowBufferSpec final_indices_dfb_spec{
        .unique_id = FINAL_INDICES_DFB,
        .entry_size = aligned_final_indices_rm_unit_size,
        .num_entries = Ht * tile_height,
        .data_format_metadata = input_indices_cb_data_format};
    DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = aligned_out0_unit_size,
        .num_entries = 1,
        .data_format_metadata = index_cb_data_format};
    DataflowBufferSpec k_dfb_spec{
        .unique_id = K_DFB, .entry_size = k_chunk_size, .num_entries = 1, .data_format_metadata = k_cb_data_format};
    DataflowBufferSpec p_dfb_spec{
        .unique_id = P_DFB, .entry_size = p_chunk_size, .num_entries = 1, .data_format_metadata = p_cb_data_format};
    DataflowBufferSpec temp_dfb_spec{
        .unique_id = TEMP_DFB,
        .entry_size = temp_chunk_size,
        .num_entries = 1,
        .data_format_metadata = temp_cb_data_format};

    // ---- Reader: streams input_values (c_0), generates index (c_2), reads input_indices ->
    //      final_indices (c_12), and the k/p chunks (c_14/c_15, relocated from the writer). ----
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp"},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_VALUES_DFB,
                 .accessor_name = "input_values",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = FINAL_INDICES_DFB,
                 .accessor_name = "final_indices",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = K_DFB, .accessor_name = "k", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = P_DFB, .accessor_name = "p", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT_VALUES_TENSOR, .accessor_name = "input_values"},
             TensorBinding{.tensor_parameter_name = INPUT_INDICES_TENSOR, .accessor_name = "input_indices"},
             TensorBinding{.tensor_parameter_name = K_TENSOR, .accessor_name = "k"},
             TensorBinding{.tensor_parameter_name = P_TENSOR, .accessor_name = "p"}},
        .compile_time_args =
            {{"Ht", Ht},
             {"Wt", Wt},
             {"final_indices_page_size", aligned_final_indices_rm_unit_size},
             {"tile_height", tile_height},
             {"use_32bit_index", static_cast<uint32_t>(use_32bit_index)},
             {"num_users", num_users},
             {"num_cores", num_cores},
             {"k_chunk_size", k_chunk_size},
             {"p_chunk_size", p_chunk_size}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ---- Writer: produces scaler_max/sum (c_3/c_17), topk_mask (c_4), temp (c_16); consumes
    //      final_indices/rand/local_vals/output_ind/k/p; assembles + DMAs output (c_13). ----
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp"},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = FINAL_INDICES_DFB,
                 .accessor_name = "final_indices",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = RAND_DFB, .accessor_name = "rand", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = LOCAL_VALS_DFB,
                 .accessor_name = "local_vals",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_IND_DFB,
                 .accessor_name = "output_ind",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = K_DFB, .accessor_name = "k", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = P_DFB, .accessor_name = "p", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = SCALER_MAX_DFB,
                 .accessor_name = "scaler_max",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SCALER_SUM_DFB,
                 .accessor_name = "scaler_sum",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TOPK_MASK_DFB,
                 .accessor_name = "topk_mask",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = TEMP_DFB, .accessor_name = "temp", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
             TensorBinding{.tensor_parameter_name = TEMP_TENSOR, .accessor_name = "temp"}},
        .compile_time_args =
            {{"final_indices_stick_size", aligned_final_indices_rm_unit_size},
             {"out_stick_size", aligned_out0_unit_size},
             {"ids_per_batch", tile_width},
             {"num_cores", num_cores},
             {"use_32bit_index", static_cast<uint32_t>(use_32bit_index)},
             {"num_users", num_users}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ---- Compute: top-k + softmax; the OUTPUT DFB (c_13) is bound here as a TERMINAL no-op
    //      CONSUMER so the writer-produced output CB has a legal cross-kernel consumer. ----
    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp"},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_VALUES_DFB,
                 .accessor_name = "input_values",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = INPUT_TRANSPOSED_DFB,
                 .accessor_name = "input_transposed",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = INPUT_TRANSPOSED_DFB,
                 .accessor_name = "input_transposed",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = INDEX_TRANSPOSED_DFB,
                 .accessor_name = "index_transposed",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = INDEX_TRANSPOSED_DFB,
                 .accessor_name = "index_transposed",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = VALUES_DFB, .accessor_name = "values", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = VALUES_DFB, .accessor_name = "values", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_IND_DFB,
                 .accessor_name = "output_ind",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TOPK_MASK_DFB,
                 .accessor_name = "topk_mask",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = SCALER_MAX_DFB,
                 .accessor_name = "scaler_max",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = SCALER_SUM_DFB,
                 .accessor_name = "scaler_sum",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = CUR_MAX_DFB, .accessor_name = "cur_max", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = CUR_MAX_DFB, .accessor_name = "cur_max", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = CUR_SUM_DFB, .accessor_name = "cur_sum", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = CUR_SUM_DFB, .accessor_name = "cur_sum", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = RAND_DFB, .accessor_name = "rand", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = LOCAL_VALS_DFB,
                 .accessor_name = "local_vals",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = TEMP_DFB, .accessor_name = "temp", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {{"Ht", Ht},
             {"Wt", Wt},
             {"logWt", static_cast<uint32_t>(std::log2(Wt))},
             {"seed", random_seed},
             {"tile_width", tile_width}},
        .hw_config = ComputeHardwareConfig{.fp32_dest_acc_en = use_32bit_index},
    };
    // INTRA self-loops (compute-kernel-internal FIFOs) + the no-op output bridge consumer.
    compute_spec.advanced_options.dfb_self_loop_connectivities = {
        {INPUT_TRANSPOSED_DFB, DFBSelfLoopConnectivity::INTRA},
        {INDEX_TRANSPOSED_DFB, DFBSelfLoopConnectivity::INTRA},
        {VALUES_DFB, DFBSelfLoopConnectivity::INTRA},
        {CUR_MAX_DFB, DFBSelfLoopConnectivity::INTRA},
        {CUR_SUM_DFB, DFBSelfLoopConnectivity::INTRA},
    };

    // ---- Per-core runtime args: only the writer needs core_id (== index in `cores`). ----
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    writer_run.runtime_arg_values.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        writer_run.runtime_arg_values.push_back({cores[i], KernelRunArgs::RuntimeArgValues{{"core_id", i}}});
    }

    WorkUnitSpec wu{
        .name = "sampling",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = core_grid,
    };

    ProgramSpec spec{
        .name = "sampling",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {input_values_dfb_spec, index_dfb_spec,      scaler_max_dfb_spec,       scaler_sum_dfb_spec,
             topk_mask_dfb_spec,    input_transposed_dfb_spec, index_transposed_dfb_spec, values_dfb_spec,
             local_vals_dfb_spec,   output_ind_dfb_spec, cur_max_dfb_spec,          cur_sum_dfb_spec,
             rand_dfb_spec,         final_indices_dfb_spec,    output_dfb_spec,           k_dfb_spec,
             p_dfb_spec,            temp_dfb_spec},
        .tensor_parameters =
            {input_values_param, input_indices_param, output_param, temp_param, k_param, p_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {writer_run};
    run_args.tensor_args = {
        {INPUT_VALUES_TENSOR, TensorArgument{std::cref(input_values_tensor)}},
        {INPUT_INDICES_TENSOR, TensorArgument{std::cref(input_indices_tensor)}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}},
        {TEMP_TENSOR, TensorArgument{std::cref(temp)}},
        {K_TENSOR, TensorArgument{std::cref(k)}},
        {P_TENSOR, TensorArgument{std::cref(p)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
