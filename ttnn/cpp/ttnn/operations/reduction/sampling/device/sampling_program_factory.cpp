// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

namespace ttnn::prim {

namespace {
namespace m2 = tt::tt_metal::experimental;
using namespace tt::tt_metal;

// Resource identifier constants
constexpr const char* K_READER = "reader";
constexpr const char* K_WRITER = "writer";
constexpr const char* K_COMPUTE = "compute";
constexpr const char* WU_MAIN = "main";

// DFB unique-ids
constexpr const char* DFB_INPUT_VALUES = "input_values";
constexpr const char* DFB_INDEX = "index";                    // c_2: per-tile index intermediate
constexpr const char* DFB_LOCAL_VALS = "cb_local_vals";       // c_1
constexpr const char* DFB_SCALER_MAX = "scaler_max";          // c_3
constexpr const char* DFB_TOPK_MASK = "topk_mask";            // c_4
constexpr const char* DFB_INPUT_TRANS = "input_transposed";   // c_5
constexpr const char* DFB_INDEX_TRANS = "index_transposed";   // c_6
constexpr const char* DFB_VALUES = "values";                  // c_7
constexpr const char* DFB_OUTPUT_IND = "output_ind";          // c_8
constexpr const char* DFB_CUR_MAX = "cur_max";                // c_9
constexpr const char* DFB_CUR_SUM = "cur_sum";                // c_10
constexpr const char* DFB_RAND_TILE = "rand_tile";            // c_11
constexpr const char* DFB_FINAL_IDX_RM = "final_indices_rm";  // c_12
constexpr const char* DFB_OUTPUT = "output";                  // c_13
constexpr const char* DFB_K = "cb_k";                         // c_14
constexpr const char* DFB_P = "cb_p";                         // c_15
constexpr const char* DFB_TEMP = "cb_temp";                   // c_16
constexpr const char* DFB_SCALER_SUM = "scaler_sum";          // c_17

// TensorParameter unique-ids
constexpr const char* TP_INPUT_VALUES = "input_values";
constexpr const char* TP_INPUT_INDICES = "input_indices";
constexpr const char* TP_K = "k";
constexpr const char* TP_P = "p";
constexpr const char* TP_TEMP = "temp";
constexpr const char* TP_OUTPUT = "output";

inline m2::DFBSpecName DfbName(const char* name) { return m2::DFBSpecName{std::string{name}}; }
inline m2::TensorParamName TpName(const char* name) { return m2::TensorParamName{std::string{name}}; }
inline m2::KernelSpecName KernelName(const char* name) { return m2::KernelSpecName{std::string{name}}; }

m2::KernelSpec::DFBBinding ProducerDFB(const char* dfb_name, const char* accessor_name) {
    return m2::KernelSpec::DFBBinding{
        .dfb_spec_name = DfbName(dfb_name),
        .accessor_name = accessor_name,
        .endpoint_type = m2::DFBEndpointType::PRODUCER};
}
m2::KernelSpec::DFBBinding ConsumerDFB(const char* dfb_name, const char* accessor_name) {
    return m2::KernelSpec::DFBBinding{
        .dfb_spec_name = DfbName(dfb_name),
        .accessor_name = accessor_name,
        .endpoint_type = m2::DFBEndpointType::CONSUMER};
}

m2::DataflowBufferSpec MakeDFB(
    const char* unique_id, uint32_t entry_size, uint32_t num_entries, tt::DataFormat data_format) {
    m2::DataflowBufferSpec dfb{
        .unique_id = DfbName(unique_id),
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format};
    return dfb;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts SamplingProgramFactory::create_program_spec(
    const SamplingParams& operation_attributes, const SamplingInputs& tensor_args, Tensor& output_tensor) {
    using namespace tt::tt_metal;

    const auto& input_values_tensor = tensor_args.input_values;
    const auto& input_indices_tensor = tensor_args.input_indices;
    const auto& k = tensor_args.k;
    const auto& p = tensor_args.p;
    const auto& temp = tensor_args.temp;

    const auto& seed = operation_attributes.seed;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    uint32_t random_seed = seed.value_or(0u);

    tt::DataFormat input_values_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_values_tensor.dtype());
    tt::DataFormat input_indices_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_indices_tensor.dtype());
    // Quasar does not support UInt16 DFB metadata; keep index intermediates in 32-bit format.
    tt::DataFormat index_cb_data_format = tt::DataFormat::Int32;
    tt::DataFormat k_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(k.dtype());
    // Quasar does not support UInt32 DFB metadata. k values are non-negative and fit in signed
    // 32-bit, so represent this staging DFB as Int32 for compatibility.
    if (k_cb_data_format == tt::DataFormat::UInt32) {
        k_cb_data_format = tt::DataFormat::Int32;
    }
    tt::DataFormat p_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(p.dtype());
    tt::DataFormat temp_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(temp.dtype());

    uint32_t input_values_tile_size = tile_size(input_values_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto* device = input_values_tensor.device();

    auto input_shape = input_values_tensor.logical_shape();
    const uint32_t tile_height = input_values_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_values_tensor.tensor_spec().tile().get_width();
    uint32_t Ht = 1;
    const uint32_t num_users = input_shape[0] * input_shape[1] * input_shape[2];
    uint32_t Wt = input_shape[3] / tile_width;
    uint32_t num_cores = num_users;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);
    if (sub_core_grids.has_value()) {
        core_grid = sub_core_grids.value();
    }
    auto cores = corerange_to_cores(core_grid, num_cores, true);

    validate_reduce_op_program_grid(
        "Sampling",
        core_grid,
        compute_with_storage_grid_size,
        sub_core_grids.has_value() ? &sub_core_grids.value() : nullptr,
        true,
        {});

    constexpr uint32_t num_cb_unit = 2;
    constexpr uint32_t cb_in_units = 2 * num_cb_unit;

    // Sizes
    tt::DataFormat scalar_df =
        (input_values_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scale_tiles = 1;
    const uint32_t scalar_tile_size = tile_size(scalar_df);
    const uint32_t rand_tile_size = tile_size(tt::DataFormat::Float16_b);

    const uint32_t final_indices_rm_unit_size = input_indices_tensor.element_size();
    const uint32_t aligned_final_indices_rm_unit_size = Wt * tile_width * final_indices_rm_unit_size;
    const uint32_t output_unit_size = output_tensor.element_size();
    const uint32_t aligned_out0_unit_size = Ht * tile_height * output_unit_size;

    const uint32_t k_chunk_size = num_cores * sizeof(uint32_t);
    const uint32_t p_chunk_size = num_cores * sizeof(uint16_t);
    const uint32_t temp_chunk_size = num_cores * sizeof(uint16_t);

    const uint32_t num_out_tiles = Ht;

    ////////////////////////////////////////////////////////////////////////////
    // DataflowBufferSpecs (mapped 1:1 with legacy CBs; no borrowed memory)
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(MakeDFB(DFB_INPUT_VALUES, input_values_tile_size, cb_in_units, input_values_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_LOCAL_VALS, input_values_tile_size, num_cb_unit, input_values_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_INDEX, index_tile_size, cb_in_units, index_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_SCALER_MAX, scalar_tile_size, scale_tiles, scalar_df));
    dfbs.push_back(MakeDFB(DFB_TOPK_MASK, input_values_tile_size, cb_in_units, input_values_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_INPUT_TRANS, input_values_tile_size, Wt, input_values_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_INDEX_TRANS, index_tile_size, Wt, index_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_VALUES, input_values_tile_size, num_cb_unit, input_values_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_OUTPUT_IND, index_tile_size, num_cb_unit, index_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_CUR_MAX, input_values_tile_size, num_out_tiles, input_values_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_CUR_SUM, input_values_tile_size, num_out_tiles, input_values_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_RAND_TILE, rand_tile_size, 1, tt::DataFormat::Float16_b));
    dfbs.push_back(
        MakeDFB(DFB_FINAL_IDX_RM, aligned_final_indices_rm_unit_size, Ht * tile_height, input_indices_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_OUTPUT, aligned_out0_unit_size, 1, index_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_K, k_chunk_size, 1, k_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_P, p_chunk_size, 1, p_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_TEMP, temp_chunk_size, 1, temp_cb_data_format));
    dfbs.push_back(MakeDFB(DFB_SCALER_SUM, scalar_tile_size, scale_tiles, scalar_df));

    ////////////////////////////////////////////////////////////////////////////
    // TensorParameters
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back({.unique_id = TpName(TP_INPUT_VALUES), .spec = input_values_tensor.tensor_spec()});
    tensor_parameters.push_back({.unique_id = TpName(TP_INPUT_INDICES), .spec = input_indices_tensor.tensor_spec()});
    tensor_parameters.push_back({.unique_id = TpName(TP_K), .spec = k.tensor_spec()});
    tensor_parameters.push_back({.unique_id = TpName(TP_P), .spec = p.tensor_spec()});
    tensor_parameters.push_back({.unique_id = TpName(TP_TEMP), .spec = temp.tensor_spec()});
    tensor_parameters.push_back({.unique_id = TpName(TP_OUTPUT), .spec = output_tensor.tensor_spec()});

    ////////////////////////////////////////////////////////////////////////////
    // Reader KernelSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader_spec;
    reader_spec.unique_id = KernelName(K_READER);
    reader_spec.source = std::filesystem::path{
        "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp"};
    reader_spec.hw_config = m2::DataMovementHardwareConfig{
        .gen1_config =
            m2::DataMovementHardwareConfig::Gen1Config{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        .gen2_config = m2::DataMovementHardwareConfig::Gen2Config{}};
    reader_spec.compile_time_args = {
        {"Ht", Ht},
        {"Wt", Wt},
        {"input_indices_page_size", aligned_final_indices_rm_unit_size},
        {"tile_height", tile_height},
        {"num_users", num_users},
    };
    reader_spec.dfb_bindings = {
        ProducerDFB(DFB_INPUT_VALUES, "cb_input_values"),
        ProducerDFB(DFB_FINAL_IDX_RM, "cb_final_indices_rm"),
        ProducerDFB(DFB_INDEX, "cb_index"),
    };
    reader_spec.tensor_bindings = {
        {.tensor_parameter_name = TpName(TP_INPUT_VALUES), .accessor_name = "values"},
        {.tensor_parameter_name = TpName(TP_INPUT_INDICES), .accessor_name = "indices"},
    };

    ////////////////////////////////////////////////////////////////////////////
    // Writer KernelSpec — single KernelSpec; core_id is per-node named RTA
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec writer_spec;
    writer_spec.unique_id = KernelName(K_WRITER);
    writer_spec.source = std::filesystem::path{
        "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp"};
    writer_spec.hw_config = m2::DataMovementHardwareConfig{
        .gen1_config =
            m2::DataMovementHardwareConfig::Gen1Config{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        .gen2_config = m2::DataMovementHardwareConfig::Gen2Config{}};
    writer_spec.compile_time_args = {
        {"final_indices_stick_size", aligned_final_indices_rm_unit_size},
        {"out_stick_size", aligned_out0_unit_size},
        {"ids_per_batch", tile_width},
        {"num_cores", num_cores},
        {"num_users", num_users},
    };
    // cb_out / cb_k / cb_p are writer-local staging — writer pushes via push_back but no
    // other kernel calls wait_front. Add ghost CONSUMER bindings on the writer itself so
    // the validator's ≥1 PRODUCER + ≥1 CONSUMER rule is satisfied without any kernel-side
    // pop_front actually happening (per-enqueue DFB state is reinitialized).
    // cb_temp: writer is producer (generate_bcast_unary_scalar pushes), compute consumes
    // via wait_front in mul_block_bcast_scalar_inplace — no ghost needed.
    // cb_mask / scaler_max / scaler_sum: writer is producer, compute consumes.
    writer_spec.dfb_bindings = {
        ProducerDFB(DFB_OUTPUT, "cb_out"),
        ConsumerDFB(DFB_OUTPUT, "cb_out"),
        ProducerDFB(DFB_TOPK_MASK, "cb_mask"),
        ProducerDFB(DFB_SCALER_MAX, "scaler_max"),
        ProducerDFB(DFB_SCALER_SUM, "scaler_sum"),
        ConsumerDFB(DFB_FINAL_IDX_RM, "final_indices_rm"),
        ConsumerDFB(DFB_LOCAL_VALS, "local_vals"),
        ConsumerDFB(DFB_OUTPUT_IND, "local_indices"),
        ConsumerDFB(DFB_RAND_TILE, "rand_tile"),
        ProducerDFB(DFB_K, "cb_k"),
        ConsumerDFB(DFB_K, "cb_k"),
        ProducerDFB(DFB_P, "cb_p"),
        ConsumerDFB(DFB_P, "cb_p"),
        ProducerDFB(DFB_TEMP, "cb_temp"),
    };
    writer_spec.tensor_bindings = {
        {.tensor_parameter_name = TpName(TP_OUTPUT), .accessor_name = "output"},
        {.tensor_parameter_name = TpName(TP_TEMP), .accessor_name = "temp"},
        {.tensor_parameter_name = TpName(TP_K), .accessor_name = "k_tensor"},
        {.tensor_parameter_name = TpName(TP_P), .accessor_name = "p_tensor"},
    };
    writer_spec.runtime_arg_schema.runtime_arg_names = {"core_id"};

    ////////////////////////////////////////////////////////////////////////////
    // Compute KernelSpec — single KernelSpec; CTAs are uniform across cores
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec compute_spec;
    compute_spec.unique_id = KernelName(K_COMPUTE);
    compute_spec.source =
        std::filesystem::path{"ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp"};
    compute_spec.hw_config = m2::ComputeHardwareConfig{};
    compute_spec.compile_time_args = {
        {"Ht", Ht},
        {"Wt", Wt},
        {"logWt", static_cast<uint32_t>(std::log2(Wt))},
        {"seed", random_seed},
        {"tile_width", tile_width},
    };
    // Compute self-loop DFBs (compute is both producer and consumer). Metal 2.0 requires
    // distinct `local_accessor_name`s for the two endpoints on the same kernel; on Gen1
    // both `dfb::*_w` and `dfb::*_r` resolve to the same underlying CB id, so the kernel
    // can keep using a single DataflowBuffer object constructed from either handle.
    auto self_loop = [&](const char* name, const char* base) {
        std::string w = std::string(base) + "_w";
        std::string r = std::string(base) + "_r";
        compute_spec.dfb_bindings.push_back(ProducerDFB(name, w.c_str()));
        compute_spec.dfb_bindings.push_back(ConsumerDFB(name, r.c_str()));
        compute_spec.advanced_options.dfb_self_loop_connectivities.emplace(
            DfbName(name), m2::DFBSelfLoopConnectivity::INTRA);
    };
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_INPUT_VALUES, "input_values"));
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_INDEX, "index"));
    self_loop(DFB_INPUT_TRANS, "input_transposed");
    self_loop(DFB_INDEX_TRANS, "index_transposed");
    self_loop(DFB_VALUES, "values");
    compute_spec.dfb_bindings.push_back(ProducerDFB(DFB_OUTPUT_IND, "output_ind"));
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_TOPK_MASK, "topk_mask"));
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_SCALER_MAX, "scaler_max"));
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_SCALER_SUM, "scaler_sum"));
    self_loop(DFB_CUR_MAX, "cur_max");
    self_loop(DFB_CUR_SUM, "cur_sum");
    compute_spec.dfb_bindings.push_back(ProducerDFB(DFB_RAND_TILE, "rand_tile"));
    compute_spec.dfb_bindings.push_back(ProducerDFB(DFB_LOCAL_VALS, "local_vals"));
    compute_spec.dfb_bindings.push_back(ConsumerDFB(DFB_TEMP, "temp"));

    ////////////////////////////////////////////////////////////////////////////
    // WorkUnitSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::WorkUnitSpec wu{
        .name = WU_MAIN,
        .kernels = {KernelName(K_READER), KernelName(K_WRITER), KernelName(K_COMPUTE)},
        .target_nodes = core_grid,  // NodeRangeSet aliases CoreRangeSet
    };

    ////////////////////////////////////////////////////////////////////////////
    // ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .name = "sampling",
        .kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)},
        .dataflow_buffers = std::move(dfbs),
        .semaphores = {},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {std::move(wu)},
    };

    ////////////////////////////////////////////////////////////////////////////
    // ProgramRunParams
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramRunArgs run_params;

    m2::ProgramRunArgs::KernelRunArgs writer_run{.kernel = KernelName(K_WRITER)};
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        writer_run.runtime_arg_values.push_back({.node = m2::NodeCoord{core.x, core.y}, .args = {{"core_id", i}}});
    }
    // Per program_run_args.hpp: "KernelRunParams must be specified for ALL kernels in
    // the ProgramSpec." Reader and compute have no named RTAs/varargs, but we still need
    // an entry (with the kernel_spec_name set) so the framework doesn't fault on the
    // missing kernel.
    run_params.kernel_run_args.push_back(m2::ProgramRunArgs::KernelRunArgs{.kernel = KernelName(K_READER)});
    run_params.kernel_run_args.push_back(std::move(writer_run));
    run_params.kernel_run_args.push_back(m2::ProgramRunArgs::KernelRunArgs{.kernel = KernelName(K_COMPUTE)});

    // Tensor args
    run_params.tensor_args.emplace(TpName(TP_INPUT_VALUES), std::cref(input_values_tensor.mesh_tensor()));
    run_params.tensor_args.emplace(TpName(TP_INPUT_INDICES), std::cref(input_indices_tensor.mesh_tensor()));
    run_params.tensor_args.emplace(TpName(TP_K), std::cref(k.mesh_tensor()));
    run_params.tensor_args.emplace(TpName(TP_P), std::cref(p.mesh_tensor()));
    run_params.tensor_args.emplace(TpName(TP_TEMP), std::cref(temp.mesh_tensor()));
    run_params.tensor_args.emplace(TpName(TP_OUTPUT), std::cref(output_tensor.mesh_tensor()));

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
