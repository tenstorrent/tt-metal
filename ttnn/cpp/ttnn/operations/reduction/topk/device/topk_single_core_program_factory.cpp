// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"

#include <filesystem>
#include <limits>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp";
constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp";

// DFB names (formerly the c_0..c_7 circular-buffer indices). Unique file-scope identifiers.
constexpr const char* DFB_INPUT = "topk_sc_input";
constexpr const char* DFB_INDEX = "topk_sc_index";
constexpr const char* DFB_TRANSPOSED_VAL = "topk_sc_transposed_val";
constexpr const char* DFB_TRANSPOSED_IND = "topk_sc_transposed_ind";
constexpr const char* DFB_RESULT_PREP_VAL = "topk_sc_result_prep_val";
constexpr const char* DFB_RESULT_PREP_IND = "topk_sc_result_prep_ind";
constexpr const char* DFB_OUTPUT_VAL = "topk_sc_output_val";
constexpr const char* DFB_OUTPUT_IND = "topk_sc_output_ind";

// Tensor-parameter names.
constexpr const char* TP_INPUT = "topk_sc_input_tensor";
constexpr const char* TP_VALUES = "topk_sc_values_tensor";
constexpr const char* TP_INDICES = "topk_sc_indices_tensor";

m2::DataflowBufferSpec dfb(const char* id, uint32_t entry, uint32_t n, tt::DataFormat f) {
    return m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{id}, .entry_size = entry, .num_entries = n, .data_format_metadata = f};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TopKDeviceOperation::TopKSingleCoreProgramFactory::create_program_artifacts(
    const TopkParams& operation_attributes,
    const TopkInputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;
    // Tensor references
    const auto& input_tensor = tensor_args.input.mesh_tensor();
    const auto& value_tensor = std::get<0>(output_tensors).mesh_tensor();
    const auto& index_tensor = std::get<1>(output_tensors).mesh_tensor();

    // Determine index output format based on dimension size constraints
    const ttnn::Shape input_shape = input_tensor.padded_shape();
    const bool uint16_output = (input_shape[args.dim] < std::numeric_limits<uint16_t>::max());

    // Data format conversions for dataflow buffer configurations
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat output_val_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    const tt::DataFormat output_ind_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    // Use bf16 for compute intermediate buffers to avoid precision loss from bfp8/bfp4
    // shared-exponent grouping during sort (e.g. a single inf in a block makes all other
    // elements in that block encode to 0, corrupting the sort result).
    const tt::DataFormat compute_cb_data_format =
        (input_cb_data_format == tt::DataFormat::Bfp8_b || input_cb_data_format == tt::DataFormat::Bfp4_b)
            ? tt::DataFormat::Float16_b
            : input_cb_data_format;

    // Calculate tile sizes for memory allocation
    const uint32_t input_tile_size = tile_size(input_cb_data_format);
    const uint32_t value_tile_size = tile_size(output_val_cb_data_format);
    const uint32_t index_tile_size = tile_size(output_ind_cb_data_format);
    const uint32_t compute_tile_size = tile_size(compute_cb_data_format);

    // Tensor shape and dimension calculations
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    // Single core selection from the provided core grid
    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(args.sub_core_grids, Ht, true);
    TT_FATAL(
        args.sub_core_grids.contains(core_range),
        "TopK single-core program core grid {} must be contained in sub_core_grids {}",
        core_range,
        args.sub_core_grids);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_range, total_number_of_cores, true);

    // Number of tiles needed to store K top elements
    const uint32_t Ktiles = tt::div_up(args.k, tile_width);

    // Pipeline Flow:
    // Input DFB -> Reader Kernel -> Transposed DFBs -> Compute Kernel -> Result Prep DFBs -> Output DFBs -> Writer
    const uint32_t num_cb_unit = 2;                         // Base unit for double buffering
    const uint32_t cb_in_units = num_cb_unit;               // 2 units total for input double buffering
    const uint32_t input_cb_tile_count = cb_in_units;       // Input stream buffer size
    const uint32_t transposed_cb_tile_count = 4;            // Transposed data staging
    const uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // Intermediate TopK results (double-buffered)
    const uint32_t output_cb_tile_count = Ktiles;           // Final output buffer

    // ---- Dataflow buffers (formerly the c_0..c_7 circular buffers). ----
    // Uses bf16 when input is bfp8/bfp4 so that the insertion sort operates at higher precision and
    // avoids shared-exponent corruption of tiles adjacent to inf values (transposed/result-prep value DFBs).
    std::vector<m2::DataflowBufferSpec> dfbs = {
        dfb(DFB_INPUT, input_tile_size, input_cb_tile_count, input_cb_data_format),
        dfb(DFB_INDEX, index_tile_size, input_cb_tile_count, output_ind_cb_data_format),
        dfb(DFB_TRANSPOSED_VAL, compute_tile_size, transposed_cb_tile_count, compute_cb_data_format),
        dfb(DFB_TRANSPOSED_IND, index_tile_size, transposed_cb_tile_count, output_ind_cb_data_format),
        dfb(DFB_RESULT_PREP_VAL, compute_tile_size, result_prep_cb_tile_count, compute_cb_data_format),
        dfb(DFB_RESULT_PREP_IND, index_tile_size, result_prep_cb_tile_count, output_ind_cb_data_format),
        dfb(DFB_OUTPUT_VAL, value_tile_size, output_cb_tile_count, output_val_cb_data_format),
        dfb(DFB_OUTPUT_IND, index_tile_size, output_cb_tile_count, output_ind_cb_data_format),
    };

    auto P = [](const char* id) { return m2::ProducerOf(m2::DFBSpecName{id}, id); };
    auto C = [](const char* id) { return m2::ConsumerOf(m2::DFBSpecName{id}, id); };

    // ---- Reader kernel (NCRISC). ----
    // GENERATE_INDICES is forced to "1" (matches the legacy reader_defines_map; the precomputed-indices
    // path is disabled per GH issue #36329), so the reader only reads the input tensor and synthesizes
    // the index tile locally — it binds only the input tensor parameter.
    m2::KernelSpec::CompilerOptions reader_opts;
    reader_opts.defines.emplace("GENERATE_INDICES", "1");
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{READER_KERNEL_PATH},
        .compiler_options = std::move(reader_opts),
        .dfb_bindings = {P(DFB_INPUT), P(DFB_INDEX)},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{TP_INPUT}, .accessor_name = "inout_tensor"}},
        .compile_time_args =
            {{"Ht", Ht},
             {"Wt", Wt},
             {"total_number_of_cores", total_number_of_cores},
             {"uint16_output", static_cast<uint32_t>(uint16_output)}},
        .runtime_arg_schema = {.runtime_arg_names = {"id", "work_per_core"}},
        // Reader on NCRISC (RISCV_1) so the two data-movement kernels don't collide on the same DM processor.
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::READER},
    };

    // ---- Writer kernel (BRISC). ----
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{WRITER_KERNEL_PATH},
        .dfb_bindings = {C(DFB_OUTPUT_VAL), C(DFB_OUTPUT_IND)},
        .tensor_bindings =
            {m2::TensorBinding{
                 .tensor_parameter_name = m2::TensorParamName{TP_VALUES}, .accessor_name = "values_tensor"},
             m2::TensorBinding{
                 .tensor_parameter_name = m2::TensorParamName{TP_INDICES}, .accessor_name = "indices_tensor"}},
        .compile_time_args = {{"Ht", Ht}, {"Kt", Ktiles}, {"total_number_of_cores", total_number_of_cores}},
        .runtime_arg_schema = {.runtime_arg_names = {"id", "work_per_core"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::WRITER},
    };

    // ---- Compute kernel. ----
    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .dfb_bindings =
            {C(DFB_INPUT),
             C(DFB_INDEX),
             P(DFB_TRANSPOSED_VAL),
             C(DFB_TRANSPOSED_VAL),
             P(DFB_TRANSPOSED_IND),
             C(DFB_TRANSPOSED_IND),
             P(DFB_RESULT_PREP_VAL),
             C(DFB_RESULT_PREP_VAL),
             P(DFB_RESULT_PREP_IND),
             C(DFB_RESULT_PREP_IND),
             P(DFB_OUTPUT_VAL),
             P(DFB_OUTPUT_IND)},
        .compile_time_args =
            {{"Ht", Ht}, {"Wt", Wt}, {"output_tiles", Ktiles}, {"largest", static_cast<uint32_t>(args.largest)}},
        .runtime_arg_schema = {.runtime_arg_names = {"work_per_core"}},
        .hw_config = m2::ComputeHardwareConfig{.fp32_dest_acc_en = !uint16_output, .dst_full_sync_en = false},
    };

    m2::ProgramSpec spec;
    spec.name = "topk_single_core";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dfbs);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{TP_INPUT}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{TP_VALUES}, .spec = value_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{TP_INDICES}, .spec = index_tensor.tensor_spec()}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "topk_single_core",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = core_range}};

    // ---- Run-args: per-core work split (start row id + tiles-per-core). The degenerate concept applies
    // the complete set on a cache miss and re-applies it on every hit. ----
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};
    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel = m2::KernelSpecName{"compute"}};

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                reader_args.runtime_arg_values.push_back({core, {{"id", id}, {"work_per_core", work_per_core}}});
                writer_args.runtime_arg_values.push_back({core, {{"id", id}, {"work_per_core", work_per_core}}});
                compute_args.runtime_arg_values.push_back({core, {{"work_per_core", work_per_core}}});
                id++;
            }
        }
    }

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.kernel_run_args.push_back(std::move(compute_args));
    run_args.tensor_args.emplace(
        m2::TensorParamName{TP_INPUT}, m2::ProgramRunArgs::TensorArgument{std::cref(input_tensor)});
    run_args.tensor_args.emplace(
        m2::TensorParamName{TP_VALUES}, m2::ProgramRunArgs::TensorArgument{std::cref(value_tensor)});
    run_args.tensor_args.emplace(
        m2::TensorParamName{TP_INDICES}, m2::ProgramRunArgs::TensorArgument{std::cref(index_tensor)});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
