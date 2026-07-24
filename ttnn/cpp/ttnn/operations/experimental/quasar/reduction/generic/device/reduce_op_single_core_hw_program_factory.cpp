// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <bit>
#include <cmath>
#include <filesystem>
#include <map>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

// Metal 2.0 port of the single-core HW reduce factory. reader (data + reduce-scaler DFB) -> compute
// (reduce<in, scaler, out>) -> writer. Reduce defines (REDUCE_OP / REDUCE_DIM / optional
// REDUCE_POST_MUL) flow through compiler_options.defines. MIN (negate) is rejected in validate() on
// Quasar (negative_tile is unported), so no fused-negate compute variant exists here.
ttnn::device_operation::ProgramArtifacts
ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    const auto& a = tensor_args.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device().arch(), operation_attributes.compute_kernel_config);

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;
    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    TT_FATAL(
        operation_attributes.dim == ReduceOpDim::HW,
        "ReduceSingleCoreHwProgramFactory supports HW dim only, got dim enum value {}",
        static_cast<int>(operation_attributes.dim));

    TT_FATAL(operation_attributes.scaler >= 0, "Scalar must be non-negative");
    float scaler = std::sqrt(operation_attributes.scaler);

    TT_FATAL(
        H % tile_height == 0 && W % tile_width == 0, "Reduce HW expects tile-aligned padded shape H={}, W={}", H, W);
    uint32_t num_tensor_tiles = NC * H * W / tile_hw;
    const uint32_t num_tensor_tiles_ht_wt = NC * Ht * Wt;
    TT_FATAL(
        num_tensor_tiles == num_tensor_tiles_ht_wt,
        "Reduce HW tile count mismatch: tile_hw path={} vs Ht*Wt path={}",
        num_tensor_tiles,
        num_tensor_tiles_ht_wt);

    CoreCoord selected_core_coord = {0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        const auto& r = operation_attributes.sub_core_grids->ranges().front();
        selected_core_coord = r.start_coord;
        TT_FATAL(
            operation_attributes.sub_core_grids->contains(selected_core_coord),
            "Selected core {} must be contained in provided sub_core_grids {}",
            selected_core_coord,
            *operation_attributes.sub_core_grids);
    }
    CoreRange core(selected_core_coord, selected_core_coord);
    CoreRangeSet core_set(core);

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // ---- Resource names ----
    const DFBSpecName IN{"in"};          // legacy c_0
    const DFBSpecName SCALER{"scaler"};  // legacy c_2
    const DFBSpecName OUT{"out"};        // legacy c_3
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = src0_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = src0_cb_data_format};
    DataflowBufferSpec scaler_dfb{
        .unique_id = SCALER,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_cb_data_format};
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = dst_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = dst_cb_data_format};

    std::vector<DataflowBufferSpec> dfbs = {in_dfb, scaler_dfb, out_dfb};

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reduce defines ----
    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils_qsr::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::HW);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : reduce_defines) {
        compute_defines.emplace(k, v);
    }
    KernelSpec::CompilerOptions::Defines reader_defines = compute_defines;

    const std::filesystem::path kdir("ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/kernels/");

    KernelSpec reader{
        .unique_id = READER,
        .source = kdir / "dataflow/reader_unary_reduce_universal_start_id_metal2.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"scaler_bits", std::bit_cast<uint32_t>(scaler)}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device().arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "dataflow/writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device().arch()),
    };

    // ---- Compute (reduce<in, scaler, out>) ----
    std::vector<DFBBinding> compute_bindings = {
        DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}};
    const std::filesystem::path compute_source = kdir / "compute/reduce_metal2.cpp";

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = compute_source,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args = {{"Ht", Ht}, {"Wt", Wt}, {"NC", NC}, {"post_mul_scaler_bits", post_mul_scaler_bits}},
        .hw_config = ttnn::to_compute_hardware_config(
            a.device().arch(),
            ttnn::ComputeKernelConfig{
                .math_fidelity = math_fidelity,
                .math_approx_mode = false,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = false}),
    };

    Group<KernelSpec> kernels = {reader, writer, compute};
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{.name = "reduce_single_core_hw", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_set}};

    uint32_t out_dim_divider = Ht * Wt;
    TT_FATAL(
        num_tensor_tiles % out_dim_divider == 0,
        "Reduce HW per-core input tiles {} must be divisible by Ht*Wt={}",
        num_tensor_tiles,
        out_dim_divider);

    KernelRunArgs::RuntimeArgValues reader_node_args =
        MakeRuntimeArgsForSingleNode(selected_core_coord, {{"num_tiles", num_tensor_tiles}, {"start_id", 0u}});
    KernelRunArgs::RuntimeArgValues writer_node_args = MakeRuntimeArgsForSingleNode(
        selected_core_coord, {{"num_pages", num_tensor_tiles / out_dim_divider}, {"start_id", 0u}});

    ProgramSpec spec{
        .name = "reduce_single_core_hw",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)}};
    run_args.tensor_args = {{INPUT, a}, {OUTPUT, output}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
