// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_program_factory.hpp"
#include <map>
#include <string>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using namespace tt::constants;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreProgramFactory::create_program_spec(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    if (!tensor_args.optional_input_tensors.empty()) {
        TT_FATAL(!tensor_args.optional_input_tensors[0].has_value(), "Bias is not supported for matmul multi core");
    }

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output = tensor_return_value.at(0);
    const auto& a_mesh = a.mesh_tensor();
    const auto& b_mesh = b.mesh_tensor();

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();

    const auto& ashape = a_mesh.padded_shape();
    const auto& bshape = b_mesh.padded_shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    tt::tt_metal::IDevice* device = &a_mesh.mutable_device();
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());
    (void)packer_l1_acc;

    const auto& cshape = output.mesh_tensor().padded_shape();  // C=A*B, N1MK*11KN->N1MN

    TT_FATAL(
        operation_attributes.program_config.has_value(),
        "program_config must be provided for MatmulMultiCoreProgramFactory");
    auto pc = std::get<operations::matmul::MatmulMultiCoreProgramConfig>(operation_attributes.program_config.value());
    if (!pc.allowed_worker_cores.has_value()) {
        auto device_grid = device->compute_with_storage_grid_size();
        pc.allowed_worker_cores =
            CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(device_grid.x - 1, device_grid.y - 1)));
    }
    auto compute_with_storage_grid_size = pc.allowed_worker_cores.value().bounding_box().grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    // C = A*B*...; MN = MK*KN
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    uint32_t last_ktile_h = 0;

    // ---- ProgramSpec (immutable) ----
    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;
    m2::ProgramSpec spec;
    spec.name = "matmul_multi_core";
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0"},
            .entry_size = in0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = in0_data_format},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in1"},
            .entry_size = in1_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = in1_data_format},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = output_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = output_data_format},
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                                        "reader_bmm_8bank_output_tiles_partitioned.cpp"},
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in0"},
                 .accessor_name = "in0",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in1"},
                 .accessor_name = "in1",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"a"}, .accessor_name = "a"},
             m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"b"}, .accessor_name = "b"}},
        .compile_time_args = {{"in0_last_ktile_w", last_ktile_w}, {"in0_last_ktile_h", last_ktile_h}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"output_tile_start_id", "num_output_tiles"},
             .common_runtime_arg_names = {"Mt", "Kt", "Nt", "MtKt", "KtNt", "batch", "bcast_B", "MtNt"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                                        "writer_bmm_8bank_interleaved_start_id_m2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"out"},
            .accessor_name = "out",
            .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"out"}, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    std::map<std::string, std::string> mm_kernel_defines;
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);
    m2::KernelSpec::CompilerOptions::Defines defines_table;
    for (const auto& [k, v] : mm_kernel_defines) {
        defines_table.insert({k, v});
    }

    const char* COMPUTE_SRC = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_m2.cpp";
    auto make_compute = [&](const std::string& id, uint32_t per_core_nt) {
        return m2::KernelSpec{
            .unique_id = m2::KernelSpecName{id},
            .source = std::filesystem::path{COMPUTE_SRC},
            .compiler_options = {.defines = defines_table},
            .dfb_bindings =
                {m2::DFBBinding{
                     .dfb_spec_name = m2::DFBSpecName{"in0"},
                     .accessor_name = "in0",
                     .endpoint_type = m2::DFBEndpointType::CONSUMER},
                 m2::DFBBinding{
                     .dfb_spec_name = m2::DFBSpecName{"in1"},
                     .accessor_name = "in1",
                     .endpoint_type = m2::DFBEndpointType::CONSUMER},
                 m2::DFBBinding{
                     .dfb_spec_name = m2::DFBSpecName{"out"},
                     .accessor_name = "out",
                     .endpoint_type = m2::DFBEndpointType::PRODUCER}},
            // bmm uses B,Mt,Nt as 3 nested loops acting as one large loop; only Nt varies per group.
            .compile_time_args = {{"batch", 1}, {"Mt", 1}, {"Kt", Kt}, {"Nt", per_core_nt}},
            .hw_config =
                m2::ComputeHardwareConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .math_approx_mode = math_approx_mode},
        };
    };
    m2::KernelSpec compute_1 = make_compute("compute_1", num_output_tiles_per_core_group_1);

    // push_back+move instead of an initializer-list assignment: a std::initializer_list is const,
    // so `= {...}` deep-copies every TensorSpec; emplacing rvalues moves them in.
    spec.tensor_parameters.reserve(3);
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"a"}, .spec = a.tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"b"}, .spec = b.tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"out"}, .spec = output.tensor_spec()});

    // Local DFBs (in0/in1/out) require their producer and consumer KernelSpecs to share the
    // SAME WorkUnitSpec(s) — every node hosting the DFB must host both endpoints. So each core
    // group gets one WorkUnitSpec containing reader + compute_<group> + writer (reader/writer
    // are shared across both groups' WorkUnitSpecs; the per-group compute differs only in its
    // Nt CTA).
    const bool has_group_2 = !core_group_2.ranges().empty();
    // push_back+move instead of `spec.kernels = {...}` / `spec.work_units = {...}`: the init-lists
    // are const, so the assignments would deep-copy every KernelSpec (source path + bindings) and
    // WorkUnitSpec (CoreRangeSet). The KernelSpec objects are not used after this (the WorkUnitSpecs
    // reference kernels by name), so they can be moved in. (target_nodes still copies core_group_*,
    // which the per-node loop below reuses for contains().)
    if (has_group_2) {
        m2::KernelSpec compute_2 = make_compute("compute_2", num_output_tiles_per_core_group_2);
        spec.kernels.reserve(4);
        spec.kernels.push_back(std::move(reader));
        spec.kernels.push_back(std::move(writer));
        spec.kernels.push_back(std::move(compute_1));
        spec.kernels.push_back(std::move(compute_2));
        spec.work_units.reserve(2);
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "g1",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"compute_1"}, m2::KernelSpecName{"writer"}},
            .target_nodes = core_group_1});
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "g2",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"compute_2"}, m2::KernelSpecName{"writer"}},
            .target_nodes = core_group_2});
    } else {
        spec.kernels.reserve(3);
        spec.kernels.push_back(std::move(reader));
        spec.kernels.push_back(std::move(writer));
        spec.kernels.push_back(std::move(compute_1));
        spec.work_units.reserve(1);
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "g1",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"compute_1"}, m2::KernelSpecName{"writer"}},
            .target_nodes = core_group_1});
    }

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    reader_run.common_runtime_arg_values = {
        {"Mt", Mt},
        {"Kt", Kt},
        {"Nt", Nt},
        {"MtKt", MtKt},
        {"KtNt", KtNt},
        {"batch", B},
        {"bcast_B", static_cast<uint32_t>(bcast_batch)},
        {"MtNt", MtNt}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    reader_run.runtime_arg_values.reserve(num_cores);
    writer_run.runtime_arg_values.reserve(num_cores);
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t per_core = 0;
        if (core_group_1.contains(core)) {
            per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        reader_run.runtime_arg_values.push_back(
            {core, {{"output_tile_start_id", num_tiles_written}, {"num_output_tiles", per_core}}});
        writer_run.runtime_arg_values.push_back({core, {{"num_pages", per_core}, {"start_id", num_tiles_written}}});
        num_tiles_written += per_core;
    }
    // push_back+move instead of `= {reader_run, writer_run}`: the init-list is const, so the
    // assignment would deep-copy every per-node runtime-arg Table a second time.
    run.kernel_run_args.reserve(2);
    run.kernel_run_args.push_back(std::move(reader_run));
    run.kernel_run_args.push_back(std::move(writer_run));
    run.tensor_args = {
        {m2::TensorParamName{"a"}, a_mesh},
        {m2::TensorParamName{"b"}, b_mesh},
        {m2::TensorParamName{"out"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
