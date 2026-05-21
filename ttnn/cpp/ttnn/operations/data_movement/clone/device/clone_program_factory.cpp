// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Clone op program factory, Metal 2.0 host-API port.
//
// Branches internally on (tilized, is_sharded) to select kernel sources and on
// convert_dtype to introduce the dtype-conversion compute kernel. The framework
// adapter (ProgramSpecMeshWorkloadFactoryAdapter) handles cache-miss /
// cache-hit dispatch.

#include <cmath>

#include "clone_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/operations/math.hpp"

#include <string>
#include <vector>

namespace ttnn::operations::data_movement::clone {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

// Unity-build hygiene: clone is currently a single-factory op, but prefix the
// kernel/work-unit unique-id constants with `C_` so they don't collide if a
// sibling factory ever lands in the same unity TU.
constexpr const char* C_READER = "reader";
constexpr const char* C_WRITER = "writer";
constexpr const char* C_COMPUTE_G1 = "compute_g1";
constexpr const char* C_COMPUTE_G2 = "compute_g2";

constexpr const char* C_WU_G1 = "wu_g1";
constexpr const char* C_WU_G2 = "wu_g2";

constexpr const char* INPUT_DFB = "input";
constexpr const char* OUTPUT_DFB = "output";

constexpr const char* INPUT_TENSOR = "input";
constexpr const char* OUTPUT_TENSOR = "output";

}  // namespace

ttnn::device_operation::ProgramArtifacts CloneOperation::ProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt;

    const auto& input = tensor_args.input;
    auto input_data_format = datatype_to_dataformat_converter(input.dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool convert_dtype = input_data_format != output_data_format;
    const bool tilized = output.layout() == Layout::TILE;

    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tilized ? tt::tile_size(data_format) : tensor.logical_shape()[-1] * tensor.element_size();
    };
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    const uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.logical_shape()[-1];

    auto output_memory_layout = output.memory_config().memory_layout();
    const bool is_sharded = output_memory_layout != TensorMemoryLayout::INTERLEAVED;

    uint32_t num_cores;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_units_per_core_group_1;
    uint32_t num_units_per_core_group_2;
    uint32_t num_cores_x;
    uint32_t num_cores_y;

    if (is_sharded) {
        auto shard_spec = output.buffer()->shard_spec();
        all_cores = shard_spec.grid();
        num_cores = all_cores.num_cores();

        auto shard_shape = shard_spec.shape();
        uint32_t shard_height = shard_shape[0];
        uint32_t shard_width = shard_shape[1];

        // For row-major sharded, the unit (stick) size must be the shard width, not the
        // full tensor width. Using tensor width causes OOB reads past the shard boundary.
        if (!tilized) {
            input_unit_size = shard_width * input.element_size();
            output_unit_size = shard_width * output.element_size();
        }

        if (tilized) {
            num_units_per_core_group_1 = (shard_height * shard_width) / TILE_HW;
        } else {
            num_units_per_core_group_1 = shard_height;
        }

        num_units_per_core_group_2 = 0;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();

        auto grid_size = all_cores.bounding_box();
        num_cores_x = grid_size.end_coord.x + 1;
        num_cores_y = grid_size.end_coord.y + 1;
    } else {
        auto compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
        num_cores_x = compute_with_storage_grid_size.x;
        num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_units_per_core_group_1_result,
             num_units_per_core_group_2_result] = split_work_to_cores(compute_with_storage_grid_size, num_units);
        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_units_per_core_group_1 = num_units_per_core_group_1_result;
        num_units_per_core_group_2 = num_units_per_core_group_2_result;
    }

    const auto alignment = input.buffer()->alignment();
    const uint32_t aligned_input_unit_size = tt::align(input_unit_size, alignment);
    const uint32_t aligned_output_unit_size = tt::align(output_unit_size, alignment);

    // ----- DataflowBufferSpecs -----

    std::vector<m2::DataflowBufferSpec> dataflow_buffers;
    // INPUT_DFB always present; when !convert_dtype, the writer binds INPUT_DFB
    // (the legacy code aliased dst_cb_id = src_cb_id in that case).
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = INPUT_DFB,
        .entry_size = aligned_input_unit_size,
        .num_entries = 2,
        .data_format_metadata = input_data_format,
    });
    if (convert_dtype) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = OUTPUT_DFB,
            .entry_size = aligned_output_unit_size,
            .num_entries = 2,
            .data_format_metadata = output_data_format,
        });
    }
    const char* writer_dfb_name = convert_dtype ? OUTPUT_DFB : INPUT_DFB;

    // ----- Reader / writer KernelSpecs -----
    //
    // Kernel source selection mirrors the legacy four-way branch.

    const std::string read_kernel_path = [&]() -> std::string {
        if (is_sharded) {
            return tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_sharded.cpp"
                           : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm_sharded.cpp";
        }
        return tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel.cpp"
                       : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm.cpp";
    }();
    const std::string write_kernel_path = [&]() -> std::string {
        if (is_sharded) {
            return tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_sharded.cpp"
                           : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp";
        }
        return tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp"
                       : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp";
    }();

    // Reader: produces INPUT_DFB from the input tensor.
    // RTA schema differs slightly by branch:
    //   tilized + interleaved: {num_tiles, start_id}
    //   tilized + sharded:     {num_tiles}
    //   row-major + interleaved: {stick_size, num_sticks, start_id}
    //   row-major + sharded:     {stick_size, num_sticks}
    std::vector<std::string> reader_rta_names;
    if (tilized) {
        reader_rta_names.push_back("num_tiles");
        if (!is_sharded) {
            reader_rta_names.push_back("start_id");
        }
    } else {
        reader_rta_names.push_back("stick_size");
        reader_rta_names.push_back("num_sticks");
        if (!is_sharded) {
            reader_rta_names.push_back("start_id");
        }
    }

    m2::KernelSpec reader{
        .unique_id = C_READER,
        .source = m2::KernelSpec::SourceFilePath{read_kernel_path},
        .dfb_bindings =
            {
                {.dfb_spec_name = INPUT_DFB,
                 .local_accessor_name = "src_dfb",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"},
            },
        .runtime_arguments_schema = {.named_runtime_args = reader_rta_names},
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    // Writer: consumes (INPUT_DFB or OUTPUT_DFB depending on convert_dtype).
    std::vector<std::string> writer_rta_names = reader_rta_names;  // same schema shape

    m2::KernelSpec writer{
        .unique_id = C_WRITER,
        .source = m2::KernelSpec::SourceFilePath{write_kernel_path},
        .dfb_bindings =
            {
                {.dfb_spec_name = writer_dfb_name,
                 .local_accessor_name = "dst_dfb",
                 .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                {.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
            },
        .runtime_arguments_schema = {.named_runtime_args = writer_rta_names},
        .config_spec =
            m2::DataMovementConfiguration{
                .gen1_data_movement_config =
                    m2::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    // ----- Compute KernelSpec(s) (only when convert_dtype = true) -----
    //
    // Per [Anti-pattern: Demoting per-group CTA to RTA], preserve per-group
    // num_units CTA — one compute KernelSpec per populated core group.

    auto make_compute = [&](const char* unique_id, uint32_t this_num_units) {
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
        return m2::KernelSpec{
            .unique_id = unique_id,
            .source =
                m2::KernelSpec::SourceFilePath{
                    "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp"},
            .dfb_bindings =
                {
                    {.dfb_spec_name = INPUT_DFB,
                     .local_accessor_name = "src_dfb",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER},
                    {.dfb_spec_name = OUTPUT_DFB,
                     .local_accessor_name = "dst_dfb",
                     .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER},
                },
            .compile_time_arg_bindings = {{"num_units", this_num_units}},
            .config_spec =
                m2::ComputeConfiguration{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .math_approx_mode = math_approx_mode,
                },
        };
    };

    const bool g2_present = !core_group_2.ranges().empty();
    std::optional<m2::KernelSpec> compute_g1;
    std::optional<m2::KernelSpec> compute_g2;
    if (convert_dtype) {
        compute_g1 = make_compute(C_COMPUTE_G1, num_units_per_core_group_1);
        if (g2_present) {
            compute_g2 = make_compute(C_COMPUTE_G2, num_units_per_core_group_2);
        }
    }

    // ----- WorkUnitSpecs -----

    std::vector<m2::WorkUnitSpec> work_units;
    {
        std::vector<m2::KernelSpecName> wu_g1_kernels = {C_READER, C_WRITER};
        if (convert_dtype) {
            wu_g1_kernels.push_back(C_COMPUTE_G1);
        }
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = C_WU_G1,
            .kernels = std::move(wu_g1_kernels),
            .target_nodes = core_group_1,
        });
    }
    if (g2_present) {
        std::vector<m2::KernelSpecName> wu_g2_kernels = {C_READER, C_WRITER};
        if (convert_dtype) {
            wu_g2_kernels.push_back(C_COMPUTE_G2);
        }
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = C_WU_G2,
            .kernels = std::move(wu_g2_kernels),
            .target_nodes = core_group_2,
        });
    }

    // ----- ProgramSpec assembly -----

    std::vector<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    if (compute_g1.has_value()) {
        kernels.push_back(std::move(*compute_g1));
    }
    if (compute_g2.has_value()) {
        kernels.push_back(std::move(*compute_g2));
    }

    m2::ProgramSpec spec{
        .program_id = "clone",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                {.unique_id = INPUT_TENSOR, .spec = input.tensor_spec()},
                {.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    // ----- ProgramRunParams -----

    m2::ProgramRunParams run_params;
    m2::ProgramRunParams::KernelRunParams reader_rp{.kernel_spec_name = C_READER};
    m2::ProgramRunParams::KernelRunParams writer_rp{.kernel_spec_name = C_WRITER};

    const uint32_t num_cores_group_1 = core_group_1.num_cores();
    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    uint32_t start_id = 0;
    for (size_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        const uint32_t num_units_per_core =
            i < num_cores_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;

        std::unordered_map<std::string, uint32_t> reader_args;
        std::unordered_map<std::string, uint32_t> writer_args;
        if (tilized) {
            reader_args["num_tiles"] = num_units_per_core;
            writer_args["num_tiles"] = num_units_per_core;
        } else {
            reader_args["stick_size"] = input_unit_size;
            reader_args["num_sticks"] = num_units_per_core;
            writer_args["stick_size"] = output_unit_size;
            writer_args["num_sticks"] = num_units_per_core;
        }
        if (!is_sharded) {
            reader_args["start_id"] = start_id;
            writer_args["start_id"] = start_id;
        }
        reader_rp.named_runtime_args.push_back(
            m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{.node = core, .args = std::move(reader_args)});
        writer_rp.named_runtime_args.push_back(
            m2::ProgramRunParams::KernelRunParams::NodeNamedRTAs{.node = core, .args = std::move(writer_args)});
        if (!is_sharded) {
            start_id += num_units_per_core;
        }
    }

    run_params.kernel_run_params = {std::move(reader_rp), std::move(writer_rp)};
    if (convert_dtype) {
        // Compute kernels have CTAs only; no per-execution RTAs.
        run_params.kernel_run_params.push_back(m2::ProgramRunParams::KernelRunParams{.kernel_spec_name = C_COMPUTE_G1});
        if (g2_present) {
            run_params.kernel_run_params.push_back(
                m2::ProgramRunParams::KernelRunParams{.kernel_spec_name = C_COMPUTE_G2});
        }
    }

    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = INPUT_TENSOR, .tensor = std::cref(input.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::operations::data_movement::clone
