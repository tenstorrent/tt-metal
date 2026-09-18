// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>

#include "moreh_getitem_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tilize_utils.hpp>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
struct IndexInfo {
    bool is_defined{};
    const ttnn::Tensor* tensor{};
    uint32_t unit_size{};
};
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::moreh::moreh_getitem {

ttnn::device_operation::ProgramArtifacts MorehGetItemOperation::MorehGetItemRmFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& input = tensor_args.input;
    const auto& index_tensors = tensor_args.index_tensors;
    const auto& output = output_tensor;
    auto index_dims = operation_attributes.index_dims;
    auto* device = input.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange allCores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    auto core_range = allCores;

    auto input_shape = input.logical_shape();
    auto output_shape = output.logical_shape();

    std::array<uint32_t, 5> new_input_shape{};
    std::array<uint32_t, 5> new_output_shape{};
    new_input_shape.fill(1);
    new_output_shape.fill(1);

    auto input_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < input_shape.rank(); index++) {
        new_input_shape[index + input_dim_offset] = input_shape[index];
    }
    auto output_dim_offset = 5 - output_shape.rank();
    for (auto index = 0; index < output_shape.rank(); index++) {
        new_output_shape[index + output_dim_offset] = output_shape[index];
    }
    ttnn::Shape input_5d_shape(new_input_shape);
    ttnn::Shape output_5d_shape(new_output_shape);

    uint32_t index_start_dim = index_dims.front();
    uint32_t index_end_dim = index_dims.back();

    Tensor input_5d = input;
    input_5d = ttnn::experimental::view(input_5d, input_5d_shape);

    IndexInfo index_info[5] = {{false}};

    for (uint32_t i = 0; i < index_tensors.size(); i++) {
        auto dim = index_dims[i] + input_dim_offset;
        const auto& index = index_tensors[i];

        index_info[dim].is_defined = true;
        index_info[dim].tensor = &index_tensors[i];
        index_info[dim].unit_size = index.padded_shape()[-1] * index.element_size();
    }

    uint32_t index_size = index_tensors.front().padded_shape()[-1];

    uint32_t input_unit_size = input_5d_shape[-1] * input_5d.element_size();
    uint32_t output_unit_size = input_unit_size;

    // split work
    uint32_t num_units = output.physical_volume() / output_shape[-1];

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_units);

    // ---- Program-scope resource names (these drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: this file and moreh_getitem_tilized_factory.cpp land in the same
    // unity-build translation unit, so no anonymous-namespace constants are introduced.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    // in0 stages one input stick: the reader fills it, the writer drains it to the output tensor.
    const DFBSpecName IN0{"in0"};
    // One DFB per index dimension the reader stages a stick of, indexed by 5-D-normalized dimension.
    // Dimension 4 has no entry: the reader's dimension loop runs 3 … 0 and never touches one.
    const std::array<DFBSpecName, 4> INDEX_DFB{
        DFBSpecName{"in1"}, DFBSpecName{"in2"}, DFBSpecName{"in3"}, DFBSpecName{"in4"}};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const std::array<TensorParamName, 5> INDEX{
        TensorParamName{"index0"},
        TensorParamName{"index1"},
        TensorParamName{"index2"},
        TensorParamName{"index3"},
        TensorParamName{"index4"}};
    constexpr std::array<const char*, 5> INDEX_ACCESSOR{"index0", "index1", "index2", "index3", "index4"};
    constexpr std::array<const char*, 4> INDEX_DFB_ACCESSOR{"in1", "in2", "in3", "in4"};
    constexpr std::array<const char*, 5> INDEX_DEFINE{
        "HAS_INDEX0", "HAS_INDEX1", "HAS_INDEX2", "HAS_INDEX3", "HAS_INDEX4"};

    ProgramSpec spec;
    spec.name = "moreh_getitem_rm";

    // ---- Dataflow buffers ----
    auto src_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    auto index_cb_data_format = datatype_to_dataformat_converter(index_tensors[0].dtype());

    auto rounded_input_page_size = round_up_to_mul32(input_unit_size);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = rounded_input_page_size,
        .num_entries = 1,
        .data_format_metadata = src_cb_data_format,
    });

    // ---- Reader kernel ----
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "s0"},
    };
    KernelSpec::CompilerOptions::Defines reader_defines;

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    // The index tensors are optional: only the dimensions the caller supplied exist in a given
    // instantiation. An absent index has no tensor to bind, so its binding is omitted entirely and the
    // matching HAS_INDEX<dim> define gates the kernel's references to the accessor and the DFB.
    for (uint32_t dim = 0; dim < INDEX_DFB.size(); dim++) {
        if (!index_info[dim].is_defined) {
            continue;
        }

        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = INDEX[dim], .spec = index_info[dim].tensor->tensor_spec()});
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = INDEX[dim], .accessor_name = INDEX_ACCESSOR[dim]});
        reader_defines.emplace(INDEX_DEFINE[dim], "1");

        auto index_page_size = round_up_to_mul32(index_info[dim].unit_size);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INDEX_DFB[dim],
            .entry_size = index_page_size,
            .num_entries = 1,
            .data_format_metadata = index_cb_data_format,
        });
        // The reader runs the whole FIFO cycle on an index DFB by itself — reserve_back, push_back,
        // wait_front, pop_front — so it is the buffer's only endpoint and binds both roles.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDEX_DFB[dim],
            .accessor_name = INDEX_DFB_ACCESSOR[dim],
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDEX_DFB[dim],
            .accessor_name = INDEX_DFB_ACCESSOR[dim],
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_kernels/reader_moreh_getitem.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {
                     // input
                     "input_stick_idx_stride_n",
                     "input_stick_idx_stride_c",
                     "input_stick_idx_stride_d",
                     "input_stick_idx_stride_h",

                     "input_size_n",
                     "input_size_c",
                     "input_size_d",
                     "input_size_h",
                     "input_size_w",

                     // index
                     "index0_is_defined",
                     "index1_is_defined",
                     "index2_is_defined",
                     "index3_is_defined",
                     "index4_is_defined",
                     "index0_stick_size",
                     "index1_stick_size",
                     "index2_stick_size",
                     "index3_stick_size",
                     "index4_stick_size",
                     "index_size",
                     "index_start_dim",
                     "index_end_dim",

                     // output
                     "output_size_n",
                     "output_size_c",
                     "output_size_d",
                     "output_size_h",
                     "output_size_w",

                     // etc
                     "start_id",
                     "num_sticks",
                     "stick_size",
                 }},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // ---- Writer kernel ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_kernels/writer_moreh_getitem.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "s0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"output_stick_size", "start_id", "num_sticks"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    // ---- Work unit (placement) ----
    spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});

    uint32_t input_stick_idx_stride_h = 1;
    uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape[3];
    uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape[2];
    uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape[1];

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    uint32_t g1_numcores = core_group_1.num_cores();

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    uint32_t start_id = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                // input
                {"input_stick_idx_stride_n", input_stick_idx_stride_n},
                {"input_stick_idx_stride_c", input_stick_idx_stride_c},
                {"input_stick_idx_stride_d", input_stick_idx_stride_d},
                {"input_stick_idx_stride_h", input_stick_idx_stride_h},

                {"input_size_n", input_5d_shape[0]},
                {"input_size_c", input_5d_shape[1]},
                {"input_size_d", input_5d_shape[2]},
                {"input_size_h", input_5d_shape[3]},
                {"input_size_w", input_5d_shape[4]},

                // index
                {"index0_is_defined", static_cast<uint32_t>(index_info[0].is_defined)},
                {"index1_is_defined", static_cast<uint32_t>(index_info[1].is_defined)},
                {"index2_is_defined", static_cast<uint32_t>(index_info[2].is_defined)},
                {"index3_is_defined", static_cast<uint32_t>(index_info[3].is_defined)},
                {"index4_is_defined", static_cast<uint32_t>(index_info[4].is_defined)},
                {"index0_stick_size", index_info[0].unit_size},
                {"index1_stick_size", index_info[1].unit_size},
                {"index2_stick_size", index_info[2].unit_size},
                {"index3_stick_size", index_info[3].unit_size},
                {"index4_stick_size", index_info[4].unit_size},
                {"index_size", index_size},
                {"index_start_dim", index_start_dim},
                {"index_end_dim", index_end_dim},

                // output
                {"output_size_n", output_5d_shape[0]},
                {"output_size_c", output_5d_shape[1]},
                {"output_size_d", output_5d_shape[2]},
                {"output_size_h", output_5d_shape[3]},
                {"output_size_w", output_5d_shape[4]},

                // etc
                {"start_id", start_id},
                {"num_sticks", num_units_per_core},
                {"stick_size", input_unit_size},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                // output
                {"output_stick_size", output_unit_size},

                // etc
                {"start_id", start_id},
                {"num_sticks", num_units_per_core},
            });

        start_id += num_units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});
    for (uint32_t dim = 0; dim < INDEX_DFB.size(); dim++) {
        if (index_info[dim].is_defined) {
            run_args.tensor_args.emplace(INDEX[dim], TensorArgument{index_info[dim].tensor->mesh_tensor()});
        }
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_getitem
