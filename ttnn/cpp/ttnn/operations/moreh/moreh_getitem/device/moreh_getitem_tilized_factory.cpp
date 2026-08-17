// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "moreh_getitem_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
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

ttnn::device_operation::ProgramArtifacts MorehGetItemOperation::MorehGetItemTilizedFactory::create_program_artifacts(
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
    auto TILE_HEIGHT = constants::TILE_HEIGHT;
    auto TILE_WIDTH = constants::TILE_WIDTH;
    auto* device = input.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange allCores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    auto core_range = allCores;

    auto input_shape = input.padded_shape();
    auto input_shape_without_padding = input.logical_shape();
    auto output_shape = output.padded_shape();
    auto output_shape_without_padding = output.logical_shape();

    std::array<uint32_t, 5> new_input_shape{};
    std::array<uint32_t, 5> new_output_shape{};
    std::array<uint32_t, 5> new_input_padded_shape{};
    std::array<uint32_t, 5> new_output_padded_shape{};

    new_input_shape.fill(1);
    new_input_padded_shape.fill(1);
    auto input_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < input_shape.rank(); index++) {
        new_input_shape[index + input_dim_offset] = input_shape_without_padding[index];
        new_input_padded_shape[index + input_dim_offset] = input_shape[index];
    }

    new_output_shape.fill(1);
    new_output_padded_shape.fill(1);
    auto output_dim_offset = 5 - input_shape.rank();
    for (auto index = 0; index < output_shape.rank(); index++) {
        new_output_shape[index + output_dim_offset] = output_shape_without_padding[index];
        new_output_padded_shape[index + output_dim_offset] = output_shape[index];
    }

    ttnn::Shape input_5d_shape(new_input_padded_shape);
    ttnn::Shape output_5d_shape(new_output_padded_shape);

    bool is_w_index_exist = false;
    for (auto dim : index_dims) {
        if (dim + input_dim_offset == 4) {
            is_w_index_exist = true;
        }
    }

    ttnn::Shape input_5d_shape_without_padding(new_input_shape);
    ttnn::Shape output_5d_shape_without_padding(new_output_shape);

    auto index_layout = index_tensors.front().layout();
    bool is_row_major_index = (index_layout == Layout::ROW_MAJOR);

    // ---- Program-scope resource names (these drive the generated dfb:: / tensor:: tokens) ----
    // Declared function-local: this file and moreh_getitem_rm_factory.cpp land in the same unity-build
    // translation unit, so no anonymous-namespace constants are introduced.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    // in0 stages one input stick: the reader fills it, the writer drains it.
    const DFBSpecName IN0{"in0"};
    // One DFB per index dimension the reader stages a tile of, indexed by 5-D-normalized dimension.
    const std::array<DFBSpecName, 5> INDEX_DFB{
        DFBSpecName{"in1"}, DFBSpecName{"in2"}, DFBSpecName{"in3"}, DFBSpecName{"in4"}, DFBSpecName{"in5"}};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const std::array<TensorParamName, 5> INDEX{
        TensorParamName{"index0"},
        TensorParamName{"index1"},
        TensorParamName{"index2"},
        TensorParamName{"index3"},
        TensorParamName{"index4"}};
    constexpr std::array<const char*, 5> INDEX_ACCESSOR{"index0", "index1", "index2", "index3", "index4"};
    constexpr std::array<const char*, 5> INDEX_DFB_ACCESSOR{"in1", "in2", "in3", "in4", "in5"};
    constexpr std::array<const char*, 5> INDEX_DEFINE{
        "HAS_INDEX0", "HAS_INDEX1", "HAS_INDEX2", "HAS_INDEX3", "HAS_INDEX4"};

    if (is_w_index_exist) {
        // compute index info
        IndexInfo index_info[5] = {{false}};

        for (uint32_t i = 0; i < index_tensors.size(); i++) {
            auto dim = index_dims[i] + input_dim_offset;
            const auto& index = index_tensors[i];

            index_info[dim].is_defined = true;
            index_info[dim].tensor = &index_tensors[i];
            index_info[dim].unit_size = index.element_size();
        }

        uint32_t index_size = index_tensors[0].logical_shape()[-1];

        uint32_t input_unit_size = input.element_size();
        uint32_t output_unit_size = output.element_size();

        uint32_t alignment_size = 32;
        uint32_t num_elements_per_alignment = alignment_size / output_unit_size;
        uint32_t num_units =
            output_5d_shape_without_padding[0] * output_5d_shape_without_padding[1] *
            output_5d_shape_without_padding[2] * output_5d_shape_without_padding[3] *
            ((output_5d_shape_without_padding[4] + num_elements_per_alignment - 1) / num_elements_per_alignment);

        uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
                split_work_to_cores_wt_core_range(core_range, num_units);

        // out1 stages the aligned run of output elements the writer assembles before pushing it out.
        const DFBSpecName OUT1{"out1"};

        ProgramSpec spec;
        spec.name = "moreh_getitem_tilize_w";

        // ---- Dataflow buffers ----
        auto src_cb_data_format = datatype_to_dataformat_converter(input.dtype());
        auto index_cb_data_format = datatype_to_dataformat_converter(index_tensors[0].dtype());
        auto output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

        auto rounded_input_page_size = round_up_to_mul32(input_unit_size);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN0,
            .entry_size = rounded_input_page_size,
            .num_entries = 1,
            .data_format_metadata = src_cb_data_format,
        });

        auto rounded_output_page_size = round_up_to_mul32(output_unit_size);
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT1,
            .entry_size = rounded_output_page_size,
            .num_entries = 1,
            .data_format_metadata = output_cb_data_format,
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

        if (is_row_major_index) {
            reader_defines.emplace("ROW_MAJOR_INDEX", "1");
        } else {
            reader_defines.emplace("TILIZE_INDEX", "1");
        }

        spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
        spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

        // The index tensors are optional: only the dimensions the caller supplied exist in a given
        // instantiation. An absent index has no tensor to bind, so its binding is omitted entirely and
        // the matching HAS_INDEX<dim> define — emitted only when the slot is defined — gates the
        // kernel's references to the accessor and the DFB. (Legacy instead passed a nullptr Buffer* for
        // an absent slot, which the descriptor API lowered to a literal 0u address the kernel never
        // dereferenced.)
        for (uint32_t dim = 0; dim < 5; dim++) {
            if (!index_info[dim].is_defined) {
                continue;
            }

            spec.tensor_parameters.push_back(
                TensorParameter{.unique_id = INDEX[dim], .spec = index_info[dim].tensor->tensor_spec()});
            reader_tensor_bindings.push_back(
                TensorBinding{.tensor_parameter_name = INDEX[dim], .accessor_name = INDEX_ACCESSOR[dim]});
            reader_defines.emplace(INDEX_DEFINE[dim], "1");

            auto index_page_size = 1024 * 4;
            spec.dataflow_buffers.push_back(DataflowBufferSpec{
                .unique_id = INDEX_DFB[dim],
                .entry_size = static_cast<uint32_t>(index_page_size),
                .num_entries = 1,
                .data_format_metadata = index_cb_data_format,
            });
            // The reader is an index DFB's only endpoint — it reserves an entry and reads the index
            // tile through the write pointer, without a matching push_back — so it binds both roles.
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
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
                      "reader_moreh_getitem_tilize_w.cpp",
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
                         "input_stick_idx_stride_w",
                         "input_size_c_without_padding",
                         "input_size_d_without_padding",
                         "input_size_h_without_padding",
                         "input_num_stick_width",
                         "input_noc_id_stride_n",
                         "input_noc_id_stride_c",
                         "input_noc_id_stride_d",
                         "input_noc_id_stride_h",

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

                         // output
                         "output_size_n",
                         "output_size_c",
                         "output_size_d",
                         "output_size_h",
                         "output_size_w",
                         "output_num_stick_width",

                         // etc
                         "start_id",
                         "num_sticks",
                         "element_size",
                         "num_elements_per_alignment",
                         "num_alignment_width",
                     }},
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        });

        // ---- Writer kernel ----
        // out1 is touched by the writer alone, and only through raw pointers and as a NoC source (no
        // FIFO calls at all), so the writer binds both of its endpoints.
        spec.kernels.push_back(KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
                      "writer_moreh_getitem_tilize_w.cpp",
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0,
                        .accessor_name = "out0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT1,
                        .accessor_name = "out1",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT1,
                        .accessor_name = "out1",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "s0"}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {
                         // output
                         "output_size_c_without_padding",
                         "output_size_d_without_padding",
                         "output_size_h_without_padding",
                         "output_size_w_without_padding",
                         "output_noc_id_stride_n",
                         "output_noc_id_stride_c",
                         "output_noc_id_stride_d",
                         "output_noc_id_stride_h",
                         "output_num_stick_width",

                         // etc
                         "start_id",
                         "num_sticks",
                         "stick_size",
                         "element_size",
                         "num_elements_per_alignment",
                         "num_alignment_width",
                     }},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        });

        // ---- Work unit (placement) ----
        spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});

        uint32_t face_width = 16;
        uint32_t input_num_stick_width = div_up(input_5d_shape_without_padding[4], face_width);
        uint32_t num_alignment_width = div_up(output_5d_shape_without_padding[4], num_elements_per_alignment);
        uint32_t output_num_stick_width = div_up(output_5d_shape_without_padding[4], face_width);

        uint32_t input_num_tile_c = input_5d_shape[1];
        uint32_t input_num_tile_d = input_5d_shape[2];
        uint32_t input_num_tile_height = input_5d_shape[3] / TILE_HEIGHT;
        uint32_t input_num_tile_width = input_5d_shape[4] / TILE_WIDTH;
        uint32_t input_noc_id_stride_h = input_num_tile_width;
        uint32_t input_noc_id_stride_d = input_noc_id_stride_h * input_num_tile_height;
        uint32_t input_noc_id_stride_c = input_noc_id_stride_d * input_num_tile_d;
        uint32_t input_noc_id_stride_n = input_noc_id_stride_c * input_num_tile_c;

        uint32_t output_num_tile_c = output_5d_shape[1];
        uint32_t output_num_tile_d = output_5d_shape[2];
        uint32_t output_num_tile_height = output_5d_shape[3] / TILE_HEIGHT;
        uint32_t output_num_tile_width = output_5d_shape[4] / TILE_WIDTH;

        uint32_t output_noc_id_stride_h = output_num_tile_width;
        uint32_t output_noc_id_stride_d = output_noc_id_stride_h * output_num_tile_height;
        uint32_t output_noc_id_stride_c = output_noc_id_stride_d * output_num_tile_d;
        uint32_t output_noc_id_stride_n = output_noc_id_stride_c * output_num_tile_c;

        uint32_t input_stick_idx_stride_w = 1;
        uint32_t input_stick_idx_stride_h = input_num_stick_width;
        uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape_without_padding[3];
        uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape_without_padding[2];
        uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape_without_padding[1];

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
                    {"input_stick_idx_stride_w", input_stick_idx_stride_w},
                    {"input_size_c_without_padding", input_5d_shape_without_padding[1]},
                    {"input_size_d_without_padding", input_5d_shape_without_padding[2]},
                    {"input_size_h_without_padding", input_5d_shape_without_padding[3]},
                    {"input_num_stick_width", input_num_stick_width},
                    {"input_noc_id_stride_n", input_noc_id_stride_n},
                    {"input_noc_id_stride_c", input_noc_id_stride_c},
                    {"input_noc_id_stride_d", input_noc_id_stride_d},
                    {"input_noc_id_stride_h", input_noc_id_stride_h},

                    {"input_size_n", input_5d_shape_without_padding[0]},
                    {"input_size_c", input_5d_shape_without_padding[1]},
                    {"input_size_d", input_5d_shape_without_padding[2]},
                    {"input_size_h", input_5d_shape_without_padding[3]},
                    {"input_size_w", input_5d_shape_without_padding[4]},

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

                    // output
                    {"output_size_n", output_5d_shape_without_padding[0]},
                    {"output_size_c", output_5d_shape_without_padding[1]},
                    {"output_size_d", output_5d_shape_without_padding[2]},
                    {"output_size_h", output_5d_shape_without_padding[3]},
                    {"output_size_w", output_5d_shape_without_padding[4]},
                    {"output_num_stick_width", output_num_stick_width},

                    // etc
                    {"start_id", start_id},
                    {"num_sticks", num_units_per_core},
                    {"element_size", input.element_size()},
                    {"num_elements_per_alignment", num_elements_per_alignment},
                    {"num_alignment_width", num_alignment_width},
                });

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {
                    // output
                    {"output_size_c_without_padding", output_5d_shape_without_padding[1]},
                    {"output_size_d_without_padding", output_5d_shape_without_padding[2]},
                    {"output_size_h_without_padding", output_5d_shape_without_padding[3]},
                    {"output_size_w_without_padding", output_5d_shape_without_padding[4]},
                    {"output_noc_id_stride_n", output_noc_id_stride_n},
                    {"output_noc_id_stride_c", output_noc_id_stride_c},
                    {"output_noc_id_stride_d", output_noc_id_stride_d},
                    {"output_noc_id_stride_h", output_noc_id_stride_h},
                    {"output_num_stick_width", output_num_stick_width},

                    // etc
                    {"start_id", start_id},
                    {"num_sticks", num_units_per_core},
                    {"stick_size", output_unit_size},
                    {"element_size", output.element_size()},
                    {"num_elements_per_alignment", num_elements_per_alignment},
                    {"num_alignment_width", num_alignment_width},
                });

            start_id += num_units_per_core;
        }

        run_args.kernel_run_args.push_back(std::move(reader_run_args));
        run_args.kernel_run_args.push_back(std::move(writer_run_args));

        run_args.tensor_args.emplace(INPUT, TensorArgument{input.mesh_tensor()});
        run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});
        for (uint32_t dim = 0; dim < 5; dim++) {
            if (index_info[dim].is_defined) {
                run_args.tensor_args.emplace(INDEX[dim], TensorArgument{index_info[dim].tensor->mesh_tensor()});
            }
        }

        return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};

    }  // compute index info

    IndexInfo index_info[5] = {{false}};

    for (uint32_t i = 0; i < index_tensors.size(); i++) {
        auto dim = index_dims[i] + input_dim_offset;
        const auto& index = index_tensors[i];

        index_info[dim].is_defined = true;
        index_info[dim].tensor = &index_tensors[i];
        index_info[dim].unit_size = index.padded_shape()[-1] * index.element_size();
    }
    uint32_t index_size = index_tensors[0].logical_shape()[-1];

    uint32_t input_unit_size = 16 * input.element_size();
    uint32_t output_unit_size = 16 * output.element_size();

    uint32_t num_units = output_5d_shape_without_padding[0] * output_5d_shape_without_padding[1] *
                         output_5d_shape_without_padding[2] * output_5d_shape_without_padding[3] *
                         ((output_5d_shape_without_padding[4] + 15) / 16);

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_units);

    ProgramSpec spec;
    spec.name = "moreh_getitem_tilize";

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

    if (is_row_major_index) {
        reader_defines.emplace("ROW_MAJOR_INDEX", "1");
    } else {
        reader_defines.emplace("TILIZE_INDEX", "1");
    }

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    // Optional index tensors, exactly as in the is_w_index_exist branch above: an undefined slot is not
    // bound and its HAS_INDEX<dim> define is not emitted, which removes the kernel's references to it.
    // A defined dimension 4 cannot reach this branch (it sets is_w_index_exist), so this reader's
    // dimension loop — and the DFB set below — stops at dimension 3.
    for (uint32_t dim = 0; dim < 5; dim++) {
        if (!index_info[dim].is_defined) {
            continue;
        }

        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = INDEX[dim], .spec = index_info[dim].tensor->tensor_spec()});
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = INDEX[dim], .accessor_name = INDEX_ACCESSOR[dim]});
        reader_defines.emplace(INDEX_DEFINE[dim], "1");

        if (dim == 4) {
            continue;
        }

        auto index_page_size = 1024 * 4;
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INDEX_DFB[dim],
            .entry_size = static_cast<uint32_t>(index_page_size),
            .num_entries = 1,
            .data_format_metadata = index_cb_data_format,
        });
        // The reader is an index DFB's only endpoint — it reserves an entry and reads the index tile
        // through the write pointer, without a matching push_back — so it binds both roles.
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
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
                  "reader_moreh_getitem_tilize.cpp",
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
                     "input_stick_idx_stride_w",
                     "input_size_c_without_padding",
                     "input_size_d_without_padding",
                     "input_size_h_without_padding",
                     "input_noc_id_stride_n",
                     "input_noc_id_stride_c",
                     "input_noc_id_stride_d",
                     "input_noc_id_stride_h",
                     "input_num_stick_width",

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

                     // output
                     "output_size_n",
                     "output_size_c",
                     "output_size_d",
                     "output_size_h",
                     "output_size_w",
                     "output_num_stick_width",

                     // etc
                     "start_id",
                     "num_sticks",
                     "stick_size",
                     "element_size",
                 }},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // ---- Writer kernel ----
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_tilized_kernels/"
                  "writer_moreh_getitem_tilize.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN0,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "s0"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {
                     // output
                     "output_size_c_without_padding",
                     "output_size_d_without_padding",
                     "output_size_h_without_padding",
                     "output_size_w_without_padding",
                     "output_noc_id_stride_n",
                     "output_noc_id_stride_c",
                     "output_noc_id_stride_d",
                     "output_noc_id_stride_h",
                     "output_num_stick_width",

                     // etc
                     "start_id",
                     "num_sticks",
                     "stick_size",
                     "element_size",
                 }},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    // ---- Work unit (placement) ----
    spec.work_units.push_back(WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores});

    uint32_t face_width = 16;
    uint32_t input_num_stick_width = div_up(input_5d_shape_without_padding[4], face_width);
    uint32_t output_num_stick_width = div_up(output_5d_shape_without_padding[4], face_width);

    uint32_t input_num_tile_c = input_5d_shape[1];
    uint32_t input_num_tile_d = input_5d_shape[2];
    uint32_t input_num_tile_height = input_5d_shape[3] / TILE_HEIGHT;
    uint32_t input_num_tile_width = input_5d_shape[4] / TILE_WIDTH;
    uint32_t input_noc_id_stride_h = input_num_tile_width;
    uint32_t input_noc_id_stride_d = input_noc_id_stride_h * input_num_tile_height;
    uint32_t input_noc_id_stride_c = input_noc_id_stride_d * input_num_tile_d;
    uint32_t input_noc_id_stride_n = input_noc_id_stride_c * input_num_tile_c;

    uint32_t output_num_tile_c = output_5d_shape[1];
    uint32_t output_num_tile_d = output_5d_shape[2];
    uint32_t output_num_tile_height = output_5d_shape[3] / TILE_HEIGHT;
    uint32_t output_num_tile_width = output_5d_shape[4] / TILE_WIDTH;

    uint32_t output_noc_id_stride_h = output_num_tile_width;
    uint32_t output_noc_id_stride_d = output_noc_id_stride_h * output_num_tile_height;
    uint32_t output_noc_id_stride_c = output_noc_id_stride_d * output_num_tile_d;
    uint32_t output_noc_id_stride_n = output_noc_id_stride_c * output_num_tile_c;

    uint32_t input_stick_idx_stride_w = 1;
    uint32_t input_stick_idx_stride_h = input_num_stick_width;
    uint32_t input_stick_idx_stride_d = input_stick_idx_stride_h * input_5d_shape_without_padding[3];
    uint32_t input_stick_idx_stride_c = input_stick_idx_stride_d * input_5d_shape_without_padding[2];
    uint32_t input_stick_idx_stride_n = input_stick_idx_stride_c * input_5d_shape_without_padding[1];

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
                {"input_stick_idx_stride_w", input_stick_idx_stride_w},
                {"input_size_c_without_padding", input_5d_shape_without_padding[1]},
                {"input_size_d_without_padding", input_5d_shape_without_padding[2]},
                {"input_size_h_without_padding", input_5d_shape_without_padding[3]},
                {"input_noc_id_stride_n", input_noc_id_stride_n},
                {"input_noc_id_stride_c", input_noc_id_stride_c},
                {"input_noc_id_stride_d", input_noc_id_stride_d},
                {"input_noc_id_stride_h", input_noc_id_stride_h},
                {"input_num_stick_width", input_num_stick_width},

                {"input_size_n", input_5d_shape_without_padding[0]},
                {"input_size_c", input_5d_shape_without_padding[1]},
                {"input_size_d", input_5d_shape_without_padding[2]},
                {"input_size_h", input_5d_shape_without_padding[3]},
                {"input_size_w", input_5d_shape_without_padding[4]},

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

                // output
                {"output_size_n", output_5d_shape[0]},
                {"output_size_c", output_5d_shape[1]},
                {"output_size_d", output_5d_shape[2]},
                {"output_size_h", output_5d_shape_without_padding[3]},
                {"output_size_w", output_5d_shape_without_padding[4]},
                {"output_num_stick_width", output_num_stick_width},

                // etc
                {"start_id", start_id},
                {"num_sticks", num_units_per_core},
                {"stick_size", input_unit_size},
                {"element_size", input.element_size()},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                // output
                {"output_size_c_without_padding", output_5d_shape_without_padding[1]},
                {"output_size_d_without_padding", output_5d_shape_without_padding[2]},
                {"output_size_h_without_padding", output_5d_shape_without_padding[3]},
                {"output_size_w_without_padding", output_5d_shape_without_padding[4]},
                {"output_noc_id_stride_n", output_noc_id_stride_n},
                {"output_noc_id_stride_c", output_noc_id_stride_c},
                {"output_noc_id_stride_d", output_noc_id_stride_d},
                {"output_noc_id_stride_h", output_noc_id_stride_h},
                {"output_num_stick_width", output_num_stick_width},

                // etc
                {"start_id", start_id},
                {"num_sticks", num_units_per_core},
                {"stick_size", output_unit_size},
                {"element_size", output.element_size()},
            });

        start_id += num_units_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    run_args.tensor_args.emplace(INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});
    for (uint32_t dim = 0; dim < 5; dim++) {
        if (index_info[dim].is_defined) {
            run_args.tensor_args.emplace(INDEX[dim], TensorArgument{index_info[dim].tensor->mesh_tensor()});
        }
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_getitem
