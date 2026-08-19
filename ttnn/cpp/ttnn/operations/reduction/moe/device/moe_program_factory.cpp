// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/moe/device/moe_program_factory.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include <cmath>
#include <string>
#include <utility>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

using namespace tt::tt_metal::experimental;

// The spec-name constants below carry a MOE_ prefix because this file is one source of the
// unity-built ttnn_op_reduction target: a unity build merges every anonymous namespace in the
// target into a single scope, and sibling factories there declare their own reader / writer /
// compute and buffer constants. The name strings the constants hold are scoped to one ProgramSpec,
// so those need no prefix.
const KernelSpecName MOE_READER{"reader"};
const KernelSpecName MOE_WRITER{"writer"};
const KernelSpecName MOE_COMPUTE{"compute"};

const TensorParamName MOE_TENSOR_INPUT{"input"};
const TensorParamName MOE_TENSOR_EXPERT_MASK{"expert_mask"};
const TensorParamName MOE_TENSOR_TOPK_MASK{"topk_mask"};
const TensorParamName MOE_TENSOR_OUTPUT{"output"};

const DFBSpecName MOE_DFB_INPUT{"input"};
const DFBSpecName MOE_DFB_EXPERT_MASK{"expert_mask"};
const DFBSpecName MOE_DFB_TOPK_MASK{"topk_mask"};
const DFBSpecName MOE_DFB_SCALE{"scale"};
const DFBSpecName MOE_DFB_INDEX{"index"};
const DFBSpecName MOE_DFB_INPUT_TRANSPOSED{"input_transposed"};
const DFBSpecName MOE_DFB_INDEX_TRANSPOSED{"index_transposed"};
const DFBSpecName MOE_DFB_VALUES{"values"};
const DFBSpecName MOE_DFB_OUTPUT_IND{"output_ind"};
const DFBSpecName MOE_DFB_CUR_MAX{"cur_max"};
const DFBSpecName MOE_DFB_CUR_SUM{"cur_sum"};
const DFBSpecName MOE_DFB_OUT{"out"};
const DFBSpecName MOE_DFB_MASKED_INPUT{"masked_input"};

}  // namespace

ttnn::device_operation::ProgramArtifacts MoeProgramFactory::create_program_artifacts(
    const MoeParams& operation_attributes, const MoeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input.mesh_tensor();
    const auto& expert_mask_tensor = tensor_args.expert_mask.mesh_tensor();
    const auto& topk_mask_tensor = tensor_args.topk_mask.mesh_tensor();
    const auto& out_tensor = output_tensor.mesh_tensor();

    const auto k = operation_attributes.k;

    const NodeCoord node{0, 0};

    tt::DataFormat input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat topk_mask_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(topk_mask_tensor.dtype());
    tt::DataFormat expert_mask_dfb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(expert_mask_tensor.dtype());
    tt::DataFormat out_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(out_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat index_dfb_data_format = tt::DataFormat::UInt16;
    tt::DataFormat value_dfb_data_format = tt::DataFormat::Float16_b;

    uint32_t input_tile_size = tile_size(input_dfb_data_format);
    uint32_t topk_mask_tile_size = tile_size(topk_mask_dfb_data_format);
    uint32_t expert_mask_tile_size = tile_size(expert_mask_dfb_data_format);
    uint32_t out_tile_size = tile_size(out_dfb_data_format);
    uint32_t scalar_tile_size = tile_size(scalar_df);
    uint32_t index_tile_size = tile_size(index_dfb_data_format);
    uint32_t value_tile_size = tile_size(value_dfb_data_format);

    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t tile_hw = input_tensor.tensor_spec().tile().get_tile_hw();
    uint32_t num_out_tiles = out_tensor.physical_volume() / tile_hw;
    uint32_t scale_tiles = 1;

    auto input_shape = input_tensor.padded_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    uint32_t Wt = input_shape[3] / tile_width;
    uint32_t Kt = (k + tile_width - 1) / tile_width;

    // for streaming in input
    uint32_t num_dfb_unit = 2;
    uint32_t dfb_in_units = 2 * num_dfb_unit;

    // values and topk_indices tensors are fully computed and buffered
    uint32_t topk_mask_dfb_units = Kt;
    uint32_t values_and_topk_indices_dfb_units = Ht * Kt;

    Group<DataflowBufferSpec> dataflow_buffers;

    // INPUT DFBs
    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_INPUT,
        .entry_size = input_tile_size,
        .num_entries = dfb_in_units,
        .data_format_metadata = input_dfb_data_format,
    });

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_EXPERT_MASK,
        .entry_size = expert_mask_tile_size,
        .num_entries = Wt,
        .data_format_metadata = expert_mask_dfb_data_format,
    });

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_TOPK_MASK,
        .entry_size = topk_mask_tile_size,
        .num_entries = topk_mask_dfb_units,
        .data_format_metadata = topk_mask_dfb_data_format,
    });

    // identity scale input
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_SCALE,
        .entry_size = scalar_tile_size,
        .num_entries = scale_tiles,
        .data_format_metadata = scalar_df,
    });

    // TOP K DFBs
    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space. This buffer carries the indices that are created in the reader kernel.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_INDEX,
        .entry_size = index_tile_size,
        .num_entries = dfb_in_units,
        .data_format_metadata = index_dfb_data_format,
    });

    // Single buffered dataflow buffer that holds the transposed input tiles.
    // The backing region is sized as Wt Float16_b tiles while an entry is one input-format tile, so
    // the entry count is that region divided by the input tile size. The two tile sizes coincide for
    // a BFLOAT16 input, which is the only input dtype the op documents.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_INPUT_TRANSPOSED,
        .entry_size = input_tile_size,
        .num_entries = (Wt * value_tile_size) / input_tile_size,
        .data_format_metadata = input_dfb_data_format,
    });

    // Single buffered dataflow buffer that holds the transposed index tiles
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_INDEX_TRANSPOSED,
        .entry_size = index_tile_size,
        .num_entries = Wt,
        .data_format_metadata = index_dfb_data_format,
    });

    // topk values
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_VALUES,
        .entry_size = value_tile_size,
        .num_entries = values_and_topk_indices_dfb_units,
        .data_format_metadata = value_dfb_data_format,
    });

    // topk indices
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_OUTPUT_IND,
        .entry_size = index_tile_size,
        .num_entries = values_and_topk_indices_dfb_units,
        .data_format_metadata = index_dfb_data_format,
    });

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_CUR_MAX,
        .entry_size = out_tile_size,
        .num_entries = num_out_tiles,
        .data_format_metadata = out_dfb_data_format,
    });

    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_CUR_SUM,
        .entry_size = out_tile_size,
        .num_entries = num_out_tiles,
        .data_format_metadata = out_dfb_data_format,
    });

    // OUTPUT DFBs
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_OUT,
        .entry_size = out_tile_size,
        .num_entries = num_out_tiles,
        .data_format_metadata = out_dfb_data_format,
    });

    // Intermediate buffer for adding input + expert_mask before sorting.
    // Takes 2 tiles because two tiles are processed and freed before the next two are loaded.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = MOE_DFB_MASKED_INPUT,
        .entry_size = input_tile_size,
        .num_entries = 2,
        .data_format_metadata = input_dfb_data_format,
    });

    const tt::ARCH arch = input_tensor.device().arch();

    KernelSpec reader{
        .unique_id = MOE_READER,
        .source = "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = MOE_DFB_INPUT,
                    .accessor_name = "input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = MOE_DFB_INDEX,
                    .accessor_name = "index",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = MOE_DFB_TOPK_MASK,
                    .accessor_name = "topk_mask",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = MOE_DFB_EXPERT_MASK,
                    .accessor_name = "expert_mask",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = MOE_TENSOR_INPUT, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = MOE_TENSOR_TOPK_MASK, .accessor_name = "topk_mask"},
                TensorBinding{.tensor_parameter_name = MOE_TENSOR_EXPERT_MASK, .accessor_name = "expert_mask"},
            },
        .compile_time_args = {{"Ht", Ht}, {"Wt", Wt}, {"K", k}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    KernelSpec writer{
        .unique_id = MOE_WRITER,
        .source = "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = MOE_DFB_OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // The writer also fills the identity scale buffer that the compute kernel's reduce
                // steps multiply by, so it is that buffer's producer as well as the output's consumer.
                DFBBinding{
                    .dfb_spec_name = MOE_DFB_SCALE,
                    .accessor_name = "scale",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = MOE_TENSOR_OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args = {{"Ht", Ht}, {"K", k}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    Group<DFBBinding> compute_dfb_bindings{
        DFBBinding{
            .dfb_spec_name = MOE_DFB_INPUT,
            .accessor_name = "input",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = MOE_DFB_EXPERT_MASK,
            .accessor_name = "expert_mask",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = MOE_DFB_TOPK_MASK,
            .accessor_name = "topk_mask",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = MOE_DFB_SCALE,
            .accessor_name = "scale",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = MOE_DFB_INDEX,
            .accessor_name = "index",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
        DFBBinding{
            .dfb_spec_name = MOE_DFB_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };

    // The remaining buffers are scratch space private to the compute kernel: it is the only kernel
    // that touches them, filling and draining each FIFO itself. Every dataflow buffer needs one
    // producer and one consumer endpoint, so that single toucher takes both roles. Both endpoints
    // share one accessor name, so the kernel drives each buffer through a single handle.
    auto bind_compute_scratch = [&compute_dfb_bindings](const DFBSpecName& dfb, std::string accessor_name) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = dfb,
            .accessor_name = accessor_name,
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = dfb,
            .accessor_name = std::move(accessor_name),
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    };
    bind_compute_scratch(MOE_DFB_INPUT_TRANSPOSED, "input_transposed");
    bind_compute_scratch(MOE_DFB_INDEX_TRANSPOSED, "index_transposed");
    bind_compute_scratch(MOE_DFB_VALUES, "values");
    bind_compute_scratch(MOE_DFB_OUTPUT_IND, "output_ind");
    bind_compute_scratch(MOE_DFB_CUR_MAX, "cur_max");
    bind_compute_scratch(MOE_DFB_CUR_SUM, "cur_sum");
    bind_compute_scratch(MOE_DFB_MASKED_INPUT, "masked_input");

    KernelSpec compute{
        .unique_id = MOE_COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp",
        // A compute kernel is built at O3; the generic default is a level lower.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"Ht", Ht},
                {"Wt", Wt},
                {"K", k},
                {"logk", static_cast<uint32_t>(std::log2(k))},
                {"logWt", static_cast<uint32_t>(std::log2(Wt))},
                {"tile_width", tile_width},
            },
        .hw_config = ComputeGen1Config{},
    };

    ProgramSpec spec{
        .name = "moe",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = MOE_TENSOR_INPUT, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = MOE_TENSOR_EXPERT_MASK, .spec = expert_mask_tensor.tensor_spec()},
                TensorParameter{.unique_id = MOE_TENSOR_TOPK_MASK, .spec = topk_mask_tensor.tensor_spec()},
                TensorParameter{.unique_id = MOE_TENSOR_OUTPUT, .spec = out_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {MOE_READER, MOE_WRITER, MOE_COMPUTE},
                    .target_nodes = node,
                },
            },
    };

    // No kernel has runtime or common runtime arguments: every value the kernels once received as a
    // runtime arg was a tensor base address, and those now ride the tensor bindings.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {MOE_TENSOR_INPUT, input_tensor},
        {MOE_TENSOR_EXPERT_MASK, expert_mask_tensor},
        {MOE_TENSOR_TOPK_MASK, topk_mask_tensor},
        {MOE_TENSOR_OUTPUT, out_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
