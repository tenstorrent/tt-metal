// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "fold_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_fold {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/kernels/reader_fold_rm.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_fold/device/kernels/writer_fold_rm.cpp";

ttnn::device_operation::ProgramArtifacts MorehFoldOperation::MultiCore::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;

    auto output_size = operation_attributes.output_size;
    auto kernel_size = operation_attributes.kernel_size;
    auto dilation = operation_attributes.dilation;
    auto padding = operation_attributes.padding;
    auto stride = operation_attributes.stride;
    auto output_shape = output.logical_shape();
    auto output_shape_rank = output.logical_shape().rank();

    std::vector<uint32_t> ls;
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t l = (((output_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1);
        ls.push_back(l);
    }
    uint32_t N = output_shape_rank == 4 ? output_shape[0] : 1;
    uint32_t C = output_shape_rank == 4 ? output_shape[1] : output_shape[0];
    uint32_t H = output_shape_rank == 4 ? output_shape[2] : output_shape[1];
    uint32_t W = output_shape_rank == 4 ? output_shape[3] : output_shape[2];
    uint32_t kernel_size_h = kernel_size[0];
    uint32_t kernel_size_w = kernel_size[1];
    uint32_t stride_h = stride[0];
    uint32_t stride_w = stride[1];
    uint32_t padding_h = padding[0];
    uint32_t padding_w = padding[1];
    uint32_t dilation_h = dilation[0];
    uint32_t dilation_w = dilation[1];
    uint32_t LH = ls[0];
    uint32_t LW = ls[1];

    IDevice* device = input.device();

    // MeshTensors for the tensor-parameter bindings (Metal 2.0 speaks MeshTensor).
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    uint32_t num_units = output.logical_volume() / output.logical_shape()[-1];

    auto grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(grid, num_units);

    auto data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t unit_size = input.element_size();
    uint32_t input_cb_page_size = unit_size * input.logical_shape()[-1];
    uint32_t output_cb_page_size = unit_size * output.logical_shape()[-1];

    // For L1 circular buffer alignment
    uint32_t aligned_input_cb_page_size = round_up_to_mul32(input_cb_page_size);
    uint32_t aligned_output_cb_page_size = round_up_to_mul32(output_cb_page_size);

    // For DRAM reads, we need DRAM-aligned size
    bool src_is_dram = input.buffer()->buffer_type() == BufferType::DRAM;
    bool is_blackhole = (device->arch() == tt::ARCH::BLACKHOLE);
    uint32_t dram_alignment = hal::get_dram_alignment();
    uint32_t dram_aligned_input_cb_page_size = tt::align(input_cb_page_size, dram_alignment);

    // The scratch DFB (legacy c_1) is only needed for a two-step DRAM-aligned read:
    // when the DRAM source page is not DRAM-aligned, or unconditionally on Blackhole.
    bool use_scratch = (src_is_dram && (input_cb_page_size % dram_alignment != 0)) || is_blackhole;

    // ---- Resource names ----
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName INPUT_CB{"input_cb"};      // legacy c_0
    const DFBSpecName OUTPUT_CB{"output_cb"};    // legacy c_16
    const DFBSpecName SCRATCH_CB{"scratch_cb"};  // legacy c_1 (conditional)
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()};

    // ---- Dataflow buffers (plain interleaved L1 CBs; no compute kernel, so format metadata is optional) ----
    Group<DataflowBufferSpec> dfbs = {
        DataflowBufferSpec{
            .unique_id = INPUT_CB,
            .entry_size = aligned_input_cb_page_size,
            .num_entries = 2,
            .data_format_metadata = data_format,
        },
        DataflowBufferSpec{
            .unique_id = OUTPUT_CB,
            .entry_size = aligned_output_cb_page_size,
            .num_entries = 2,
            .data_format_metadata = data_format,
        },
    };
    // Scratch CB for DRAM alignment. On Blackhole, always use two-step read for DRAM.
    if (use_scratch) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = SCRATCH_CB,
            .entry_size = dram_aligned_input_cb_page_size,
            .num_entries = 4,
            .data_format_metadata = data_format,
        });
    }

    // ---- Reader kernel ----
    // c_0 (input): reader is the sole toucher (full FIFO on the reader) -> self-loop (PRODUCER + CONSUMER).
    // c_16 (output): reader is the producer (reserve_back/push_back).
    // c_1 (scratch): reader is the sole toucher (raw get_write_ptr peek) -> self-loop; only when allocated.
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{.dfb_spec_name = INPUT_CB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = INPUT_CB, .accessor_name = "input", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUTPUT_CB, .accessor_name = "output", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    if (use_scratch) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = SCRATCH_CB, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = SCRATCH_CB, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    // Kernel defines
    KernelSpec::CompilerOptions::Defines reader_defines;
    switch (input.dtype()) {
        case DataType::BFLOAT16: reader_defines.insert({"DTYPE_BFLOAT16", "1"}); break;
        case DataType::FLOAT32: reader_defines.insert({"DTYPE_FLOAT32", "1"}); break;
        default: break;
    }
    // The conditionally-bound scratch DFB is gated kernel-side by this define.
    if (use_scratch) {
        reader_defines.insert({"HAS_SCRATCH_CB", "1"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = READER_KERNEL_PATH,
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"N",
                  "C",
                  "H",
                  "W",
                  "kernel_size_h",
                  "kernel_size_w",
                  "stride_h",
                  "stride_w",
                  "padding_h",
                  "padding_w",
                  "dilation_h",
                  "dilation_w",
                  "LH",
                  "LW",
                  "input_cb_page_size",
                  "dram_aligned_input_cb_page_size",
                  "output_cb_page_size",
                  "start_id",
                  "num_units_per_core",
                  "aligned"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer kernel ----
    // c_16 (output): writer is the consumer (wait_front/pop_front).
    KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUTPUT_CB, .accessor_name = "output", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"output_cb_page_size", "start_id", "num_units_per_core"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Runtime args per core ----
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    uint32_t g1_numcores = core_group_1.num_cores();

    // Alignment info for the kernel. Loop-invariant (same on every node); legacy computed it identically
    // per core. On Blackhole, always use two-step read for DRAM.
    uint32_t aligned = (src_is_dram ? (input_cb_page_size % dram_alignment == 0) : 1);
    aligned = aligned && !is_blackhole;

    KernelRunArgs reader_kra{.kernel = READER};
    KernelRunArgs writer_kra{.kernel = WRITER};

    uint32_t start_id = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        AddRuntimeArgsForNode(
            reader_kra.runtime_arg_values,
            core,
            {{"N", N},
             {"C", C},
             {"H", H},
             {"W", W},
             {"kernel_size_h", kernel_size_h},
             {"kernel_size_w", kernel_size_w},
             {"stride_h", stride_h},
             {"stride_w", stride_w},
             {"padding_h", padding_h},
             {"padding_w", padding_w},
             {"dilation_h", dilation_h},
             {"dilation_w", dilation_w},
             {"LH", LH},
             {"LW", LW},
             {"input_cb_page_size", input_cb_page_size},
             {"dram_aligned_input_cb_page_size", dram_aligned_input_cb_page_size},
             {"output_cb_page_size", aligned_output_cb_page_size},
             {"start_id", start_id},
             {"num_units_per_core", num_units_per_core},
             {"aligned", aligned}});

        AddRuntimeArgsForNode(
            writer_kra.runtime_arg_values,
            core,
            {{"output_cb_page_size", aligned_output_cb_page_size},
             {"start_id", start_id},
             {"num_units_per_core", num_units_per_core}});

        start_id += num_units_per_core;
    }

    // ---- Assemble ----
    ProgramSpec spec;
    spec.name = "moreh_fold_rm";
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.dataflow_buffers = std::move(dfbs);
    spec.tensor_parameters = {std::move(input_param), std::move(output_param)};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_kra), std::move(writer_kra)};
    run_args.tensor_args.insert({INPUT, input_mesh_tensor});
    run_args.tensor_args.insert({OUTPUT, output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_fold
