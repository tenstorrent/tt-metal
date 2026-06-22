// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_cn_program_factory.hpp"
#include "transpose_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// DFB / kernel / tensor names for the CN factory's ProgramSpec.
const DFBSpecName CN_CB{"cn_cb"};
const KernelSpecName CN_READER{"cn_reader"};
const KernelSpecName CN_WRITER{"cn_writer"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeCNProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    auto input_shape = input_tensor.padded_shape();
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t page_shape[2] = {TILE_WIDTH, TILE_HEIGHT};
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        page_shape[0] = 1;
        page_shape[1] = input_shape[-1];
    }
    uint32_t page_size = page_shape[0] * page_shape[1];
    uint32_t stick_size = (row_major) ? page_shape[1] * input_tensor.element_size() : tt::tile_size(cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    uint32_t num_tensor_pages = input_tensor.physical_volume() / page_size;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_pages);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_input_pages = 2;

    // -------------------------------------------------------------------------
    // ProgramSpec
    // -------------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "transpose_cn";

    // Single DFB shared by the reader (producer) and writer (consumer): the legacy
    // c_0 CB. entry_size = stick_size (one page), num_entries = 2 (double buffer).
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = CN_CB,
        .entry_size = stick_size,
        .num_entries = num_input_pages,
        .data_format_metadata = cb_data_format,
    });

    // Tensor parameters. RuntimeTensorShape in the legacy factory → mirror the
    // existing runtime-shape relaxation via dynamic_tensor_shape (faithful mirror,
    // not a new relaxation decision).
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT,
        .spec = input_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    });
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT,
        .spec = output_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    });

    // Defines (CN_RM define preserved verbatim).
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (row_major) {
        reader_defines.emplace("CN_RM", "1");
        writer_defines.emplace("CN_RM", "1");
    }

    // Reader KernelSpec.
    KernelSpec reader{
        .unique_id = CN_READER,
        .source =
            std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                  "reader_unary_transpose_cn_interleaved_start_id.cpp"),
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = CN_CB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
            },
        .compile_time_args =
            {
                {"page_size", src0_buffer->aligned_page_size()},
                {"read_size", stick_size},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"N", "C", "HtWt", "batch_step", "channel_step", "num_pages", "start_id", "hw", "n"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // Writer KernelSpec.
    KernelSpec writer{
        .unique_id = CN_WRITER,
        .source =
            std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                  "writer_unary_transpose_cn_interleaved_start_id.cpp"),
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = CN_CB, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"page_size", dst_buffer->aligned_page_size()},
                {"write_size", stick_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    // Single WorkUnit: reader + writer run together across the full grid.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "cn_wu",
        .kernels = {CN_READER, CN_WRITER},
        .target_nodes = total_cores,
    });

    // -------------------------------------------------------------------------
    // ProgramRunArgs
    // -------------------------------------------------------------------------
    ProgramRunArgs run_args;

    // Per-core runtime args (identical traversal to the legacy factory).
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t Wt = W / page_shape[1];
    uint32_t Ht = H / page_shape[0];
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_pages;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    KernelRunArgs reader_run{.kernel = CN_READER};
    KernelRunArgs writer_run{.kernel = CN_WRITER};
    reader_run.runtime_arg_values.reserve(num_cores_total);
    writer_run.runtime_arg_values.reserve(num_cores_total);

    for (uint32_t i = 0, num_pages_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        }

        uint32_t hw = num_pages_read % HtWt;
        uint32_t curr_c = num_pages_read / HtWt;
        uint32_t n = curr_c % N;
        uint32_t start_tile = num_pages_read + (curr_c * batch_step) - (curr_c / N * channel_step);

        reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args =
                {
                    {"N", N},
                    {"C", C},
                    {"HtWt", HtWt},
                    {"batch_step", batch_step},
                    {"channel_step", channel_step},
                    {"num_pages", num_pages_per_core},
                    {"start_id", start_tile},
                    {"hw", hw},
                    {"n", n},
                },
        });
        writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args =
                {
                    {"num_pages", num_pages_per_core},
                    {"start_id", num_pages_read},
                },
        });

        num_pages_read += num_pages_per_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run));
    run_args.kernel_run_args.push_back(std::move(writer_run));

    run_args.tensor_args.emplace(INPUT, input_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
