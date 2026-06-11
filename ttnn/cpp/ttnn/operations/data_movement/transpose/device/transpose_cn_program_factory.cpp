// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_cn_program_factory.hpp"
#include "transpose_utils.hpp"

#include <filesystem>

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Unique file-scope constants per .cpp (unity build) — prefix with the variant name.
constexpr const char* TRANSPOSE_CN_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
    "reader_unary_transpose_cn_interleaved_start_id_m2.cpp";
constexpr const char* TRANSPOSE_CN_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
    "writer_unary_transpose_cn_interleaved_start_id_m2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeCNProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
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

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec;
    spec.name = "transpose_cn";

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()}};

    // Local L1 DFB: reader produces, writer consumes (one producer + one consumer in the same work unit).
    spec.dataflow_buffers = {m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"src0"},
        .entry_size = stick_size,
        .num_entries = num_input_pages,
        .data_format_metadata = cb_data_format,
    }};

    // CN_RM define selects the sharded-aware multi-page split helper in both kernels (formerly an
    // unconditional positional define in the legacy descriptor).
    m2::KernelSpec::CompilerOptions reader_opts;
    m2::KernelSpec::CompilerOptions writer_opts;
    if (row_major) {
        reader_opts.defines.emplace("CN_RM", "1");
        writer_opts.defines.emplace("CN_RM", "1");
    }

    // Named compile-time args (formerly positional CTAs): page_size / read_size for the reader,
    // page_size / write_size for the writer.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{TRANSPOSE_CN_READER_KERNEL_PATH},
        .compiler_options = std::move(reader_opts),
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"src0"}, "src0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .compile_time_args = {{"page_size", src0_buffer->aligned_page_size()}, {"read_size", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"N", "C", "HtWt", "batch_step", "channel_step", "num_pages", "start_id", "hw", "n"}},
        // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — so the two data-movement kernels don't
        // collide on the same DM processor.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{TRANSPOSE_CN_WRITER_KERNEL_PATH},
        .compiler_options = std::move(writer_opts),
        .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"src0"}, "src0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .compile_time_args = {{"page_size", dst_buffer->aligned_page_size()}, {"write_size", stick_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "transpose_cn",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
        .target_nodes = total_cores}};

    ////////////////////////////////////////////////////////////////////////////
    //                      Per-core runtime args (full grid)
    ////////////////////////////////////////////////////////////////////////////
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t Wt = W / page_shape[1];
    uint32_t Ht = H / page_shape[0];
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_pages;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    // Need to set runtime args for all cores, not just the ones doing work.
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

        reader_args.runtime_arg_values.push_back(
            {core,
             {{"N", N},
              {"C", C},
              {"HtWt", HtWt},
              {"batch_step", batch_step},
              {"channel_step", channel_step},
              {"num_pages", num_pages_per_core},
              {"start_id", start_tile},
              {"hw", hw},
              {"n", n}}});
        writer_args.runtime_arg_values.push_back(
            {core, {{"num_pages", num_pages_per_core}, {"start_id", num_pages_read}}});

        num_pages_read += num_pages_per_core;
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input_tensor.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output_tensor.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
