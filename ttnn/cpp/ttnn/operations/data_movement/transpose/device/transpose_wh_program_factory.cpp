// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Emit per-core runtime args for the tiled (TILE-layout) WH path. Reproduces the legacy
// emit_runtime_args_wh_tiled loop verbatim; only the dispatch channel changes (named RTAs).
void emit_runtime_args_wh_tiled(
    m2::KernelRunArgs& reader_run,
    m2::KernelRunArgs& compute_run,
    m2::KernelRunArgs& writer_run,
    const Tensor& input_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2];

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    auto HtWt = Ht * Wt;

    reader_run.runtime_arg_values.reserve(num_cores_total);
    compute_run.runtime_arg_values.reserve(num_cores_total);
    writer_run.runtime_arg_values.reserve(num_cores_total);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;
        }

        uint32_t h = num_tiles_read % Ht;
        uint32_t w = num_tiles_read / Ht % Wt;

        reader_run.runtime_arg_values.push_back(
            {core,
             {{"num_tiles", num_tiles_per_core},
              {"start_id", tt::round_down(num_tiles_read, HtWt) + (h * Wt) + w},
              {"start_ht", h},
              {"start_wt", w},
              {"Ht", Ht},
              {"Wt", Wt},
              {"HtWt", HtWt}}});

        compute_run.runtime_arg_values.push_back({core, {{"NHtWt", num_tiles_per_core}}});

        writer_run.runtime_arg_values.push_back(
            {core, {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_read}}});

        num_tiles_read += num_tiles_per_core;
    }
}

// Emit per-core runtime args for the row-major (RM) WH path. Reproduces the legacy
// emit_runtime_args_wh_rm loop verbatim; only the dispatch channel changes (named RTAs).
void emit_runtime_args_wh_rm(
    m2::KernelRunArgs& reader_run,
    m2::KernelRunArgs& compute_run,
    m2::KernelRunArgs& writer_run,
    const Tensor& input_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_hw_blocks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_hw_blocks_per_core_group_2) {
    auto input_shape = input_tensor.logical_shape();

    uint32_t W = input_shape[3], H = input_shape[2];

    reader_run.runtime_arg_values.reserve(num_cores_total);
    compute_run.runtime_arg_values.reserve(num_cores_total);
    writer_run.runtime_arg_values.reserve(num_cores_total);

    for (uint32_t i = 0, num_sticks_read = 0, num_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_hw_blocks_per_core;

        if (core_group_1.contains(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_2;
        } else {
            num_hw_blocks_per_core = 0;
        }

        reader_run.runtime_arg_values.push_back(
            {core, {{"start_id", num_sticks_read}, {"num_hw_blocks_per_core", num_hw_blocks_per_core}}});

        compute_run.runtime_arg_values.push_back({core, {{"num_hw_blocks_per_core", num_hw_blocks_per_core}}});

        writer_run.runtime_arg_values.push_back(
            {core, {{"start_id", num_sticks_write}, {"num_hw_blocks_per_core", num_hw_blocks_per_core}}});

        num_sticks_read += num_hw_blocks_per_core * H;
        num_sticks_write += num_hw_blocks_per_core * W;
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeWHProgramFactory::create_program_spec(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t NC = input_tensor.logical_shape()[1] * input_tensor.logical_shape()[0];
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32 ||
                            src0_cb_data_format == tt::DataFormat::Int32 ||
                            src0_cb_data_format == tt::DataFormat::UInt32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "transpose_wh";

    // src0 DFB (legacy CB c_0) and output DFB (legacy CB c_16). The RM path adds a "tilize"
    // intermediate DFB (legacy CB c_24). The legacy c_25 ("im2", marked TODO REMOVE) is dead — no
    // kernel reads or writes it on the interleaved RM path — so it is omitted (a Metal 2.0 DFB
    // requires >=1 producer and >=1 consumer; binding a never-touched buffer is impossible). See
    // METAL2_PORT_REPORT.md "Open items".
    uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    uint32_t num_output_tiles = row_major ? ht * 2 : 2;

    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = src0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src0_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
        },
    };
    if (row_major) {
        uint32_t num_im_tiles = ht * wt;
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"tilize"},
            .entry_size = src0_single_tile_size,
            .num_entries = num_im_tiles,
            .data_format_metadata = src0_cb_data_format,
        });
    }

    // ---- Kernel sources (selected by layout) ----
    const std::filesystem::path reader_source =
        row_major ? std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                           "reader_unary_transpose_wh_interleaved_start_id_rm.cpp"}
                  : std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                          "reader_unary_transpose_wh_interleaved_start_id.cpp"};
    const std::filesystem::path writer_source =
        row_major ? std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                           "writer_unary_transpose_wh_interleaved_start_id_rm.cpp"}
                  // Forked from eltwise/unary/.../writer_unary_interleaved_start_id.cpp (shared, unmigrated).
                  : std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                          "writer_unary_interleaved_start_id_m2.cpp"};
    const std::filesystem::path compute_source =
        row_major ? std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
                                          "transpose_wh_rm_m2.cpp"}
                  : std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
                                          "transpose_wh_m2.cpp"};

    // ---- Reader ----
    // The legacy factory built the input accessor with TensorAccessorArgs(RuntimeTensorShape) and
    // plumbed the buffer address through an RTA; both collapse to the TensorBinding below. See
    // METAL2_PORT_REPORT.md "Open items" for the dynamic_tensor_shape consideration.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = reader_source,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "input"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    if (row_major) {
        reader.compile_time_args = {
            {"Ht", ht},
            {"H_per_tile", H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT},
            {"H_per_tile_last", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
            {"Wt", wt},
            {"W", W},
            {"HtWt", ht * wt},
            {"W_size_bytes", W * input_tensor.element_size()},
            {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH},
        };
        reader.runtime_arg_schema.runtime_arg_names = {"start_id", "num_hw_blocks_per_core"};
    } else {
        reader.runtime_arg_schema.runtime_arg_names = {
            "num_tiles", "start_id", "start_ht", "start_wt", "Ht", "Wt", "HtWt"};
    }

    // ---- Writer ----
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = writer_source,
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"out"},
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };
    if (row_major) {
        writer.compile_time_args = {
            {"Ht", ht},
            {"H", H},
            {"Wt", wt},
            {"W_per_tile", W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH},
            {"W_per_tile_last", W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH},
            {"HtWt", ht * wt},
            {"H_size_bytes", H * output_tensor.element_size()},
            {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT},
        };
        writer.runtime_arg_schema.runtime_arg_names = {"start_id", "num_hw_blocks_per_core"};
    } else {
        // Legacy writer_unary_interleaved_start_id carried the output CB index as CTA slot 0;
        // that magic index is replaced by the dfb::out binding.
        writer.runtime_arg_schema.runtime_arg_names = {"num_pages", "start_id"};
    }

    // ---- Compute ----
    m2::ComputeHardwareConfig compute_hw{.fp32_dest_acc_en = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Keep the source DFB in full Float32 on the unpack-to-dest path. In the row-major kernel,
        // the tile-formatted intermediate ("tilize", legacy c_24) also feeds the transpose, so it
        // needs the same treatment.
        compute_hw.unpack_to_dest_mode.insert(
            {m2::DFBSpecName{"src0"}, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
        if (row_major) {
            compute_hw.unpack_to_dest_mode.insert(
                {m2::DFBSpecName{"tilize"}, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
        }
    }

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = compute_source,
        .hw_config = compute_hw,
    };
    if (row_major) {
        compute.dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"src0"},
                .accessor_name = "src0",
                .endpoint_type = m2::DFBEndpointType::CONSUMER,
            },
            // "tilize" is a producer-and-consumer intermediate within the compute kernel
            // (tilize writes it, the transpose reads it back) → self-loop binding.
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"tilize"},
                .accessor_name = "tilize",
                .endpoint_type = m2::DFBEndpointType::PRODUCER,
            },
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"tilize"},
                .accessor_name = "tilize",
                .endpoint_type = m2::DFBEndpointType::CONSUMER,
            },
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"out"},
                .accessor_name = "out",
                .endpoint_type = m2::DFBEndpointType::PRODUCER,
            },
        };
        compute.compile_time_args = {{"Ht", ht}, {"Wt", wt}, {"HtWt", ht * wt}};
        compute.runtime_arg_schema.runtime_arg_names = {"num_hw_blocks_per_core"};
        if (input_tensor.dtype() == DataType::UINT32 || input_tensor.dtype() == DataType::INT32) {
            compute.compiler_options.defines = {{"DST_ACCUM_MODE", "1"}};
        }
    } else {
        compute.dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"src0"},
                .accessor_name = "src0",
                .endpoint_type = m2::DFBEndpointType::CONSUMER,
            },
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"out"},
                .accessor_name = "out",
                .endpoint_type = m2::DFBEndpointType::PRODUCER,
            },
        };
        compute.runtime_arg_schema.runtime_arg_names = {"NHtWt"};
    }

    // push_back+move instead of init-list assignment (init-list elements are const -> deep copy).
    spec.kernels.reserve(3);
    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));
    spec.kernels.push_back(std::move(compute));
    spec.tensor_parameters.reserve(2);
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()});
    // reader/compute/writer share the src0/tilize/out DFBs, so they must share one WorkUnitSpec.
    // The legacy factory launches all three kernels on the full grid (total_cores); no-op cores get
    // num_tiles/num_hw_blocks = 0.
    spec.work_units.reserve(1);
    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "transpose_wh",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = total_cores,
    });

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs compute_run{.kernel = m2::KernelSpecName{"compute"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    if (row_major) {
        emit_runtime_args_wh_rm(
            reader_run,
            compute_run,
            writer_run,
            input_tensor,
            num_cores_total,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    } else {
        emit_runtime_args_wh_tiled(
            reader_run,
            compute_run,
            writer_run,
            input_tensor,
            num_cores_total,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    }

    run.kernel_run_args.reserve(3);
    run.kernel_run_args.push_back(std::move(reader_run));
    run.kernel_run_args.push_back(std::move(compute_run));
    run.kernel_run_args.push_back(std::move(writer_run));
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"output"}, output_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
