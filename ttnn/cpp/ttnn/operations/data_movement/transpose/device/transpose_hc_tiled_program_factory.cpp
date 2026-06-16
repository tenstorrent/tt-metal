// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Per-core runtime args for the reader + writer kernels. Reproduces the legacy
// emit_runtime_args_hc_tiled loop verbatim; only the dispatch channel changes (named RTAs). The
// legacy buffer-address RTA (reader/writer slot 0) is replaced by the TensorBindings and dropped.
void emit_runtime_args_hc_tiled(
    m2::KernelRunArgs& reader_run,
    m2::KernelRunArgs& writer_run,
    const Tensor& input_tensor,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1];
    uint32_t HW = H * W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW_bytes = C * HW * input_tensor.element_size();

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ct = C / TILE_HEIGHT;
    uint32_t CtHWt = Ct * H * Wt;
    uint32_t CtWt = Ct * Wt;

    reader_run.runtime_arg_values.reserve(num_cores_total);
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

        uint32_t h = num_tiles_read / CtWt % H;
        uint32_t ct = num_tiles_read / Wt % Ct;

        reader_run.runtime_arg_values.push_back(
            {core,
             {{"WT", Wt},
              {"H", H},
              {"CT", Ct},
              {"HW_bytes", HW_bytes},
              {"CHW_bytes", CHW_bytes},
              {"start_id", num_tiles_read},
              {"num_tiles", num_tiles_per_core},
              {"batch_addr", num_tiles_read / CtHWt * CHW_bytes},
              {"h", h},
              {"htWT", h / TILE_HEIGHT * Wt},
              {"ct", ct},
              {"ctoffs", ct * TILE_HEIGHT * HW_bytes},
              {"wt", num_tiles_read % Wt}}});

        writer_run.runtime_arg_values.push_back(
            {core, {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_read}}});

        num_tiles_read += num_tiles_per_core;
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCTiledProgramFactory::create_program_spec(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    uint32_t sub_tile_line_bytes = 16 * input_tensor.element_size();
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "transpose_hc_tiled");
    log_debug(tt::LogOp, "sub_tile_line_bytes: {}", sub_tile_line_bytes);
    log_debug(tt::LogOp, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // check if we need to allocate a scratch buffer
    // The kernel reads several 16 element face lines (32B for BFLOAT16) from different input tiles to form a single
    // output tile, one output tile at a time Each face line is 32 bytes, so if our minimum read alignment is greater
    // than that (64B for Blackhole) then we will have reads from unaligned face-lines into differently aligned
    // destination face-lines
    // TODO: noc_async_write only require 16B alignment for both DRAM and L1 for Blackhole, so instead of reading in
    // face-lines from C tiles to form a single tile, we can load a single tile and then write out its face-lines to C
    // tiles
    uint32_t alignment = dst_buffer->alignment();
    bool misaligned = alignment > sub_tile_line_bytes;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "transpose_hc_tiled";

    // src0 DFB (legacy CB c_0): the reader produces output tiles into it, the writer consumes them.
    uint32_t num_input_tiles = 2;
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = cb_data_format,
        },
    };

    // need some scratch memory here - if we need data from a misaligned address then we need to read from the
    // nearest aligned address and then copy the data to the correct location.
    // This is a "fake CB" (legacy c_1) — the kernel uses it purely as an address source (get_write_ptr,
    // then direct memory copy), with no real FIFO producer/consumer. It is bound as a self-loop DFB on
    // the reading kernel (PRODUCER + CONSUMER) to satisfy the validator's producer-and-consumer rule.
    // See METAL2_PORT_REPORT.md "Open items". Only present (and only referenced kernel-side) when
    // MISALIGNED.
    if (misaligned) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"scratch"},
            .entry_size = alignment,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        });
    }

    // The legacy factory built the input accessor with TensorAccessorArgs and plumbed the buffer
    // address through RTA slot 0; both collapse to the TensorBinding below. The reader's CB index
    // (c_0) is bound as dfb::src0; the misaligned scratch (c_1) as a self-loop dfb::scratch.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "reader_unary_transpose_hc_interleaved_partitioned.cpp"},
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
        .compile_time_args =
            {
                {"SUBTILE_LINE_BYTES", sub_tile_line_bytes},
                {"FLOAT32_DTYPE", cb_data_format == tt::DataFormat::Float32 ? 1u : 0u},
                {"ALIGNMENT", alignment},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"WT",
                     "H",
                     "CT",
                     "HW_bytes",
                     "CHW_bytes",
                     "start_id",
                     "num_tiles",
                     "batch_addr",
                     "h",
                     "htWT",
                     "ct",
                     "ctoffs",
                     "wt"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    if (misaligned) {
        // Self-loop the fake scratch CB on the single reading kernel so the validator is satisfied
        // (nothing is actually produced/consumed as a FIFO — it is an address-source borrow).
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"scratch"},
            .accessor_name = "scratch",
            .endpoint_type = m2::DFBEndpointType::PRODUCER,
        });
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"scratch"},
            .accessor_name = "scratch",
            .endpoint_type = m2::DFBEndpointType::CONSUMER,
        });
        reader.compiler_options.defines = {{"MISALIGNED", "1"}};
    }

    // Writer: forked from eltwise/unary/.../writer_unary_interleaved_start_id.cpp (shared, unmigrated);
    // reuses the existing transpose-local writer_unary_interleaved_start_id_m2.cpp. The legacy writer
    // carried the output CB index (= c_0, the shared src0 DFB) as CTA slot 0; that magic index is
    // replaced by the dfb::out binding (accessor_name "out", consuming the src0 DFB).
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()},
    };
    // reader (produces src0) and writer (consumes src0) share one WorkUnitSpec — every node hosting
    // the DFB hosts both endpoints. The legacy factory launches both on the full grid (total_cores);
    // no-op cores get num_tiles = 0.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "transpose_hc_tiled",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = total_cores,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    emit_runtime_args_hc_tiled(
        reader_run,
        writer_run,
        input_tensor,
        num_cores_total,
        num_cores_y,
        core_group_1,
        num_tiles_per_core_group_1,
        core_group_2,
        num_tiles_per_core_group_2);

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"output"}, output_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
