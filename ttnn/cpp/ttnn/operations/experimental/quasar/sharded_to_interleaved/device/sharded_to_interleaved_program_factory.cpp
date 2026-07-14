// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// Spec resource names. Prefixed to stay distinct under unity builds
// (Pattern: Unity-build hygiene for anonymous-namespace symbols).
const DFBSpecName S2I_INPUT_DFB{"s2i_input"};
const DFBSpecName S2I_OUTPUT_DFB{"s2i_output"};

const TensorParamName S2I_INPUT{"s2i_input"};
const TensorParamName S2I_OUTPUT{"s2i_output"};

const KernelSpecName S2I_READER{"s2i_reader"};
const KernelSpecName S2I_WRITER{"s2i_writer"};
const KernelSpecName S2I_COMPUTE{"s2i_compute"};

}  // namespace

ttnn::device_operation::ProgramArtifacts ShardedToInterleavedProgramFactory::create_program_artifacts(
    const ShardedToInterleavedParams& operation_attributes,
    const ShardedToInterleavedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    const uint32_t num_slices = operation_attributes.num_slices;
    const uint32_t slice_index = operation_attributes.slice_index;
    const bool is_l1_aligned = true;

    uint32_t num_units_per_shard = 0;
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t num_units_per_shard_width = 0;
    uint32_t num_units_per_shard_height = 0;
    uint32_t num_units_offset = 0;
    uint32_t num_units_per_row = 0;
    uint32_t num_units_height = 0;
    uint32_t num_units_per_shard_height_last = 0;
    uint32_t num_units_per_shard_width_last = 0;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_cores_unpadded = num_cores;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    CoreCoord end_core = cores[num_cores - 1];
    if (output.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        num_units_height = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        num_units_offset = 1;
        num_units_height = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            output_unit_size - (round_up(num_units_per_row, output_unit_size) - num_units_per_row);
    }

    // re-calculate end_core in the case shard grid is larger than used grid
    if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_unpadded = div_up(num_units_height, num_units_per_shard_height);
    } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
        if (output.layout() == Layout::TILE) {
            num_cores_unpadded = div_up(num_units_per_row, num_units_per_shard_width);
        } else {
            num_cores_unpadded = div_up(num_units_per_row, output_unit_size);
        }
    }
    end_core = cores[num_cores_unpadded - 1];

    // Create CoreRangeSet for only the cores that will be used (fixes NOC error when grid > data)
    CoreRangeSet used_cores = num_cores_unpadded < num_cores
                                  ? select_from_corerangeset(all_cores, 0, num_cores_unpadded - 1, rm_orientation)
                                  : all_cores;

    bool convert_df = input_cb_data_format != output_cb_data_format;

    uint32_t num_input_units = num_units_per_shard;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
    uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);

    bool is_tile = (output.layout() == Layout::TILE);

    // ---- Build the ProgramSpec ----
    ProgramSpec spec;
    spec.name = "sharded_to_interleaved";

    // Tensor parameters (typed bindings replace the legacy buffer-address writer RTA slot 0).
    spec.tensor_parameters = {
        TensorParameter{.unique_id = S2I_INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = S2I_OUTPUT, .spec = output.tensor_spec()},
    };

    // Dataflow buffers.
    // INPUT DFB: always present. Borrowed onto the (sharded-L1) input buffer so the resident
    // shard is the DFB's backing memory (legacy dynamic-CB rebinding via cb.buffer = src_buffer).
    DataflowBufferSpec input_dfb{
        .unique_id = S2I_INPUT_DFB,
        .entry_size = input_page_size,
        .num_entries = num_input_units,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = S2I_INPUT,
    };
    spec.dataflow_buffers.push_back(input_dfb);

    // OUTPUT DFB: only when a data-format conversion compute kernel is inserted. Plain L1
    // staging buffer the compute kernel produces and the writer consumes.
    if (convert_df) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = S2I_OUTPUT_DFB,
            .entry_size = output_page_size,
            .num_entries = num_input_units,
            .data_format_metadata = output_cb_data_format,
        });
    }

    // The writer consumes the converted OUTPUT DFB when converting, else the INPUT DFB directly
    // (legacy out_cb_index == src0_cb_index when no conversion).
    const DFBSpecName writer_in_dfb = convert_df ? S2I_OUTPUT_DFB : S2I_INPUT_DFB;

    // Reader kernel: produces the resident input shard into the borrowed INPUT DFB (fake-push).
    KernelSpec reader{
        .unique_id = S2I_READER,
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };
    reader.source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/device/kernels/dataflow/"
        "reader_unary_sharded.cpp";
    reader.dfb_bindings = {ProducerOf(S2I_INPUT_DFB, "in0")};
    reader.runtime_arg_schema = {.runtime_arg_names = {"num_units"}};

    // Writer kernel: consumes the writer-input DFB and writes interleaved output.
    KernelSpec writer{
        .unique_id = S2I_WRITER,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = S2I_OUTPUT, .accessor_name = "dst"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };
    writer.dfb_bindings = {ConsumerOf(writer_in_dfb, "out")};
    if (is_tile) {
        writer.source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/device/kernels/dataflow/"
            "writer_unary_sharded_blocks_interleaved_start_id.cpp";
        writer.runtime_arg_schema = {
            .runtime_arg_names = {
                "block_height_tiles",
                "block_width_tiles",
                "unpadded_block_height_tiles",
                "unpadded_block_width_tiles",
                "output_width_tiles",
                "block_num_tiles",
                "start_id_offset",
                "start_id_base"}};
    } else {
        writer.source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/device/kernels/dataflow/"
            "writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp";
        writer.runtime_arg_schema = {
            .runtime_arg_names = {
                "block_height",
                "block_width_bytes",
                "padded_block_width_bytes",
                "input_width_offset_bytes",
                "start_id"}};
    }

    spec.kernels.push_back(reader);
    spec.kernels.push_back(writer);

    // Optional compute kernel for data-format conversion: consumes INPUT DFB, produces OUTPUT DFB.
    if (convert_df) {
        spec.kernels.push_back(KernelSpec{
            .unique_id = S2I_COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/device/kernels/compute/"
                      "eltwise_copy.cpp",
            .dfb_bindings = {ConsumerOf(S2I_INPUT_DFB, "in0"), ProducerOf(S2I_OUTPUT_DFB, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
            .hw_config = ttnn::to_compute_hardware_config(
                input.device()->arch(),
                ttnn::ComputeKernelConfig{.math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false}),
        });
    }

    // Single work unit: every used core runs the same kernel set; per-core variation is via RTAs.
    Group<KernelSpecName> wu_kernels = {S2I_READER, S2I_WRITER};
    if (convert_df) {
        wu_kernels.push_back(S2I_COMPUTE);
    }
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = wu_kernels, .target_nodes = used_cores}};

    // ---- Build the ProgramRunArgs (per-core runtime args) ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = S2I_READER};
    KernelRunArgs writer_run{.kernel = S2I_WRITER};
    KernelRunArgs compute_run{.kernel = S2I_COMPUTE};

    // Reader run-time args: identical on every used core.
    for (const auto& core_range : used_cores.ranges()) {
        for (const auto& core : core_range) {
            reader_run.runtime_arg_values.push_back(
                KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_units", num_units_per_shard}}});
        }
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    for (uint32_t core_idx = 0; core_idx < num_cores_unpadded; core_idx++) {
        const auto& core = cores[core_idx];
        uint32_t shard_height = num_units_per_shard_height;
        uint32_t shard_width = is_tile ? num_units_per_shard_width : output_unit_size;
        if (is_tile) {
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            // Writer run-time args (buffer-address slot 0 is gone — bound via TensorParameter).
            writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"block_height_tiles", num_units_per_shard_height},
                    {"block_width_tiles", num_units_per_shard_width},
                    {"unpadded_block_height_tiles", shard_height},
                    {"unpadded_block_width_tiles", shard_width},
                    {"output_width_tiles", num_units_offset},
                    {"block_num_tiles", num_units_per_shard},
                    {"start_id_offset", curr_idx_h + curr_idx_w},
                    {"start_id_base", starting_idx_h}}});

            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            uint32_t l1_alignment = hal::get_l1_alignment();
            uint32_t padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment());
            if (is_blackhole or is_l1_aligned) {
                if (!dst_is_dram or is_l1_aligned) {
                    padded_shard_width = tt::align(output_unit_size, l1_alignment);
                }
            }
            // Writer run-time args (buffer-address slot 0 is gone — bound via TensorParameter;
            // legacy slot 1 `num_units_per_row` was emitted but never read by the kernel, so it
            // is dropped here to match the kernel's actual reads).
            writer_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
                .node = core,
                .args = {
                    {"block_height", shard_height},
                    {"block_width_bytes", shard_width},
                    {"padded_block_width_bytes", padded_shard_width},
                    {"input_width_offset_bytes", curr_idx_w},
                    {"start_id", curr_idx_h}}});

            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }

        if (convert_df) {
            compute_run.runtime_arg_values.push_back(
                KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_units", num_units_per_shard}}});
        }
    }

    run_args.kernel_run_args.push_back(reader_run);
    run_args.kernel_run_args.push_back(writer_run);
    if (convert_df) {
        run_args.kernel_run_args.push_back(compute_run);
    }

    // Tensor arguments: reference the same MeshTensors the parameters were declared from.
    run_args.tensor_args.emplace(S2I_INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(S2I_OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
