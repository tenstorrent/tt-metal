// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// Spec resource names. Prefixed to stay distinct under unity builds
// (Pattern: Unity-build hygiene for anonymous-namespace symbols).
const DFBSpecName I2S_INPUT_DFB{"i2s_input"};
const DFBSpecName I2S_OUTPUT_DFB{"i2s_output"};
const DFBSpecName I2S_SCRATCH_DFB{"i2s_scratch"};

const TensorParamName I2S_INPUT{"i2s_input"};
const TensorParamName I2S_OUTPUT{"i2s_output"};

const KernelSpecName I2S_READER{"i2s_reader"};
const KernelSpecName I2S_WRITER{"i2s_writer"};
const KernelSpecName I2S_COMPUTE{"i2s_compute"};

}  // namespace

// Hardcoded for non-partial interleaved_to_sharded operation
// to keep backward compatibility after migration to new infra
// https://github.com/tenstorrent/tt-metal/issues/32752
constexpr uint32_t num_slices = 1;
constexpr uint32_t slice_index = 0;

ttnn::device_operation::ProgramArtifacts InterleavedToShardedProgramFactory::create_program_artifacts(
    const InterleavedToShardedParams& /*operation_attributes*/,
    const InterleavedToShardedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    // Keep explicit bool init to match legacy behavior which forced it true
    bool keep_l1_aligned = true;  // operation_attributes.keep_l1_aligned;

    uint32_t num_units_per_shard = 0;
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t num_units_per_shard_width = 0;
    uint32_t num_units_per_shard_height = 0;
    uint32_t num_units_offset = 0;
    uint32_t num_units_per_row = 0;
    uint32_t num_units_per_shard_height_last = 0;
    uint32_t num_units_per_shard_width_last = 0;
    uint32_t padded_offset_bytes = 0;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = output.shard_spec().value();
    auto shard_strategy = output.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto cores = get_optimal_worker_cores_for_sharded_tensor(output);
    auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));
    CoreCoord end_core = cores.back();

    bool convert_df = input_cb_data_format != output_cb_data_format;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);

    if (input.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        TT_FATAL(
            shard_spec.shape[0] % TILE_HEIGHT == 0 && shard_spec.shape[1] % TILE_WIDTH == 0,
            "Shard shape {} must be tile {}x{} sized!",
            shard_spec.shape,
            TILE_HEIGHT,
            TILE_WIDTH);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width -
            (tt::round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
        padded_offset_bytes = (num_units_per_shard_width - num_units_per_shard_width_last) * input_unit_size;
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        num_units_offset = 1;
        uint32_t num_units_height = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        // TODO: Use a different variable name. Units refers to pages, but this is being used as size
        num_units_per_shard_width_last =
            input_unit_size - (tt::round_up(num_units_per_row, input_unit_size) - num_units_per_row);
        // Adjust accordingly to l1 alignment, do it for all archs
        if (keep_l1_aligned) {
            padded_offset_bytes = tt::align(input_unit_size, hal::get_l1_alignment());
        } else {
            padded_offset_bytes = tt::align(input_unit_size, input.buffer()->alignment());
        }
    }

    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());

    uint32_t dram_alignment = hal::get_dram_alignment();
    uint32_t l1_alignment = hal::get_l1_alignment();
    uint32_t num_trids = 4;

    // The scratch DFB is only used by the stick-layout (ROW_MAJOR) reader. Match the legacy
    // condition that decided whether the scratch CB was created.
    bool use_scratch = (src_is_dram && (input_unit_size % dram_alignment != 0)) || is_blackhole || keep_l1_aligned;
    uint32_t scratch_cb_page_size = tt::align(input_unit_size + dram_alignment, dram_alignment);

    uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());

    bool is_tile = (input.layout() == Layout::TILE);

    // ---- Build the ProgramSpec ----
    ProgramSpec spec;
    spec.name = "interleaved_to_sharded";

    // Tensor parameters (typed bindings replace the legacy buffer-address RTA slot 0).
    spec.tensor_parameters = {
        TensorParameter{.unique_id = I2S_INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = I2S_OUTPUT, .spec = output.tensor_spec()},
    };

    // Dataflow buffers.
    // OUTPUT DFB: always present. Borrowed onto the output buffer when the destination is
    // sharded-L1 (legacy dynamic-CB rebinding via cb.buffer); plain L1 DFB when dst is DRAM.
    DataflowBufferSpec output_dfb{
        .unique_id = I2S_OUTPUT_DFB,
        .entry_size = output_page_size,
        .num_entries = num_input_units,
        .data_format_metadata = output_cb_data_format,
    };
    if (!dst_is_dram) {
        output_dfb.borrowed_from = I2S_OUTPUT;
    }
    spec.dataflow_buffers.push_back(output_dfb);

    // INPUT DFB: only when a data-format conversion compute kernel is inserted.
    if (convert_df) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = I2S_INPUT_DFB,
            .entry_size = input_page_size,
            .num_entries = num_input_units,
            .data_format_metadata = input_cb_data_format,
        });
    }

    // SCRATCH DFB: only on the stick-layout reader path, used as a TRID staging buffer.
    if (!is_tile && use_scratch) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = I2S_SCRATCH_DFB,
            .entry_size = scratch_cb_page_size,
            .num_entries = num_trids,
            .data_format_metadata = input_cb_data_format,
        });
    }

    // Reader kernel. Produces into INPUT_DFB (when converting) or directly into OUTPUT_DFB.
    const DFBSpecName reader_out_dfb = convert_df ? I2S_INPUT_DFB : I2S_OUTPUT_DFB;
    KernelSpec reader{
        .unique_id = I2S_READER,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = I2S_INPUT, .accessor_name = "src"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };
    if (is_tile) {
        reader.source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/device/kernels/dataflow/"
            "reader_unary_sharded_blocks_interleaved_start_id.cpp";
        // tile_bytes: the in0 DFB's tile size (== the bound DFB's data-format tile size).
        // Passed as a CTA because the kernel needs it as a compile-time constant (template
        // arg) and the device-side get_tile_size() is not arch-portable to Quasar.
        reader.compile_time_args = {
            {"num_readers", all_cores.num_cores()}, {"tile_bytes", convert_df ? input_unit_size : output_unit_size}};
        reader.dfb_bindings = {ProducerOf(reader_out_dfb, "in0")};
        reader.runtime_arg_schema = {
            .runtime_arg_names = {
                "block_height_tiles",
                "block_width_tiles",
                "padded_offset_bytes",
                "input_width_offset_tiles",
                "block_num_tiles",
                "start_id_offset",
                "start_id_base"}};
    } else {
        reader.source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/device/kernels/dataflow/"
            "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp";
        reader.compile_time_args = {{"num_trids", num_trids}};
        // SCRATCH is a self-loop on the reader (produces and consumes its own staging buffer).
        reader.dfb_bindings = {
            ProducerOf(reader_out_dfb, "in0"),
            DFBBinding{
                .dfb_spec_name = I2S_SCRATCH_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = I2S_SCRATCH_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
        };
        reader.runtime_arg_schema = {
            .runtime_arg_names = {
                "block_height",
                "block_width_bytes",
                "padded_block_width_bytes",
                "aligned",
                "aligned_input_width_offset_bytes",
                "aligned_block_width_bytes",
                "aligned_offset",
                "start_id"}};
    }

    // Writer kernel.
    KernelSpec writer{
        .unique_id = I2S_WRITER,
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };
    if (dst_is_dram) {
        writer.dfb_bindings = {ConsumerOf(I2S_OUTPUT_DFB, "out")};
        writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = I2S_OUTPUT, .accessor_name = "dst"}};
        if (is_tile) {
            writer.source =
                "ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/device/kernels/dataflow/"
                "writer_unary_sharded_blocks_start_id.cpp";
            writer.runtime_arg_schema = {
                .runtime_arg_names = {
                    "block_height_tiles",
                    "block_width_tiles",
                    "padded_offset",
                    "block_width_padded_num_tiles",
                    "output_width_tiles",
                    "start_id_offset",
                    "start_id_base"}};
        } else {
            writer.source =
                "ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/device/kernels/dataflow/"
                "writer_unary_sharded_stick_layout_start_id.cpp";
            writer.runtime_arg_schema = {
                .runtime_arg_names = {
                    "block_height",
                    "block_width_bytes",
                    "padded_block_width_bytes",
                    "start_id",
                    "output_width_in_pages"}};
        }
    } else {
        writer.source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/device/kernels/dataflow/"
            "writer_unary_sharded.cpp";
        writer.dfb_bindings = {ConsumerOf(I2S_OUTPUT_DFB, "out")};
        writer.runtime_arg_schema = {.runtime_arg_names = {"num_units"}};
    }

    spec.kernels.push_back(reader);
    spec.kernels.push_back(writer);

    // Optional compute kernel for data-format conversion: consumes INPUT_DFB, produces OUTPUT_DFB.
    if (convert_df) {
        spec.kernels.push_back(KernelSpec{
            .unique_id = I2S_COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/device/kernels/compute/"
                      "eltwise_copy.cpp",
            .dfb_bindings = {ConsumerOf(I2S_INPUT_DFB, "in0"), ProducerOf(I2S_OUTPUT_DFB, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
            .hw_config = ttnn::to_compute_hardware_config(
                input.device()->arch(),
                ttnn::ComputeKernelConfig{.math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false}),
        });
    }

    // Single work unit: every core runs the same kernel set; per-core variation is via RTAs.
    Group<KernelSpecName> wu_kernels = {I2S_READER, I2S_WRITER};
    if (convert_df) {
        wu_kernels.push_back(I2S_COMPUTE);
    }
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = wu_kernels, .target_nodes = all_cores}};

    // ---- Build the ProgramRunArgs (per-core runtime args) ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = I2S_READER};
    KernelRunArgs writer_run{.kernel = I2S_WRITER};
    KernelRunArgs compute_run{.kernel = I2S_COMPUTE};

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(input, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    for (const auto& core : cores) {
        uint32_t curr_num_units_per_shard = num_units_per_shard;
        KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
        KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
        KernelRunArgs::RuntimeArgValues& compute_rtas = compute_run.runtime_arg_values;
        if (is_tile) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            uint32_t padded_offset = 0;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core == end_core) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core == end_core) {
                    shard_width = num_units_per_shard_width_last;
                    padded_offset = padded_offset_bytes;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                        padded_offset = padded_offset_bytes;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                        padded_offset = padded_offset_bytes;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            curr_num_units_per_shard = shard_height * num_units_per_shard_width;

            // Reader run-time args (buffer-address slot 0 is gone — bound via TensorParameter).
            AddRuntimeArgsForNode(
                reader_rtas,
                core,
                {
                    {"block_height_tiles", shard_height},
                    {"block_width_tiles", shard_width},
                    {"padded_offset_bytes", padded_offset},
                    {"input_width_offset_tiles", num_units_offset},
                    {"block_num_tiles", curr_num_units_per_shard},
                    {"start_id_offset", curr_idx_h + curr_idx_w},
                    {"start_id_base", starting_idx_h},
                });

            // Writer run-time args
            uint32_t pad_offset = (num_units_per_shard_width - shard_width) * output_unit_size;
            if (dst_is_dram) {
                AddRuntimeArgsForNode(
                    writer_rtas,
                    core,
                    {
                        {"block_height_tiles", shard_height},
                        {"block_width_tiles", shard_width},
                        {"padded_offset", pad_offset},
                        {"block_width_padded_num_tiles", curr_num_units_per_shard},
                        {"output_width_tiles", num_units_offset},
                        {"start_id_offset", curr_idx_h + curr_idx_w},
                        {"start_id_base", starting_idx_h},
                    });
            } else {
                writer_rtas["num_units"][core] = curr_num_units_per_shard;
            }

            // Update indexing
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = input_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                    curr_num_units_per_shard = shard_height * num_units_per_shard_width;
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
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                }
            }

            bool aligned = false;
            if (src_is_dram) {
                aligned = (curr_idx_w % dram_alignment == 0) && (padded_offset_bytes % dram_alignment == 0);
            } else if (is_blackhole) {
                aligned = (curr_idx_w % l1_alignment == 0) && (padded_offset_bytes % l1_alignment == 0);
            } else {
                aligned = true;
            }
            uint32_t aligned_width_offset = 0;
            uint32_t aligned_shard_width = 0;
            uint32_t aligned_offset = 0;
            if (!aligned) {
                // TODO: is this right, leaving non BH case the same for now, should investigate
                if (!is_blackhole) {
                    aligned_width_offset = tt::round_down(curr_idx_w, dram_alignment);
                } else {
                    if (src_is_dram) {
                        aligned_width_offset = tt::round_down(curr_idx_w, dram_alignment);
                    } else {
                        aligned_width_offset = tt::round_down(curr_idx_w, l1_alignment);
                    }
                }
                aligned_offset = curr_idx_w - aligned_width_offset;
                aligned_shard_width = aligned_offset + shard_width;
            } else {
                aligned_width_offset = curr_idx_w;
                aligned_shard_width = shard_width;
                aligned_offset = 0;
            }

            // Reader run-time args (buffer-address slot 0 is gone — bound via TensorParameter;
            // legacy slot 1 `num_units_per_row` was emitted but never read by the kernel, so it
            // is dropped here to match the kernel's actual reads).
            AddRuntimeArgsForNode(
                reader_rtas,
                core,
                {
                    {"block_height", shard_height},
                    {"block_width_bytes", shard_width},
                    {"padded_block_width_bytes", padded_offset_bytes},
                    {"aligned", static_cast<uint32_t>(aligned)},
                    {"aligned_input_width_offset_bytes", aligned_width_offset},
                    {"aligned_block_width_bytes", aligned_shard_width},
                    {"aligned_offset", aligned_offset},
                    {"start_id", curr_idx_h},
                });

            // Writer run-time args
            if (dst_is_dram) {
                uint32_t page_id_within_row = curr_idx_w / input_unit_size;
                uint32_t output_width_in_pages = tt::div_up(num_units_per_row, input_unit_size);
                uint32_t start_id = (curr_idx_h * output_width_in_pages) + page_id_within_row;
                AddRuntimeArgsForNode(
                    writer_rtas,
                    core,
                    {
                        {"block_height", shard_height},
                        {"block_width_bytes", shard_width},
                        {"padded_block_width_bytes", padded_offset_bytes},
                        {"start_id", start_id},
                        {"output_width_in_pages", output_width_in_pages},
                    });
            } else {
                writer_rtas["num_units"][core] = curr_num_units_per_shard;
            }

            // Update indexing
            curr_idx_w += input_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
        if (convert_df) {
            compute_rtas["num_units"][core] = curr_num_units_per_shard;
        }
    }

    run_args.kernel_run_args.push_back(reader_run);
    run_args.kernel_run_args.push_back(writer_run);
    if (convert_df) {
        run_args.kernel_run_args.push_back(compute_run);
    }

    // Tensor arguments: reference the same MeshTensors the parameters were declared from.
    run_args.tensor_args.emplace(I2S_INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(I2S_OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
