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
#include <tt-metalium/math.hpp>
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

    // Extents in destination units: tiles for tile layout, sticks x bytes for row-major. A row-major
    // page is one logical row, so its extent comes from the logical shape, not the padded one.
    uint32_t num_units_per_shard = 0;
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t tensor_h = 0;
    uint32_t tensor_w = 0;
    uint32_t shard_h = 0;
    uint32_t shard_w = 0;

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto& all_cores = shard_spec.grid;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);

    if (output.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        tensor_h = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        tensor_w = input.padded_shape()[-1] / TILE_WIDTH;
        shard_h = shard_spec.shape[0] / TILE_HEIGHT;
        shard_w = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = shard_h * shard_w;
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        tensor_h = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        tensor_w = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        shard_h = shard_spec.shape[0];
        shard_w = output_unit_size;
        num_units_per_shard = shard_h;
    }

    const uint32_t height_shards = div_up(tensor_h, shard_h);
    const uint32_t width_shards = div_up(tensor_w, shard_w);
    const uint32_t num_active_cores = height_shards * width_shards;

    // A grid provisioned wider than the data leaves cores holding only padding; leave them out.
    const CoreCoord grid_origin = all_cores.bounding_box().start_coord;
    CoreRangeSet used_cores;
    if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
        const uint32_t grid_x = rm_orientation ? width_shards : height_shards;
        const uint32_t grid_y = rm_orientation ? height_shards : width_shards;
        used_cores =
            CoreRangeSet(CoreRange(grid_origin, CoreCoord{grid_origin.x + grid_x - 1, grid_origin.y + grid_y - 1}));
    } else {
        used_cores = num_active_cores < all_cores.num_cores()
                         ? select_from_corerangeset(all_cores, 0, num_active_cores - 1, rm_orientation)
                         : all_cores;
    }

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
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
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
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
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
            reader_run.runtime_arg_values["num_units"][core] = num_units_per_shard;
        }
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(output, num_slices, slice_index);
    // Source stride: the shard's row pitch in L1, which stays the full shard width.
    uint32_t padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment());
    if (is_blackhole or is_l1_aligned) {
        if (!dst_is_dram or is_l1_aligned) {
            padded_shard_width = tt::align(output_unit_size, hal::get_l1_alignment());
        }
    }

    KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
    for (uint32_t sh = 0; sh < height_shards; sh++) {
        for (uint32_t sw = 0; sw < width_shards; sw++) {
            // Height sharding has a single column and width sharding a single row, so one index is 0.
            const CoreCoord core =
                shard_strategy == TensorMemoryLayout::BLOCK_SHARDED
                    ? CoreCoord{grid_origin.x + (rm_orientation ? sw : sh), grid_origin.y + (rm_orientation ? sh : sw)}
                    : cores[sh + sw];
            const uint32_t h0 = sh * shard_h;
            const uint32_t w0 = sw * shard_w;
            // Clipping to the tensor is what keeps the write inside the destination page.
            const uint32_t shard_height = std::min(shard_h, tensor_h - h0);
            const uint32_t shard_width = std::min(shard_w, tensor_w - w0);

            if (is_tile) {
                AddRuntimeArgsForNode(
                    writer_rtas,
                    core,
                    {
                        {"block_height_tiles", shard_h},
                        {"block_width_tiles", shard_w},
                        {"unpadded_block_height_tiles", shard_height},
                        {"unpadded_block_width_tiles", shard_width},
                        {"output_width_tiles", tensor_w},
                        {"block_num_tiles", num_units_per_shard},
                        {"start_id_offset", h0 * tensor_w + w0},
                        {"start_id_base", starting_idx_h},
                    });
            } else {
                AddRuntimeArgsForNode(
                    writer_rtas,
                    core,
                    {
                        {"block_height", shard_height},
                        {"block_width_bytes", shard_width},
                        {"padded_block_width_bytes", padded_shard_width},
                        {"input_width_offset_bytes", w0},
                        {"start_id", h0},
                    });
            }

            if (convert_df) {
                compute_run.runtime_arg_values["num_units"][core] = num_units_per_shard;
            }
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
