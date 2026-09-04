// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm.hpp"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::data_movement {

namespace {

// Sub-row chunking: pair-batched NOC transfers per stick (last = `last_chunk_size`) so a wide row fits L1.
struct ChunkingParams {
    uint32_t chunk_size;
    uint32_t num_chunks_per_stick;
    uint32_t last_chunk_size;
};

// The reader / writer scalars that vary per node, plus the reader's runtime vararg block.
struct RmPerNodeArgs {
    uint32_t start_id = 0;
    uint32_t num_sticks_per_core = 0;
    uint32_t num_sticks_per_core_read = 0;
    uint32_t num_read_per_barrier = 0;
    // 3 * num_dims entries: num_unpadded_sticks_per_dim, num_padded_sticks_per_dim, id_per_dim.
    std::vector<uint32_t> reader_varargs;
    uint32_t num_sticks_written = 0;
};

// Program-wide scalars every node receives identically. Legacy emitted these per node as runtime
// args and the port keeps them there: moving a per-node arg to a common one changes dispatch
// semantics, which is a separate cleanup, not port work.
struct RmSharedArgs {
    uint32_t unpadded_row_size_bytes = 0;
    uint32_t unpadded_row_size_bytes_offset = 0;
    uint32_t num_dims = 0;
    uint32_t misalignment = 0;
    uint32_t src_offset_bytes = 0;
};

inline std::pair<RmSharedArgs, std::vector<RmPerNodeArgs>> get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const std::vector<CoreCoord>& all_cores_vec,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size,
    const ChunkingParams& chunking) {
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;
    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, alignment);

    RmSharedArgs shared{
        .unpadded_row_size_bytes = unpadded_row_size_bytes,
        .unpadded_row_size_bytes_offset = unpadded_row_size_bytes_offset,
        .num_dims = num_dims,
        .misalignment = misalignment,
        // Byte offset of the slice's W-begin within a row, rounded down to the source buffer's
        // alignment; the leftover `misalignment` bytes are trimmed on device.
        .src_offset_bytes = begins_bytes - misalignment,
    };

    std::vector<RmPerNodeArgs> per_node;
    per_node.reserve(all_cores_vec.size());

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    uint32_t num_sticks_written = 0;
    for (const auto& core : all_cores_vec) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            num_sticks_per_core = 0;
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            if (chunking.num_chunks_per_stick > 1) {
                num_sticks_per_core_read = num_sticks_per_core;
                // Match `compute_dfb_size`: nrpb=2 only when num_chunks is even, else 1 to avoid ring-wrap straddle.
                num_read_per_barrier = (chunking.num_chunks_per_stick % 2 == 0) ? 2 : 1;
            } else {
                auto num_sticks_per_core_pad32 = round_up_to_mul32(num_sticks_per_core);
                num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(
                    num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, max_read_size);
                num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
            }
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        RmPerNodeArgs node_args{
            .start_id = start_id,
            .num_sticks_per_core = num_sticks_per_core,
            .num_sticks_per_core_read = num_sticks_per_core_read,
            .num_read_per_barrier = num_read_per_barrier,
            .num_sticks_written = num_sticks_written,
        };
        // Three num_dims-long blocks, in the order the kernel walks them.
        node_args.reader_varargs.reserve(num_dims * 3);
        node_args.reader_varargs.insert(
            node_args.reader_varargs.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
        node_args.reader_varargs.insert(
            node_args.reader_varargs.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());
        node_args.reader_varargs.insert(node_args.reader_varargs.end(), id_per_dim.begin(), id_per_dim.end());

        num_sticks_written += num_sticks_per_core;
        per_node.push_back(std::move(node_args));
    }

    return {shared, std::move(per_node)};
}

constexpr uint32_t MAX_READ_SIZE = 4096;
constexpr uint32_t CHUNK_TARGET_BYTES = 8192;  // NOC-friendly chunk when splitting a wide row

struct SliceDfbSizing {
    uint32_t dfb_entry_size;
    uint32_t num_read_per_barrier;
    uint32_t misalignment;
    ChunkingParams chunking;
};

// Chunks sub-row when double-buffered row page overflows L1; gated on misalignment==0
// (misalign path uses a whole-stick memmove that doesn't compose with chunk boundaries).
SliceDfbSizing compute_dfb_size(
    const Tensor& input,
    const Tensor& output,
    const Shape& output_tensor_start,
    const uint32_t num_sticks_per_core_group_1,
    const uint32_t num_sticks_per_core_group_2) {
    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    const auto single_alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    auto alignment = single_alignment;

    uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    const uint32_t unpadded_row_size_bytes = output.padded_shape()[-1] * input.element_size();
    const uint32_t stick_size_aligned = tt::round_up(unpadded_row_size_bytes, alignment);

    const uint32_t l1_budget = ttnn::operations::data_movement::get_max_l1_space(input);

    SliceDfbSizing s{
        .dfb_entry_size = stick_size_aligned,
        .num_read_per_barrier = 0,
        .misalignment = misalignment,
        .chunking = {stick_size_aligned, 1, stick_size_aligned},
    };

    const bool needs_chunking = (misalignment == 0) && (static_cast<uint64_t>(2u) * stick_size_aligned > l1_budget) &&
                                (stick_size_aligned > alignment);
    uint32_t stride_for_merge = tt::round_up(unpadded_row_size_bytes, single_alignment);

    if (needs_chunking) {
        // l1_budget/8 leaves headroom for reader + writer DFBs pair-batched (each 4*chunk_size).
        uint32_t max_chunk = std::min<uint32_t>(CHUNK_TARGET_BYTES, static_cast<uint32_t>(l1_budget / 8));
        max_chunk = (max_chunk / alignment) * alignment;
        TT_FATAL(
            max_chunk >= alignment,
            "ttnn::slice: L1 budget {} B too small for sub-row chunking (alignment {} B)",
            l1_budget,
            alignment);

        uint32_t num_chunks = (unpadded_row_size_bytes + max_chunk - 1) / max_chunk;
        // Odd num_chunks with nrpb=2 straddles the 4-page ring on the next stick — try an aligned
        // shrink to reach even; commit only if it lands, else let the nrpb=1 fallback do the work.
        constexpr uint32_t nrpb = 2;
        if ((num_chunks % nrpb) != 0) {
            const uint32_t target_n = num_chunks + (nrpb - (num_chunks % nrpb));
            uint32_t new_max = unpadded_row_size_bytes / target_n;
            new_max = (new_max / alignment) * alignment;
            const uint32_t candidate_num_chunks =
                (new_max >= alignment) ? (unpadded_row_size_bytes + new_max - 1) / new_max : 0;
            if (new_max >= alignment && (candidate_num_chunks % nrpb) == 0) {
                max_chunk = new_max;
                num_chunks = candidate_num_chunks;
            }
        }

        const uint32_t remainder = unpadded_row_size_bytes % max_chunk;
        s.chunking = {
            .chunk_size = max_chunk,
            .num_chunks_per_stick = num_chunks,
            .last_chunk_size = (remainder == 0) ? max_chunk : remainder,
        };
        s.dfb_entry_size = max_chunk;
        stride_for_merge = max_chunk;
    }

    TT_FATAL(
        static_cast<uint64_t>(2u) * s.dfb_entry_size <= l1_budget,
        "ttnn::slice: required DFB size {} B exceeds per-core L1 budget {} B "
        "(row_bytes={}, misalignment={}); consider slicing along a non-width dim",
        2u * s.dfb_entry_size,
        l1_budget,
        unpadded_row_size_bytes,
        misalignment);

    const uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2
                                         ? num_sticks_per_core_group_1
                                         : num_sticks_per_core_group_2;
    if (num_input_pages != 0) {
        if (needs_chunking) {
            // Fallback when the shrink above couldn't reach an even num_chunks: nrpb=1 makes a straddle impossible.
            s.num_read_per_barrier = (s.chunking.num_chunks_per_stick % 2 == 0) ? 2 : 1;
        } else {
            auto num_sticks_per_core_pad32 = round_up_to_mul32(num_input_pages);
            uint32_t num_sticks_per_core_read =
                tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, stride_for_merge, MAX_READ_SIZE);
            s.num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }
    }

    return s;
}

// The port drops the TensorAccessor page-size override that this factory used to pass as a third
// constructor argument; Metal 2.0 supplies the buffer's aligned page size instead and offers no
// override.
//
// On an *interleaved* accessor the substitution is inert whatever the two values are: the accessor
// realigns the passed page size up to the allocator alignment, so the old argument (the true logical
// row) and the new implicit one (that row rounded up) address identically. Only a *sharded* accessor
// uses the value verbatim, and there the two must actually agree — which they do for a
// HEIGHT-sharded buffer by construction, and for a BLOCK/WIDTH-sharded one only while the shard row
// is alignment-aligned. `ttnn::slice` guarantees that by resharding any tensor that would violate it
// before this factory is reached, but MeshPartition builds these programs off select_program_factory
// and never passes through that guard, so pin the invariant here rather than leaving it implicit.
void check_accessor_page_size(const Tensor& tensor, uint32_t row_size_bytes, const char* role) {
    if (!tensor.memory_config().is_sharded()) {
        return;
    }
    const uint32_t passed = per_shard_page_size_bytes(tensor, row_size_bytes);
    const uint32_t implicit = tensor.buffer()->aligned_page_size();
    TT_FATAL(
        passed == implicit,
        "SliceRmProgramFactory: {} per-shard page size ({} B) must equal the buffer's aligned page size ({} B); "
        "a sub-aligned shard row must be resharded before reaching this factory.",
        role,
        passed,
        implicit);
}

}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

namespace {

// Function-local for the same unity-build reason as the other slice factories.
struct RmSpecNames {
    KernelSpecName reader{"reader"};
    KernelSpecName writer{"writer"};
    DFBSpecName src0{"src0"};
    TensorParamName input{"input"};
    TensorParamName output{"output"};
};

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceRmProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const RmSpecNames names;

    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    const uint32_t padded_row_size_bytes = input.padded_shape()[-1] * input.element_size();
    const uint32_t unpadded_row_size_bytes = output.padded_shape()[-1] * input.element_size();
    ttnn::operations::data_movement::check_accessor_page_size(input, padded_row_size_bytes, "input");
    ttnn::operations::data_movement::check_accessor_page_size(output, unpadded_row_size_bytes, "output");

    // DFB sizing (incl. chunking) derives from padded_shape + slice_start + alignment, all of which
    // fold into compute_program_hash(), so cache entries stay distinct per unique sizing.
    const auto sizing = ttnn::operations::data_movement::compute_dfb_size(
        input, output, args.slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    const DataflowBufferSpec src0_dfb{
        .unique_id = names.src0,
        .entry_size = sizing.dfb_entry_size,
        .num_entries = sizing.num_read_per_barrier * 2,
        .data_format_metadata = dfb_data_format,
    };

    const TensorParameter input_param{.unique_id = names.input, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = names.output, .spec = output.tensor_spec()};

    const auto all_cores_vec = corerange_to_cores(all_cores);
    auto [shared, per_node] = ttnn::operations::data_movement::get_slice_runtime_args_rm(
        input,
        output,
        args.slice_start,
        all_cores_vec,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        ttnn::operations::data_movement::MAX_READ_SIZE,
        sizing.chunking);

    const KernelSpec reader{
        .unique_id = names.reader,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.src0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = names.input, .accessor_name = "src"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"unpadded_stick_size",
                     "stick_size_offset",
                     "num_dims",
                     "misalignment",
                     "start_id",
                     "num_sticks_per_core",
                     "num_sticks_per_core_read",
                     "num_read_per_barrier",
                     "chunk_size",
                     "num_chunks_per_stick",
                     "last_chunk_size",
                     "src_offset_bytes"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = shared.num_dims * 3},
    };

    const KernelSpec writer{
        .unique_id = names.writer,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.src0,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = names.output, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"stick_size",
                     "stick_size_offset",
                     "num_sticks_per_core",
                     "num_sticks_per_core_read",
                     "num_read_per_barrier",
                     "start_id",
                     "chunk_size",
                     "num_chunks_per_stick",
                     "last_chunk_size"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    KernelRunArgs reader_run_args{.kernel = names.reader};
    KernelRunArgs writer_run_args{.kernel = names.writer};
    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        const auto& node = all_cores_vec[i];
        const auto& node_args = per_node[i];

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            node,
            {{"unpadded_stick_size", shared.unpadded_row_size_bytes},
             {"stick_size_offset", shared.unpadded_row_size_bytes_offset},
             {"num_dims", shared.num_dims},
             {"misalignment", shared.misalignment},
             {"start_id", node_args.start_id},
             {"num_sticks_per_core", node_args.num_sticks_per_core},
             {"num_sticks_per_core_read", node_args.num_sticks_per_core_read},
             {"num_read_per_barrier", node_args.num_read_per_barrier},
             {"chunk_size", sizing.chunking.chunk_size},
             {"num_chunks_per_stick", sizing.chunking.num_chunks_per_stick},
             {"last_chunk_size", sizing.chunking.last_chunk_size},
             {"src_offset_bytes", shared.src_offset_bytes}});
        reader_run_args.advanced_options.runtime_varargs[node] = node_args.reader_varargs;

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            node,
            {{"stick_size", shared.unpadded_row_size_bytes},
             {"stick_size_offset", shared.unpadded_row_size_bytes_offset},
             {"num_sticks_per_core", node_args.num_sticks_per_core},
             {"num_sticks_per_core_read", node_args.num_sticks_per_core_read},
             {"num_read_per_barrier", node_args.num_read_per_barrier},
             {"start_id", node_args.num_sticks_written},
             {"chunk_size", sizing.chunking.chunk_size},
             {"num_chunks_per_stick", sizing.chunking.num_chunks_per_stick},
             {"last_chunk_size", sizing.chunking.last_chunk_size}});
    }

    ProgramSpec spec{
        .name = "slice_rm",
        .kernels = {reader, writer},
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "slice",
            .kernels = {names.reader, names.writer},
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{names.input, input.mesh_tensor()}, {names.output, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmProgramFactory::override_runtime_arguments(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const RmSpecNames names;

    // The legacy refresh for this factory re-pointed the two buffer addresses and nothing else; both
    // now travel as tensor bindings, so re-supplying them is the whole job.
    ProgramRunArgs run_args;
    run_args.tensor_args = {{names.input, tensor_args.input.mesh_tensor()}, {names.output, output.mesh_tensor()}};
    return run_args;
}

}  // namespace ttnn::prim
