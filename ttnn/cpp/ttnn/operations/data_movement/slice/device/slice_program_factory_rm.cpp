// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_metal2_names.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm.hpp"

#include <optional>
#include <tuple>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

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

// The scalars every core shares, and the per-core ones. The legacy factory delivered both as one
// positional vector per core; named arguments make the split explicit, and the per-dim blocks stay a
// variable-length run because their length tracks the tensor rank.
struct SliceRmSharedArgs {
    uint32_t unpadded_stick_size;
    uint32_t stick_size_offset;
    uint32_t num_dims;
    uint32_t misalignment;
    uint32_t src_offset_bytes;
    std::vector<uint32_t> num_unpadded_sticks_per_dim;
    std::vector<uint32_t> num_padded_sticks_per_dim;
};

struct SliceRmPerCoreArgs {
    uint32_t start_id;
    uint32_t num_sticks_per_core;
    uint32_t num_sticks_per_core_read;
    uint32_t num_read_per_barrier;
    uint32_t num_sticks_written;
    std::vector<uint32_t> id_per_dim;
};

inline std::pair<SliceRmSharedArgs, std::vector<SliceRmPerCoreArgs>> get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores,
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

    SliceRmSharedArgs shared{
        .unpadded_stick_size = unpadded_row_size_bytes,
        .stick_size_offset = unpadded_row_size_bytes_offset,
        .num_dims = num_dims,
        .misalignment = misalignment,
        // Byte offset of the slice's W-begin within a row, rounded down to the source buffer's
        // alignment; the kernel trims the leftover `misalignment` bytes on device.
        .src_offset_bytes = begins_bytes - misalignment,
        .num_unpadded_sticks_per_dim = num_unpadded_sticks_per_dim,
        .num_padded_sticks_per_dim = num_padded_sticks_per_dim,
    };

    std::vector<SliceRmPerCoreArgs> per_core;
    per_core.reserve(num_cores);

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

        per_core.push_back(SliceRmPerCoreArgs{
            .start_id = start_id,
            .num_sticks_per_core = num_sticks_per_core,
            .num_sticks_per_core_read = num_sticks_per_core_read,
            .num_read_per_barrier = num_read_per_barrier,
            .num_sticks_written = num_sticks_written,
            .id_per_dim = id_per_dim,
        });
        num_sticks_written += num_sticks_per_core;
    }

    return {std::move(shared), std::move(per_core)};
}

constexpr uint32_t MAX_READ_SIZE = 4096;
constexpr uint32_t CHUNK_TARGET_BYTES = 8192;  // NOC-friendly chunk when splitting a wide row

struct SliceDfbSizing {
    uint32_t dfb_page_size;
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
        .dfb_page_size = stick_size_aligned,
        .num_read_per_barrier = 0,
        .misalignment = misalignment,
        .chunking = {stick_size_aligned, 1, stick_size_aligned},
    };

    const bool needs_chunking = (misalignment == 0) && (static_cast<uint64_t>(2u) * stick_size_aligned > l1_budget) &&
                                (stick_size_aligned > alignment);
    uint32_t stride_for_merge = tt::round_up(unpadded_row_size_bytes, single_alignment);

    if (needs_chunking) {
        // l1_budget/8 leaves headroom for the reader and writer buffers pair-batched (each 4*chunk_size).
        uint32_t max_chunk = std::min<uint32_t>(CHUNK_TARGET_BYTES, static_cast<uint32_t>(l1_budget / 8));
        max_chunk = (max_chunk / alignment) * alignment;
        TT_FATAL(
            max_chunk >= alignment,
            "ttnn::slice: L1 budget {} B too small for sub-row chunking (alignment {} B)",
            l1_budget,
            alignment);

        uint32_t num_chunks = (unpadded_row_size_bytes + max_chunk - 1) / max_chunk;
        // Odd num_chunks with nrpb=2 straddles the 4-page ring on the next stick. Try an aligned
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
        s.dfb_page_size = max_chunk;
        stride_for_merge = max_chunk;
    }

    TT_FATAL(
        static_cast<uint64_t>(2u) * s.dfb_page_size <= l1_budget,
        "ttnn::slice: required buffer size {} B exceeds per-core L1 budget {} B "
        "(row_bytes={}, misalignment={}); consider slicing along a non-width dim",
        2u * s.dfb_page_size,
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

// Both RM kernels build their TensorAccessor from the tensor binding, so each strides by the aligned
// page size the binding supplies. That is interchangeable with the per-shard page size they need only
// where the two agree: exactly, on a sharded buffer, whose accessor strides by the value verbatim and
// whose `noc_async_*_sharded` splits pages by it; and up to rounding on an interleaved one, whose
// accessor rounds the page size up to the allocator alignment internally before using it as a stride,
// so a raw row and a pre-rounded one coincide. On a block/width-sharded buffer that reduces to the
// shard row being a multiple of the buffer alignment, which `has_subaligned_shard_row` guarantees for
// anything arriving via ttnn::slice. MeshPartition, though, builds these programs straight off
// select_program_factory and never sees that guard.
void check_accessor_page_size(const Tensor& t, uint32_t row_bytes, const char* role) {
    const auto* buffer = t.buffer();
    const uint32_t alignment = buffer->alignment();
    const uint32_t aligned_page_size = static_cast<uint32_t>(buffer->aligned_page_size());
    const uint32_t per_shard = per_shard_page_size_bytes(t, row_bytes);
    const uint32_t effective = t.memory_config().is_sharded() ? per_shard : tt::round_up(per_shard, alignment);
    TT_FATAL(
        effective == aligned_page_size,
        "ttnn::slice: {} per-shard page size {} B disagrees with the accessor's aligned page size {} B "
        "({} B is not a multiple of the {} B buffer alignment). Reach this op through ttnn::slice, which "
        "reshards such tensors, rather than building the program factory directly.",
        role,
        effective,
        aligned_page_size,
        per_shard,
        alignment);
}

}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

namespace {

// The one dataflow buffer this factory declares: the reader stages sticks (or sub-row chunks) here
// and the writer drains them. The identifier is factory-prefixed for the unity build.
const DFBSpecName RM_STICKS{"sticks"};

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceRmProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using slice_metal2::INPUT;
    using slice_metal2::OUTPUT;
    using slice_metal2::READER;
    using slice_metal2::WRITER;

    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // The kernels take the accessor's page size rather than a runtime one, so pin the equivalence
    // here: a route that skips slice.cpp's resharding guard fails loudly instead of striding by the
    // wrong page.
    ttnn::operations::data_movement::check_accessor_page_size(
        input, input.padded_shape()[-1] * input.element_size(), "input");
    ttnn::operations::data_movement::check_accessor_page_size(
        output, output.padded_shape()[-1] * input.element_size(), "output");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    // Buffer sizing (incl. chunking) derives from padded_shape + slice_start + alignment, all of which
    // fold into compute_program_hash(), so cache entries stay distinct per unique layout.
    const auto sizing = ttnn::operations::data_movement::compute_dfb_size(
        input, output, args.slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    DataflowBufferSpec sticks_dfb{
        .unique_id = RM_STICKS,
        .entry_size = sizing.dfb_page_size,
        .num_entries = sizing.num_read_per_barrier * 2,
        .data_format_metadata = dfb_data_format,
    };

    auto all_cores_vec = corerange_to_cores(all_cores);
    auto [shared, per_core] = ttnn::operations::data_movement::get_slice_runtime_args_rm(
        input,
        output,
        args.slice_start,
        num_cores,
        all_cores_vec,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        ttnn::operations::data_movement::MAX_READ_SIZE,
        sizing.chunking);

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RM_STICKS,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
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
        // Three rank-length blocks: num_unpadded_sticks, num_padded_sticks, and the per-dim walk
        // counters the kernel advances as it reads.
        .advanced_options = {.num_runtime_varargs = 3 * shared.num_dims},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RM_STICKS,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "dst",
                },
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

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        const auto& core = all_cores_vec[i];
        const auto& c = per_core[i];

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"unpadded_stick_size", shared.unpadded_stick_size},
             {"stick_size_offset", shared.stick_size_offset},
             {"num_dims", shared.num_dims},
             {"misalignment", shared.misalignment},
             {"start_id", c.start_id},
             {"num_sticks_per_core", c.num_sticks_per_core},
             {"num_sticks_per_core_read", c.num_sticks_per_core_read},
             {"num_read_per_barrier", c.num_read_per_barrier},
             {"chunk_size", sizing.chunking.chunk_size},
             {"num_chunks_per_stick", sizing.chunking.num_chunks_per_stick},
             {"last_chunk_size", sizing.chunking.last_chunk_size},
             {"src_offset_bytes", shared.src_offset_bytes}});

        AdvancedKernelRunArgs::Varargs reader_varargs;
        reader_varargs.reserve(3 * shared.num_dims);
        reader_varargs.insert(
            reader_varargs.end(), shared.num_unpadded_sticks_per_dim.begin(), shared.num_unpadded_sticks_per_dim.end());
        reader_varargs.insert(
            reader_varargs.end(), shared.num_padded_sticks_per_dim.begin(), shared.num_padded_sticks_per_dim.end());
        reader_varargs.insert(reader_varargs.end(), c.id_per_dim.begin(), c.id_per_dim.end());
        reader_run_args.advanced_options.runtime_varargs[core] = std::move(reader_varargs);

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"stick_size", shared.unpadded_stick_size},
             {"stick_size_offset", shared.stick_size_offset},
             {"num_sticks_per_core", c.num_sticks_per_core},
             {"num_sticks_per_core_read", c.num_sticks_per_core_read},
             {"num_read_per_barrier", c.num_read_per_barrier},
             {"start_id", c.num_sticks_written},
             {"chunk_size", sizing.chunking.chunk_size},
             {"num_chunks_per_stick", sizing.chunking.num_chunks_per_stick},
             {"last_chunk_size", sizing.chunking.last_chunk_size}});
    }

    ProgramSpec spec{
        .name = "slice_rm",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(sticks_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmProgramFactory::override_runtime_arguments(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Every scalar this factory emits is shape-derived and cache-keyed, so a hit can only differ in
    // where the two tensors live. That matches the set the ported-from override re-pointed.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {slice_metal2::INPUT, tensor_args.input.mesh_tensor()},
        {slice_metal2::OUTPUT, output.mesh_tensor()},
    };
    return run_args;
}

}  // namespace ttnn::prim
