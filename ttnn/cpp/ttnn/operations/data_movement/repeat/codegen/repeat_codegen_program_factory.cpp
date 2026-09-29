// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr const char* kReaderSequenced =
    "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp";
constexpr const char* kWriterInterleaved =
    "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp";

// SEQ_REPEAT, see common/kernels/codegen/sequencers.h.
constexpr uint32_t kSeqRepeat = 1;

// The direct outer-axis reader stages one source tile at a time.
constexpr uint32_t kDirectCbDepth = 1;

// Blackhole worker caps for an outer-axis TILE repeat into sharded storage. A full-grid producer set
// repeatedly crossing a small shard grid saturates the NoC links into it; fewer workers keep enough
// read parallelism while cutting that many-to-few injection pressure. A WIDTH_SHARDED output below
// kWidthShardedCapMinOutPages stays uncapped: there the fixed per-worker overhead dominates.
constexpr uint32_t kCoreCapMinRepeats = 4;
constexpr uint32_t kBlockShardedOuterRepeatCores = 100;
constexpr uint32_t kWidthShardedOuterRepeatCores = 64;
constexpr uint32_t kWidthShardedCapMinOutPages = 1024;

// Axes 2 and 3 of the 4D page map are H and W; anything below is an outer axis.
constexpr uint32_t kFirstTileAxis = 2;

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

std::optional<uint32_t> tuned_core_cap(const Tensor& input, const Tensor& output, const RepeatCodegenParams& params) {
    if (input.device()->arch() != tt::ARCH::BLACKHOLE || input.layout() != ttnn::TILE_LAYOUT ||
        params.rep_dim >= kFirstTileAxis || params.num_repeats < kCoreCapMinRepeats) {
        return std::nullopt;
    }
    const auto out_layout = output.memory_config().memory_layout();
    if (out_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        return kBlockShardedOuterRepeatCores;
    }
    if (out_layout == TensorMemoryLayout::WIDTH_SHARDED && params.total_out_pages >= kWidthShardedCapMinOutPages) {
        return kWidthShardedOuterRepeatCores;
    }
    return std::nullopt;
}

CoreSplit split_work(const Tensor& input, uint32_t total_work, std::optional<uint32_t> max_cores = std::nullopt) {
    MeshDevice* device = input.device();
    auto grid_size = device->compute_with_storage_grid_size();
    // Column-major enumeration (row_wise=false). When the work fills the grid the order does not
    // matter. A small split, such as a few sticks spread over a mostly idle grid, lands on the first
    // cores of the first columns, which sit next to a DRAM column, so every page range takes
    // fewer NoC hops than the row-major order would give it.
    const auto grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        max_cores.has_value() ? tt::tt_metal::split_work_to_cores(
                                    num_cores_to_corerangeset(
                                        std::min({*max_cores, total_work, grid_cores}), grid_size, /*row_wise=*/false),
                                    total_work,
                                    /*row_wise=*/false)
                              : tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
    return CoreSplit{
        .all_cores = all_cores,
        .cores_in_order = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .work_per_core_1 = work_per_core_1,
        .work_per_core_2 = work_per_core_2,
    };
}

uint32_t work_for_core(const CoreSplit& split, const CoreCoord& core) {
    if (split.core_group_1.contains(core)) {
        return split.work_per_core_1;
    }
    if (split.core_group_2.contains(core)) {
        return split.work_per_core_2;
    }
    return 0;
}

}  // namespace

uint64_t static_l1_window(const Tensor& input) {
    MeshDevice* device = input.device();
    const uint64_t base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint64_t ceiling = static_cast<uint64_t>(device->l1_size_per_core());
    return ceiling > base ? ceiling - base : 0;
}

uint32_t spec_aligned_page_bytes(const Tensor& device_tensor, const TensorSpec& spec) {
    const uint32_t alignment = device_tensor.device()->allocator()->get_alignment(spec.memory_config().buffer_type());
    return tt::round_up(static_cast<uint32_t>(spec.compute_page_size_bytes()), alignment);
}

RepeatPageMap derive_page_map(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats) {
    const auto& shape = input.logical_shape();
    RepeatPageMap map;
    if (input.layout() == ttnn::TILE_LAYOUT) {
        const std::array<uint32_t, 4> dim_pages = {
            shape[0],
            shape[1],
            tt::div_up(shape[2], tt::constants::TILE_HEIGHT),
            tt::div_up(shape[3], tt::constants::TILE_WIDTH)};
        map.lower_pages = 1;
        for (uint32_t d = rep_dim + 1; d < 4; ++d) {
            map.lower_pages *= dim_pages[d];
        }
        map.rep_dim_pages = dim_pages[rep_dim];
        map.total_out_pages = dim_pages[0] * dim_pages[1] * dim_pages[2] * dim_pages[3] * num_repeats;
        return map;
    }
    // A ROW_MAJOR page is one stick.
    map.stick_size = shape[3] * input.element_size();
    const uint32_t sticks = shape[0] * shape[1] * shape[2];
    if (rep_dim == 3) {
        // The repeat widens each stick and leaves the page count alone.
        map.total_out_pages = sticks;
        return map;
    }
    const std::array<uint32_t, 4> dim_pages = {shape[0], shape[1], shape[2], 1};
    map.lower_pages = 1;
    for (uint32_t d = rep_dim + 1; d < 4; ++d) {
        map.lower_pages *= dim_pages[d];
    }
    map.rep_dim_pages = dim_pages[rep_dim];
    map.total_out_pages = sticks * num_repeats;
    return map;
}

std::optional<RepeatRmCbPlan> live_rm_cb_plan(const Tensor& input, uint32_t slot_bytes) {
    if (input.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        return std::nullopt;
    }
    return plan_rm_cb(slot_bytes, ttnn::operations::data_movement::get_max_l1_space(input));
}

ProgramDescriptor RepeatCodegenProgramFactory::create_descriptor(
    const RepeatCodegenParams& operation_attributes,
    const RepeatCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src_buffer != nullptr, "RepeatCodegen input must be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "RepeatCodegen output must be allocated on device!");

    const bool is_row_major = input.layout() == ttnn::ROW_MAJOR_LAYOUT;
    const bool is_last_dim_rm = is_row_major && operation_attributes.rep_dim == 3;
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    ProgramDescriptor desc;

    // An outer-axis TILE repeat into L1 reads each source tile once and writes all of its copies,
    // instead of re-reading the source once per output page. Only placements whose destination pages
    // the accessor maps one tile per page qualify.
    const auto out_layout = output.memory_config().memory_layout();
    const bool direct_outer_tile =
        !is_row_major && operation_attributes.rep_dim < kFirstTileAxis && dst_buffer->buffer_type() == BufferType::L1 &&
        (out_layout == TensorMemoryLayout::INTERLEAVED || out_layout == TensorMemoryLayout::WIDTH_SHARDED);
    if (direct_outer_tile) {
        const uint32_t tile_bytes = tt::tile_size(cb_data_format);
        const uint32_t total_in_pages = operation_attributes.total_out_pages / operation_attributes.num_repeats;
        const CoreSplit split = split_work(input, total_in_pages, tuned_core_cap(input, output, operation_attributes));

        desc.cbs.push_back(CBDescriptor{
            .total_size = kDirectCbDepth * tile_bytes,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = 0,
                .data_format = cb_data_format,
                .page_size = tile_bytes,
            }}},
        });

        std::vector<uint32_t> reader_ct_args = {
            0, tile_bytes, operation_attributes.lower_pages, operation_attributes.rep_dim_pages};
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
        TensorAccessorArgs(*dst_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/reader_repeat_outer_tile_direct.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.config = ReaderConfigDescriptor{};

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            reader_desc.emplace_runtime_args(
                core, {src_buffer, dst_buffer, start, n, operation_attributes.num_repeats});
            start += n;
        }

        desc.kernels.push_back(std::move(reader_desc));
        return desc;
    }

    const CoreSplit split = split_work(
        input,
        operation_attributes.total_out_pages,
        is_row_major ? std::nullopt : tuned_core_cap(input, output, operation_attributes));

    if (!is_row_major) {
        // TILE-interleaved path: shared pluggable sequencer reader (seq_id=1 == SEQ_REPEAT)
        // + interleaved writer.
        const uint32_t page_size = static_cast<uint32_t>(dst_buffer->aligned_page_size());

        desc.cbs.push_back(CBDescriptor{
            .total_size = kRepeatCbDepth * page_size,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = 0,
                .data_format = cb_data_format,
                .page_size = page_size,
            }}},
        });

        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source = kReaderSequenced;
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqRepeat},
            {"cb_id", 0},
            {"batch", kRepeatBatch},
            // reader_tile_interleaved_unified.cpp unconditionally reads this named
            // arg in kernel_main() (not gated by SEQ_ID), falling back to the
            // TensorAccessorArgs page size when 0. It must be supplied even though
            // the repeat sequencer never consults it.
            {"src_page_pitch", 0},
        };
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {0, page_size};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_ct_args.push_back(kRepeatBatch);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source = kWriterInterleaved;
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            reader_desc.emplace_runtime_args(
                core,
                {src_buffer,
                 n,
                 start,
                 operation_attributes.num_repeats,
                 operation_attributes.lower_pages,
                 operation_attributes.rep_dim_pages});
            writer_desc.emplace_runtime_args(core, {dst_buffer, n, start});
            start += n;
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        return desc;
    }

    // ROW_MAJOR paths. Each side moves its own buffer type's aligned page, and DRAM and L1 align
    // differently, so a slot holds whichever is larger. On the higher-dim path input and output share
    // one stick; on the last-dim path the output page is `num_repeats` input sticks wide.
    const uint32_t in_aligned = static_cast<uint32_t>(src_buffer->aligned_page_size());
    const uint32_t out_aligned = static_cast<uint32_t>(dst_buffer->aligned_page_size());
    const uint32_t slot_size = rm_slot_bytes(in_aligned, out_aligned);
    // The key sized this slot from the output spec, before it could see the output buffer.
    TT_ASSERT(out_aligned == spec_aligned_page_bytes(input, output.tensor_spec()));
    // Sized to the L1 left free right now, which is what the program-cache key was hashed with.
    const auto cb_plan = live_rm_cb_plan(input, slot_size);
    TT_FATAL(
        cb_plan.has_value(),
        "RepeatCodegen: a {}-byte row-major stick leaves no room for a two-slot CB in free L1",
        slot_size);

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_plan->depth * slot_size,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = cb_data_format,
            .page_size = slot_size,
        }}},
    });

    // The last-dim leg widens each stick, so it has its own reader. A higher-dim leg copies whole
    // sticks and is the TILE leg's page-map read over one-stick pages: the shared reader derives the
    // source pitch from the accessor and the L1 stride from the CB, and clamps each read to the slot.
    KernelDescriptor reader_desc;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.config = ReaderConfigDescriptor{};
    if (is_last_dim_rm) {
        std::vector<uint32_t> reader_ct_args = {operation_attributes.stick_size, in_aligned, slot_size};
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
        reader_ct_args.push_back(0);  // cb_id
        reader_ct_args.push_back(operation_attributes.num_repeats);
        reader_ct_args.push_back(cb_plan->batch);
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/reader_repeat_last_dim_rm.cpp";
        reader_desc.compile_time_args = std::move(reader_ct_args);
    } else {
        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
        reader_desc.kernel_source = kReaderSequenced;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqRepeat},
            {"cb_id", 0},
            {"batch", cb_plan->batch},
            {"src_page_pitch", 0},
        };
    }

    // The writer takes its L1 stride from the CB and clamps each transfer to the destination page, so
    // it needs only the requested transfer size: the output's aligned page.
    std::vector<uint32_t> writer_ct_args = {0, out_aligned};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(cb_plan->batch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterInterleaved;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);
        if (is_last_dim_rm) {
            reader_desc.emplace_runtime_args(core, {src_buffer, n, start});
        } else {
            reader_desc.emplace_runtime_args(
                core,
                {src_buffer,
                 n,
                 start,
                 operation_attributes.num_repeats,
                 operation_attributes.lower_pages,
                 operation_attributes.rep_dim_pages});
        }
        writer_desc.emplace_runtime_args(core, {dst_buffer, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
