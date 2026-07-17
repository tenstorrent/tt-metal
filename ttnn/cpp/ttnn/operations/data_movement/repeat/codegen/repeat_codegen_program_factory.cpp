// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"

#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Matches ops/repeat/spec.py's READ_BATCH / WRITE_BATCH. kCbDepth lives in
// the header as kRepeatCbDepth since repeat_codegen_supported.cpp's
// L1-capacity gate needs the same value.
constexpr uint32_t kReadBatch = 4;
constexpr uint32_t kWriteBatch = 4;

// SEQ_REPEAT, see codegen/kernels/sequencers.h.
constexpr uint32_t kSeqRepeat = 1;

struct CoreSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores_in_order;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t work_per_core_1 = 0;
    uint32_t work_per_core_2 = 0;
};

CoreSplit split_work(const Tensor& input, uint32_t total_work) {
    IDevice* device = input.device();
    auto grid_size = device->compute_with_storage_grid_size();
    // row_wise=false (column-major core enumeration) to match the generator's
    // split_cores()/emit_per_core_rt(), which always calls ttnn.split_work_to_cores
    // and corerange_to_cores at their row_wise=False default. This is a no-op for
    // work counts that fill the whole grid, but for the small per-core-page RM
    // cases (a handful of sticks spread over a mostly-idle grid) the enumeration
    // order picks a different physical core per page range, which changes NOC
    // hop distance to the DRAM channel enough to show up as a device-time delta.
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, total_work, /*row_wise=*/false);
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

    const CoreSplit split = split_work(input, operation_attributes.total_out_pages);
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    ProgramDescriptor desc;

    if (!is_row_major) {
        // TILE-interleaved path: shared pluggable sequencer reader (seq_id=1 == SEQ_REPEAT)
        // + interleaved writer. Mirrors ops/repeat/spec.py::build_repeat_tile / _plan.
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
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/reader_tile_interleaved_unified.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.named_compile_time_args = {
            {"seq_id", kSeqRepeat},
            {"cb_id", 0},
            {"batch", kReadBatch},
        };
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {0, page_size};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_ct_args.push_back(kWriteBatch);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/writer_interleaved.cpp";
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

    if (is_last_dim_rm) {
        // ROW_MAJOR last-dim (within-stick) path. Mirrors
        // ops/repeat/spec.py::build_repeat_last_dim_rm_factory.
        const uint32_t in_stick_size = operation_attributes.stick_size;
        const uint32_t in_aligned = static_cast<uint32_t>(src_buffer->aligned_page_size());
        const uint32_t out_aligned = static_cast<uint32_t>(dst_buffer->aligned_page_size());

        desc.cbs.push_back(CBDescriptor{
            .total_size = kRepeatCbDepth * out_aligned,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = 0,
                .data_format = cb_data_format,
                .page_size = out_aligned,
            }}},
        });

        std::vector<uint32_t> reader_ct_args = {in_stick_size, in_aligned, out_aligned};
        TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
        reader_ct_args.push_back(0);  // cb_id
        reader_ct_args.push_back(operation_attributes.num_repeats);
        reader_ct_args.push_back(kReadBatch);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/reader_repeat_last_dim_rm.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct_args);
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct_args = {0, out_aligned, out_aligned};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_ct_args.push_back(kWriteBatch);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/writer_repeat_rm.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        uint32_t start = 0;
        for (const auto& core : split.cores_in_order) {
            const uint32_t n = work_for_core(split, core);
            reader_desc.emplace_runtime_args(core, {src_buffer, n, start});
            writer_desc.emplace_runtime_args(core, {dst_buffer, n, start});
            start += n;
        }

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        return desc;
    }

    // ROW_MAJOR higher-dim path. Mirrors ops/repeat/spec.py::build_repeat_rm_factory.
    // Input and output share the same last-dim width on this branch (only a non-last
    // dim is repeated), so one aligned page pitch serves reader, writer, and CB.
    const uint32_t aligned_page_size = static_cast<uint32_t>(src_buffer->aligned_page_size());

    desc.cbs.push_back(CBDescriptor{
        .total_size = kRepeatCbDepth * aligned_page_size,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = cb_data_format,
            .page_size = aligned_page_size,
        }}},
    });

    std::vector<uint32_t> reader_ct_args = {aligned_page_size, aligned_page_size};
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
    reader_ct_args.push_back(0);  // cb_id
    reader_ct_args.push_back(operation_attributes.num_repeats);
    reader_ct_args.push_back(operation_attributes.lower_pages);
    reader_ct_args.push_back(operation_attributes.rep_dim_pages);
    reader_ct_args.push_back(kReadBatch);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/reader_repeat_higherdim_rm.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct_args = {0, aligned_page_size, aligned_page_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    writer_ct_args.push_back(kWriteBatch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/writer_repeat_rm.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t start = 0;
    for (const auto& core : split.cores_in_order) {
        const uint32_t n = work_for_core(split, core);
        reader_desc.emplace_runtime_args(core, {src_buffer, n, start});
        writer_desc.emplace_runtime_args(core, {dst_buffer, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
