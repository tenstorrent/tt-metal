// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_program_factory.hpp"

#include <variant>
#include <vector>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

// Matches READ_BATCH / WRITE_BATCH / _CB_DEPTH in ops/repeat_interleave/spec.py and
// ops/repeat/spec.py (shared TILE builder).
constexpr uint32_t kReadBatch = 4;
constexpr uint32_t kWriteBatch = 4;
constexpr uint32_t kCbDepth = 8;              // max(2 * max(READ_BATCH, WRITE_BATCH), 8)
constexpr uint32_t kSeqRepeatInterleave = 9;  // sequencers.h SEQ_REPEAT_INTERLEAVE
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/codegen/kernels";

std::string kernel_path(const char* name) { return std::string(kKernelDir) + "/" + name; }

// Result of the standard split_work_to_cores() 2-group work split.
struct CoreSplit {
    uint32_t num_cores;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1;
    uint32_t units_per_core_group_2;
};

CoreSplit split_work(IDevice* device, uint32_t total_work) {
    const CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, cg1, cg2, upc1, upc2] = split_work_to_cores(grid, total_work);
    return CoreSplit{num_cores, all_cores, cg1, cg2, upc1, upc2};
}

// Per-core RT emission shared by all three kernel variants: reader RT is always
// [src_addr, n, start] plus an (optionally empty) fixed tail; writer RT is always
// [dst_addr, n, start] (matches assemble_rm_positional / build_repeat_tile's
// rt_closure — only the TILE path has a non-empty reader tail).
void emit_per_core_rt(
    const CoreSplit& split,
    KernelDescriptor& reader_desc,
    KernelDescriptor& writer_desc,
    Buffer* src_buffer,
    Buffer* dst_buffer,
    const std::vector<uint32_t>& reader_tail = {}) {
    const auto cores = corerange_to_cores(split.all_cores, split.num_cores);
    const uint32_t g1_num_cores = split.core_group_1.num_cores();
    uint32_t start = 0;
    for (uint32_t i = 0; i < cores.size(); i++) {
        const uint32_t n = (i < g1_num_cores) ? split.units_per_core_group_1 : split.units_per_core_group_2;
        std::vector<std::variant<uint32_t, Buffer*>> reader_args{src_buffer, n, start};
        for (uint32_t v : reader_tail) {
            reader_args.emplace_back(v);
        }
        reader_desc.emplace_runtime_args(cores[i], reader_args);
        writer_desc.emplace_runtime_args(cores[i], std::vector<std::variant<uint32_t, Buffer*>>{dst_buffer, n, start});
        start += n;
    }
}

}  // namespace

// Transliterated from ops/repeat_interleave/spec.py (build_repeat_tile via ops/repeat/spec.py
// for the TILE path, build_repeat_interleave_rm_factory / build_repeat_interleave_lastdim_rm for
// ROW_MAJOR). Layout selects TILE vs ROW_MAJOR; within ROW_MAJOR, operation_attributes.rep_dim == 3
// (the 4D-padded last-dim marker set by the host free function) selects the lastdim kernel,
// otherwise the higher-dim (whole-stick) kernel.
tt::tt_metal::ProgramDescriptor RepeatInterleaveCodegenProgramFactory::create_descriptor(
    const RepeatInterleaveCodegenParams& operation_attributes,
    const RepeatInterleaveCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;
    IDevice* device = input.device();
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "repeat_interleave: output buffer should be allocated on device!");

    ProgramDescriptor desc;

    if (input.layout() == Layout::TILE) {
        const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
        const uint32_t raw_tile_size = tt::tile_size(cb_data_format);
        const uint32_t aligned_tile_size = tt::align(raw_tile_size, tt::tt_metal::hal::get_dram_alignment());

        const CoreSplit split = split_work(device, operation_attributes.total_out_pages);

        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbDepth * aligned_tile_size,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = 0,
                .data_format = cb_data_format,
                .page_size = aligned_tile_size,
            }}},
        });

        std::vector<uint32_t> reader_ct;
        TensorAccessorArgs(*src_buffer).append_to(reader_ct);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source = kernel_path("reader_tile_interleaved_unified.cpp");
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct);
        reader_desc.named_compile_time_args = {{"seq_id", kSeqRepeatInterleave}, {"cb_id", 0}, {"batch", kReadBatch}};
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct = {0, aligned_tile_size};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct);
        writer_ct.push_back(kWriteBatch);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source = kernel_path("writer_interleaved.cpp");
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct);
        writer_desc.config = WriterConfigDescriptor{};

        emit_per_core_rt(
            split,
            reader_desc,
            writer_desc,
            src_buffer,
            dst_buffer,
            {operation_attributes.num_repeats, operation_attributes.lower_pages, operation_attributes.rep_dim_pages});

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        return desc;
    }

    // ROW_MAJOR. rep_dim == 3 marks the (4D-padded) last dim -> within-stick (W) path.
    if (operation_attributes.rep_dim == 3) {
        const uint32_t stick_in = operation_attributes.stick_size;
        const uint32_t stick_out = operation_attributes.stick_size_out;
        const uint32_t elem_size = input.element_size();
        const uint32_t w_in = stick_in / elem_size;
        const uint32_t out_align = dst_buffer->alignment();
        const uint32_t aligned_out = tt::align(stick_out, out_align);

        const CoreSplit split = split_work(device, operation_attributes.total_out_pages);

        desc.cbs.push_back(CBDescriptor{
            .total_size = kCbDepth * stick_out,
            .core_ranges = split.all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = 0,
                .data_format = datatype_to_dataformat_converter(input.dtype()),
                .page_size = stick_out,
            }}},
        });

        std::vector<uint32_t> reader_ct = {stick_out, stick_in, w_in, operation_attributes.num_repeats, elem_size};
        TensorAccessorArgs(*src_buffer).append_to(reader_ct);
        reader_ct.push_back(0);  // cb_id
        reader_ct.push_back(kReadBatch);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source = kernel_path("reader_repeat_interleave_lastdim_rm.cpp");
        reader_desc.core_ranges = split.all_cores;
        reader_desc.compile_time_args = std::move(reader_ct);
        reader_desc.config = ReaderConfigDescriptor{};

        std::vector<uint32_t> writer_ct = {0, stick_out, aligned_out};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct);
        writer_ct.push_back(kWriteBatch);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source = kernel_path("writer_repeat_interleave_rm.cpp");
        writer_desc.core_ranges = split.all_cores;
        writer_desc.compile_time_args = std::move(writer_ct);
        writer_desc.config = WriterConfigDescriptor{};

        emit_per_core_rt(split, reader_desc, writer_desc, src_buffer, dst_buffer);

        desc.kernels.push_back(std::move(reader_desc));
        desc.kernels.push_back(std::move(writer_desc));
        return desc;
    }

    // ROW_MAJOR higher-dim (whole-stick, outer/H) path.
    const uint32_t stick_size = operation_attributes.stick_size;
    const uint32_t aligned_page_size = tt::align(stick_size, src_buffer->alignment());

    const CoreSplit split = split_work(device, operation_attributes.total_out_pages);

    desc.cbs.push_back(CBDescriptor{
        .total_size = kCbDepth * stick_size,
        .core_ranges = split.all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = datatype_to_dataformat_converter(input.dtype()),
            .page_size = stick_size,
        }}},
    });

    std::vector<uint32_t> reader_ct = {stick_size, aligned_page_size};
    TensorAccessorArgs(*src_buffer).append_to(reader_ct);
    reader_ct.push_back(0);  // cb_id
    reader_ct.push_back(operation_attributes.num_repeats);
    reader_ct.push_back(operation_attributes.lower_pages);
    reader_ct.push_back(operation_attributes.rep_dim_pages);
    reader_ct.push_back(kReadBatch);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kernel_path("reader_repeat_interleave_rm.cpp");
    reader_desc.core_ranges = split.all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {0, stick_size, aligned_page_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct);
    writer_ct.push_back(kWriteBatch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kernel_path("writer_repeat_interleave_rm.cpp");
    writer_desc.core_ranges = split.all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    emit_per_core_rt(split, reader_desc, writer_desc, src_buffer, dst_buffer);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
