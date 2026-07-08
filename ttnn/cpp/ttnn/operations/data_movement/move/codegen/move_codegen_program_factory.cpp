// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_codegen_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

// Transliterated from tt-dm-codegen's ops/identity/spec.py (build_identity_tile / build_identity_rm)
// via the shared ProgramFactory.assemble() host pipeline (common/codegen_common/factory/*.py). Only
// the non-sharded interleaved TILE/ROW_MAJOR identity-copy paths are in scope (manifests/move.yaml).

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

constexpr uint32_t kCbId = 0;
constexpr uint32_t kSeqIdentity = 0;   // sequencers.h SEQ_IDENTITY
constexpr uint32_t kModeSequenced = 4;  // reader_stick_interleaved_unified.cpp MODE_SEQUENCED

constexpr const char* kReaderTile =
    "ttnn/cpp/ttnn/operations/data_movement/move/codegen/kernels/reader_tile_interleaved_unified.cpp";
constexpr const char* kWriterInterleaved =
    "ttnn/cpp/ttnn/operations/data_movement/move/codegen/kernels/writer_interleaved.cpp";
constexpr const char* kReaderStick =
    "ttnn/cpp/ttnn/operations/data_movement/move/codegen/kernels/reader_stick_interleaved_unified.cpp";
constexpr const char* kWriterRm =
    "ttnn/cpp/ttnn/operations/data_movement/move/codegen/kernels/writer_rm_interleaved.cpp";

// reader_stick_interleaved_unified.cpp's kernel_main is not a template: every `if constexpr`
// branch (including MODE_TILEROW_PAD) still has its get_named_compile_time_arg_val() calls
// compiled, so those names must resolve even when MODE_SEQUENCED never reads them. Mirrors
// tt-dm-codegen's builder_utils._merge_tilerow_pad_defaults, which every caller of this reader
// picks up automatically; our host code has to add them explicitly.
KernelDescriptor::NamedCompileTimeArgs stick_reader_named_ct(
    uint32_t stick_bytes, uint32_t aligned_page_size, uint32_t read_batch) {
    return {
        {"mode", kModeSequenced},
        {"cb_id", kCbId},
        {"stick_bytes", stick_bytes},
        {"aligned_page_size", aligned_page_size},
        {"seq_id", kSeqIdentity},
        {"batch", read_batch},
        {"nabatch", 1u},
        // MODE_TILEROW_PAD defaults (unused by MODE_SEQUENCED, but must exist):
        {"elem_size", 2u},
        {"tile_height", 32u},
        {"tile_row_shift_bits", 0u},
        {"num_pages_in_row", 1u},
        {"unpadded_X_bytes", 0u},
        {"valid_last_page_bytes", 0u},
        {"page_size", 32u},
    };
}

// Assigns per-core reader/writer runtime args over `all_cores` (== core_group_1 ∪ core_group_2, the
// cores split_work_to_cores actually assigned work to). `writer_rt` builds the writer's full RT-arg
// list for a working core, since the RM writer's ABI ([dst, n, 1, 1, start]) differs from the tile
// writer's ([dst, n, start]) — mirrors the Python factory's `Writer.rt` full-closure override.
template <typename WriterRtFn>
void assign_runtime_args(
    KernelDescriptor& reader_desc,
    KernelDescriptor& writer_desc,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t work_per_core_group_1,
    uint32_t work_per_core_group_2,
    Buffer* in_buffer,
    Buffer* out_buffer,
    WriterRtFn&& writer_rt) {
    const auto work_groups = {
        std::make_pair(core_group_1, work_per_core_group_1), std::make_pair(core_group_2, work_per_core_group_2)};
    uint32_t start = 0;
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                if (work_per_core > 0) {
                    reader_desc.emplace_runtime_args(core, {in_buffer, work_per_core, start});
                    writer_rt(writer_desc, core, out_buffer, work_per_core, start);
                } else {
                    // Idle core: literal zeros so no BufferBinding is registered for a core the
                    // kernel never touches.
                    reader_desc.emplace_runtime_args(core, {0u, 0u, 0u});
                    writer_rt(writer_desc, core, nullptr, 0u, 0u);
                }
                start += work_per_core;
            }
        }
    }
}

// build_identity_tile (ops/identity/spec.py): reader_tile_interleaved_unified.cpp ->
// writer_interleaved.cpp, one CB shared by both kernels.
ProgramDescriptor build_tile_descriptor(
    const MoveCodegenOperationAttributes& attrs, const Tensor& input, Tensor& output) {
    auto* device = input.device();
    Buffer* in_buffer = input.buffer();
    Buffer* out_buffer = output.buffer();

    const auto compute_grid = device->compute_with_storage_grid_size();
    const CoreRangeSet full_grid = num_cores_to_corerangeset(compute_grid.x * compute_grid.y, compute_grid, true);
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
            split_work_to_cores(full_grid, attrs.total_pages, true);
    (void)num_cores;

    ProgramDescriptor desc;

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = attrs.cb_depth * attrs.page_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbId,
            .data_format = data_format,
            .page_size = attrs.page_bytes,
        }}},
    });

    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(*in_buffer).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderTile;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.named_compile_time_args = {
        {"seq_id", kSeqIdentity},
        {"cb_id", kCbId},
        {"batch", attrs.read_batch},
    };
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {kCbId, attrs.page_bytes};
    TensorAccessorArgs(*out_buffer).append_to(writer_ct);
    writer_ct.push_back(attrs.write_batch);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterInterleaved;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    assign_runtime_args(
        reader_desc,
        writer_desc,
        core_group_1,
        core_group_2,
        work_per_core_group_1,
        work_per_core_group_2,
        in_buffer,
        out_buffer,
        [](KernelDescriptor& writer, const CoreCoord& core, Buffer* out, uint32_t n, uint32_t start) {
            if (out != nullptr) {
                writer.emplace_runtime_args(core, {out, n, start});
            } else {
                writer.emplace_runtime_args(core, {0u, 0u, 0u});
            }
        });

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

// build_identity_rm (ops/identity/spec.py): reader_stick_interleaved_unified.cpp (MODE_SEQUENCED) ->
// writer_rm_interleaved.cpp. The CB page is the raw (unaligned) stick size; only the DRAM/L1
// TensorAccessor page pitch (attrs.page_bytes) is alignment-padded.
ProgramDescriptor build_rm_descriptor(const MoveCodegenOperationAttributes& attrs, const Tensor& input, Tensor& output) {
    auto* device = input.device();
    Buffer* in_buffer = input.buffer();
    Buffer* out_buffer = output.buffer();

    const auto compute_grid = device->compute_with_storage_grid_size();
    const CoreRangeSet full_grid = num_cores_to_corerangeset(compute_grid.x * compute_grid.y, compute_grid, true);
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
            split_work_to_cores(full_grid, attrs.total_pages, true);
    (void)num_cores;

    const uint32_t width = input.padded_shape()[-1];
    const uint32_t stick_bytes = width * input.element_size();

    ProgramDescriptor desc;

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = attrs.cb_depth * stick_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kCbId,
            .data_format = data_format,
            .page_size = stick_bytes,
        }}},
    });

    std::vector<uint32_t> reader_ct;
    TensorAccessorArgs(*in_buffer).append_to(reader_ct);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderStick;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.named_compile_time_args = stick_reader_named_ct(stick_bytes, attrs.page_bytes, attrs.read_batch);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {kCbId, stick_bytes, attrs.page_bytes};
    TensorAccessorArgs(*out_buffer).append_to(writer_ct);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterRm;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    assign_runtime_args(
        reader_desc,
        writer_desc,
        core_group_1,
        core_group_2,
        work_per_core_group_1,
        work_per_core_group_2,
        in_buffer,
        out_buffer,
        [](KernelDescriptor& writer, const CoreCoord& core, Buffer* out, uint32_t n, uint32_t start) {
            // writer_rm_interleaved.cpp RT ABI: [dst, num_reads, sticks_per_read, sticks_per_cb_push,
            // start] — build_identity_rm calls it with sticks_per_read=1, sticks_per_cb_push=1.
            if (out != nullptr) {
                writer.emplace_runtime_args(core, {out, n, 1u, 1u, start});
            } else {
                writer.emplace_runtime_args(core, {0u, 0u, 0u, 0u, 0u});
            }
        });

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace

tt::tt_metal::ProgramDescriptor MoveCodegenProgramFactory::create_descriptor(
    const MoveCodegenOperationAttributes& operation_attributes,
    const MoveCodegenTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input_tensor;
    if (input.layout() == Layout::TILE) {
        return build_tile_descriptor(operation_attributes, input, tensor_return_value);
    }
    TT_FATAL(
        input.layout() == Layout::ROW_MAJOR,
        "MoveCodegenProgramFactory: unsupported layout (supported_by_codegen() should have rejected this)");
    return build_rm_descriptor(operation_attributes, input, tensor_return_value);
}

}  // namespace ttnn::prim
