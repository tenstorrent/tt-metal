// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ckv_bfp4_rows_program_factory.hpp"

#include <atomic>
#include <cstddef>
#include <string>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/ckv_bfp4_rows_args.hpp"

namespace ttnn::operations::ckv_bfp4_rows {

using namespace tt::tt_metal;
namespace rows_args = ::ckv_bfp4_rows;

namespace {

constexpr const char* kRowsKernelDir = "ttnn/cpp/ttnn/operations/ckv_bfp4_rows/device/kernels/";

std::string rows_kernel_path(const char* name) { return std::string(kRowsKernelDir) + name; }

CBDescriptor rows_cb(uint32_t id, tt::DataFormat format, uint32_t page, uint32_t pages, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = page * pages,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(id), .data_format = format, .page_size = page}},
    };
}

std::atomic<std::size_t>& programs_built_counter() {
    static std::atomic<std::size_t> counter{0};
    return counter;
}

}  // namespace

std::size_t programs_built() { return programs_built_counter().load(); }

ProgramDescriptor CkvBfp4RowsProgramFactory::create_descriptor(
    const CkvBfp4RowsParams& attrs, const CkvBfp4RowsInputs& tensor_args, Tensor& output) {
    programs_built_counter().fetch_add(1);
    TT_FATAL(
        rows_args::cb::TILED < hal::get_num_dataflow_buffers(),
        "ckv_bfp4_rows: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        rows_args::cb::TILED,
        hal::get_num_dataflow_buffers());

    const uint32_t gx = attrs.grid_x;
    const uint32_t gy = attrs.grid_y;
    const CoreRangeSet cores(CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1)));
    const uint32_t tiles_per_band = rows_args::WIDTH / tt::constants::TILE_WIDTH;
    const uint32_t t_bf16 = tt::tt_metal::tile_size(DataType::BFLOAT16);

    ProgramDescriptor desc;
    desc.cbs.push_back(rows_cb(rows_args::cb::IN, tt::DataFormat::Float16_b, t_bf16, 2 * tiles_per_band, cores));
    desc.cbs.push_back(rows_cb(rows_args::cb::TILED, tt::DataFormat::Float16_b, t_bf16, tiles_per_band, cores));
    desc.cbs.push_back(rows_cb(
        rows_args::cb::ROWS, tt::DataFormat::Bfp4_b, tt::constants::TILE_HEIGHT * rows_args::ROW_BYTES, 2, cores));

    const KernelDescriptor::Defines share = {
        {"CKV_ROWS_GX", std::to_string(gx)}, {"CKV_ROWS_NC", std::to_string(gx * gy)}};

    KernelDescriptor reader;
    reader.kernel_source = rows_kernel_path("ckv_bfp4_rows_reader.cpp");
    reader.core_ranges = cores;
    reader.compile_time_args.assign(rows_args::reader_ct::COUNT, 0u);
    reader.compile_time_args[rows_args::reader_ct::ROW_BYTES] = rows_args::SRC_ROW_BYTES;
    TensorAccessorArgs(*tensor_args.src.buffer()).append_to(reader.compile_time_args);
    reader.defines = share;
    reader.emplace_common_runtime_args({tensor_args.src.buffer(), attrs.src_row0, attrs.num_bands});
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = rows_kernel_path("ckv_bfp4_writer.cpp");
    writer.core_ranges = cores;
    writer.compile_time_args.assign(rows_args::writer_ct::COUNT, 0u);
    writer.compile_time_args[rows_args::writer_ct::CB_ROWS] = rows_args::cb::ROWS;
    writer.compile_time_args[rows_args::writer_ct::CB_ROUNDED] = rows_args::cb::ROUNDED;
    writer.compile_time_args[rows_args::writer_ct::TILES_PER_BAND] = tiles_per_band;
    writer.compile_time_args[rows_args::writer_ct::ROW_BYTES] = rows_args::ROW_BYTES;
    TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.defines = share;
    writer.defines.emplace_back("CKV_ROWS_IN", "1");
    writer.emplace_common_runtime_args({output.buffer(), attrs.num_bands, attrs.row0});
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = rows_kernel_path("ckv_bfp4_compute.cpp");
    compute.core_ranges = cores;
    compute.compile_time_args.assign(rows_args::compute_ct::COUNT, 0u);
    compute.compile_time_args[rows_args::compute_ct::CB_IN] = rows_args::cb::IN;
    compute.compile_time_args[rows_args::compute_ct::CB_ROWS] = rows_args::cb::ROWS;
    compute.compile_time_args[rows_args::compute_ct::CB_ROUNDED] = rows_args::cb::ROUNDED;
    compute.compile_time_args[rows_args::compute_ct::CB_TILED] = rows_args::cb::TILED;
    compute.defines = share;
    compute.defines.emplace_back("CKV_ROWS_IN", "1");
    compute.defines.emplace_back("CKV_PACK_PRECISE", "1");
    compute.common_runtime_args = {attrs.num_bands};
    compute.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = true,
        .bfp8_pack_precise = true,
    };

    TT_FATAL(
        reader.common_runtime_args.size() == rows_args::reader_common::COUNT &&
            writer.common_runtime_args.size() == rows_args::writer_common::COUNT &&
            compute.common_runtime_args.size() == rows_args::compute_common::COUNT,
        "ckv_bfp4_rows: common runtime argument layout is out of sync with ckv_bfp4_rows_args.hpp");

    desc.kernels.resize(rows_args::kernel_index::COUNT);
    desc.kernels[rows_args::kernel_index::READER] = std::move(reader);
    desc.kernels[rows_args::kernel_index::WRITER] = std::move(writer);
    desc.kernels[rows_args::kernel_index::COMPUTE] = std::move(compute);
    return desc;
}

void CkvBfp4RowsProgramFactory::override_runtime_arguments(
    Program& program,
    const CkvBfp4RowsParams& attrs,
    const CkvBfp4RowsInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key pins grid_x / grid_y and the tensor geometry; the addresses and the per-call values
    // (src_row0, num_bands, row0, left out of the key) are all common args, rewritten here.
    auto& reader = GetCommonRuntimeArgs(program, rows_args::kernel_index::READER);
    reader[rows_args::reader_common::SRC_ADDR] = tensor_args.src.buffer()->address();
    reader[rows_args::reader_common::SRC_ROW0] = attrs.src_row0;
    reader[rows_args::reader_common::NUM_BANDS] = attrs.num_bands;
    auto& writer = GetCommonRuntimeArgs(program, rows_args::kernel_index::WRITER);
    writer[rows_args::writer_common::ROWS_ADDR] = output.buffer()->address();
    writer[rows_args::writer_common::NUM_BANDS] = attrs.num_bands;
    writer[rows_args::writer_common::ROW0] = attrs.row0;
    GetCommonRuntimeArgs(program, rows_args::kernel_index::COMPUTE)[rows_args::compute_common::NUM_BANDS] =
        attrs.num_bands;
}

}  // namespace ttnn::operations::ckv_bfp4_rows
