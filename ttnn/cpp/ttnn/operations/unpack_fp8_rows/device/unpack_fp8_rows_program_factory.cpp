// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unpack_fp8_rows_program_factory.hpp"

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "kernels/unpack_fp8_rows_args.hpp"

namespace ttnn::operations::unpack_fp8_rows {

using namespace tt::tt_metal;
namespace args = ::unpack_fp8_rows;

namespace {

constexpr const char* kUnpackKernelDir = "ttnn/cpp/ttnn/operations/unpack_fp8_rows/device/kernels/";

std::string unpack_kernel_path(const char* name) { return std::string(kUnpackKernelDir) + name; }

CBDescriptor unpack_cb(uint32_t index, DataType dtype, uint32_t page_size, uint32_t pages, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = page_size * pages,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = cb_dataformat_for(dtype),
            .page_size = page_size,
        }},
    };
}

}  // namespace

ProgramDescriptor UnpackFp8RowsProgramFactory::create_descriptor(
    const UnpackFp8RowsParams& attrs, const UnpackFp8RowsInputs& tensors, Tensor& output) {
    const uint32_t gx = attrs.grid_x;
    const uint32_t ncores = attrs.ncores;
    TT_FATAL(1 <= ncores && ncores <= gx * attrs.grid_y, "({}, {}, {})", ncores, gx, attrs.grid_y);
    TT_FATAL(
        args::cb::RSTAGE < hal::get_num_dataflow_buffers(),
        "unpack_fp8_rows: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        args::cb::RSTAGE,
        hal::get_num_dataflow_buffers());

    std::vector<CoreRange> ranges;
    std::vector<CoreCoord> cores;
    ranges.reserve(ncores);
    cores.reserve(ncores);
    for (uint32_t i = 0; i < ncores; ++i) {
        const CoreCoord core(i % gx, i / gx);
        cores.push_back(core);
        ranges.emplace_back(core, core);
    }
    const CoreRangeSet crs = CoreRangeSet(ranges).merge_ranges();

    const uint32_t stage = (tt::constants::TILE_HEIGHT * args::ROW_BYTES) + args::STAGE_ALIGN;
    ProgramDescriptor desc;
    desc.cbs.push_back(unpack_cb(args::cb::E, DataType::FP8_E4M3, tile_size(DataType::FP8_E4M3), args::E_PAGES, crs));
    desc.cbs.push_back(unpack_cb(args::cb::S, DataType::FLOAT32, tile_size(DataType::FLOAT32), args::S_PAGES, crs));
    desc.cbs.push_back(
        unpack_cb(args::cb::OUT, DataType::BFLOAT8_B, tile_size(DataType::BFLOAT8_B), args::OUT_PAGES, crs));
    desc.cbs.push_back(unpack_cb(args::cb::RSTAGE, DataType::UINT8, stage, 1, crs));

    KernelDescriptor::Defines defines;
    if (attrs.common_values) {
        defines.emplace_back("UNPACK_COMMON_VALUES", "1");
    }

    KernelDescriptor reader;
    reader.kernel_source = unpack_kernel_path("unpack_fp8_rows_reader.cpp");
    reader.core_ranges = crs;
    reader.compile_time_args.assign(args::reader_ct::COUNT, 0u);
    reader.compile_time_args[args::reader_ct::CB_E] = args::cb::E;
    reader.compile_time_args[args::reader_ct::CB_S] = args::cb::S;
    reader.compile_time_args[args::reader_ct::CB_STAGE] = args::cb::RSTAGE;
    reader.compile_time_args[args::reader_ct::GX] = gx;
    reader.compile_time_args[args::reader_ct::NCORES] = ncores;
    TensorAccessorArgs(*tensors.rows.buffer()).append_to(reader.compile_time_args);
    reader.defines = defines;
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = unpack_kernel_path("unpack_fp8_rows_writer.cpp");
    writer.core_ranges = crs;
    writer.compile_time_args.assign(args::writer_ct::COUNT, 0u);
    writer.compile_time_args[args::writer_ct::CB_OUT] = args::cb::OUT;
    writer.compile_time_args[args::writer_ct::GX] = gx;
    writer.compile_time_args[args::writer_ct::NCORES] = ncores;
    TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.defines = defines;
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = unpack_kernel_path("unpack_fp8_rows_compute.cpp");
    compute.core_ranges = crs;
    compute.compile_time_args.assign(args::compute_ct::COUNT, 0u);
    compute.compile_time_args[args::compute_ct::CB_E] = args::cb::E;
    compute.compile_time_args[args::compute_ct::CB_S] = args::cb::S;
    compute.compile_time_args[args::compute_ct::CB_OUT] = args::cb::OUT;
    compute.compile_time_args[args::compute_ct::GX] = gx;
    compute.compile_time_args[args::compute_ct::NCORES] = ncores;
    compute.defines = defines;
    ComputeConfigDescriptor cfg{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .bfp8_pack_precise = true,
        .math_approx_mode = false,
    };
    cfg.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    cfg.unpack_to_dest_mode[args::cb::S] = UnpackToDestMode::UnpackToDestFp32;  // the scales exact in DEST
    compute.config = cfg;

    if (attrs.common_values) {
        // reader_common: ROWS_ADDR, SRC_PAGE0, NBANDS; writer_common: OUT_ADDR, OUT_PAGE0, NBANDS.
        reader.emplace_common_runtime_args({tensors.rows.buffer(), attrs.src_page0, attrs.nbands});
        writer.emplace_common_runtime_args({output.buffer(), attrs.out_page0, attrs.nbands});
        TT_FATAL(
            reader.common_runtime_args.size() == args::reader_common::COUNT &&
                writer.common_runtime_args.size() == args::writer_common::COUNT,
            "unpack_fp8_rows: common runtime argument layout is out of sync with unpack_fp8_rows_args.hpp");
        compute.common_runtime_args.assign(args::compute_common::COUNT, 0u);
        compute.common_runtime_args[args::compute_common::NBANDS] = attrs.nbands;
    } else {
        reader.emplace_common_runtime_args({tensors.rows.buffer()});  // ROWS_ADDR
        writer.emplace_common_runtime_args({output.buffer()});        // OUT_ADDR
        TT_FATAL(
            reader.common_runtime_args.size() == args::reader_common::COUNT_ADDR_ONLY &&
                writer.common_runtime_args.size() == args::writer_common::COUNT_ADDR_ONLY,
            "unpack_fp8_rows: common runtime argument layout is out of sync with unpack_fp8_rows_args.hpp");
        reader.runtime_args.reserve(ncores);
        writer.runtime_args.reserve(ncores);
        compute.runtime_args.reserve(ncores);
        for (const CoreCoord& core : cores) {
            KernelDescriptor::CoreRuntimeArgs r(args::reader_rt::COUNT, 0u);
            r[args::reader_rt::SRC_PAGE0] = attrs.src_page0;
            r[args::reader_rt::NBANDS] = attrs.nbands;
            reader.runtime_args.emplace_back(core, std::move(r));
            KernelDescriptor::CoreRuntimeArgs w(args::writer_rt::COUNT, 0u);
            w[args::writer_rt::OUT_PAGE0] = attrs.out_page0;
            w[args::writer_rt::NBANDS] = attrs.nbands;
            writer.runtime_args.emplace_back(core, std::move(w));
            KernelDescriptor::CoreRuntimeArgs c(args::compute_rt::COUNT, 0u);
            c[args::compute_rt::NBANDS] = attrs.nbands;
            compute.runtime_args.emplace_back(core, std::move(c));
        }
    }

    desc.kernels.resize(args::kernel_index::COUNT);
    desc.kernels[args::kernel_index::READER] = std::move(reader);
    desc.kernels[args::kernel_index::WRITER] = std::move(writer);
    desc.kernels[args::kernel_index::COMPUTE] = std::move(compute);
    return desc;
}

void UnpackFp8RowsProgramFactory::override_runtime_arguments(
    Program& program,
    const UnpackFp8RowsParams& attrs,
    const UnpackFp8RowsInputs& tensors,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key pins the grid, ncores, common_values and the specs; a hit writes the two addresses and the
    // per-call values (src_page0, out_page0, nbands) the key leaves out.
    auto& reader_common = GetCommonRuntimeArgs(program, args::kernel_index::READER);
    auto& writer_common = GetCommonRuntimeArgs(program, args::kernel_index::WRITER);
    const uint32_t rows_addr = tensors.rows.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();
    if (attrs.common_values) {
        auto& compute_common = GetCommonRuntimeArgs(program, args::kernel_index::COMPUTE);
        // The program's common args are the record of what the last completed hit wrote (nothing else writes
        // them, and the noexcept stores below cannot stop part-way): a call that repeats it writes nothing.
        if (reader_common[args::reader_common::ROWS_ADDR] == rows_addr &&
            reader_common[args::reader_common::SRC_PAGE0] == attrs.src_page0 &&
            reader_common[args::reader_common::NBANDS] == attrs.nbands &&
            writer_common[args::writer_common::OUT_ADDR] == out_addr &&
            writer_common[args::writer_common::OUT_PAGE0] == attrs.out_page0 &&
            writer_common[args::writer_common::NBANDS] == attrs.nbands &&
            compute_common[args::compute_common::NBANDS] == attrs.nbands) {
            return;
        }
        reader_common[args::reader_common::ROWS_ADDR] = rows_addr;
        writer_common[args::writer_common::OUT_ADDR] = out_addr;
        reader_common[args::reader_common::SRC_PAGE0] = attrs.src_page0;
        reader_common[args::reader_common::NBANDS] = attrs.nbands;
        writer_common[args::writer_common::OUT_PAGE0] = attrs.out_page0;
        writer_common[args::writer_common::NBANDS] = attrs.nbands;
        compute_common[args::compute_common::NBANDS] = attrs.nbands;
        return;
    }
    // Per-core values: every core holds the same three values (create_descriptor writes them alike and every
    // completed hit below rewrites them all; nothing else writes them, and the writes cannot throw part-way), so
    // core (0, 0), on which every program runs, holds what the last call wrote. A call that repeats it, addresses
    // included, has nothing to write: it returns before the O(cores) walk.
    const CoreCoord first_core{0, 0};
    auto& reader_first = GetRuntimeArgs(program, args::kernel_index::READER, first_core);
    auto& writer_first = GetRuntimeArgs(program, args::kernel_index::WRITER, first_core);
    auto& compute_first = GetRuntimeArgs(program, args::kernel_index::COMPUTE, first_core);
    const bool same_addresses = reader_common[args::reader_common::ROWS_ADDR] == rows_addr &&
                                writer_common[args::writer_common::OUT_ADDR] == out_addr;
    const bool same_reader = reader_first[args::reader_rt::SRC_PAGE0] == attrs.src_page0 &&
                             reader_first[args::reader_rt::NBANDS] == attrs.nbands;
    const bool same_writer = writer_first[args::writer_rt::OUT_PAGE0] == attrs.out_page0 &&
                             writer_first[args::writer_rt::NBANDS] == attrs.nbands;
    const bool same_compute = compute_first[args::compute_rt::NBANDS] == attrs.nbands;
    if (same_addresses && same_reader && same_writer && same_compute) {
        return;
    }
    reader_common[args::reader_common::ROWS_ADDR] = rows_addr;
    writer_common[args::writer_common::OUT_ADDR] = out_addr;
    auto& reader_grid = GetRuntimeArgs(program, args::kernel_index::READER);
    for (auto& column : reader_grid) {
        for (auto& a : column) {
            if (a.size() > 0) {
                a[args::reader_rt::SRC_PAGE0] = attrs.src_page0;
                a[args::reader_rt::NBANDS] = attrs.nbands;
            }
        }
    }
    auto& writer_grid = GetRuntimeArgs(program, args::kernel_index::WRITER);
    for (auto& column : writer_grid) {
        for (auto& a : column) {
            if (a.size() > 0) {
                a[args::writer_rt::OUT_PAGE0] = attrs.out_page0;
                a[args::writer_rt::NBANDS] = attrs.nbands;
            }
        }
    }
    auto& compute_grid = GetRuntimeArgs(program, args::kernel_index::COMPUTE);
    for (auto& column : compute_grid) {
        for (auto& a : column) {
            if (a.size() > 0) {
                a[args::compute_rt::NBANDS] = attrs.nbands;
            }
        }
    }
}

}  // namespace ttnn::operations::unpack_fp8_rows
