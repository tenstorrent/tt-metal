// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b_v3/kv_rows_update/device/kv_rows_update_device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/kv_rows_update/device/kv_rows_update_table.hpp"

#include <set>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::llama31_8b_v3 {

using namespace tt::tt_metal;
namespace table = kv_rows_update_table;

namespace {

constexpr const char* KDIR = "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/kv_rows_update/device/kernels/";

uint32_t row_index(uint32_t rows) {
    for (uint32_t i = 0; i < table::kRows.size(); ++i) {
        if (table::kRows[i] == rows) {
            return i;
        }
    }
    TT_THROW("llama31_8b_v3.kv_rows_update: rows {} not in the table", rows);
}

CoreRangeSet cores_of(int kv) {
    std::set<CoreRange> ranges;
    for (const auto& c : table::kCores) {
        if (kv < 0 || c.kv == kv) {
            ranges.insert(CoreRange(CoreCoord(c.x, c.y), CoreCoord(c.x, c.y)));
        }
    }
    return CoreRangeSet(ranges);
}

}  // namespace

ProgramDescriptor KvRowsUpdateProgramFactory::create_descriptor(
    const KvRowsUpdateParams&, const KvRowsUpdateInputs& a, Tensor&) {
    const auto rows = static_cast<uint32_t>(a.row_positions.size());
    const uint32_t ri = row_index(rows);
    const uint32_t tiles = table::kTiles[ri];
    const uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t cb_pk = tt::CBIndex::c_16;
    const uint32_t cb_meta = tt::CBIndex::c_20;
    const uint32_t cb_scr = tt::CBIndex::c_21;
    const CoreRangeSet all = cores_of(-1);

    ProgramDescriptor desc;
    const auto cb = [&](uint32_t index, tt::DataFormat format, uint32_t page, uint32_t pages) {
        desc.cbs.push_back(
            CBDescriptor{
                .total_size = pages * page,
                .core_ranges = all,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(index),
                    .data_format = format,
                    .page_size = page,
                }}},
            });
    };
    cb(cb_in, tt::DataFormat::Float16_b, table::kBf16TileBytes, tiles);
    cb(cb_pk, tt::DataFormat::Bfp8_b, table::kBf8TileBytes, tiles);
    cb(cb_meta, tt::DataFormat::Float16_b, table::kMetaBytes, 1);
    cb(cb_scr, tt::DataFormat::Float16_b, table::kScrBytes[ri], 1);

    const Tensor* heads[2] = {&a.k, &a.v};
    const Tensor* caches[2] = {&a.keys, &a.values};
    for (int kv = 0; kv < 2; ++kv) {
        const CoreRangeSet group = cores_of(kv);

        std::vector<uint32_t> rct = {
            rows,
            table::kMaxBlocks,
            table::kBlock,
            table::kKvHeads,
            cb_in,
            cb_meta,
            cb_scr,
            table::kBf8TileBytes,
            table::kBf16TileBytes};
        TensorAccessorArgs(heads[kv]->buffer()).append_to(rct);
        TensorAccessorArgs(caches[kv]->buffer()).append_to(rct);
        TensorAccessorArgs(a.row_positions.front().buffer()).append_to(rct);
        TensorAccessorArgs(a.page_table.buffer()).append_to(rct);
        KernelDescriptor reader;
        reader.kernel_source = std::string(KDIR) + "kv_rows_update_reader.cpp";
        reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader.core_ranges = group;
        reader.compile_time_args = std::move(rct);
        reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};

        std::vector<uint32_t> wct = {
            rows, table::kBlock, table::kKvHeads, cb_pk, cb_meta, cb_scr, table::kBf8TileBytes};
        TensorAccessorArgs(caches[kv]->buffer()).append_to(wct);
        KernelDescriptor writer;
        writer.kernel_source = std::string(KDIR) + "kv_rows_update_writer.cpp";
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = group;
        writer.compile_time_args = std::move(wct);
        writer.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};

        for (const auto& c : table::kCores) {
            if (c.kv != kv) {
                continue;
            }
            const CoreCoord core(c.x, c.y);
            KernelDescriptor::RTArgList r;
            r.push_back(heads[kv]->buffer());
            r.push_back(caches[kv]->buffer());
            r.push_back(a.page_table.buffer());
            r.push_back(static_cast<uint32_t>(c.head));
            r.push_back(static_cast<uint32_t>(c.col));
            for (const auto& p : a.row_positions) {
                r.push_back(p.buffer());
            }
            reader.emplace_runtime_args(core, r);
            KernelDescriptor::RTArgList w;
            w.push_back(caches[kv]->buffer());
            w.push_back(static_cast<uint32_t>(c.head));
            w.push_back(static_cast<uint32_t>(c.col));
            writer.emplace_runtime_args(core, w);
        }
        desc.kernels.push_back(std::move(reader));
        desc.kernels.push_back(std::move(writer));
    }

    KernelDescriptor compute;
    compute.kernel_source = std::string(KDIR) + "kv_rows_update_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all;
    compute.compile_time_args = {tiles, cb_in, cb_pk};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false};
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::llama31_8b_v3
