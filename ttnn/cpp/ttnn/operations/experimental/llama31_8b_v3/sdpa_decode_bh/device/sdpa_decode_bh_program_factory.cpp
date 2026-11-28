// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_bh_device_operation.hpp"
#include "sdpa_decode_bh_table.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::experimental::llama31_8b_v3::sdpa_decode_bh_detail {

namespace {

using namespace sdpa_decode_bh_table;

void put_rt(
    KernelDescriptor& kd,
    const uint16_t (*cores)[2],
    const uint32_t* off,
    const uint32_t* rt,
    const uint32_t (*bind)[3],
    uint32_t nbind,
    const int64_t (*slot)[5],
    uint32_t nslot,
    size_t ncores,
    Buffer* const* bufs,
    uint32_t scale_bits,
    uint32_t nblk,
    uint32_t ptw) {
    uint32_t b = 0;
    uint32_t s = 0;
    for (uint32_t c = 0; c < ncores; ++c) {
        KernelDescriptor::RTArgList a;
        a.reserve(off[c + 1] - off[c]);
        for (uint32_t i = off[c]; i < off[c + 1]; ++i) {
            if (b < nbind && bind[b][0] == c && bind[b][1] == i - off[c]) {
                a.push_back(bufs[bind[b][2]]);
                ++b;
            } else if (s < nslot && slot[s][0] == i) {
                a.push_back(
                    slot[s][4] ? scale_bits : static_cast<uint32_t>(slot[s][1] * nblk + slot[s][2] * ptw + slot[s][3]));
                ++s;
            } else {
                a.push_back(rt[i]);
            }
        }
        kd.emplace_runtime_args(CoreCoord(cores[c][0], cores[c][1]), a);
    }
}

ProgramDescriptor build_m0_d64(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m0_d64) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 57344;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u, 1u, 1024u, 2u, 2u,   0u,    64u,   1u, 64u,   1u, 8u,  1u, 1024u, 1u, 8u,    1u, 1u, 1u, 0u,
            0u, 4u, 0u,    0u, 0u,   4096u, 0u,    0u, 2048u, 0u, 32u, 2u, 0u,    0u, 1024u, 0u, 0u, 1u, 2048u,
            1u, 1u, 2u,    2u, 258u, 2u,    1088u, 2u, 1088u, 0u, 0u,  2u, 64u,   2u, 4096u, 0u, 0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[52] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R2", "4");
        kd.defines.emplace_back("SDB_R3", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m0_d64_0,
            kOff_m0_d64_0,
            kRt_m0_d64_0,
            kBind_m0_d64_0,
            kNBind_m0_d64_0,
            kSlot_m0_d64_0,
            kNSlot_m0_d64_0,
            std::size(kCore_m0_d64_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 2u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_W0", "1");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m0_d64_1,
            kOff_m0_d64_1,
            kRt_m0_d64_1,
            kBind_m0_d64_1,
            kNBind_m0_d64_1,
            kSlot_m0_d64_1,
            kNSlot_m0_d64_1,
            std::size(kCore_m0_d64_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 2u, 2u, 1u, 0u, 2u, 0u, 0u, 0u, 0u, 0u,          2u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1040187392u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "2");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C0", "1");
        kd.defines.emplace_back("SDB_C1", "1");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C4", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m0_d64_2,
            kOff_m0_d64_2,
            kRt_m0_d64_2,
            kBind_m0_d64_2,
            kNBind_m0_d64_2,
            kSlot_m0_d64_2,
            kNSlot_m0_d64_2,
            std::size(kCore_m0_d64_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m0_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m0_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u, 1u, 1024u, 4u, 4u,   0u,    64u,   1u, 64u,   1u, 8u,  1u, 1024u, 1u, 8u,    1u, 1u, 1u, 0u,
            0u, 4u, 0u,    0u, 0u,   8192u, 0u,    0u, 2048u, 0u, 32u, 2u, 0u,    0u, 1024u, 0u, 0u, 1u, 2048u,
            1u, 1u, 4u,    4u, 258u, 2u,    1088u, 2u, 1088u, 0u, 0u,  2u, 64u,   2u, 4096u, 0u, 0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[52] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R2", "4");
        kd.defines.emplace_back("SDB_R3", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m0_d128_0,
            kOff_m0_d128_0,
            kRt_m0_d128_0,
            kBind_m0_d128_0,
            kNBind_m0_d128_0,
            kSlot_m0_d128_0,
            kNSlot_m0_d128_0,
            std::size(kCore_m0_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_W0", "1");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m0_d128_1,
            kOff_m0_d128_1,
            kRt_m0_d128_1,
            kBind_m0_d128_1,
            kNBind_m0_d128_1,
            kSlot_m0_d128_1,
            kNSlot_m0_d128_1,
            std::size(kCore_m0_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C0", "1");
        kd.defines.emplace_back("SDB_C1", "1");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C4", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m0_d128_2,
            kOff_m0_d128_2,
            kRt_m0_d128_2,
            kBind_m0_d128_2,
            kNBind_m0_d128_2,
            kSlot_m0_d128_2,
            kNSlot_m0_d128_2,
            std::size(kCore_m0_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m2_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m2_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u, 1u, 1024u, 4u, 4u,        0u,    64u,   1u, 64u,   1u, 8u,  1u, 1024u, 1u, 8u,    1u, 1u, 1u, 0u,
            0u, 4u, 0u,    0u, 0u,        8192u, 0u,    0u, 2048u, 0u, 32u, 2u, 0u,    0u, 1024u, 0u, 0u, 1u, 2048u,
            1u, 2u, 8u,    4u, 33685762u, 2u,    1088u, 2u, 1088u, 0u, 0u,  2u, 64u,   2u, 4096u, 0u, 0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[52] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.defines.emplace_back("SDB_M", "2");
        kd.defines.emplace_back("SDB_MQ", "1,2,2,2");
        kd.defines.emplace_back("SDB_MG", "4");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m2_d128_0,
            kOff_m2_d128_0,
            kRt_m2_d128_0,
            kBind_m2_d128_0,
            kNBind_m2_d128_0,
            kSlot_m2_d128_0,
            kNSlot_m2_d128_0,
            std::size(kCore_m2_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_M", "2");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m2_d128_1,
            kOff_m2_d128_1,
            kRt_m2_d128_1,
            kBind_m2_d128_1,
            kNBind_m2_d128_1,
            kSlot_m2_d128_1,
            kNSlot_m2_d128_1,
            std::size(kCore_m2_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C0", "1");
        kd.defines.emplace_back("SDB_C1", "1");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C4", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        kd.defines.emplace_back("SDB_M", "2");
        kd.defines.emplace_back("SDB_N", "8");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m2_d128_2,
            kOff_m2_d128_2,
            kRt_m2_d128_2,
            kBind_m2_d128_2,
            kNBind_m2_d128_2,
            kSlot_m2_d128_2,
            kNSlot_m2_d128_2,
            std::size(kCore_m2_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m3_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m3_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u,        1u,   1024u, 4u,    4u, 0u,    64u, 1u, 64u, 1u,    8u,    1u,    1024u, 1u,
            8u,        1u,   1u,    1u,    0u, 0u,    4u,  0u, 0u,  0u,    8192u, 0u,    0u,    2048u,
            0u,        32u,  2u,    0u,    0u, 1024u, 0u,  0u, 1u,  2048u, 1u,    3u,    12u,   4u,
            33685762u, 770u, 2u,    1088u, 2u, 1088u, 0u,  0u, 2u,  64u,   2u,    4096u, 0u,    0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[53] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.defines.emplace_back("SDB_M", "3");
        kd.defines.emplace_back("SDB_MQ", "1,2,2,2,3,2");
        kd.defines.emplace_back("SDB_MG", "4");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m3_d128_0,
            kOff_m3_d128_0,
            kRt_m3_d128_0,
            kBind_m3_d128_0,
            kNBind_m3_d128_0,
            kSlot_m3_d128_0,
            kNSlot_m3_d128_0,
            std::size(kCore_m3_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_M", "3");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m3_d128_1,
            kOff_m3_d128_1,
            kRt_m3_d128_1,
            kBind_m3_d128_1,
            kNBind_m3_d128_1,
            kSlot_m3_d128_1,
            kNSlot_m3_d128_1,
            std::size(kCore_m3_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C0", "1");
        kd.defines.emplace_back("SDB_C1", "1");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C4", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        kd.defines.emplace_back("SDB_M", "3");
        kd.defines.emplace_back("SDB_N", "12");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m3_d128_2,
            kOff_m3_d128_2,
            kRt_m3_d128_2,
            kBind_m3_d128_2,
            kNBind_m3_d128_2,
            kSlot_m3_d128_2,
            kNSlot_m3_d128_2,
            std::size(kCore_m3_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m4_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m4_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u,        1u,        1024u, 4u,    4u, 0u,    64u, 1u, 64u, 1u,    8u,    1u,    1024u, 1u,
            8u,        1u,        1u,    1u,    0u, 0u,    4u,  0u, 0u,  0u,    8192u, 0u,    0u,    2048u,
            0u,        32u,       2u,    0u,    0u, 1024u, 0u,  0u, 1u,  2048u, 1u,    4u,    16u,   4u,
            33685762u, 67240706u, 2u,    1088u, 2u, 1088u, 0u,  0u, 2u,  64u,   2u,    4096u, 0u,    0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[53] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.defines.emplace_back("SDB_M", "4");
        kd.defines.emplace_back("SDB_MQ", "1,2,2,2,3,2,4,2");
        kd.defines.emplace_back("SDB_MG", "4");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m4_d128_0,
            kOff_m4_d128_0,
            kRt_m4_d128_0,
            kBind_m4_d128_0,
            kNBind_m4_d128_0,
            kSlot_m4_d128_0,
            kNSlot_m4_d128_0,
            std::size(kCore_m4_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_M", "4");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m4_d128_1,
            kOff_m4_d128_1,
            kRt_m4_d128_1,
            kBind_m4_d128_1,
            kNBind_m4_d128_1,
            kSlot_m4_d128_1,
            kNSlot_m4_d128_1,
            std::size(kCore_m4_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C0", "1");
        kd.defines.emplace_back("SDB_C1", "1");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C4", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        kd.defines.emplace_back("SDB_M", "4");
        kd.defines.emplace_back("SDB_N", "16");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m4_d128_2,
            kOff_m4_d128_2,
            kRt_m4_d128_2,
            kBind_m4_d128_2,
            kNBind_m4_d128_2,
            kSlot_m4_d128_2,
            kNSlot_m4_d128_2,
            std::size(kCore_m4_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m5_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m5_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u, 1u,    1024u, 4u,    4u, 0u, 64u, 1u,    64u, 1u,    8u,  1u, 1024u,     1u,        8u,
            1u, 1u,    1u,    0u,    0u, 4u, 0u,  0u,    0u,  8192u, 0u,  0u, 2048u,     0u,        32u,
            2u, 0u,    0u,    1024u, 0u, 0u, 1u,  2048u, 1u,  5u,    20u, 4u, 33685762u, 67240706u, 1282u,
            2u, 1088u, 2u,    1088u, 0u, 0u, 2u,  64u,   2u,  4096u, 0u,  0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[54] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.defines.emplace_back("SDB_M", "5");
        kd.defines.emplace_back("SDB_MQ", "1,2,2,2,3,2,4,2,5,2");
        kd.defines.emplace_back("SDB_MG", "4");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m5_d128_0,
            kOff_m5_d128_0,
            kRt_m5_d128_0,
            kBind_m5_d128_0,
            kNBind_m5_d128_0,
            kSlot_m5_d128_0,
            kNSlot_m5_d128_0,
            std::size(kCore_m5_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_M", "5");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m5_d128_1,
            kOff_m5_d128_1,
            kRt_m5_d128_1,
            kBind_m5_d128_1,
            kNBind_m5_d128_1,
            kSlot_m5_d128_1,
            kNSlot_m5_d128_1,
            std::size(kCore_m5_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        kd.defines.emplace_back("SDB_M", "5");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m5_d128_2,
            kOff_m5_d128_2,
            kRt_m5_d128_2,
            kBind_m5_d128_2,
            kNBind_m5_d128_2,
            kSlot_m5_d128_2,
            kNSlot_m5_d128_2,
            std::size(kCore_m5_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m6_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m6_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u, 1u,    1024u, 4u,    4u, 0u, 64u, 1u,    64u, 1u,    8u,  1u, 1024u,     1u,        8u,
            1u, 1u,    1u,    0u,    0u, 4u, 0u,  0u,    0u,  8192u, 0u,  0u, 2048u,     0u,        32u,
            2u, 0u,    0u,    1024u, 0u, 0u, 1u,  2048u, 1u,  6u,    24u, 4u, 33685762u, 67240706u, 100795650u,
            2u, 1088u, 2u,    1088u, 0u, 0u, 2u,  64u,   2u,  4096u, 0u,  0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[54] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.defines.emplace_back("SDB_M", "6");
        kd.defines.emplace_back("SDB_MQ", "1,2,2,2,3,2,4,2,5,2,6,2");
        kd.defines.emplace_back("SDB_MG", "4");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m6_d128_0,
            kOff_m6_d128_0,
            kRt_m6_d128_0,
            kBind_m6_d128_0,
            kNBind_m6_d128_0,
            kSlot_m6_d128_0,
            kNSlot_m6_d128_0,
            std::size(kCore_m6_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_M", "6");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m6_d128_1,
            kOff_m6_d128_1,
            kRt_m6_d128_1,
            kBind_m6_d128_1,
            kNBind_m6_d128_1,
            kSlot_m6_d128_1,
            kNSlot_m6_d128_1,
            std::size(kCore_m6_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        kd.defines.emplace_back("SDB_M", "6");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m6_d128_2,
            kOff_m6_d128_2,
            kRt_m6_d128_2,
            kBind_m6_d128_2,
            kNBind_m6_d128_2,
            kSlot_m6_d128_2,
            kNSlot_m6_d128_2,
            std::size(kCore_m6_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m7_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m7_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u,    1u, 1024u, 4u,    4u,    0u, 64u, 1u,    64u, 1u,    8u,    1u, 1024u,     1u,        8u,
            1u,    1u, 1u,    0u,    0u,    4u, 0u,  0u,    0u,  8192u, 0u,    0u, 2048u,     0u,        32u,
            2u,    0u, 0u,    1024u, 0u,    0u, 1u,  2048u, 1u,  7u,    28u,   4u, 33685762u, 67240706u, 100795650u,
            1794u, 2u, 1088u, 2u,    1088u, 0u, 0u,  2u,    64u, 2u,    4096u, 0u, 0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[55] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.defines.emplace_back("SDB_M", "7");
        kd.defines.emplace_back("SDB_MQ", "1,2,2,2,3,2,4,2,5,2,6,2,7,2");
        kd.defines.emplace_back("SDB_MG", "4");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m7_d128_0,
            kOff_m7_d128_0,
            kRt_m7_d128_0,
            kBind_m7_d128_0,
            kNBind_m7_d128_0,
            kSlot_m7_d128_0,
            kNSlot_m7_d128_0,
            std::size(kCore_m7_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_M", "7");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m7_d128_1,
            kOff_m7_d128_1,
            kRt_m7_d128_1,
            kBind_m7_d128_1,
            kNBind_m7_d128_1,
            kSlot_m7_d128_1,
            kNSlot_m7_d128_1,
            std::size(kCore_m7_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        kd.defines.emplace_back("SDB_M", "7");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m7_d128_2,
            kOff_m7_d128_2,
            kRt_m7_d128_2,
            kBind_m7_d128_2,
            kNBind_m7_d128_2,
            kSlot_m7_d128_2,
            kNSlot_m7_d128_2,
            std::size(kCore_m7_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_m8_d128(
    Buffer* const* bufs, uint32_t scale_bits, uint32_t nblk, uint32_t ptw, MeshDevice* device) {
    for (const auto& p : kPhys_m8_d128) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b_v3.sdpa_decode_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
            p[0],
            p[1],
            v.x,
            v.y,
            p[2],
            p[3]);
    }
    ProgramDescriptor desc;
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 34816;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 64;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 15, .data_format = static_cast<tt::DataFormat>(8), .page_size = 64, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9,
            .data_format = static_cast<tt::DataFormat>(8),
            .page_size = static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0),
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 24,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 25,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 26,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 27,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 28,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 29,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 30,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 31,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 21,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 22,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 23,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 16,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 17,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 18,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 86016;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 19,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 20,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = TileDescriptor(32, 32, false)});
        desc.cbs.push_back(std::move(cb));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 0,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 1,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = 2,
        .core_type = static_cast<tt::CoreType>(12),
        .core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        .initial_value = 0});
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/reader_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u,    1u,    1024u, 4u,    4u,  0u,  64u,       1u,        64u,        1u,         8u, 1u,
            1024u, 1u,    8u,    1u,    1u,  1u,  0u,        0u,        4u,         0u,         0u, 0u,
            8192u, 0u,    0u,    2048u, 0u,  32u, 2u,        0u,        0u,         1024u,      0u, 0u,
            1u,    2048u, 1u,    8u,    32u, 4u,  33685762u, 67240706u, 100795650u, 167905026u, 2u, 1088u,
            2u,    1088u, 0u,    0u,    2u,  64u, 2u,        4096u,     0u,         0u};
        kd.compile_time_args[2] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[12] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[33] =
            static_cast<uint32_t>(1 * static_cast<int64_t>(nblk) + 0 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[55] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 4 * static_cast<int64_t>(ptw) + 0);
        kd.defines.emplace_back("SDB_R0", "1");
        kd.defines.emplace_back("SDB_R1", "1");
        kd.defines.emplace_back("SDB_R4", "1");
        kd.defines.emplace_back("SDB_M", "8");
        kd.defines.emplace_back("SDB_MQ", "1,2,2,2,3,2,4,2,5,2,6,2,7,2,10,2");
        kd.defines.emplace_back("SDB_MG", "4");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_m8_d128_0,
            kOff_m8_d128_0,
            kRt_m8_d128_0,
            kBind_m8_d128_0,
            kNBind_m8_d128_0,
            kSlot_m8_d128_0,
            kNSlot_m8_d128_0,
            std::size(kCore_m8_d128_0),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/writer_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1u, 1024u, 4u, 0u, 1065369472u, 64u, 64u, 0u, 1u,  0u, 32u, 8u,   8u,
                                              1u, 8u,    1u, 2u, 1u,          4u,  1u,  0u, 32u, 0u, 2u,  2048u};
        kd.compile_time_args[1] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.named_compile_time_args.emplace_back("reducers_per_output", 8u);
        kd.defines.emplace_back("SDB_M", "8");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_m8_d128_1,
            kOff_m8_d128_1,
            kRt_m8_d128_1,
            kBind_m8_d128_1,
            kNBind_m8_d128_1,
            kSlot_m8_d128_1,
            kNSlot_m8_d128_1,
            std::size(kCore_m8_d128_1),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/"
            "sdpa_flash_decode_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))});
        kd.compile_time_args =
            KernelDescriptor::CompileTimeArgs{1024u, 4u, 4u, 1u, 0u, 4u, 0u, 0u, 0u, 0u, 0u,          4u, 1u,  1u,
                                              1u,    8u, 1u, 1u, 0u, 0u, 4u, 0u, 1u, 0u, 1035273459u, 0u, 32u, 0u};
        kd.compile_time_args[0] =
            static_cast<uint32_t>(0 * static_cast<int64_t>(nblk) + 1 * static_cast<int64_t>(ptw) + 0);
        kd.compile_time_args[24] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("DYNAMIC_CHUNK_SIZE", "1");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("SDB_C2", "1");
        kd.defines.emplace_back("SDB_C3", "1");
        kd.defines.emplace_back("SDB_C5", "1");
        kd.defines.emplace_back("SDB_C6", "1");
        kd.defines.emplace_back("SDB_M", "8");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(2);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = true;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_m8_d128_2,
            kOff_m8_d128_2,
            kRt_m8_d128_2,
            kBind_m8_d128_2,
            kNBind_m8_d128_2,
            kSlot_m8_d128_2,
            kNSlot_m8_d128_2,
            std::size(kCore_m8_d128_2),
            bufs,
            scale_bits,
            nblk,
            ptw);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

}  // namespace

ProgramDescriptor SdpaDecodeDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& page_table_tensor = tensor_args.page_table_tensor;
    const uint32_t rows = operation_attributes.multi_pos ? input_tensor_q.padded_shape()[1] : 0u;
    const float scale =
        operation_attributes.scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1])));
    Buffer* bufs[] = {
        input_tensor_q.buffer(),
        tensor_args.k.buffer(),
        tensor_args.v.value().buffer(),
        tensor_return_value.buffer(),
        tensor_args.cur_pos_tensor.has_value() ? tensor_args.cur_pos_tensor->buffer() : nullptr,
        page_table_tensor.has_value() ? page_table_tensor->buffer() : nullptr,
        tensor_args.attn_mask.has_value() ? tensor_args.attn_mask->buffer() : nullptr,
        tensor_args.attention_sink.has_value() ? tensor_args.attention_sink->buffer() : nullptr,
        nullptr};
    MeshDevice* device = input_tensor_q.device();
    const uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
    const uint32_t dh = input_tensor_q.padded_shape()[-1];
    const uint32_t nblk = tensor_args.k.padded_shape()[0];
    const uint32_t ptw = page_table_tensor.has_value() ? page_table_tensor->padded_shape()[-1] : 0u;
    if (rows == 0u && dh == 64u) {
        return build_m0_d64(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 0u && dh == 128u) {
        return build_m0_d128(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 2u && dh == 128u) {
        return build_m2_d128(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 3u && dh == 128u) {
        return build_m3_d128(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 4u && dh == 128u) {
        return build_m4_d128(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 5u && dh == 128u) {
        return build_m5_d128(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 6u && dh == 128u) {
        return build_m6_d128(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 7u && dh == 128u) {
        return build_m7_d128(bufs, scale_bits, nblk, ptw, device);
    }
    if (rows == 8u && dh == 128u) {
        return build_m8_d128(bufs, scale_bits, nblk, ptw, device);
    }
    TT_THROW("llama31_8b_v3.sdpa_decode_bh: no table for this configuration");
}

}  // namespace ttnn::experimental::llama31_8b_v3::sdpa_decode_bh_detail
