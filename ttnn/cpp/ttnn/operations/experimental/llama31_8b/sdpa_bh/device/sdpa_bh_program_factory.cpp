// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/sdpa_bh/device/sdpa_bh_device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/sdpa_bh/device/sdpa_bh_table.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::llama31_8b {

namespace {

using namespace tt::tt_metal;
using namespace sdpa_bh_table;

void put_rt(
    KernelDescriptor& kd,
    const uint16_t (*cores)[2],
    const uint32_t* off,
    const uint32_t* rt,
    const uint32_t (*bind)[3],
    uint32_t nbind,
    const uint32_t* scale_rt,
    uint32_t nscale,
    size_t ncores,
    Buffer* const* bufs,
    uint32_t scale_bits) {
    uint32_t b = 0;
    uint32_t s = 0;
    for (uint32_t c = 0; c < ncores; ++c) {
        KernelDescriptor::RTArgList a;
        a.reserve(off[c + 1] - off[c]);
        for (uint32_t i = off[c]; i < off[c + 1]; ++i) {
            if (b < nbind && bind[b][0] == c && bind[b][1] == i - off[c]) {
                a.push_back(bufs[bind[b][2]]);
                ++b;
            } else if (s < nscale && scale_rt[s] == i) {
                a.push_back(scale_bits);
                ++s;
            } else {
                a.push_back(rt[i]);
            }
        }
        kd.emplace_runtime_args(CoreCoord(cores[c][0], cores[c][1]), a);
    }
}

ProgramDescriptor build_c0(Buffer* const* bufs, uint32_t scale_bits, IDevice* device) {
    for (const auto& p : kPhys_c0) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b.sdpa_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
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
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 4, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 16384;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6, .data_format = static_cast<tt::DataFormat>(0), .page_size = 4096, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 16384;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 16384;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(0),
            .page_size = 4096,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(0),
            .page_size = 4096,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 13,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8704;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 14,
            .data_format = static_cast<tt::DataFormat>(6),
            .page_size = 1088,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/sdpa_bh/device/kernels/reader_sdpa_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u,  32u,  8u, 8u,          32u,         32u,         32u,         4u, 4u,    2u,          16u, 2u,
            16u, 110u, 1u, 0u,          0u,          0u,          0u,          0u, 0u,    0u,          0u,  0u,
            0u,  2u,   0u, 0u,          0u,          0u,          0u,          0u, 1u,    0u,          0u,  0u,
            2u,  0u,   0u, 2u,          1088u,       2u,          1088u,       2u, 1088u, 0u,          0u,  0u,
            0u,  0u,   0u, 0u,          0u,          0u,          0u,          0u, 0u,    0u,          0u,  0u,
            1u,  2u,   3u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 0u, 0u,    4294967295u, 0u};
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("Q_UNIT_TABLE_SIZE", "5");
        kd.defines.emplace_back("REDUCE_GRANULARITY", "2");
        kd.defines.emplace_back("STATS_GRANULARITY", "2");
        kd.defines.emplace_back("SUB_EXP_GRANULARITY", "2");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_c0_0,
            kOff_c0_0,
            kRt_c0_0,
            kBind_c0_0,
            kNBind_c0_0,
            kScaleRt_c0_0,
            kNScaleRt_c0_0,
            std::size(kCore_c0_0),
            bufs,
            scale_bits);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/sdpa_bh/device/kernels/writer_sdpa_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u, 32u,   32u, 1024u, 4u, 2u, 16u, 2u, 16u, 1065369472u, 110u, 1u, 0u, 0u, 0u,
            0u, 1u,    0u,  1u,    0u, 1u, 0u,  0u, 0u,  0u,          0u,   0u, 0u, 1u, 1u,
            2u, 1088u, 0u,  0u,    0u, 0u, 3u,  4u, 5u,  4294967295u, 14u,  0u, 0u, 0u};
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("Q_UNIT_TABLE_SIZE", "5");
        kd.defines.emplace_back("REDUCE_GRANULARITY", "2");
        kd.defines.emplace_back("STATS_GRANULARITY", "2");
        kd.defines.emplace_back("SUB_EXP_GRANULARITY", "2");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_c0_1,
            kOff_c0_1,
            kRt_c0_1,
            kBind_c0_1,
            kNBind_c0_1,
            kScaleRt_c0_1,
            kNScaleRt_c0_1,
            std::size(kCore_c0_1),
            bufs,
            scale_bits);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b/sdpa_bh/device/kernels/compute_sdpa_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            32u,         4u, 4u, 2u,          16u,         2u,  16u, 4u, 2u,  2u, 1u,  1u,  2u,  2u,  1u, 2u, 2u,
            1u,          0u, 0u, 0u,          1035273459u, 0u,  0u,  0u, 32u, 0u, 1u,  0u,  0u,  0u,  1u, 2u, 3u,
            4294967295u, 4u, 5u, 4294967295u, 4294967295u, 14u, 6u,  7u, 8u,  9u, 10u, 11u, 12u, 13u, 0u, 2u, 1088u};
        kd.compile_time_args[21] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("Q_UNIT_TABLE_SIZE", "5");
        kd.defines.emplace_back("REDUCE_GRANULARITY", "2");
        kd.defines.emplace_back("STATS_GRANULARITY", "2");
        kd.defines.emplace_back("SUB_EXP_GRANULARITY", "2");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(4);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = false;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_c0_2,
            kOff_c0_2,
            kRt_c0_2,
            kBind_c0_2,
            kNBind_c0_2,
            kScaleRt_c0_2,
            kNScaleRt_c0_2,
            std::size(kCore_c0_2),
            bufs,
            scale_bits);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

ProgramDescriptor build_c1(Buffer* const* bufs, uint32_t scale_bits, IDevice* device) {
    for (const auto& p : kPhys_c1) {
        const CoreCoord v = device->worker_core_from_logical_core(CoreCoord(p[0], p[1]));
        TT_FATAL(
            v.x == p[2] && v.y == p[3],
            "llama31_8b.sdpa_bh: core ({}, {}) maps to ({}, {}), table has ({}, {})",
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
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 0, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 1, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 17408;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 2, .data_format = static_cast<tt::DataFormat>(6), .page_size = 1088, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 3, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 4, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 2048;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 5, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 16384;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 6, .data_format = static_cast<tt::DataFormat>(0), .page_size = 4096, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 16384;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 7, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 16384;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 8, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 9, .data_format = static_cast<tt::DataFormat>(5), .page_size = 2048, .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 10,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 11,
            .data_format = static_cast<tt::DataFormat>(0),
            .page_size = 4096,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8192;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 12,
            .data_format = static_cast<tt::DataFormat>(0),
            .page_size = 4096,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 4096;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 13,
            .data_format = static_cast<tt::DataFormat>(5),
            .page_size = 2048,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        CBDescriptor cb;
        cb.total_size = 8704;
        cb.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = 14,
            .data_format = static_cast<tt::DataFormat>(6),
            .page_size = 1088,
            .tile = std::nullopt});
        desc.cbs.push_back(std::move(cb));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/sdpa_bh/device/kernels/reader_sdpa_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u,  32u,  8u, 8u,          32u,         32u,         32u,         4u, 4u,    2u,          16u, 2u,
            16u, 110u, 1u, 0u,          0u,          0u,          0u,          0u, 0u,    0u,          0u,  0u,
            0u,  2u,   0u, 0u,          0u,          0u,          0u,          0u, 1u,    0u,          0u,  0u,
            2u,  0u,   0u, 2u,          1088u,       2u,          1088u,       2u, 1088u, 0u,          0u,  0u,
            0u,  0u,   0u, 0u,          0u,          0u,          0u,          0u, 0u,    0u,          0u,  0u,
            1u,  2u,   3u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 0u, 0u,    4294967295u, 0u};
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("Q_UNIT_TABLE_SIZE", "5");
        kd.defines.emplace_back("REDUCE_GRANULARITY", "2");
        kd.defines.emplace_back("STATS_GRANULARITY", "2");
        kd.defines.emplace_back("SUB_EXP_GRANULARITY", "2");
        kd.config = ReaderConfigDescriptor{};
        put_rt(
            kd,
            kCore_c1_0,
            kOff_c1_0,
            kRt_c1_0,
            kBind_c1_0,
            kNBind_c1_0,
            kScaleRt_c1_0,
            kNScaleRt_c1_0,
            std::size(kCore_c1_0),
            bufs,
            scale_bits);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source = "ttnn/cpp/ttnn/operations/experimental/llama31_8b/sdpa_bh/device/kernels/writer_sdpa_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            1u, 32u,   32u, 1024u, 4u, 2u, 16u, 2u, 16u, 1065369472u, 110u, 1u, 0u, 0u, 0u,
            0u, 1u,    0u,  1u,    0u, 1u, 0u,  1u, 0u,  0u,          0u,   0u, 0u, 1u, 1u,
            2u, 1088u, 0u,  0u,    0u, 0u, 3u,  4u, 5u,  4294967295u, 14u,  0u, 0u, 0u};
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("Q_UNIT_TABLE_SIZE", "5");
        kd.defines.emplace_back("REDUCE_GRANULARITY", "2");
        kd.defines.emplace_back("STATS_GRANULARITY", "2");
        kd.defines.emplace_back("SUB_EXP_GRANULARITY", "2");
        kd.config = WriterConfigDescriptor{};
        put_rt(
            kd,
            kCore_c1_1,
            kOff_c1_1,
            kRt_c1_1,
            kBind_c1_1,
            kNBind_c1_1,
            kScaleRt_c1_1,
            kNScaleRt_c1_1,
            std::size(kCore_c1_1),
            bufs,
            scale_bits);
        desc.kernels.push_back(std::move(kd));
    }
    {
        KernelDescriptor kd;
        kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/llama31_8b/sdpa_bh/device/kernels/compute_sdpa_bh.cpp";
        kd.source_type = static_cast<KernelDescriptor::SourceType>(0);
        kd.core_ranges = CoreRangeSet(std::vector<CoreRange>{CoreRange(CoreCoord(0, 0), CoreCoord(10, 9))});
        kd.compile_time_args = KernelDescriptor::CompileTimeArgs{
            32u,         4u, 4u, 2u,          16u,         2u,  16u, 4u, 2u,  2u, 1u,  1u,  2u,  2u,  1u, 2u, 2u,
            1u,          0u, 0u, 0u,          1035273459u, 0u,  0u,  0u, 32u, 0u, 1u,  0u,  0u,  0u,  1u, 2u, 3u,
            4294967295u, 4u, 5u, 4294967295u, 4294967295u, 14u, 6u,  7u, 8u,  9u, 10u, 11u, 12u, 13u, 0u, 2u, 1088u};
        kd.compile_time_args[21] = scale_bits;
        kd.defines.emplace_back("DHT_GRANULARITY", "4");
        kd.defines.emplace_back("EXP_APPROX_MODE", "0");
        kd.defines.emplace_back("Q_UNIT_TABLE_SIZE", "5");
        kd.defines.emplace_back("REDUCE_GRANULARITY", "2");
        kd.defines.emplace_back("STATS_GRANULARITY", "2");
        kd.defines.emplace_back("SUB_EXP_GRANULARITY", "2");
        {
            ComputeConfigDescriptor cc;
            cc.math_fidelity = static_cast<decltype(cc.math_fidelity)>(4);
            cc.fp32_dest_acc_en = true;
            cc.dst_full_sync_en = false;
            cc.bfp8_pack_precise = false;
            cc.math_approx_mode = false;
            cc.enable_trisc2_rvv = false;
            kd.config = cc;
        }
        put_rt(
            kd,
            kCore_c1_2,
            kOff_c1_2,
            kRt_c1_2,
            kBind_c1_2,
            kNBind_c1_2,
            kScaleRt_c1_2,
            kNScaleRt_c1_2,
            std::size(kCore_c1_2),
            bufs,
            scale_bits);
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}

}  // namespace

tt::tt_metal::ProgramDescriptor SdpaBhProgramFactory::create_descriptor(
    const SdpaBhOperation::operation_attributes_t& operation_attributes,
    const SdpaBhOperation::tensor_args_t& tensor_args,
    SdpaBhOperation::tensor_return_value_t& tensor_return_value) {
    const auto& q = tensor_args.q;
    const float scale = operation_attributes.scale.value_or(1.0f / std::sqrt(static_cast<float>(q.padded_shape()[-1])));
    Buffer* bufs[] = {
        q.buffer(), tensor_args.k.buffer(), tensor_args.v.value().buffer(), tensor_return_value.buffer(), nullptr};
    IDevice* device = q.device();
    const uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (!operation_attributes.output_concat_heads) {
        return build_c0(bufs, scale_bits, device);
    }
    if (operation_attributes.output_concat_heads) {
        return build_c1(bufs, scale_bits, device);
    }
    TT_THROW("llama31_8b.sdpa_bh: no table for this configuration");
}

}  // namespace ttnn::experimental::llama31_8b
