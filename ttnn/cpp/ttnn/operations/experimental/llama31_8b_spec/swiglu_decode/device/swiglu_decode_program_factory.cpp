// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b_spec/swiglu_decode/device/swiglu_decode_device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b_spec/swiglu_decode/device/swiglu_decode_table.hpp"

#include <set>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::llama31_8b_spec {

using namespace tt::tt_metal;
namespace table = swiglu_decode_table;

namespace {

constexpr const char* KDIR = "ttnn/cpp/ttnn/operations/experimental/llama31_8b_spec/swiglu_decode/device/kernels/";
constexpr uint32_t kBf16TileBytes = 2048;
constexpr uint32_t kBf8TileBytes = 1088;

}  // namespace

ProgramDescriptor SwigluDecodeProgramFactory::create_descriptor(
    const SwigluDecodeParams&, const SwigluDecodeInputs& tensor_args, Tensor& output) {
    auto* device = tensor_args.gate.device();
    const uint32_t cb_a = tt::CBIndex::c_0;
    const uint32_t cb_b = tt::CBIndex::c_1;
    const uint32_t cb_out = tt::CBIndex::c_16;

    std::set<CoreRange> ranges;
    for (const auto& c : table::kCores) {
        ranges.insert(CoreRange(CoreCoord(c.x, c.y), CoreCoord(c.x, c.y)));
    }
    const CoreRangeSet cores(ranges);

    ProgramDescriptor desc;
    const auto cb = [&](uint32_t index, tt::DataFormat format, uint32_t page) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = table::kMaxTiles * page,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = format,
                .page_size = page,
            }}},
        });
    };
    cb(cb_a, tt::DataFormat::Float16_b, kBf16TileBytes);
    cb(cb_b, tt::DataFormat::Float16_b, kBf16TileBytes);
    cb(cb_out, tt::DataFormat::Bfp8_b, kBf8TileBytes);

    KernelDescriptor reader;
    reader.kernel_source = std::string(KDIR) + "swiglu_decode_reader.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = {cb_a, cb_b};
    reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};

    KernelDescriptor writer;
    writer.kernel_source = std::string(KDIR) + "swiglu_decode_writer.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = {cb_out};
    writer.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};

    KernelDescriptor compute;
    compute.kernel_source = std::string(KDIR) + "swiglu_decode_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {cb_a, cb_b, cb_out};
    compute.config = ComputeConfigDescriptor{};

    for (const auto& c : table::kCores) {
        const CoreCoord core(c.x, c.y);
        std::vector<uint32_t> srcs;
        for (uint32_t i = 0; i < c.tiles; ++i) {
            const auto& s = c.src[i];
            const CoreCoord phys = device->worker_core_from_logical_core(CoreCoord(s.x, s.y));
            srcs.push_back((static_cast<uint32_t>(phys.x) << 16) | (static_cast<uint32_t>(phys.y) << 8) | s.slot);
        }
        KernelDescriptor::RTArgList r;
        r.push_back(tensor_args.gate.buffer());
        r.push_back(tensor_args.up.buffer());
        r.push_back(static_cast<uint32_t>(c.tiles));
        KernelDescriptor::RTArgList w;
        w.push_back(output.buffer());
        w.push_back(static_cast<uint32_t>(c.tiles));
        for (uint32_t v : srcs) {
            r.push_back(v);
            w.push_back(v);
        }
        reader.emplace_runtime_args(core, r);
        writer.emplace_runtime_args(core, w);
        KernelDescriptor::RTArgList k;
        k.push_back(static_cast<uint32_t>(c.tiles));
        compute.emplace_runtime_args(core, k);
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::llama31_8b_spec
