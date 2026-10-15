// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_untilizer.hpp"

#include <bit>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>

#include "kernels/dataflow/combine_fabric2d_kernel_interface.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

constexpr const char* kComputeDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/compute/";

constexpr uint32_t align_cb(uint32_t addr) { return (addr + 63u) & ~63u; }

struct UntilizerCb {
    uint32_t index;
    tt::DataFormat format;
    uint32_t page_size;
    uint32_t total_size;
};

// In declaration order, which is layout order.
std::vector<UntilizerCb> untilizer_cbs(const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args) {
    const uint32_t token = token_size_bytes(args, tensor_args);
    const uint32_t tile = tile_size_bytes(tensor_args);
    const uint32_t block = untilize_block_tiles(args, tensor_args);
    const bool fp8 = args.fp8_scale_block != 0;
    std::vector<UntilizerCb> cbs{
        // The packer converts as it writes, so the ring is in the wire's format.
        {cmbf2d::UNT_CB_OUT,
         fp8 ? tt::DataFormat::Fp8_e4m3 : tt::DataFormat::Float16_b,
         token,
         cmbf2d::UNT_RING_BATCHES * cmbf2d::UNT_BATCH_ROWS * token},
        {cmbf2d::UNT_CB_BATCHES, tt::DataFormat::UInt32, BATCH_COUNT_PAGE_BYTES, BATCH_COUNT_PAGE_BYTES},
        // In the dispatched buffer's own format, so the unpacker dequantises a bfp8 tile on its way through.
        {cmbf2d::UNT_CB_IN,
         tt::tt_metal::datatype_to_dataformat_converter(tensor_args.dispatched_buffer.dtype()),
         tile,
         2 * block * tile},
    };
    if (fp8) {
        // One tile per scale block of the input window, double-buffered. A power-of-two 1/scale is exact in bf16.
        constexpr uint32_t bf16_tile = tt::constants::TILE_HW * sizeof(uint16_t);
        const uint32_t scale_tiles = 2 * (block / FP8_SCALE_BLOCK_TILES);
        cbs.push_back({cmbf2d::UNT_CB_INV_SCALE, tt::DataFormat::Float16_b, bf16_tile, scale_tiles * bf16_tile});
        cbs.push_back({cmbf2d::UNT_CB_SCALE, tt::DataFormat::Float16_b, bf16_tile, scale_tiles * bf16_tile});
    }
    return cbs;
}

}  // namespace

uint32_t untilizer_cbs_end(uint32_t base, const CombineFabric2dParams& args, const CombineFabric2dInputs& tensor_args) {
    uint32_t end = align_cb(base);
    for (const auto& cb : untilizer_cbs(args, tensor_args)) {
        end = align_cb(end + cb.total_size);
    }
    return end;
}

void add_untilizer_cbs(
    tt::tt_metal::ProgramDescriptor& desc,
    const CombineFabric2dParams& args,
    const CombineFabric2dInputs& tensor_args,
    const tt::tt_metal::CoreRangeSet& core,
    tt::tt_metal::Buffer* arena) {
    uint32_t offset = 0;
    for (const auto& cb : untilizer_cbs(args, tensor_args)) {
        tt::tt_metal::CBDescriptor d{
            .total_size = cb.total_size,
            .core_ranges = core,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb.index),
                .data_format = cb.format,
                .page_size = cb.page_size,
            }}}};
        if (arena != nullptr) {
            d.buffer = arena;
            d.address_offset = offset;
            offset = align_cb(offset + cb.total_size);
        }
        desc.cbs.push_back(std::move(d));
    }
}

tt::tt_metal::KernelDescriptor untilizer_compute_kernel(
    const CombineFabric2dParams& args,
    const CombineFabric2dInputs& tensor_args,
    const tt::tt_metal::CoreRangeSet& core) {
    tt::tt_metal::KernelDescriptor kernel;
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = core;
    kernel.compile_time_args = {
        cmbf2d::UNT_CB_IN,
        cmbf2d::UNT_CB_OUT,
        cmbf2d::UNT_CB_BATCHES,
        tiles_per_token_row(tensor_args),
        untilize_block_tiles(args, tensor_args),
        cmbf2d::UNT_BATCH_ROWS};
    if (args.fp8_scale_block == 0) {
        kernel.kernel_source = std::string(kComputeDir) + "untilize_combine_fabric2d.cpp";
        kernel.config = tt::tt_metal::ComputeConfigDescriptor{};
        return kernel;
    }
    namespace q = per_token_cast_to_fp8;
    kernel.kernel_source = std::string(kComputeDir) + "untilize_quantize_combine_fabric2d.cpp";
    kernel.compile_time_args.insert(
        kernel.compile_time_args.end(),
        {cmbf2d::UNT_CB_INV_SCALE,
         cmbf2d::UNT_CB_SCALE,
         // A token in fp8 tile columns: the packer's row stride, scales included.
         token_size_bytes(args, tensor_args) / tt::constants::TILE_WIDTH,
         std::bit_cast<uint32_t>(q::SCALE_CLAMP_MIN),
         std::bit_cast<uint32_t>(1.0f / q::E4M3_MAX_NORMAL)});
    // Blackhole packs FP8_E4M3 only out of a 32-bit DEST. HiFi2 already makes x * (1/scale) exact, since a
    // power-of-two 1/scale has no low mantissa bits.
    kernel.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi2, .fp32_dest_acc_en = true};
    return kernel;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
