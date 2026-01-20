// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/grid_sample/device/grid_sample_utils.hpp"
#include "ttnn/operations/pool/grid_sample/device/grid_sample_device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::prim {

bool should_use_split_reader(
    const Tensor& input_tensor, const Tensor& grid_tensor, bool use_precomputed_grid, const std::string& mode) {
    // Split reader is only compatible with a sharded grid tensor
    if (mode == "nearest") {
        return true;
    }

    if (!grid_tensor.is_sharded()) {
        return false;
    }

    // In the case when the grid is not precomputed, majority of processing time goes to computing the coordinates and
    // the weights Processing time is in most cases halved, as both NCRISC and BRISC calculate these weights and
    // coordinates
    if (!use_precomputed_grid) {
        return true;
    }

    // As one of the NoCs is significantly slower than the other when it comes to DRAM reads, we avoid using split
    // reader when input tensor is in DRAM
    if (input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM) {
        return false;
    }

    // Get device architecture
    tt::tt_metal::IDevice* device = input_tensor.device();
    const auto arch = device->arch();

    // On wormhole, the bottleneck is always the reading of the input image, so split reader is beneficial
    if (arch == tt::ARCH::WORMHOLE_B0) {
        return true;
    }

    // On blackhole, for a lower number of channels, the bottleneck is the reading of the input image, so split reader
    // is benefitial On higher number of channels, the bottleneck is on the unpacker side, where using split reader also
    // adds additional overhead, so it slows down the program
    if (arch == tt::ARCH::BLACKHOLE) {
        const uint32_t input_channels = input_tensor.padded_shape()[-1];
        return input_channels <= 224;
    }

    // Default case for other architectures, currently should be unreachable
    return false;
}

uint32_t get_grid_batching_factor(const Tensor& grid_tensor, bool use_precomputed_grid, const std::string& mode) {
    uint32_t elements_per_point;
    if (use_precomputed_grid) {
        elements_per_point =
            (mode == "nearest") ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST : PRECOMPUTED_GRID_ELEMENTS_PER_POINT;
    } else {
        elements_per_point = STANDARD_GRID_ELEMENTS_PER_POINT;
    }
    return grid_tensor.logical_shape()[-1] / elements_per_point;
}

uint32_t get_aligned_stick_size(const ttnn::Shape& shape, const Tensor& tensor) {
    const uint32_t stick_nbytes = shape[-1] * tensor.element_size();
    const uint32_t alignment = tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                   ? tt::tt_metal::hal::get_dram_alignment()
                                   : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(stick_nbytes, alignment);
}

}  // namespace ttnn::prim
