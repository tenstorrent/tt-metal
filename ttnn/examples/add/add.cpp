// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include <array>
#include <cstdint>

using namespace ttnn;

int main(int /*argc*/, char** /*argv*/) {
    float scalar = 3.0f;

    auto device = open_mesh_device(/*device_id=*/0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    uint32_t h = 32;
    uint32_t w = 64;
    std::array<uint32_t, 2> dimensions = {h, w};
    Shape shape(dimensions);

    const auto input_tensor = zeros(shape, DataType::BFLOAT16, TILE_LAYOUT, *device);
    const auto output_tensor = input_tensor + scalar;

    return 0;
}
