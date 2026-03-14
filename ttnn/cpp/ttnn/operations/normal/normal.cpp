// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "normal.hpp"

#include <cmath>
#include <random>

#include "ttnn/operations/rand/rand.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {

Tensor normal(
    const ttnn::Shape& shape,
    MeshDevice& device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config,
    float mean,
    float stddev,
    std::optional<uint32_t> seed) {
    TT_FATAL(stddev >= 0.0f, "[ttnn::normal] stddev must be non-negative, got {}.", stddev);

    // Box-Muller transform: z = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
    // u1 in (eps, 1] to avoid log(0); u2 in [0, 1)
    const uint32_t effective_seed = seed.value_or(std::random_device{}());
    auto u1 = ttnn::rand(shape, device, DataType::FLOAT32, Layout::TILE, memory_config, 1e-6f, 1.0f, effective_seed);
    auto u2 = ttnn::rand(shape, device, DataType::FLOAT32, Layout::TILE, memory_config, 0.0f, 1.0f, effective_seed + 1);

    auto log_u1 = ttnn::log(u1, false, memory_config);
    auto neg2log = ttnn::multiply(log_u1, -2.0f, std::nullopt, memory_config);
    auto r = ttnn::sqrt(neg2log, false, memory_config);

    auto two_pi_u2 = ttnn::multiply(u2, static_cast<float>(2.0 * M_PI), std::nullopt, memory_config);
    auto cos_u2 = ttnn::cos(two_pi_u2, memory_config);

    auto z = ttnn::multiply(r, cos_u2, std::nullopt, memory_config);

    if (stddev != 1.0f) {
        z = ttnn::multiply(z, stddev, std::nullopt, memory_config);
    }
    if (mean != 0.0f) {
        z = ttnn::add(z, mean, std::nullopt, memory_config);
    }

    if (dtype != DataType::FLOAT32) {
        z = ttnn::typecast(z, dtype);
    }
    if (layout != Layout::TILE) {
        z = ttnn::to_layout(z, layout);
    }
    return z;
}

}  // namespace ttnn
