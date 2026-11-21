// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/quasar/permute/device/permute_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

namespace {

// DM threads per cluster: all six harts Quasar leaves to kernels (DM0 / DM1 are reserved), or the count in
// TTNN_QSR_PERMUTE_THREADS (experiments only; validated by the device operation).
uint32_t permute_threads() {
    uint32_t threads = ttnn::prim::qsr::kPermuteMaxThreads;
    if (const char* env = std::getenv("TTNN_QSR_PERMUTE_THREADS")) {
        unsigned value = 0;
        int consumed = 0;
        const bool parsed = std::strchr(env, '-') == nullptr && std::sscanf(env, "%u%n", &value, &consumed) == 1 &&
                            static_cast<size_t>(consumed) == std::strlen(env);
        TT_FATAL(parsed, "TTNN_QSR_PERMUTE_THREADS must be a thread count, got '{}'", env);
        threads = value;
    }
    return threads;
}

}  // namespace

Tensor permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
    const auto& shape = input_tensor.logical_shape();
    const size_t rank = shape.rank();
    TT_FATAL(
        dims.size() == rank,
        "Quasar permute: {} dims given for a rank {} tensor (logical shape {})",
        dims.size(),
        rank,
        shape);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Quasar permute: the input must be on device");

    ttsl::SmallVector<uint32_t> normalized(rank);
    bool identity = true;
    for (size_t k = 0; k < rank; ++k) {
        const int64_t dim = dims[k] < 0 ? dims[k] + static_cast<int64_t>(rank) : dims[k];
        TT_FATAL(
            dim >= 0 && dim < static_cast<int64_t>(rank),
            "Quasar permute: dims[{}] = {} is out of range for rank {}",
            k,
            dims[k],
            rank);
        normalized[k] = static_cast<uint32_t>(dim);
        identity = identity && normalized[k] == k;
    }

    const MemoryConfig output_memory_config = memory_config.value_or(input_tensor.memory_config());
    if (identity && output_memory_config == input_tensor.memory_config()) {
        return input_tensor;
    }
    return ttnn::prim::qsr::permute(input_tensor, normalized, output_memory_config, pad_value, permute_threads());
}

}  // namespace ttnn::operations::experimental::quasar
