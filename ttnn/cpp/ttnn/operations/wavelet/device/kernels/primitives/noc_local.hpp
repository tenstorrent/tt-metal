// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

namespace ttnn::operations::wavelet::kernels::primitives {

struct LocalNocCoordinates {
    uint32_t noc_x;
    uint32_t noc_y;
};

[[nodiscard]] inline __attribute__((always_inline)) LocalNocCoordinates local_noc_coordinates(const Noc& noc) {
    const uint8_t noc_id = noc.get_noc_id();
    return {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id]};
}

[[nodiscard]] inline __attribute__((always_inline)) auto local_noc_source(
    const LocalNocCoordinates& coordinates, const uint32_t address) -> noc_traits_t<UnicastEndpoint>::src_args_type {
    return {.noc_x = coordinates.noc_x, .noc_y = coordinates.noc_y, .addr = address};
}

[[nodiscard]] inline __attribute__((always_inline)) auto local_noc_source(const Noc& noc, const uint32_t address)
    -> noc_traits_t<UnicastEndpoint>::src_args_type {
    return local_noc_source(local_noc_coordinates(noc), address);
}

[[nodiscard]] inline __attribute__((always_inline)) auto local_noc_destination(
    const LocalNocCoordinates& coordinates, const uint32_t address) -> noc_traits_t<UnicastEndpoint>::dst_args_type {
    return {.noc_x = coordinates.noc_x, .noc_y = coordinates.noc_y, .addr = address};
}

[[nodiscard]] inline __attribute__((always_inline)) auto local_noc_destination(const Noc& noc, const uint32_t address)
    -> noc_traits_t<UnicastEndpoint>::dst_args_type {
    return local_noc_destination(local_noc_coordinates(noc), address);
}

}  // namespace ttnn::operations::wavelet::kernels::primitives
