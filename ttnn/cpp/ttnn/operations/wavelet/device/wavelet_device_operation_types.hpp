// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/operations/wavelet/common/boundary.hpp"
#include "ttnn/operations/wavelet/generated/schemes/registry.hpp"
#include "ttnn/operations/wavelet/wavelet_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

using Lwt1DOutputSpecs = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
using Lwt1DOutputs = std::tuple<Tensor, Tensor>;

using Lwt2DOutputSpecs =
    std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
// Forward 2D subbands are ordered as (LL, LH, HL, HH).
using Lwt2DOutputs = std::tuple<Tensor, Tensor, Tensor, Tensor>;

struct Lwt1DParams {
    operations::wavelet::SchemeId scheme_id;
    operations::wavelet::BoundaryMode boundary_mode;
    uint32_t available_l1_bytes;
    MemoryConfig output_memory_config;
};

struct Lwt1DInputs {
    const Tensor& input;
    const std::optional<Lwt1DOutputs>& preallocated_outputs;
};

struct Ilwt1DParams {
    operations::wavelet::SchemeId scheme_id;
    operations::wavelet::BoundaryMode boundary_mode;
    uint32_t original_length;
    uint32_t available_l1_bytes;
    MemoryConfig output_memory_config;
};

struct Ilwt1DInputs {
    const Tensor& approximation;
    const Tensor& detail;
    const std::optional<Tensor>& preallocated_output;
};

struct Lwt2DParams {
    operations::wavelet::SchemeId scheme_id;
    operations::wavelet::BoundaryMode boundary_mode;
    uint32_t available_l1_bytes;
    MemoryConfig output_memory_config;
};

struct Lwt2DInputs {
    const Tensor& input;
    const std::optional<std::array<Tensor, 4>>& preallocated_outputs;
};

struct Ilwt2DParams {
    operations::wavelet::SchemeId scheme_id;
    operations::wavelet::BoundaryMode boundary_mode;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t available_l1_bytes;
    MemoryConfig output_memory_config;
};

struct Ilwt2DInputs {
    const Tensor& ll;
    const Tensor& lh;
    const Tensor& hl;
    const Tensor& hh;
    const std::optional<Tensor>& preallocated_output;
};

}  // namespace ttnn::prim
