// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental::ccl {

// Shared validation for ring/line reduce-scatter ops: checks topology, alignment,
// storage, num_links, rank, dim, scatter-dim divisibility, memory layout, and
// optional pre-allocated output tensor constraints.
void reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    ttnn::ccl::Topology topology,
    uint32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor);

// Shared validation for an optional pre-allocated intermediate tensor: checks storage,
// layout/dtype/page_config compatibility with input, optional mem config match, and
// that block-sharded intermediates use L1. The caller is responsible for any additional
// shape constraints (e.g. single-batch requirement for strided RS).
void validate_intermediate_tensor(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& intermediate_tensor,
    const std::optional<ttnn::MemoryConfig>& optional_intermediate_mem_config);

}  // namespace ttnn::experimental::ccl
