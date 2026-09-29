// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Re-exported from combine_fabric2d: which chip owes which run to whom is a property of the ring, not of
// the op that walks it, and both ops now name chunks with the same ChunkDescriptor.

#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_assignments.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

namespace cmbf2d_host = ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;

using cmbf2d_host::Assignment;
using cmbf2d_host::forwarding_chunks;
using cmbf2d_host::generate_assignments;
using cmbf2d_host::relay_chunks_per_stream;

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
