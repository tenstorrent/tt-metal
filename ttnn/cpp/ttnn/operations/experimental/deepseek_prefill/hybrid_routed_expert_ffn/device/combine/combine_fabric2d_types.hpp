// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The overlapped fork shares combine_fabric2d's parameter and input structs rather than redeclaring them:
// both ops link into one binary, so a second definition differing by a field would be an ODR violation
// rather than a build error. The overlap-only fields live in that one definition, defaulted.

#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_types.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::BATCH_COUNT_PAGE_BYTES;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dInputs;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dParams;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::dispatched_is_tiled;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::DramBuffers;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::HandshakePeer;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::KernelPlan;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::L1Layout;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::my_dg_index;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::num_dispatch_groups;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::num_routed_experts;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::ReaderUntilizers;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::ring_extent;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::tile_size_bytes;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::tiles_per_token_row;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::token_size_bytes;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::untilize_block_tiles;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::UNTILIZE_MAX_BLOCK_TILES;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::UntilizerPlan;

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
