// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Re-exported from combine_fabric2d. The overlapped fork's geometry, plans and constants are the same
// ones; the fields only it fills (the collector's counts, the id table, the ready count) sit unused in
// the standalone op rather than in a second definition that would differ by ODR.

#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_kernel_interface.hpp"

#ifndef KERNEL_BUILD

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::BATCH_COUNT_PAGE_BYTES;
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

namespace hyb_cmbf2d {
namespace op = ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine;
}  // namespace hyb_cmbf2d

#endif

namespace hyb_cmbf2d {

// Chunk layout: one definition for both ops, re-exported through cmbf2d.
using ::cmbf2d::CHUNK_WORDS;
using ::cmbf2d::ChunkDescriptor;

using ::cmbf2d::align_control;
using ::cmbf2d::ASSIGNMENT_WORDS;
using ::cmbf2d::BATCH;
using ::cmbf2d::CMD_END;
using ::cmbf2d::CMD_FINAL_WRITE;
using ::cmbf2d::CMD_FORWARD;
using ::cmbf2d::CMD_FORWARD_END;
using ::cmbf2d::expert_table_row_stride;
using ::cmbf2d::FORWARDING_METADATA_SIZE;
using ::cmbf2d::FWD_EXTRA_BYTES;
using ::cmbf2d::FwdMetadata;
using ::cmbf2d::META_PAD_STRIDE;
using ::cmbf2d::META_PREFETCH;
using ::cmbf2d::NUM_L1_SLOTS;
using ::cmbf2d::SCHED_FWD;
using ::cmbf2d::UNT_BATCH_ROWS;
using ::cmbf2d::UNT_CB_BATCHES;
using ::cmbf2d::UNT_CB_IN;
using ::cmbf2d::UNT_CB_OUT;
using ::cmbf2d::UNT_PEER_WORDS;
using ::cmbf2d::UNT_RING_BATCHES;

}  // namespace hyb_cmbf2d
