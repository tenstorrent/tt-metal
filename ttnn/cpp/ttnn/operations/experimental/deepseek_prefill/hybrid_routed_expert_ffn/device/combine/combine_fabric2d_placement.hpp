// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Re-exported from combine_fabric2d: where the streams, untilizers and collector sit is decided the same
// way for both ops. The overlapped caller asks for the collector with decide_placement's with_collector.

#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_placement.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

namespace cmbf2d_host = ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d;

using cmbf2d_host::CollectorPlacement;
using cmbf2d_host::DevicePlacement;
using cmbf2d_host::MeshPlacement;
using cmbf2d_host::StreamId;
using cmbf2d_host::StreamPlacement;
using cmbf2d_host::StreamPlacements;
using cmbf2d_host::UntilizerGroups;
using cmbf2d_host::UntilizerPlacement;

using cmbf2d_host::DEFAULT_UNTILIZERS_PER_GROUP;
using cmbf2d_host::MAX_UNTILIZERS_PER_GROUP;
using cmbf2d_host::UNTILIZER_GROUPS;

using cmbf2d_host::decide_placement;
using cmbf2d_host::make_stream_id;
using cmbf2d_host::stream_count;
using cmbf2d_host::stream_is_cw;
using cmbf2d_host::untilizer_group_of;
using cmbf2d_host::untilizers_per_group;

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
