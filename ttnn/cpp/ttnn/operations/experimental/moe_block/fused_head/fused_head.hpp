// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.experimental.moe_block.fused_head: the fused head: x all-gather, shared expert + reduce-scatter, router, routing exchange + planner, dispatch prep, latent (one program a chip).
// Placeholder of the shared base; the fused_head agent owns this directory.

#pragma once

namespace ttnn::operations::experimental::moe_block::fused_head {}  // namespace ttnn::operations::experimental::moe_block::fused_head
