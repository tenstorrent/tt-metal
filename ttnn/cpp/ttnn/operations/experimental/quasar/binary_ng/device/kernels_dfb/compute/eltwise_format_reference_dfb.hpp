// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../kernels/compute/eltwise_utils_common.hpp"

// Named-buffer counterpart of QSR_BINARY_SRCA_FORMAT_CB. Use the factory's
// bindings rather than assuming numeric operand IDs. Broadcast llk_post has
// pre_lhs's format when it replaces LHS; optimized broadcast admission excludes
// format-changing intermediates. Startup and SFPU chunk exit preserve this
// physical-LHS format, including tensor-scalar kernels.
// An inactive post_lhs binding does not exist: #if must discard the name entirely.
#if HAS_ACTIVATIONS(LHS)
#define QSR_BINARY_SRCA_FORMAT_DFB static_cast<uint32_t>(dfb::post_lhs)
#else
#define QSR_BINARY_SRCA_FORMAT_DFB static_cast<uint32_t>(dfb::pre_lhs)
#endif
