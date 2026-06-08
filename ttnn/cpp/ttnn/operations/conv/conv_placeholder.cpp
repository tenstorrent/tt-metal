// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TODO(nuked-op conv2d): placeholder translation unit.
// The conv2d / conv1d / conv_transpose2d operations were removed for eval.
// This file exists only so the ttnn_op_conv library has at least one source
// and links cleanly. Delete it once conv2d (and friends) are recreated and
// their real sources are restored to sources.cmake.

namespace ttnn::operations::conv {}  // namespace ttnn::operations::conv
