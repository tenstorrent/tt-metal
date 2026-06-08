// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TODO(nuked-op pool): placeholder translation unit.
// The pool ops (generic max/avg pool, upsample, grid_sample, rotate) were removed
// for eval. The shared device-kernel headers under device/kernels/ are intentionally
// kept (used by data_movement/fold, experimental/cnn, conv3d, padded_slice, ...).
// This file exists only so the ttnn_op_pool library has at least one source and
// links cleanly. Delete it once the pool ops are recreated and their real sources
// are restored to sources.cmake.

namespace ttnn::operations::pool {}  // namespace ttnn::operations::pool
