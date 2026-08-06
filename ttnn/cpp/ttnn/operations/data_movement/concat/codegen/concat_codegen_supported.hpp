// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::data_movement::concat_codegen {

// Placeholder correctness gate for ConcatCodegen; unconditionally false until
// phase 4a fills in the real per-input/dim/dtype/layout predicate transcribed
// from ops/concat/spec.py.
bool supported_by_codegen();

}  // namespace ttnn::operations::data_movement::concat_codegen
