// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::data_movement {

// Selects which host implementation a data_movement op dispatches to:
// Auto (default) routes in-scope, non-demoted cases to codegen and everything else to native;
// Native always uses the existing native prim; Codegen always uses the codegen prim.
enum class ImplementationSelector { Auto, Native, Codegen };

}  // namespace ttnn::operations::data_movement
