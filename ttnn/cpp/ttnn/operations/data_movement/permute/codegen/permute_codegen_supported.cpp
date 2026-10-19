// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_supported.hpp"

namespace ttnn::operations::data_movement::permute_codegen {

bool supported_by_codegen(const Tensor& /*input*/, const ttsl::SmallVector<uint32_t>& /*dims*/) { return false; }

bool is_demoted(const Tensor& /*input*/, const ttsl::SmallVector<uint32_t>& /*dims*/) { return false; }

}  // namespace ttnn::operations::data_movement::permute_codegen
