// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace topk_common_args {
// These values are shared by every core running the corresponding kernel.
enum Reader { input_address = 0, metadata_address = 1, search_length = 2, input_row_bytes = 3 };
enum Compute { compute_search_length = 0 };
enum Writer { output_address = 0 };
}  // namespace topk_common_args
