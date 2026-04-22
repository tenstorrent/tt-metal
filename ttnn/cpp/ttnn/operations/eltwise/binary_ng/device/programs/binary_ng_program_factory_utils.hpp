// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::operations::binary_ng::program {
bool is_fp32_dest_acc_en(tt::DataFormat a_data_format, tt::DataFormat b_data_format, tt::DataFormat c_data_format);

}  // namespace ttnn::operations::binary_ng::program
