// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_program_factory_utils.hpp"

namespace ttnn::operations::binary_ng::program {

bool is_fp32_dest_acc_en(
    const tt::DataFormat a_data_format, const tt::DataFormat b_data_format, const tt::DataFormat c_data_format) {
    return c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
           c_data_format == tt::DataFormat::Float32 ||
           (a_data_format == tt::DataFormat::Float32 && b_data_format == tt::DataFormat::Float32) ||
           (a_data_format == tt::DataFormat::Int32 && b_data_format == tt::DataFormat::Int32) ||
           (a_data_format == tt::DataFormat::UInt32 && b_data_format == tt::DataFormat::UInt32);
}

}  // namespace ttnn::operations::binary_ng::program
