// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_program_factory_utils.hpp"

namespace ttnn::operations::binary_ng::program {

bool is_fp32_dest_acc_en(
    const tt::DataFormat a_data_format, const tt::DataFormat b_data_format, const tt::DataFormat c_data_format) {
    // fp32 dest accumulation must be enabled whenever any input or output is fp32, otherwise
    // loading fp32 tiles into a DST configured for bf16 produces tile-aligned corruption
    // for broadcast multiply (issue 43196).
    return c_data_format == tt::DataFormat::UInt32 || c_data_format == tt::DataFormat::Int32 ||
           c_data_format == tt::DataFormat::Float32 || a_data_format == tt::DataFormat::Float32 ||
           b_data_format == tt::DataFormat::Float32 ||
           (a_data_format == tt::DataFormat::Int32 && b_data_format == tt::DataFormat::Int32) ||
           (a_data_format == tt::DataFormat::UInt32 && b_data_format == tt::DataFormat::UInt32);
}

}  // namespace ttnn::operations::binary_ng::program
