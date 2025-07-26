// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "where_device_operation.hpp"
#include "ttnn/tensor/types.hpp"

#include <optional>
#include <string>

namespace ttnn::operations::ternary {

enum class KernelName {
    ReaderNoBcastTTT,
    ReaderNoBcastTST,
    ReaderNoBcastTTS,
    ReaderNoBcastTSS,
    WriterNoBcastTTT,
    WriterNoBcastTST,
    WriterNoBcastTTS,
    WriterNoBcastTSS,
    ComputeNoBcastTTT,
    ComputeNoBcastTST,
    ComputeNoBcastTTS,
    ComputeNoBcastTSS,
};

struct WhereKernelConfig {
    WhereKernelConfig(WhereVariant where_variant);

    KernelName reader_kernel;
    KernelName compute_kernel;
    KernelName writer_kernel;
};

std::string get_kernel_file_path(KernelName kernel_name);

uint32_t pack_scalar_runtime_arg(float scalar, DataType dtype, bool is_quant_op);

std::map<std::string, std::string> make_dataflow_defines(DataType dtype, DataType b_dtype);

}  // namespace ttnn::operations::ternary
