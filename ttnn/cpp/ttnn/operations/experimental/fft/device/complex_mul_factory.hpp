// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ComplexMulFactory — ProgramDescriptor program factory for the
// complex_mul op (ROW_MAJOR elementwise complex multiply of two
// same-shape complex tensors).  Reuses apply_twiddles_compute and
// apply_twiddles_writer verbatim; the only new kernel is the reader
// (complex_mul_reader.cpp) which loads B from DRAM rather than
// generating it on-the-fly from a delta lookup.

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "complex_mul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct ComplexMulFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ComplexMulParams& operation_attributes,
        const ComplexMulTensorArgs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
