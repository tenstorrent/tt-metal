// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_program_factory_general.hpp"

namespace ttnn::prim {

struct SoftmaxProgramFactoryGeneralHSmall : SoftmaxProgramFactoryGeneral {
    static cached_program_t create(const SoftmaxParams&, const SoftmaxInputs&, Tensor&);
};

}  // namespace ttnn::prim
