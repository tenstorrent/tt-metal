// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The overlapped fork shares combine_fabric2d's parameter and input structs rather than redeclaring them:
// both ops link into one binary, so a second definition differing by a field would be an ODR violation
// rather than a build error. The overlap-only fields live in that one definition, defaulted.

#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dInputs;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dParams;

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
