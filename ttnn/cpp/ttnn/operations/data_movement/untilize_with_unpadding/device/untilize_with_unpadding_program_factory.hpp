// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks untilize_with_unpadding_single_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

operation::ProgramWithCallbacks untilize_with_unpadding_multi_core_interleaved(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

// This purely supports input block shard -> output interleaved for now
operation::ProgramWithCallbacks untilize_with_unpadding_multi_core_sharded(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

operation::ProgramWithCallbacks untilize_with_unpadding_multi_core(
    const Tensor& a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en);

}  // namespace ttnn::operations::data_movement::detail
