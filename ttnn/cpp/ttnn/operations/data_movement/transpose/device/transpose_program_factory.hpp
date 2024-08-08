// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_wh_multi_core_sharded_rm(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_hc_multi_core_sharded(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks transpose_cn_multi_core(const Tensor &a, Tensor &output);

} // namespace ttnn::operations::reduction::detail
