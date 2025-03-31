// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks transpose_wh_multi_core(const Tensor& a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks transpose_wh_multi_core_sharded(const Tensor& a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks transpose_wh_multi_core_sharded_rm(const Tensor& a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks transpose_hc_multi_core(
    const Tensor& a, Tensor& output, const std::optional<float>& pad_value);
tt::tt_metal::operation::ProgramWithCallbacks transpose_hc_multi_core_tiled_interleaved(
    const Tensor& a, Tensor& output, const std::optional<float>& pad_value);
tt::tt_metal::operation::ProgramWithCallbacks transpose_hc_multi_core_sharded(const Tensor& a, Tensor& output);
tt::tt_metal::operation::ProgramWithCallbacks transpose_cn_multi_core(const Tensor& a, Tensor& output);

}  // namespace ttnn::operations::data_movement::detail
