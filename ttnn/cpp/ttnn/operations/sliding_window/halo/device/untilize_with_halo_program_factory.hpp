// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation_types.hpp"

namespace ttnn::prim {

struct UntilizeWithHaloProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle src_cb{};
        tt::tt_metal::CBHandle out_cb{};
        tt::tt_metal::CBHandle padding_config_cb0{};
        tt::tt_metal::CBHandle padding_config_cb1{};
        tt::tt_metal::CBHandle gather_config_cb0{};
        tt::tt_metal::CBHandle gather_config_cb1{};
        tt::tt_metal::DeviceStorage padding_config_storage0;
        tt::tt_metal::DeviceStorage padding_config_storage1;
        tt::tt_metal::DeviceStorage gather_config_storage0;
        tt::tt_metal::DeviceStorage gather_config_storage1;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const HaloParams& operation_attributes, const Tensor& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const HaloParams& operation_attributes,
        const Tensor& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
