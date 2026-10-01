// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/arch.hpp>

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

// The checks a program config must pass for a matmul's inputs, as pure functions of their specs: each returns
// the first rule broken, or empty. The device op throws what they return; the program config selection uses
// them to keep only configs the device op accepts.
namespace ttnn::prim {

// What the checks read of the device
struct DeviceDesc {
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0;
    CoreCoord grid;                                  // compute_with_storage_grid_size
    bool has_sub_devices = false;                    // the device has sub-device ids
    std::optional<CoreRangeSet> sub_device_workers;  // the worker cores of the matmul's sub-device, if it has one
};

// A matmul call as specs: its inputs (A, then B and, for the multi-tensor 1D path, further weights), the bias,
// its normalized attributes (create_matmul_attributes) and the device
struct MatmulSpecs {
    std::vector<tt::tt_metal::TensorSpec> inputs;
    std::optional<tt::tt_metal::TensorSpec> bias;
    MatmulParams attributes;
    DeviceDesc device;

    const tt::tt_metal::TensorSpec& a() const { return inputs.at(0); }
    const tt::tt_metal::TensorSpec& b() const { return inputs.at(1); }
};

MatmulSpecs matmul_specs(
    const std::vector<Tensor>& input_tensors, const std::optional<const Tensor>& bias, const MatmulParams& attributes);

// The first rule `program_config` (normalized: normalize_program_config) breaks for this matmul, or empty
std::string program_config_error(
    const MatmulSpecs& specs, const operations::matmul::MatmulProgramConfig& program_config);

}  // namespace ttnn::prim
