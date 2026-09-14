// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdn_spec_tloop_proto_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct GdnSpecTloopProtoProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const GdnSpecTloopProtoParams& attrs, const GdnSpecTloopProtoInputs& in, Tensor& output);
};

}  // namespace ttnn::experimental::prim
