// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "plusone_program_factory.hpp"
#include "plusone_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts PlusOneProgramFactory::create_program_artifacts(
    const PlusoneParams& operation_attributes, const Tensor& input, Tensor& /*tensor_return_value*/) {
    const MeshTensor& input_mesh_tensor = input.mesh_tensor();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.element_size();

    CoreRangeSet all_cores = CoreRangeSet(std::vector{CoreRange({0, 0}, {0, 0})});
    if (operation_attributes.sub_core_grids.has_value()) {
        all_cores = operation_attributes.sub_core_grids.value();
    }

    const auto& input_shape = input.padded_shape();
    uint32_t W = input_shape[-1];
    uint32_t H = 1;
    if (!input.is_sharded() && input_shape.size() > 1) {
        for (uint32_t i = 0; i < input_shape.size() - 1; ++i) {
            H *= input_shape[i];
        }
    }

    uint32_t num_input_units = W;
    auto* src_buffer = input.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const uint32_t page_alignment =
        src_is_dram ? tt::tt_metal::hal::get_dram_alignment() : tt::tt_metal::hal::get_l1_alignment();
    uint32_t aligned_input_page_size = tt::align(num_input_units * input_unit_size, page_alignment);

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const DFBSpecName IN0{"in0"};
    const KernelSpecName READER{"reader"};

    // The input tensor is a Program-scope resource when the kernel touches it:
    //  - DRAM (interleaved): via TensorAccessor (accessor path, gated by SRC0_IS_DRAM).
    //  - sharded (L1): as the DFB's borrowed backing memory.
    // For an L1-interleaved input (neither DRAM nor sharded — the pre-existing
    // "unhandled" anomaly) the kernel operates on uninitialized L1 scratch and never
    // references the input, so no TensorParameter is declared. Behavior is preserved.
    const bool needs_tensor_param = src_is_dram || input.is_sharded();

    // ---- Dataflow buffer (legacy c_0) ----
    // When the input is sharded, borrow the DFB from the input buffer so the framework
    // re-applies the globally-allocated address on a program-cache hit. Otherwise the
    // DFB is plain L1 scratch. The reader uses it purely as an address source (raw
    // get_write_ptr, no FIFO ops), so it is a single-toucher sync-free CB → self-loop.
    DataflowBufferSpec in0_dfb{
        .unique_id = IN0,
        .entry_size = aligned_input_page_size,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    };
    if (input.is_sharded()) {
        in0_dfb.borrowed_from = INPUT;
    }

    // ---- Reader kernel ----
    // Self-loop: the sole toucher binds the DFB as both PRODUCER and CONSUMER (one
    // accessor name). Legacy CTA slots 0 (cb index) and 1 (src_is_dram) are gone: the
    // CB index becomes the DFB binding, and src_is_dram becomes the SRC0_IS_DRAM define
    // (it gates the conditional TensorAccessor binding). The Buffer* RTA and the
    // TensorAccessorArgs plumbing are replaced by the tensor binding.
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .compile_time_args =
            {
                {"stick_size", aligned_input_page_size},
                {"W", W},
                {"H", H},
                {"skip_negative_entries", operation_attributes.skip_negative_entries},
            },
        .hw_config = ttnn::create_reader_datamovement_config(input_mesh_tensor.device().arch()),
    };
    if (src_is_dram) {
        // Accessor path (DRAM): bind the input tensor and enable the NoC transfers.
        reader.tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}};
        reader.compiler_options.defines = {{"SRC0_IS_DRAM", "1"}};
    }

    // ---- Assemble the spec ----
    ProgramSpec spec;
    spec.name = "plusone";
    spec.kernels = {std::move(reader)};
    spec.dataflow_buffers = {std::move(in0_dfb)};
    if (needs_tensor_param) {
        spec.tensor_parameters = {TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()}};
    }
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER}, .target_nodes = all_cores}};

    // ---- Run args ----
    // The reader has no runtime args (the Buffer* address is now carried by the tensor
    // binding); provide an empty per-kernel entry to satisfy the "a KernelRunArgs for
    // every kernel" contract.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{.kernel = READER}};
    if (needs_tensor_param) {
        run_args.tensor_args.insert({INPUT, input_mesh_tensor});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
