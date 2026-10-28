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
    // Interleaved input is not split across cores, so each extra core would add one more increment
    // to every value. Run it on one core.
    if (!input.is_sharded()) {
        all_cores = CoreRangeSet(CoreRange(corerange_to_cores(all_cores, 1, true).front()));
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
    const ScratchpadSpecName IN0_SCRATCH{"in0"};
    const KernelSpecName READER{"reader"};

    // The kernel always uses the input tensor:
    //  - interleaved (DRAM or L1): copied in and out over the NoC (NEEDS_NOC_COPY).
    //  - sharded (L1): the DFB borrows the shard's memory.
    const bool needs_noc_copy = !input.is_sharded();

    // ---- Reader kernel ----
    // Legacy CTA slots 0 (cb index) and 1 (src_is_dram) are gone: the CB index becomes the
    // scratchpad / DFB binding below, and src_is_dram is replaced by the NEEDS_NOC_COPY define
    // (set for interleaved input; it gates the TensorAccessor binding). The Buffer* RTA and the
    // TensorAccessorArgs plumbing are replaced by the tensor binding.
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp",
        .compile_time_args =
            {
                {"stick_size", aligned_input_page_size},
                {"W", W},
                {"H", H},
                {"skip_negative_entries", operation_attributes.skip_negative_entries},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // ---- Working buffer (legacy c_0) ----
    ProgramSpec spec;
    if (needs_noc_copy) {
        // Interleaved: the reader copies each stick into a private scratchpad over the NoC,
        // increments it there and writes it back; it never uses the buffer as a FIFO. (Formerly a
        // self-loop DFB. Quasar rejects a DFB that one data-movement kernel both produces and
        // consumes, and rejects UInt32 as a DFB data format; a scratchpad has no data format.)
        spec.scratchpads = {ScratchpadSpec{.unique_id = IN0_SCRATCH, .size_per_node = aligned_input_page_size}};
        reader.scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = IN0_SCRATCH, .accessor_name = "in0"}};
        reader.tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}};
        reader.compiler_options.defines = {{"NEEDS_NOC_COPY", "1"}};
    } else {
        // Sharded: borrow the DFB from the input buffer so the framework re-applies the shard
        // address on a program-cache hit; the reader increments the shard in place. The reader is
        // the DFB's only toucher, so it binds it as both PRODUCER and CONSUMER (a self-loop, which
        // Quasar rejects).
        DataflowBufferSpec in0_dfb{
            .unique_id = IN0,
            .entry_size = aligned_input_page_size,
            .num_entries = 1,
            .data_format_metadata = input_cb_data_format,
        };
        in0_dfb.borrowed_from = INPUT;
        spec.dataflow_buffers = {std::move(in0_dfb)};
        reader.dfb_bindings = {
            DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
        };
    }

    // ---- Assemble the spec ----
    spec.name = "plusone";
    spec.kernels = {std::move(reader)};
    spec.tensor_parameters = {TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()}};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER}, .target_nodes = all_cores}};

    // ---- Run args ----
    // The reader has no runtime args (the Buffer* address is now carried by the tensor
    // binding); provide an empty per-kernel entry to satisfy the "a KernelRunArgs for
    // every kernel" contract.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{.kernel = READER}};
    run_args.tensor_args.insert({INPUT, input_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
