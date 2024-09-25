// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include "tt_metal/common/core_coord.h"
#include "eth_l1_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include <sstream>
#include <type_traits>

#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul/device/all_gather_matmul_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt::constants;

namespace ttnn {

using namespace experimental::ccl;
using Tensors = std::vector<Tensor>;

// Used to hold the return values for setup_datacopy
struct DatacopyParams {
    std::vector<CoreCoord> datacopy_cores_noc;
    std::vector<uint32_t> datacopy_signal_semaphore_ids;
    std::optional<operation::OverrideRuntimeArgumentsCallback<Tensors>> datacopy_override_runtime_arguments_callback;
};

DatacopyParams setup_datacopy(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& all_gather_output_tensor,
    Tensor& datacopy_output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,

    CoreCoord datacopy_core_coord,
    const ttnn::experimental::ccl::MatmulFusedOpSignaler& matmul_fused_op_signaler
) {

    std::size_t num_edm_buffers_per_channel = 2;
    if (user_defined_num_buffers_per_channel.has_value()) {
        // Override with user defined value
        num_edm_buffers_per_channel = user_defined_num_buffers_per_channel.value();
    }
    const auto& device = input_tensor.device();
    auto const& all_gather_config = ttnn::AllGatherConfig(input_tensor, all_gather_output_tensor, dim, ring_size, num_links, topology, num_edm_buffers_per_channel, true, user_defined_num_workers);
    const uint32_t num_transfers = 4;

    auto tensor_slicer = ttnn::ccl::InterleavedRingAllGatherTensorSlicer (
        input_tensor,
        all_gather_output_tensor,
        dim,
        ring_index
    );


    // Select cores for datacopy (single core for now)
    CoreRangeSet datacopy_workers = CoreRangeSet({CoreRange(datacopy_core_coord)});
    std::vector<CoreCoord> all_datacopy_cores = corerange_to_cores(datacopy_workers, std::nullopt, true);
    std::vector<CoreCoord> all_datacopy_cores_noc;
    for (auto core : all_datacopy_cores) {
        all_datacopy_cores_noc.push_back(device->worker_core_from_logical_core(core));
    }

    // Setup semaphores used to signal datacopy. TODO: instead of datacopy, this should be matmul cores
    // Dir0: first half of all gather (clockwise), Dir1: second half of all gather (counter-clockwise)
    auto datacopy_signal_semaphore_id_dir0 = CreateSemaphore(program, datacopy_workers, 0);
    auto datacopy_signal_semaphore_id_dir1 = CreateSemaphore(program, datacopy_workers, 0);

    // Setup args for the kernel
    const uint32_t tile_size = 32;
    const uint32_t page_size = all_gather_output_tensor.buffer()->page_size();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(all_gather_output_tensor.get_dtype());


    auto all_gather_output_buffer = all_gather_output_tensor.buffer();
    auto datacopy_output_buffer = datacopy_output_tensor.buffer();

    bool all_gather_output_is_dram = all_gather_output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool datacopy_output_is_dram = datacopy_output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t last_output_page_offset = (ring_size - 1) * tensor_slicer.output_page_offset;
    uint32_t num_rows = input_tensor.get_legacy_shape()[2] / tile_size ;
    bool is_clockwise_dir = true; // Specifically for the first half of the all gather

    uint32_t datacopy_buffer_size = 200;

    // Compile time args
    std::vector<uint32_t> datacopy_ct_args = {
        static_cast<uint32_t>(all_gather_output_is_dram),
        static_cast<uint32_t>(datacopy_output_is_dram),
        static_cast<uint32_t>(num_transfers),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(ring_index),
        static_cast<uint32_t>(ring_size),
        static_cast<uint32_t>(all_gather_output_tensor.get_legacy_shape()[3] / tile_size), // tesnor width
        static_cast<uint32_t>(all_gather_output_tensor.get_legacy_shape()[2] / tile_size), // tensor height
        static_cast<uint32_t>(tensor_slicer.num_cols), // tensor slice width in tiles
        static_cast<uint32_t>(num_rows), // tnesor slice height in tiles
        static_cast<uint32_t>(tensor_slicer.output_page_offset),
        static_cast<uint32_t>(last_output_page_offset),
        static_cast<bool>(is_clockwise_dir),
        static_cast<uint32_t>(datacopy_signal_semaphore_id_dir0),
        static_cast<uint32_t>(datacopy_signal_semaphore_id_dir1),
        static_cast<uint32_t>(datacopy_buffer_size),
        static_cast<uint32_t>(matmul_fused_op_signaler.num_fused_op_cores_to_signal)
    };

    uint32_t cb_id_in0 = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_in0_config =
        tt::tt_metal::CircularBufferConfig(
            page_size * datacopy_buffer_size, {{cb_id_in0, cb_data_format}})
            .set_page_size(cb_id_in0, page_size);
    auto cb_input = tt::tt_metal::CreateCircularBuffer(program, datacopy_workers, cb_in0_config);

    // Runtime args
    std::vector<uint32_t> datacopy_rt_args = {
        static_cast<uint32_t>(all_gather_output_buffer->address()),
        static_cast<uint32_t>(datacopy_output_buffer->address()),
        static_cast<uint32_t>(matmul_fused_op_signaler.fused_op_receiver_signal_semaphores[0]),
        static_cast<uint32_t>(matmul_fused_op_signaler.fused_op_receiver_signal_semaphores[1]),
    };

    // Push the matmul core NOC coordinates
    for (auto coord : matmul_fused_op_signaler.fused_op_receiver_cores_noc) {
        datacopy_rt_args.push_back(static_cast<uint32_t>(coord.x));
        datacopy_rt_args.push_back(static_cast<uint32_t>(coord.y));
    }

    std::map<string, string> kernel_defines = {
        {"TILED_LAYOUT", "1"},
        {"INTERLEAVED_MEM_LAYOUT", "1"}
    };


    // Create the kernel
    tt::tt_metal::KernelHandle datacopy_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul/device/kernels/datacopy.cpp",
        datacopy_workers,
        tt::tt_metal::WriterDataMovementConfig(datacopy_ct_args, kernel_defines));

    // Set runtime args
    tt::tt_metal::SetRuntimeArgs(
        program,
        datacopy_kernel_id,
        datacopy_workers,
        datacopy_rt_args
    );

    auto override_runtime_arguments_callback = [datacopy_kernel_id, all_datacopy_cores] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto datacopy_output_buffer = output_tensors[2].buffer();
        auto all_gather_output_buffer = output_tensors[0].buffer();

        auto &cached_args = GetRuntimeArgs(program, datacopy_kernel_id);

        for (auto core : all_datacopy_cores) {
            auto &cached_rt_args = cached_args.at(core.x).at(core.y);

            cached_rt_args[0] = static_cast<uint32_t>(all_gather_output_buffer->address());
            cached_rt_args[1] = static_cast<uint32_t>(datacopy_output_buffer->address());
        }

    };

    // Return the core coordinates and semaphore address
    return {
        .datacopy_cores_noc = all_datacopy_cores_noc,
        .datacopy_signal_semaphore_ids = {datacopy_signal_semaphore_id_dir0, datacopy_signal_semaphore_id_dir1},
        .datacopy_override_runtime_arguments_callback = override_runtime_arguments_callback
    };
}


// For ring all-gather, we can send sub-sections of input tensor in opposite directions
// For linear all-gather though, we must ensure we send full tensors in BOTH directions
//   (in other words, disable the "bidirectional" send flag)
operation::ProgramWithCallbacks experimental::all_gather_matmul_multi_core_with_workers(
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    Tensor& datacopy_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,

    /* All Gather Params */
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology,
    const CoreCoord core_grid_offset,


    /* Matmul Params */
    const std::optional<const Tensor> bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig program_config,
    bool untilize_out

) {
    tt::tt_metal::Program program{};
    bool use_datacopy = false; /* Enable for debugging purposes */

    ////////////// Params for fused op signalers //////////////

    auto tensor_slicer = ttnn::ccl::InterleavedRingAllGatherTensorSlicer (
        input_tensor,
        all_gather_output_tensor,
        dim,
        ring_index
    );
    bool is_clockwise_direction = true;
    const uint32_t num_transfers = 4;
    const uint32_t weight_tensor_width = weight_tensor.get_legacy_shape()[3] / 32;

    ////////////////////////////////////////////////////////

    // Create a matmul signal info object that gets populated by the matmul kernel
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler = ttnn::experimental::ccl::MatmulFusedOpSignaler();
    matmul_fused_op_signaler->init_all_gather(
        num_transfers,
        ring_size,
        ring_index,
        tensor_slicer.num_cols,
        tensor_slicer.output_page_offset,
        is_clockwise_direction,
        tensor_slicer.num_cols * weight_tensor_width /* weight_output_page_offset: stride across a tensor slice in the weight_tensor */
    );

    // Matmul
    std::optional<operation::ProgramWithCallbacks> matmul_program_with_callbacks;
    std::optional<operation::OverrideRuntimeArgumentsCallback<Tensors>> matmul_override_runtime_arguments_callback;

    std::visit([&] (const auto& config) {
        using ProgramConfigType = std::decay_t<decltype(config)>;
        if (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
            matmul_program_with_callbacks = operations::matmul::matmul_multi_core_reuse_mcast_2d_optimized_helper(
                program,
                all_gather_output_tensor,
                weight_tensor,
                bias,
                matmul_output_tensor,
                bcast_batch,
                compute_kernel_config,
                config,
                untilize_out,
                matmul_fused_op_signaler
            );
            matmul_override_runtime_arguments_callback = matmul_program_with_callbacks->override_runtime_arguments_callback;
        } else if (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
            matmul_program_with_callbacks = operations::matmul::matmul_multi_core_reuse_mcast_1d_optimized_helper(
                program,
                all_gather_output_tensor,
                weight_tensor,
                bias,
                matmul_output_tensor,
                bcast_batch,
                compute_kernel_config,
                config,
                untilize_out,
                matmul_fused_op_signaler
            );
            matmul_override_runtime_arguments_callback = matmul_program_with_callbacks->override_runtime_arguments_callback;
        } else {
            TT_THROW("Unsupported MatmulProgramConfig type. Needs to be 1D or 2D Multicast.");
        }
    }, program_config);

    if (!matmul_program_with_callbacks.has_value()) {
        TT_THROW("Matmul program with callbacks not created");
    }


    // Datacopy
    const CoreCoord datacopy_core_coord = {0, 7}; // Pick a location that doesn't overlap with all_gather/matmul
    DatacopyParams datacopy_params;
    if (use_datacopy) {
        datacopy_params = setup_datacopy(
            matmul_program_with_callbacks->program,
            input_tensor,
            all_gather_output_tensor,
            datacopy_output_tensor,
            dim,
            num_links,
            ring_size,
            ring_index,
            topology,
            user_defined_num_workers,
            user_defined_num_buffers_per_channel,
            datacopy_core_coord,
            matmul_fused_op_signaler.value()
        );
    }

    // Create the all gather fused op signaler
    std::optional<AllGatherFusedOpSignaler> all_gather_fused_op_signaler = AllGatherFusedOpSignaler();
    if (use_datacopy) {
        all_gather_fused_op_signaler->init_fused_op(
            datacopy_params.datacopy_cores_noc,
            datacopy_params.datacopy_signal_semaphore_ids
        );
    } else {
        all_gather_fused_op_signaler->init_fused_op(
            matmul_fused_op_signaler->fused_op_receiver_cores_noc,
            matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
            matmul_fused_op_signaler->fused_op_signaler_mode
        );
    }

    // All Gather
    operation::ProgramWithCallbacks program_with_callbacks = ttnn::all_gather_multi_core_with_workers_helper(
        matmul_program_with_callbacks->program,
        input_tensor,
        all_gather_output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        receiver_device_id,
        sender_device_id,
        topology,
        user_defined_num_workers,
        user_defined_num_buffers_per_channel,
        all_gather_fused_op_signaler,
        core_grid_offset);
    const auto all_gather_override_runtime_arguments_callback = program_with_callbacks.override_runtime_arguments_callback;



    // Fuse the override runtime arguments callbacks
    auto override_runtime_arguments_callback = [use_datacopy, all_gather_override_runtime_arguments_callback, matmul_override_runtime_arguments_callback, datacopy_params] (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        if (matmul_override_runtime_arguments_callback.has_value()) {
            matmul_override_runtime_arguments_callback.value()(
                operation,
                program,
                {input_tensors[1], input_tensors[2]}, /* all gather output tensor, weight tensor */
                optional_input_tensors,
                {output_tensors[1]} /* matmul output tensor */
            );
        }

        if (all_gather_override_runtime_arguments_callback.has_value()) {
            all_gather_override_runtime_arguments_callback.value()(
                operation,
                program,
                {input_tensors[0], output_tensors[0]}, /* input tensor, all gather output tensor */
                optional_input_tensors,
                {output_tensors[0]} /* all gather output tensor */
            );
        }

        if (use_datacopy && datacopy_params.datacopy_override_runtime_arguments_callback.has_value()) {
            datacopy_params.datacopy_override_runtime_arguments_callback.value()(
                operation,
                program,
                {input_tensors[0], output_tensors[0]}, /* input tensor, all gather output tensor */
                optional_input_tensors,
                {output_tensors[2]} /* datacopy output tensor */
            );
        }
    };

    program_with_callbacks.override_runtime_arguments_callback = override_runtime_arguments_callback;

    return program_with_callbacks;
}

}  // namespace ttnn
