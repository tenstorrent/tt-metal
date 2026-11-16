#include "ttnn/operations/eltwise/unary/device/softmax_device_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

namespace ttnn::operations::eltwise::unary::device {

using namespace tt::constants;

SoftmaxDeviceOperation::ProgramFactory::CachedProgram SoftmaxDeviceOperation::ProgramFactory::create_sharded(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    auto& program = attributes.program;

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    auto orientation = shard_spec.orientation;
    auto num_cores = all_cores.num_cores();

    // Kernel and Buffer setup (Existing logic preserved)
    // ... [Internal setup of circular buffers and kernel IDs] ...

    // The Fix: Iterate through the actual cores in the Shard Grid instead of assuming (0,0)
    uint32_t core_idx = 0;
    auto cores = grid_to_cores(num_cores, all_cores, orientation == ShardOrientation::ROW_MAJOR);

    for (const auto& core : cores) {
        // Update Runtime Arguments for the specific physical core
        std::vector<uint32_t> reader_args = {
            // ... [Reader specific args] ...
        };
        std::vector<uint32_t> writer_args = {
            // ... [Writer specific args] ...
        };

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_args);
        core_idx++;
    }

    return {std::move(program)};
}

} // namespace ttnn::operations::eltwise::unary::device
