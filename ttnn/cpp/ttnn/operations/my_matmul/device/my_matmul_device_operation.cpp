#include "my_matmul_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::my_matmul {

// 1) Always the single-core factory.
MyMatmulDeviceOperation::program_factory_t MyMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    // TODO: define when to use which

    // return SingleCore{};
    return MultiCore{};
}

// 2) Validate. Keep it meaningful but light — this is where you catch caller mistakes.
void MyMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;

    // Both operands must live on-device (the kernels read from DRAM buffers).
    TT_FATAL(
        a.storage_type() == tt::tt_metal::StorageType::DEVICE && b.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "my_matmul: both inputs must be on-device tensors");
    TT_FATAL(a.device() == b.device(), "my_matmul: both inputs must be on the same device");

    // The naive kernel operates on 32x32 tiles, so both inputs must be tile-laid-out.
    TT_FATAL(a.layout() == Layout::TILE && b.layout() == Layout::TILE, "my_matmul: both inputs must be in TILE layout");

    // The compute kernel and CBs are configured for bfloat16.
    TT_FATAL(
        a.dtype() == DataType::BFLOAT16 && b.dtype() == DataType::BFLOAT16, "my_matmul: both inputs must be bfloat16");

    // Shapes must be matmul-compatible: A is [.., M, K], B is [.., K, N] -> the K's must match.
    const auto& as = a.padded_shape();
    const auto& bs = b.padded_shape();
    uint32_t Ka = as[as.rank() - 1];
    uint32_t Kb = bs[bs.rank() - 2];
    TT_FATAL(Ka == Kb, "my_matmul: inner dimensions must match (A's K = {} vs B's K = {})", Ka, Kb);

    // The 32x32-tile hardware requires each relevant dim to be a whole number of tiles.
    TT_FATAL(
        as[as.rank() - 2] % tt::constants::TILE_HEIGHT == 0 && Ka % tt::constants::TILE_WIDTH == 0 &&
            bs[bs.rank() - 1] % tt::constants::TILE_WIDTH == 0,
        "my_matmul: M, K, and N must each be divisible by the tile size");
}

// 3) Output spec: shape [M, N], TILE layout, same dtype as A.
MyMatmulDeviceOperation::spec_return_value_t MyMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    const auto& as = a.logical_shape();
    const auto& bs = b.logical_shape();
    uint32_t M = as[as.rank() - 2];
    uint32_t N = bs[bs.rank() - 1];
    return TensorSpec(
        ttnn::Shape{M, N}, tt::tt_metal::TensorLayout(a.dtype(), tt::tt_metal::PageConfig(a.layout()), MemoryConfig{}));
}

// 4) Allocate the output tensor on the same device as A.
MyMatmulDeviceOperation::tensor_return_value_t MyMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(attrs, tensor_args);
    return create_device_tensor(spec, tensor_args.input_tensor_a.device());
}

}  // namespace ttnn::operations::my_matmul

// The primitive launcher: bundle attributes + tensor args and hand off to the framework.
namespace ttnn::prim {
ttnn::operations::my_matmul::MyMatmulDeviceOperation::tensor_return_value_t my_matmul(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    using Op = ttnn::operations::my_matmul::MyMatmulDeviceOperation;
    auto attrs = Op::operation_attributes_t{};
    auto args = Op::tensor_args_t{input_tensor_a, input_tensor_b};
    return ttnn::device_operation::launch<Op>(attrs, args);
}
}  // namespace ttnn::prim
