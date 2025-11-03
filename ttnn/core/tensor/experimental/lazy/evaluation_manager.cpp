#include "ttnn/experimental/lazy/evaluation_manager.hpp"
#include "ttnn/experimental/lazy/graph_utils.hpp"
#include "ttnn/experimental/lazy/lazy_tensor.hpp"

#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/old_infra_device_operation.hpp"
#include <unordered_map>
#include <enchantum/enchantum.hpp>

namespace ttnn::experimental::lazy {

namespace {
// This is an example of how we're going to traverse nodes in graph later
void log_unary_operation(const std::shared_ptr<LazyTensor>& tensor) {
    auto op = tensor->op();
    auto* unary_op = dynamic_cast<LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>*>(op.get());
    auto attributes = unary_op->attributes();
    log_info(
        tt::LogTest,
        "[Evaluate] Unary operation[id={}, type={}]",
        tensor->id(),
        enchantum::to_string(attributes.op_chain.at(0).type()));
}
void log_binary_ng_operation(const std::shared_ptr<LazyTensor>& tensor) {
    auto op = tensor->op();
    auto* binary_ng_op =
        dynamic_cast<LazyDeviceOperation<ttnn::operations::binary_ng::BinaryNgDeviceOperation>*>(op.get());
    auto attributes = binary_ng_op->attributes();
    log_info(
        tt::LogTest,
        "[Evaluate] Binary operation[id={}, type={}]",
        tensor->id(),
        enchantum::to_string(attributes.binary_op_type));
}

void log_old_infra_operation(const std::shared_ptr<LazyTensor>& tensor) {
    auto op = tensor->op();
    // TODO: We should probably capture old infra operation differently to skip casting it to OldInfraDeviceOperation
    auto* old_infra_op = dynamic_cast<
        LazyDeviceOperation<tt::tt_metal::operation::OldInfraDeviceOperation<tt::tt_metal::operation::Tensors>>*>(
        op.get());
    auto attributes = old_infra_op->attributes();
    log_info(tt::LogTest, "[Evaluate] Old infra operation[id={}, type={}]", tensor->id(), attributes.get_type_name());
}

std::unordered_map<tt::stl::hash::hash_t, std::function<void(const std::shared_ptr<LazyTensor>&)>> operation_log_map = {
    {tt::stl::hash::type_hash<ttnn::operations::unary::UnaryDeviceOperation>, log_unary_operation},
    {tt::stl::hash::type_hash<ttnn::operations::binary_ng::BinaryNgDeviceOperation>, log_binary_ng_operation},
    {tt::stl::hash::type_hash<tt::tt_metal::operation::OldInfraDeviceOperation<tt::tt_metal::operation::Tensors>>,
     log_old_infra_operation},
};
}  // namespace

void evaluate(const std::shared_ptr<LazyTensor>& lazy_tensor) {
    auto sorted_tensors = GraphUtils::topological_sort(lazy_tensor);
    for (auto& tensor : sorted_tensors) {
        if (tensor->op()) {
            if (operation_log_map.find(tensor->op()->operation_type_id()) != operation_log_map.end()) {
                operation_log_map[tensor->op()->operation_type_id()](tensor);
            } else {
                log_info(
                    tt::LogTest,
                    "[Evaluate] {} operation[id={}, type_id={}]",
                    tensor->op()->name(),
                    tensor->id(),
                    tensor->op()->operation_type_id());
            }
        } else {
            log_info(tt::LogTest, "[Evaluate] Unknown operation[id={}, type_id=???]", tensor->id());
        }

        tensor->evaluate();
    }
}

}  // namespace ttnn::experimental::lazy
