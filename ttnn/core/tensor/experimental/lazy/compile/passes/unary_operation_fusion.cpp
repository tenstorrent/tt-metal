#include "ttnn/experimental/lazy/compile/passes/unary_operation_fusion.hpp"
#include "ttnn/experimental/lazy/lazy_device_operation.hpp"
#include "ttnn/experimental/lazy/lazy_tensor.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/experimental/lazy/graph_utils.hpp"
#include <enchantum/enchantum.hpp>

namespace ttnn::experimental::lazy {

namespace compile {
std::string UnaryOperationsFusionPass::name() const { return "UnaryOperationsFusionPass"; }

void UnaryOperationsFusionPass::run(const ttnn::Tensor& tensor) {
    auto sorted_tensors = GraphUtils::topological_sort(tensor.lazy());

    // Track which tensors have been fused (so we don't fuse them again)
    std::unordered_set<LazyTensorId> fused_tensors;

    for (auto& current_tensor : sorted_tensors) {
        // Skip if already fused
        if (fused_tensors.find(current_tensor->id()) != fused_tensors.end()) {
            continue;
        }

        // Check if this is a unary operation
        auto op = current_tensor->op();
        if (!op) {
            continue;
        }

        auto* unary_op = dynamic_cast<LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>*>(op.get());
        if (!unary_op) {
            continue;
        }

        // Check if this unary operation has exactly one input
        const auto& inputs = current_tensor->op_inputs();
        if (inputs->size() != 1) {
            continue;
        }

        // Check if the input is also a unary operation
        auto input_tensor = inputs->at(0);
        auto input_op = input_tensor->op();
        if (!input_op) {
            continue;
        }

        auto* input_unary_op =
            dynamic_cast<LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>*>(input_op.get());
        if (!input_unary_op) {
            continue;
        }

        // Found a chain! Now traverse backwards to find all consecutive unary ops
        std::vector<LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>*> chain;
        std::vector<std::shared_ptr<LazyTensor>> chain_tensors;

        auto current = current_tensor;
        while (true) {
            auto curr_op = current->op();
            if (!curr_op) {
                break;
            }

            auto* curr_unary_op =
                dynamic_cast<LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>*>(curr_op.get());
            if (!curr_unary_op) {
                break;
            }

            chain.push_back(curr_unary_op);
            chain_tensors.push_back(current);

            const auto& curr_inputs = current->op_inputs();
            if (curr_inputs->size() != 1) {
                break;
            }

            current = curr_inputs->at(0);
        }

        // If we have more than one unary op in the chain, fuse them
        if (chain.size() > 1) {
            // Reverse the chain to go from first to last
            std::reverse(chain.begin(), chain.end());
            std::reverse(chain_tensors.begin(), chain_tensors.end());

            // Merge all op_chains
            std::vector<ttnn::operations::unary::EltwiseUnaryWithParam> merged_op_chain;
            for (auto* unary_op_ptr : chain) {
                const auto& attrs = unary_op_ptr->attributes();
                merged_op_chain.insert(merged_op_chain.end(), attrs.op_chain.begin(), attrs.op_chain.end());
            }

            // Log the fusion
            log_info(tt::LogTest, "[UnaryOperationsFusionPass] Fusing {} unary operations:", chain.size());
            for (size_t i = 0; i < chain.size(); ++i) {
                const auto& attrs = chain[i]->attributes();
                log_info(
                    tt::LogTest,
                    "  [{}] Tensor id={}, op_chain size={}",
                    i,
                    chain_tensors[i]->id(),
                    attrs.op_chain.size());
                for (const auto& op : attrs.op_chain) {
                    log_info(tt::LogTest, "      - {}", enchantum::to_string(op.type()));
                }
            }
            log_info(tt::LogTest, "  Merged op_chain size: {}", merged_op_chain.size());

            // Get the attributes from the last operation (use its output config)
            auto* last_unary_op = chain.back();
            auto old_attributes = last_unary_op->attributes();

            // Get the output specs from the last operation (since that's what our fused op will output)
            auto output_specs = last_unary_op->compute_output_specs();

            // Get the input of the first operation in the chain (for graph dependency update)
            auto first_op_inputs = chain_tensors[0]->op_inputs();
            auto first_op_tensor_args =
                std::any_cast<ttnn::operations::unary::UnaryDeviceOperation::tensor_args_t>(first_op_inputs->get());

            // Create new attributes with merged op_chain
            auto merged_attributes = ttnn::operations::unary::operation_attributes_t{
                .op_chain = merged_op_chain,
                .output_dtype = old_attributes.output_dtype,
                .output_memory_config = old_attributes.output_memory_config,
                .fp32_dest_acc_en = old_attributes.fp32_dest_acc_en,
                .preserve_fp32_precision = old_attributes.preserve_fp32_precision,
                .bfp8_pack_precise = old_attributes.bfp8_pack_precise};

            // Create a new LazyDeviceOperation with the merged attributes and metadata from existing ops
            auto new_fused_op = std::make_shared<LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>>(
                merged_attributes, first_op_tensor_args, "fused_unary_operation");

            // Update the last tensor in the chain to use the new operation and new input
            chain_tensors.back()->set_op(new_fused_op);
            chain_tensors.back()->set_op_inputs(first_op_inputs);

            // Mark all tensors in the chain as fused
            for (const auto& t : chain_tensors) {
                fused_tensors.insert(t->id());
            }

            log_info(tt::LogTest, "[UnaryOperationsFusionPass] Fusion complete!");
        }
    }
}
}  // namespace compile
}  // namespace ttnn::experimental::lazy
