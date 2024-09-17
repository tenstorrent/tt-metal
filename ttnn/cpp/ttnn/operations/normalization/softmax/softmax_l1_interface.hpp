#pragma once

#include <optional>

#include "ttnn/cpp/ttnn/operations/common/l1_interface_common.hpp"

class SoftmaxOpL1Usage {
   public:
    SoftmaxOpL1Usage(
        const L1InterfaceOperandParams& input, int dim_arg, const std::optional<L1InterfaceOperandParams>& output);

    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const;
    std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const;

   protected:
    bool should_tilize_input() const;
    uint32_t calculate_block_size_impl(const L1InterfaceOperandParams& input) const;
    uint32_t get_input_cb_size() const;
    uint32_t get_output_cb_size() const;
    std::vector<uint32_t> get_intermediate_cb_sizes() const;
    uint32_t get_tilize_cb_size() const;

    L1InterfaceOperandParams input;
    L1InterfaceOperandParams output;
    int dim_arg;

    uint32_t block_size;
};

class SoftmaxOpL1UsageFactory {
   public:
    SoftmaxOpL1UsageFactory() = delete;
    static std::unique_ptr<SoftmaxOpL1Usage> Make(
        const L1InterfaceOperandParams& input,
        int dim_arg,
        const std::optional<L1InterfaceOperandParams>& output = std::nullopt);
};
