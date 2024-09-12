#pragma once

#include "ttnn/cpp/ttnn/operations/common/l1_interface_common.hpp"

class SoftmaxOpL1Usage {
   public:
    SoftmaxOpL1Usage(const L1InterfaceOpParams& input, int dim_arg);

    std::vector<std::tuple<uint32_t, uint32_t>> get_circular_buffer_l1_allocations_per_core() const;
    std::vector<std::tuple<uint32_t, uint32_t>> get_tensor_l1_allocations_per_core() const;

   protected:
    bool should_tilize_input() const;

    L1InterfaceOpParams input;
    L1InterfaceOpParams output;
    int dim_arg;
};

class SoftmaxOpL1UsageFactory {
   public:
    SoftmaxOpL1UsageFactory() = delete;
    static std::unique_ptr<SoftmaxOpL1Usage> Make(const L1InterfaceOpParams& input, int dim_arg);
};
