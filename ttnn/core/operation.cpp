// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/ttnn/operation.hpp"
#include <array>
#include <tuple>

namespace tt::tt_metal::operation {

template <typename OutputTensorsT>
OpPerformanceModelGeneral<OutputTensorsT>::OpPerformanceModelGeneral(
    Tensors input_tensors, const OutputTensors& output_tensors, int ideal_compute_cycles) :
    ideal_compute_cycles(ideal_compute_cycles) {
    const auto& t = input_tensors.at(0);
    const auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : ARCH::WORMHOLE_B0;

    // Get clock rate dynamically from device
    float clock_rate_ghz;
    if (t.storage_type() == StorageType::DEVICE) {
        int freq_mhz = t.device()->get_clock_rate_mhz();
        clock_rate_ghz = freq_mhz / 1000.0f;
    } else {
        // Fallback for non-device tensors (defaults to WH)
        clock_rate_ghz = 1.0;
    }
    this->ideal_compute_ns = std::ceil(ideal_compute_cycles / clock_rate_ghz);

    // WH L1 Bisection bandwidth: 512 B/cycle = sqrt(64) * 32 B/cycle * 2
    // 512 * 1.0 GHz = 512 GB/s
    // WH DRAM bandwidth: 258 GB/s = 21.5 GB/s * 6 channels * 2 banks

    // BH DRAM bandwidth: 512 GB/s (32GB GDDR6 @ 16 GT/sec)

    float peak_dram_bw;
    if (arch == ARCH::WORMHOLE_B0) {
        peak_dram_bw = 6 * 2 * 21.5;  // 258 GB/s
    } else if (arch == ARCH::BLACKHOLE) {
        peak_dram_bw = 512.0;  // 512 GB/s
    } else {
        TT_THROW("Unsupported architecture for OpPerformanceModel");
    }

    auto tensor_ns = [peak_dram_bw](const Tensor& t) {
        int size_bytes = t.physical_volume() * t.element_size();
        if (t.memory_config().is_dram()) {
            return size_bytes / peak_dram_bw / 1024 / 1024 / 1024 * 1000 * 1000 * 1000;
        }
        if (t.memory_config().is_l1()) {
            return 1.0f;  // TODO: figure out better modelling scheme for L1->L1 Transfers
        }
        return 0.0f;
    };

    for (const auto& t : input_tensors) {
        this->inputs_bytes.push_back(t.physical_volume() * t.element_size());
        if (tensor_ns(t) > this->ideal_bandwidth_ns) {
            this->ideal_bandwidth_ns = tensor_ns(t);
        }
    }
    if constexpr (std::is_same_v<OutputTensors, Tensors>) {
        for (const auto& t : output_tensors) {
            this->outputs_bytes.push_back(t.physical_volume() * t.element_size());
            if (tensor_ns(t) > this->ideal_bandwidth_ns) {
                this->ideal_bandwidth_ns = tensor_ns(t);
            }
        }
    } else if constexpr (std::is_same_v<OutputTensors, Tensor>) {
        this->outputs_bytes.push_back(output_tensors.physical_volume() * output_tensors.element_size());
        auto output_ns = tensor_ns(output_tensors);
        if (output_ns > this->ideal_bandwidth_ns) {
            this->ideal_bandwidth_ns = output_ns;
        }
    } else if constexpr (
        std::is_same_v<OutputTensors, std::array<std::vector<Tensor>, 2>> ||
        std::is_same_v<OutputTensors, std::array<Tensor, 2>>) {
        // Handle std::array types - iterate over array elements
        for (const auto& element : output_tensors) {
            if constexpr (std::is_same_v<decltype(element), const std::vector<Tensor>&>) {
                // Array of vectors - process each tensor in each vector
                for (const auto& t : element) {
                    this->outputs_bytes.push_back(t.physical_volume() * t.element_size());
                    if (tensor_ns(t) > this->ideal_bandwidth_ns) {
                        this->ideal_bandwidth_ns = tensor_ns(t);
                    }
                }
            } else {
                // Array of tensors - process each tensor
                this->outputs_bytes.push_back(element.physical_volume() * element.element_size());
                if (tensor_ns(element) > this->ideal_bandwidth_ns) {
                    this->ideal_bandwidth_ns = tensor_ns(element);
                }
            }
        }
    } else if constexpr (std::is_same_v<OutputTensors, std::tuple<Tensor, Tensor>>) {
        // Handle std::tuple<Tensor, Tensor>
        auto process_tuple_element = [&](const auto& t) {
            this->outputs_bytes.push_back(t.physical_volume() * t.element_size());
            if (tensor_ns(t) > this->ideal_bandwidth_ns) {
                this->ideal_bandwidth_ns = tensor_ns(t);
            }
        };
        process_tuple_element(std::get<0>(output_tensors));
        process_tuple_element(std::get<1>(output_tensors));
    } else {
        // Handle optional types
        for (const auto& ot : output_tensors) {
            if (!ot.has_value()) {
                continue;
            }
            auto& t = ot.value();
            this->outputs_bytes.push_back(t.physical_volume() * t.element_size());
            if (tensor_ns(t) > this->ideal_bandwidth_ns) {
                this->ideal_bandwidth_ns = tensor_ns(t);
            }
        }
    }

    this->ideal_ns = std::max(this->ideal_compute_ns, this->ideal_bandwidth_ns);
}

// Note: Only instantiate the constructor for common types.
// get_input_bws/get_output_bws are kept inline in header to avoid needing
// explicit instantiation for every custom result type.
template OpPerformanceModelGeneral<Tensors>::OpPerformanceModelGeneral(Tensors, const Tensors&, int);
template OpPerformanceModelGeneral<Tensor>::OpPerformanceModelGeneral(Tensors, const Tensor&, int);
template OpPerformanceModelGeneral<OptionalTensors>::OpPerformanceModelGeneral(Tensors, const OptionalTensors&, int);

}  // namespace tt::tt_metal::operation
