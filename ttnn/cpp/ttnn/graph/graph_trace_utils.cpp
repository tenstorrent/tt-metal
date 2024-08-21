// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_trace_utils.hpp"

#include "graph_processor.hpp"

#include "tt_metal/common/assert.hpp"

#include <unordered_set>
#include <string>


namespace ttnn::graph {

uint32_t extract_peak_memory_usage(const nlohmann::json& trace) {
    uint32_t total_cb = 0;
    uint32_t total_buffer = 0;
    uint32_t peak_memory_usage = 0;
    std::vector<std::string> current_op;

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];

        if (v["name"] == "function_start") {
            if (current_op.empty()) {
                while (++i < trace.size()) {
                    const auto& inner_v = trace[i];
                    if (inner_v["name"] == "buffer" && inner_v["params"]["type"] == "L1") {
                        total_buffer += std::stoi(inner_v["params"]["size"].get<std::string>());
                    } else if (inner_v["name"].get<std::string>().find("tensor") != std::string::npos) {
                        continue;
                    } else {
                        break;
                    }
                }
                --i;  // adjust for loop increment
            }
            current_op.push_back(v["params"]["name"]);
        } else if (v["name"] == "circular_buffer_allocate") {
            total_cb += stoi(v["params"]["size"].get<std::string>());
        } else if (v["name"] == "circular_buffer_deallocate_all") {
            total_cb = 0;
        } else if (v["name"] == "buffer_allocate" && v["params"]["type"] == "L1") {
            total_buffer += stoi(v["params"]["size"].get<std::string>());
        } else if (v["name"] == "buffer_deallocate") {
            auto connection = v["connections"][0].get<int>();
            auto buffer = trace[connection];
            if(buffer["params"]["type"] == "L1") {
                total_buffer -= stoi(buffer["params"]["size"].get<std::string>());
            }
        } else if (v["name"] == "function_end") {
            current_op.pop_back();
        }

        peak_memory_usage = std::max(peak_memory_usage, total_cb + total_buffer);
    }

    return peak_memory_usage;
}

// Returns count of intermediate and output tensors
std::pair<uint32_t, uint32_t> count_intermediate_and_output_tensors(const nlohmann::json& trace) {
    bool first_begin_found = false;
    bool last_end_found = false;

    std::unordered_set<int> intermediate_tensors;
    std::unordered_set<int> output_tensors;

    int first_begin_index = -1;
    int last_end_index = -1;

    for (int i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];
        if (v["name"] == "function_start" && !first_begin_found) {
            first_begin_found = true;
            first_begin_index = i;
        } else if (v["name"] == "function_end") {
            last_end_found = true;
            last_end_index = i;

            if("create_device_tensor") {
                auto id = v["connections"][0].get<int>();
                intermediate_tensors.insert(id);
            }
        }
    }

    TT_ASSERT(first_begin_found);
    TT_ASSERT(last_end_found);

    for(int index : trace[last_end_index]["connections"]) {
        // It can be tensor or some other node like
        if(trace[index]["name"].get<std::string>().find("tensor") != std::string::npos) {
            output_tensors.insert(index);
        }
    }

    for(int index : output_tensors) {
        intermediate_tensors.erase(index);
    }

    // Return the counts of intermediate and output tensors
    return {intermediate_tensors.size(), output_tensors.size()};
}

std::vector<std::string> extract_calltrace(const nlohmann::json& trace){
    std::vector<std::string> op_calls;
    size_t i = 0;

    while (i < trace.size()) {
        const auto& v = trace[i];
        i++;

        if (v["name"] == "function_start") {
            op_calls.push_back(v["params"]["name"]);
        }
    }

    return op_calls;
}

size_t extract_output_L1_size(const nlohmann::json& trace)
{
    // Lambda to find the last 'function_end' node
    auto find_function_end_node = [](const auto& trace) -> const nlohmann::json& {
        for(int i = trace.size() - 1; i >= 0; --i) {
            const auto& v = trace[i];
            if (v["name"] == "function_end") {
                return v;
            }
        }
        throw std::runtime_error("function_end node not found");
    };

    const auto& function_end_node = find_function_end_node(trace);

    // Lambda to extract output tensors from the 'function_end' node
    auto extract_output_tensors = [&trace](const auto& function_end_node) {
        std::unordered_set<uint32_t> output;
        for (const auto& output_id : function_end_node["connections"]) {
            if (trace[output_id]["name"].get<std::string>().find("tensor") != std::string::npos) {
                output.insert(output_id);
            }
        }
        return output;
    };

    const auto output_tensors = extract_output_tensors(function_end_node);

    // Calculate the total size of L1 buffers for output tensors
    size_t output_size = 0;
    for (const auto& node : trace) {
        if (node["name"] == "buffer" && node["params"]["type"] == "L1") {
            for (const auto& tensor_id : node["connections"]) {
                if (output_tensors.find(tensor_id) != output_tensors.end()) {
                    output_size += node["params"]["size"].get<size_t>();
                }
            }
        }
    }

    return output_size;
}


} // namespace ttnn::graph
