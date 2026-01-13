// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/graph/graph_argument_serializer.hpp"
#include <functional>
#include <vector>

namespace ttnn::graph {

// Deferred registration queue - solves SIOF by deferring registrations until singleton is ready
class GraphArgumentRegistrationQueue {
public:
    using RegistrationFunc = std::function<void()>;

    // Get the queue singleton (safe, constructed on first use)
    static GraphArgumentRegistrationQueue& instance() {
        static GraphArgumentRegistrationQueue queue;
        return queue;
    }

    // Add a registration function to the queue
    void enqueue(RegistrationFunc func) { registrations_.push_back(std::move(func)); }

    // Execute all queued registrations (called by GraphArgumentSerializer::initialize())
    void execute_all() {
        for (auto& func : registrations_) {
            func();
        }
        registrations_.clear();
    }

private:
    GraphArgumentRegistrationQueue() = default;
    std::vector<RegistrationFunc> registrations_;
};

// Helper for static registration
template <typename T>
struct GraphArgumentAutoRegistrar {
    GraphArgumentAutoRegistrar() {
        GraphArgumentRegistrationQueue::instance().enqueue(
            []() { GraphArgumentSerializer::register_argument_type<T>(); });
    }
};

}  // namespace ttnn::graph

// Macro for automatic type registration via static initialization
// This is safe because it queues the registration, which executes during controlled initialization
// Use variadic args to handle types with commas (templates, variants)
#define TTNN_REGISTER_GRAPH_ARG_CAT(a, b) a##b
#define TTNN_REGISTER_GRAPH_ARG_UNIQUE(prefix, counter) TTNN_REGISTER_GRAPH_ARG_CAT(prefix, counter)

#define TTNN_REGISTER_GRAPH_ARG(...)                                                              \
    namespace {                                                                                   \
    static ::ttnn::graph::GraphArgumentAutoRegistrar<__VA_ARGS__> TTNN_REGISTER_GRAPH_ARG_UNIQUE( \
        ttnn_graph_arg_registrar_, __COUNTER__);                                                  \
    }
