// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A C ABI over trace replay and the host<->device sockets, for host runtimes in other languages that run in the
// same process as ttnn (e.g. a Rust decode loop). Python builds the model and captures the traces, then hands
// the C++ objects over as raw addresses (``native_handle``); the foreign runtime calls these functions on its own
// threads, with no GIL. Every function catches C++ exceptions and reports them through ``err`` instead.

#include "host_runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>

#include <nanobind/nanobind.h>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/trace.hpp"

#define TTNN_HR_API __attribute__((visibility("default")))

namespace {

using tt::tt_metal::distributed::D2HSocket;
using tt::tt_metal::distributed::H2DSocket;
using tt::tt_metal::distributed::MeshDevice;

// Bumped whenever a signature below changes, so a runtime built against another ABI refuses to start.
constexpr uint32_t kAbiVersion = 1;

void set_error(char* err, size_t err_len, const char* message) {
    if (err == nullptr || err_len == 0) {
        return;
    }
    const size_t n = std::min(err_len - 1, std::strlen(message));
    std::memcpy(err, message, n);
    err[n] = '\0';
}

template <typename F>
int guarded(char* err, size_t err_len, F&& body) {
    try {
        return body();
    } catch (const std::exception& e) {
        set_error(err, err_len, e.what());
    } catch (...) {
        set_error(err, err_len, "unknown C++ exception");
    }
    return -1;
}

std::optional<uint32_t> timeout_of(uint32_t timeout_ms) {
    return timeout_ms == 0 ? std::nullopt : std::optional<uint32_t>(timeout_ms);
}

}  // namespace

extern "C" {

TTNN_HR_API uint32_t ttnn_hr_abi_version() { return kAbiVersion; }

// Non-blocking replay of ``trace_id`` on ``mesh_device``'s command queue ``cq_id``: returns once the replay is
// enqueued. Calls on different mesh devices (e.g. the submeshes of one parent) may run on different threads.
TTNN_HR_API int ttnn_hr_execute_trace(void* mesh_device, uint32_t trace_id, uint8_t cq_id, char* err, size_t err_len) {
    return guarded(err, err_len, [&] {
        ttnn::operations::trace::execute_trace(
            static_cast<MeshDevice*>(mesh_device),
            tt::tt_metal::distributed::MeshTraceId(trace_id),
            ttnn::QueueId(cq_id),
            /*blocking=*/false);
        return 0;
    });
}

TTNN_HR_API uint32_t ttnn_hr_h2d_page_size(void* socket) { return static_cast<H2DSocket*>(socket)->get_page_size(); }

// 1 if ``num_bytes`` fit in the FIFO now, 0 if not, -1 on error.
TTNN_HR_API int ttnn_hr_h2d_has_space(void* socket, uint32_t num_bytes, char* err, size_t err_len) {
    return guarded(err, err_len, [&] { return static_cast<H2DSocket*>(socket)->has_space(num_bytes) ? 1 : 0; });
}

// Blocks while the FIFO is full, until the device drains enough pages.
TTNN_HR_API int ttnn_hr_h2d_write(void* socket, const void* data, uint32_t num_pages, char* err, size_t err_len) {
    return guarded(err, err_len, [&] {
        static_cast<H2DSocket*>(socket)->write(const_cast<void*>(data), num_pages);
        return 0;
    });
}

// A connector to the socket another handle exported as ``socket_id`` (``H2DSocket::connect``); free it with
// ``ttnn_hr_h2d_destroy``. ``timeout_ms`` 0 takes the default.
TTNN_HR_API void* ttnn_hr_h2d_connect(const char* socket_id, uint32_t timeout_ms, char* err, size_t err_len) {
    void* out = nullptr;
    guarded(err, err_len, [&] {
        out = H2DSocket::connect(socket_id, timeout_of(timeout_ms)).release();
        return 0;
    });
    return out;
}

TTNN_HR_API void ttnn_hr_h2d_destroy(void* socket) { delete static_cast<H2DSocket*>(socket); }

TTNN_HR_API uint32_t ttnn_hr_d2h_page_size(void* socket) { return static_cast<D2HSocket*>(socket)->get_page_size(); }

// Pages readable now, or -1 on error.
TTNN_HR_API int ttnn_hr_d2h_pages_available(void* socket, char* err, size_t err_len) {
    return guarded(err, err_len, [&] { return static_cast<int>(static_cast<D2HSocket*>(socket)->pages_available()); });
}

// Blocks until ``num_pages`` are in, copies them out and returns their space to the device.
TTNN_HR_API int ttnn_hr_d2h_read(void* socket, void* data, uint32_t num_pages, char* err, size_t err_len) {
    return guarded(err, err_len, [&] {
        static_cast<D2HSocket*>(socket)->read(data, num_pages, /*notify_sender=*/true);
        return 0;
    });
}

TTNN_HR_API void* ttnn_hr_d2h_connect(const char* socket_id, uint32_t timeout_ms, char* err, size_t err_len) {
    void* out = nullptr;
    guarded(err, err_len, [&] {
        out = D2HSocket::connect(socket_id, timeout_of(timeout_ms)).release();
        return 0;
    });
    return out;
}

TTNN_HR_API void ttnn_hr_d2h_destroy(void* socket) { delete static_cast<D2HSocket*>(socket); }

}  // extern "C"

namespace ttnn::host_runtime {

void py_module(nb::module_& mod) {
    const char* doc = R"doc(
        The address of the C++ object behind ``obj`` (a MeshDevice, H2DSocket or D2HSocket), for a host runtime
        that calls the ``ttnn_hr_*`` C functions of this library on it. The Python object must outlive every use.
    )doc";
    mod.def("native_handle", [](MeshDevice& obj) { return reinterpret_cast<std::uintptr_t>(&obj); }, doc);
    mod.def("native_handle", [](H2DSocket& obj) { return reinterpret_cast<std::uintptr_t>(&obj); }, doc);
    mod.def("native_handle", [](D2HSocket& obj) { return reinterpret_cast<std::uintptr_t>(&obj); }, doc);
}

}  // namespace ttnn::host_runtime
