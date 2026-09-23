// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefetcher_pipe.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::prefetcher_pipe {

using tt::tt_metal::experimental::PrefetcherPipe;
using tt::tt_metal::experimental::PrefetcherPipeSpace;
using tt::tt_metal::experimental::PrefetcherPipeSpaceConfig;

void py_module_types(nb::module_& mod) {
    nb::class_<PrefetcherPipe>(mod, "PrefetcherPipe", R"doc(
        One sender -> receivers PrefetcherPipe carved from a PrefetcherPipeSpace.

        A Program uses it through the Metal 2.0 host API (ProgramRunArgs.advanced_options.prefetcher_pipe_args)
        or through a ttnn operation that takes a pipe argument. Keep the pipe alive for as long
        as any program uses it; letting it go returns its cores to the space. A pipe holds a
        reference to its PrefetcherPipeSpace, so the space stays alive while the pipe does.
    )doc")
        .def("buffer_address", &PrefetcherPipe::buffer_address, "Base address of the data ring on every pipe core.")
        .def("config_address", &PrefetcherPipe::config_address, "Base address of the config page on every pipe core.")
        .def("ring_size", &PrefetcherPipe::ring_size, "Per-core data ring size in bytes.")
        .def("config_page_size", &PrefetcherPipe::config_page_size, "Config page size in bytes.")
        .def("sender_core", &PrefetcherPipe::sender_core)
        .def("sender_cores", &PrefetcherPipe::sender_cores, nb::rv_policy::reference_internal)
        .def("receiver_cores", &PrefetcherPipe::receiver_cores, nb::rv_policy::reference_internal)
        .def("all_cores", &PrefetcherPipe::all_cores, nb::rv_policy::reference_internal);

    nb::class_<PrefetcherPipeSpace>(mod, "PrefetcherPipeSpace", R"doc(
        A reservation of persistent L1 (data ring + config page) on sender_cores ∪ receiver_domain
        from which PrefetcherPipes are carved. Every pipe carved from one space shares the space's
        buffer_address() and config_address(). The space owns the L1; every pipe carved from it
        keeps the space alive, so the L1 is released once the space and all its pipes are gone.
    )doc")
        .def("buffer_address", &PrefetcherPipeSpace::buffer_address)
        .def("config_address", &PrefetcherPipeSpace::config_address)
        .def("ring_size", &PrefetcherPipeSpace::ring_size)
        .def("config_page_size", &PrefetcherPipeSpace::config_page_size)
        .def("max_receivers_per_pipe", &PrefetcherPipeSpace::max_receivers_per_pipe)
        .def("sender_cores", &PrefetcherPipeSpace::sender_cores, nb::rv_policy::reference_internal)
        .def("receiver_domain", &PrefetcherPipeSpace::receiver_domain, nb::rv_policy::reference_internal)
        .def("reservation_cores", &PrefetcherPipeSpace::reservation_cores, nb::rv_policy::reference_internal)
        .def(
            "unclaimed_cores",
            &PrefetcherPipeSpace::unclaimed_cores,
            "Reservation cores not currently claimed by a live pipe.")
        .def(
            "create_pipe",
            &PrefetcherPipeSpace::create_pipe,
            // The C++ space must outlive its pipes (a pipe does not own its space); tying the
            // returned pipe to the Python space object enforces that from Python, and keeps the
            // device alive through the space as well.
            nb::keep_alive<0, 1>(),
            nb::arg("sender"),
            nb::arg("receivers"),
            R"doc(
                Carve one pipe. `sender` must be one of the space's sender_cores and `receivers` a
                non-empty subset of receiver_domain with at most max_receivers_per_pipe cores, none
                of which is claimed by a live pipe. Writes the pipe's config pages; allocates
                nothing.

                Args:
                    sender (CoreCoord): the sender worker core.
                    receivers (CoreRangeSet): the receiver worker cores.

                Returns:
                    PrefetcherPipe
            )doc")
        .def(
            "create_pipes",
            [](nb::handle self_obj,
               const std::vector<std::pair<tt::tt_metal::CoreCoord, tt::tt_metal::CoreRangeSet>>& pipes) {
                auto& self = nb::cast<PrefetcherPipeSpace&>(self_obj);
                // Same lifetime tie as create_pipe, applied per element: nb::keep_alive<0, 1> cannot
                // be used here because the return value is a Python list, which is not
                // weak-referenceable, so nanobind would raise at call time.
                nb::list out;
                for (PrefetcherPipe& pipe : self.create_pipes(pipes)) {
                    nb::object pipe_obj = nb::cast(std::move(pipe));
                    nb::detail::keep_alive(pipe_obj.ptr(), self_obj.ptr());
                    out.append(std::move(pipe_obj));
                }
                return out;
            },
            nb::arg("pipes"),
            R"doc(
                Carve several disjoint pipes in one call. The batch is validated as a whole before
                anything is claimed.

                Args:
                    pipes (List[Tuple[CoreCoord, CoreRangeSet]]): (sender, receivers) per pipe.

                Returns:
                    List[PrefetcherPipe]
            )doc");
}

void py_module(nb::module_& mod) {
    // Binds tt_metal's CreatePrefetcherPipeSpace directly; the config struct is flattened into
    // keyword arguments and the reservation runs under a ttnn-named allocation context. Worker
    // senders only (DRAM-sender pipes are not exposed here).
    mod.def(
        "create_prefetcher_pipe_space",
        [](const tt::tt_metal::distributed::MeshDevice& mesh_device,
           const tt::tt_metal::CoreRangeSet& sender_cores,
           const tt::tt_metal::CoreRangeSet& receiver_domain,
           uint32_t ring_size,
           uint32_t max_receivers_per_pipe,
           tt::tt_metal::BufferType buffer_type) {
            auto guard = tt::tt_metal::make_allocation_context_guard("ttnn.experimental.create_prefetcher_pipe_space");
            return tt::tt_metal::experimental::CreatePrefetcherPipeSpace(
                mesh_device,
                PrefetcherPipeSpaceConfig{
                    .sender_cores = sender_cores,
                    .num_dram_senders = 0,
                    .receiver_domain = receiver_domain,
                    .ring_size = ring_size,
                    .max_receivers_per_pipe = max_receivers_per_pipe,
                    .buffer_type = buffer_type,
                });
        },
        nb::keep_alive<0, 1>(),
        nb::arg("mesh_device"),
        nb::arg("sender_cores"),
        nb::arg("receiver_domain"),
        nb::arg("ring_size"),
        nb::arg("max_receivers_per_pipe"),
        nb::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Reserve persistent L1 for PrefetcherPipes on sender_cores ∪ receiver_domain.

            Must run before any program places program-local L1 on those cores. receiver_domain is
            every core that MAY later be carved as a receiver, not the 1:N map; pipes are carved
            afterwards with PrefetcherPipeSpace.create_pipe / create_pipes.

            Args:
                mesh_device (MeshDevice): the mesh device to reserve on.
                sender_cores (CoreRangeSet): worker cores that may act as pipe senders.
                receiver_domain (CoreRangeSet): worker cores that may act as pipe receivers.
                ring_size (int): per-core data ring size in bytes (multiple of the L1 alignment).
                max_receivers_per_pipe (int): largest receiver count of any pipe carved here, at
                    most receiver_domain.num_cores().
                buffer_type (BufferType): must be L1 (the default); the persistent arena lives in
                    worker L1 and no other buffer type is accepted.

            Returns:
                PrefetcherPipeSpace
        )doc");
}

}  // namespace ttnn::prefetcher_pipe
