// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The combine's options: the arrangement the planner (combine_auto) picked, the same for every combine data-movement
// kernel of a call. The host (combine_program.cpp combine_options) gives every combine data-movement kernel all of them
// as named compile-time args; the kernels branch on them with `if constexpr`.
//
//   named arg                              | meaning
//   ---------------------------------------+-------------------------------------------------------------------------
//   accumulators_on_ports                  | the accumulators run on the port cores: the port writer also writes the
//                                          | finished output rows (no separate accumulator writer)
//   per_chip_accumulator_table             | the accumulators sit on different cores on each chip: a port looks a home
//                                          | chip's accumulator cores up in a DRAM table (else one list for every chip)
//   shared_landing                         | rows land in slots each accumulator grants to sources on demand (else in
//                                          | per-source rings with credits)
//   shared_landing_grants_per_source       | shared landing: grants a source may hold at an accumulator at once (1 off)
//   shared_landing_grant_ids_offset        | shared landing: the grant ids' offset in a port core's state (0 off)
//   shared_landing_grant_snapshots_offset  | shared landing: the forwarded grant snapshots' offset there (0 off)
//   landing_on_flat_expert_cores           | the landing slots sit in the L1 the flat expert leaves free on its cores
//                                          | (CombineLanding::FlatCores; else in the accumulators' own L1)
//   landing_reads_in_flight                | landing on the flat expert cores: an accumulator's row reads in flight
//
// Example: separate accumulator cores, the same on every chip, the shared landing on the flat expert cores with 4
// grants a source: accumulators_on_ports 0, per_chip_accumulator_table 0, shared_landing 1,
// shared_landing_grants_per_source 4, shared_landing_grant_ids_offset / _grant_snapshots_offset the state layout's
// (combine_state_layout), landing_on_flat_expert_cores 1, landing_reads_in_flight 2.
#pragma once

#include <cstdint>

namespace combine_options {

constexpr bool accumulators_on_ports = get_named_compile_time_arg_val("accumulators_on_ports") != 0;
constexpr bool per_chip_accumulator_table = get_named_compile_time_arg_val("per_chip_accumulator_table") != 0;
constexpr bool shared_landing = get_named_compile_time_arg_val("shared_landing") != 0;
constexpr uint32_t shared_landing_grants_per_source =
    get_named_compile_time_arg_val("shared_landing_grants_per_source");
constexpr uint32_t shared_landing_grant_ids_offset = get_named_compile_time_arg_val("shared_landing_grant_ids_offset");
constexpr uint32_t shared_landing_grant_snapshots_offset =
    get_named_compile_time_arg_val("shared_landing_grant_snapshots_offset");
constexpr bool landing_on_flat_expert_cores = get_named_compile_time_arg_val("landing_on_flat_expert_cores") != 0;
constexpr uint32_t landing_reads_in_flight = get_named_compile_time_arg_val("landing_reads_in_flight");

static_assert(shared_landing_grants_per_source >= 1);
static_assert(landing_reads_in_flight >= 1);

}  // namespace combine_options
