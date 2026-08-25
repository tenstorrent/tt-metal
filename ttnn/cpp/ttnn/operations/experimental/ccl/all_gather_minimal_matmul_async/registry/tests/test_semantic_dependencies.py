# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
MANIFEST = (
    REPOSITORY_ROOT
    / "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/semantic_dependencies.txt"
)
REQUIRED_SEMANTIC_CLOSURE = tuple(
    sorted(
        {
            "ttnn/api/ttnn/config.hpp",
            "ttnn/api/ttnn/operation.hpp",
            "ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp",
            "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp",
            "ttnn/cpp/ttnn/operations/ccl/ccl_host_types.hpp",
            "ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp",
            "ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.hpp",
            "ttnn/cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp",
            "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp",
            "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp",
            "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp",
            "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp",
            "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp",
            "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp",
            "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp",
            "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async_nanobind.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async_nanobind.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation_types.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_program_factory.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_program_factory.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in1_sender_out.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/matmul_dataflow_common.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_config_registry.cpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_config_registry.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_registry_descriptor.hpp",
            "ttnn/cpp/ttnn/operations/experimental/ccl/sources.cmake",
            "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp",
            "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp",
        }
    )
)


def load_manifest() -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_manifest_is_canonical_sorted_unique_and_existing() -> None:
    entries = load_manifest()
    assert entries
    assert entries == tuple(sorted(set(entries)))
    for entry in entries:
        path = PurePosixPath(entry)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert str(path) == entry
        assert (REPOSITORY_ROOT / entry).is_file()


def test_manifest_closes_over_reviewed_agmm_runtime_semantics() -> None:
    # Generated table data is attested by its own content digest. Everything
    # that defines the native key, replay, dispatch, launch, or kernel semantics
    # is pinned here explicitly so a source addition requires a reviewable test
    # update rather than silently weakening the semantic digest.
    assert load_manifest() == REQUIRED_SEMANTIC_CLOSURE
