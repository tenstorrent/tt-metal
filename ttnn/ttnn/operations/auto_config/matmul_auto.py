# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-optimal matmul API -- ttnn.matmul_auto

A torch.matmul-like API that automatically selects the most performant
matmul configuration for both single-device and multi-device inputs.

Usage:
    import ttnn
    from ttnn.operations.auto_config import matmul_auto

    # Simple usage (like torch.matmul)
    output = matmul_auto(input_a, input_b)

    # With bias and activation
    output = matmul_auto(input_a, input_b, bias=bias, activation="relu")

    # Accepts host weights -- moves to device automatically
    output = matmul_auto(input_a, host_weights, device=device)

    # Benchmark mode: run all candidates and report timing table
    output = matmul_auto(input_a, input_b, benchmark_mode=True)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import ttnn

from ttnn.operations.auto_config.base import (
    AutoConfigSelector,
    ConfigCandidate,
    SelectionResult,
)
from ttnn.operations.auto_config.candidate_generator import generate_matmul_candidates
from ttnn.operations.auto_config.config_cache import ConfigCache
from ttnn.operations.auto_config.constraint_validator import validate_candidate
from ttnn.operations.auto_config.feature_extraction import (
    extract_matmul_features,
    get_cache_key_from_features,
)
from ttnn.operations.auto_config.scorer.heuristic import HeuristicScorer

logger = logging.getLogger(__name__)

# Global singleton instances
_global_cache: Optional[ConfigCache] = None
_global_scorer: Optional[HeuristicScorer] = None


def _get_global_cache() -> ConfigCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = ConfigCache()
    return _global_cache


def _get_global_scorer() -> HeuristicScorer:
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = HeuristicScorer()
    return _global_scorer


class MatmulAutoConfig(AutoConfigSelector):
    """
    Auto-config selector specialized for matmul operations.

    Implements the full selection pipeline:
    1. Feature extraction from input tensors
    2. Cache lookup for previously-seen signatures
    3. Candidate generation from all config families
    4. Constraint validation (L1, grid, subblock)
    5. Heuristic scoring
    6. Selection and caching

    If all candidates fail validation (edge-case shapes, unusual dtypes),
    the selector returns None to signal that the caller should fall back
    to default ttnn.matmul with no program_config.
    """

    def __init__(self, cache: Optional[ConfigCache] = None, scorer: Optional[HeuristicScorer] = None):
        self._cache = cache or _get_global_cache()
        self._scorer = scorer or _get_global_scorer()
        super().__init__(cache=self._cache, scorer=self._scorer)

    def extract_features(self, *tensors, **kwargs) -> Dict[str, Any]:
        """Extract matmul features from input tensors."""
        if len(tensors) < 2:
            raise ValueError("matmul_auto requires at least 2 input tensors")
        return extract_matmul_features(
            tensors[0],
            tensors[1],
            bias=kwargs.get("bias"),
            transpose_a=kwargs.get("transpose_a", False),
            transpose_b=kwargs.get("transpose_b", False),
            activation=kwargs.get("activation"),
            dtype=kwargs.get("dtype"),
            memory_config=kwargs.get("memory_config"),
        )

    def get_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key from matmul features (includes arch and grid)."""
        return get_cache_key_from_features(features)

    def generate_candidates(self, features: Dict[str, Any]) -> List[ConfigCandidate]:
        """Generate all matmul config candidates."""
        return generate_matmul_candidates(features)

    def validate(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a matmul config candidate."""
        return validate_candidate(
            candidate.config,
            candidate.config_family,
            candidate.backend,
            features,
        )

    def score(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score a matmul config candidate."""
        return self._scorer.score(candidate, features)

    def _generate_suggestions(
        self, features: Dict[str, Any], candidates: List[ConfigCandidate]
    ) -> List[str]:
        """
        Generate user-facing suggestions for better performance.

        This satisfies the bounty requirement: "give suggestions to the user
        that a different layout would be more performant."
        """
        suggestions = []
        selected = candidates[0] if candidates else None
        selected_score = selected.score if selected else 0.0

        # Suggestion 1: Resharding input_tensor_a could help
        if not features["is_a_sharded"]:
            if selected and selected.config_family == "MultiCast1D":
                mcast_in0 = selected.params.get("mcast_in0", False)
                if mcast_in0:
                    suggestions.append(
                        f"[matmul_auto] Suggestion: resharding input_tensor_a to "
                        f"WIDTH_SHARDED could improve throughput for this "
                        f"(M={features['M']}, K={features['K']}, N={features['N']}) matmul "
                        f"by reducing data movement overhead."
                    )
                else:
                    suggestions.append(
                        f"[matmul_auto] Suggestion: resharding input_tensor_a to "
                        f"HEIGHT_SHARDED could improve throughput for this "
                        f"(M={features['M']}, K={features['K']}, N={features['N']}) matmul "
                        f"by reducing data movement overhead."
                    )

        # Suggestion 2: BLOCK_SHARDED could improve 2D multicast
        if not features["is_a_sharded"] and features.get("is_square", False):
            has_2d = any(c.config_family == "MultiCast2D" and c.score > selected_score * 0.9 for c in candidates)
            if has_2d:
                suggestions.append(
                    f"[matmul_auto] Suggestion: resharding input_tensor_a to "
                    f"BLOCK_SHARDED could enable more efficient 2D multicast for this "
                    f"(M={features['M']}, K={features['K']}, N={features['N']}) matmul."
                )

        # Suggestion 3: Check if minimal_matmul offers better throughput
        valid_dtypes = {"DataType.BFLOAT16", "DataType.BFLOAT8_B"}
        if features["dtype_a"] in valid_dtypes and features["layout_a"] == "Layout.TILE":
            has_minimal_winner = any(
                c.backend == "minimal_matmul" and c.score > selected_score * 1.2
                for c in candidates
            )
            if has_minimal_winner:
                suggestions.append(
                    "[matmul_auto] Suggestion: ttnn.experimental.minimal_matmul may offer "
                    "better throughput for this input combination."
                )

        # Suggestion 4: DRAM interleaved could unlock DRAM-sharded path
        if features["is_a_sharded"] and "DRAM" not in features.get("buffer_type_a", ""):
            has_dram_sharded = any(c.config_family == "DRAMSharded" for c in candidates)
            if not has_dram_sharded:
                suggestions.append(
                    f"[matmul_auto] Suggestion: moving input_tensor_a to DRAM_MEMORY_CONFIG "
                    f"could unlock the DRAM-sharded matmul path for this "
                    f"(M={features['M']}, K={features['K']}, N={features['N']}) matmul."
                )

        return suggestions


def _execute_with_config(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    result: SelectionResult,
    *,
    bias: Optional[ttnn.Tensor] = None,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> ttnn.Tensor:
    """Execute matmul with the selected configuration."""
    selected = result.selected_config
    backend = selected.backend

    if backend == "minimal_matmul":
        # Use the experimental minimal_matmul API
        try:
            fused_activation = None
            if activation:
                # Map string activation to UnaryWithParam if needed
                fused_activation = None  # TODO: map activation string

            output = ttnn.experimental.minimal_matmul(
                input_tensor_a,
                input_tensor_b,
                bias_tensor=bias,
                fused_activation=fused_activation,
                config=selected.config,
                memory_config=memory_config,
                dtype=dtype,
            )
            return output
        except Exception as e:
            logger.warning(f"minimal_matmul failed ({e}), falling back to ttnn.matmul")
            # Fall through to standard matmul

    # Standard ttnn.matmul
    program_config = selected.config if selected.config_family != "MinimalMatmul" else None
    if bias is not None:
        output = ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            memory_config=memory_config,
            dtype=dtype,
            program_config=program_config,
            activation=activation,
        )
    else:
        output = ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            memory_config=memory_config,
            dtype=dtype,
            program_config=program_config,
            activation=activation,
        )

    return output


def _execute_fallback(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> ttnn.Tensor:
    """
    Fallback execution: use default ttnn.matmul with no program_config.

    This is the always-valid path used when all auto-selected configs fail.
    """
    logger.warning("[matmul_auto] All candidates failed validation, using default ttnn.matmul (no program_config)")
    if bias is not None:
        return ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            memory_config=memory_config,
            dtype=dtype,
            activation=activation,
        )
    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        activation=activation,
    )


def _execute_multi_device(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    features: Dict[str, Any],
    result: SelectionResult,
    *,
    bias: Optional[ttnn.Tensor] = None,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> ttnn.Tensor:
    """
    Handle multi-device matmul execution with automatic CCL selection.

    Determines whether to use:
    - all_gather + matmul
    - matmul + reduce_scatter
    - Fused all_gather_matmul_async
    - Fused matmul_reduce_scatter_async
    - strided_all_gather_minimal_matmul_async
    """
    device = input_tensor_a.device()
    is_a_sharded = features.get("is_a_sharded", False)
    is_b_sharded = features.get("is_b_sharded", False)
    config = result.config

    # Multi-device strategy selection logic:
    # 1. Row-Parallel (B sharded along K): Requires reduce_scatter on output
    # 2. Column-Parallel (B sharded along N): Replicated A, Sharded B along N. No internal CCL in matmul.
    # 3. Data-Parallel (A sharded along M, B replicated): Requires all_gather on A before matmul.

    try:
        # Case A: Fused strided all_gather + minimal_matmul
        if result.backend == "minimal_matmul" and hasattr(ttnn.experimental, "strided_all_gather_minimal_matmul_async"):
            # This is a specialized path for Llama-like architectures
            # It performs AG on input_a followed by MM
            logger.info("Using ttnn.experimental.strided_all_gather_minimal_matmul_async")
            all_gather_out, mm_out = ttnn.experimental.strided_all_gather_minimal_matmul_async(
                input_tensor_a,
                input_tensor_b,
                dim=0, # Default to dim 0 for AG
                memory_config=memory_config,
            )
            return mm_out

        # Case B: Fused matmul + reduce_scatter
        # Typically used in Row-Parallel linear layers
        if is_b_sharded and hasattr(ttnn.experimental, "matmul_reduce_scatter_async"):
             logger.info("Using ttnn.experimental.matmul_reduce_scatter_async")
             # Returns (matmul_out, reduce_scatter_out)
             mm_out, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
                 input_tensor_a,
                 input_tensor_b,
                 scatter_dim=0,
                 math_op=ttnn.ReduceType.Sum,
                 memory_config=memory_config,
                 program_config=config,
             )
             return rs_out

        # Case C: Fused all_gather + matmul
        # Typically used in Data-Parallel or Column-Parallel where A needs AG
        if is_a_sharded and not is_b_sharded and hasattr(ttnn.experimental, "all_gather_matmul_async"):
            logger.info("Using ttnn.experimental.all_gather_matmul_async")
            output = ttnn.experimental.all_gather_matmul_async(
                input_tensor_a,
                input_tensor_b,
                dim=0,
                memory_config=memory_config,
                program_config=config,
            )
            return output

    except Exception as e:
        logger.warning(f"Fused CCL path failed, falling back to standard execution: {e}")

    # Fallback: Let TTNN's standard multi-device tensor dispatch handle it
    # or perform separate CCL if necessary (though usually dispatch is enough if tensors are correctly sharded)
    return _execute_with_config(
        input_tensor_a,
        input_tensor_b,
        result,
        bias=bias,
        activation=activation,
        dtype=dtype,
        memory_config=memory_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


def _ensure_on_device(
    tensor: ttnn.Tensor,
    device: Optional[ttnn.Device],
) -> ttnn.Tensor:
    """
    Ensure a tensor is on device. Handles host weights by moving to device.

    This satisfies the bounty requirement: "accepting host weights."
    """
    try:
        # If tensor already has a device, it's on device
        tensor.device()
        return tensor
    except Exception:
        # Tensor is on host — move to device
        if device is None:
            raise ValueError(
                "matmul_auto received a host tensor but no device was specified. "
                "Pass device= to matmul_auto() when using host weights."
            )
        return ttnn.to_device(tensor, device)


def matmul_auto(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    device: Optional[ttnn.Device] = None,
    bias: Optional[ttnn.Tensor] = None,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    force_program_config: Optional[Any] = None,
    benchmark_mode: bool = False,
) -> ttnn.Tensor:
    """
    Auto-optimal matmul -- a torch.matmul-like API for TTNN.

    Automatically selects the most performant matmul configuration for the
    given inputs. Supports both single-device and multi-device tensors.

    Features:
    - Accepts host weights and handles to_device internally
    - Detects single-device vs multi-device from input tensors
    - For multi-device: automatically selects CCL flow
    - Caches optimal configs for cross-session reuse
    - Logs suggestions when a different input layout would improve perf
    - Falls back to default ttnn.matmul when all candidates fail

    Args:
        input_tensor_a: First input tensor (activations).
        input_tensor_b: Second input tensor (weights). Can be on host.
        device: Target device. Required if input_tensor_b is on host.
            Inferred from input_tensor_a if not provided.
        bias: Optional bias tensor.
        activation: Optional fused activation (e.g., "relu", "gelu").
        dtype: Output data type.
        memory_config: Output memory configuration.
        transpose_a: Whether to transpose input A.
        transpose_b: Whether to transpose input B.
        force_program_config: Manual override for the program config.
            If provided, bypasses auto-selection entirely.
        benchmark_mode: If True, run all valid candidates and report timing.

    Returns:
        Output tensor with the result of the matmul operation.
    """
    # Infer device from input_tensor_a if not provided
    if device is None:
        try:
            device = input_tensor_a.device()
        except Exception:
            pass

    # Handle host weights — move to device automatically
    input_tensor_a = _ensure_on_device(input_tensor_a, device)
    input_tensor_b = _ensure_on_device(input_tensor_b, device)
    if bias is not None:
        bias = _ensure_on_device(bias, device)

    # Manual override path
    if force_program_config is not None:
        if bias is not None:
            return ttnn.linear(
                input_tensor_a,
                input_tensor_b,
                bias=bias,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                memory_config=memory_config,
                dtype=dtype,
                program_config=force_program_config,
                activation=activation,
            )
        return ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            memory_config=memory_config,
            dtype=dtype,
            program_config=force_program_config,
            activation=activation,
        )

    # Auto-selection path
    selector = MatmulAutoConfig()

    result = selector.select(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        activation=activation,
        dtype=dtype,
        memory_config=memory_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )

    # Fallback chain: if all candidates fail, use default ttnn.matmul
    if result is None or result.selected_config is None:
        return _execute_fallback(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            activation=activation,
            dtype=dtype,
            memory_config=memory_config,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    # Log selection info
    selected = result.selected_config
    logger.info(
        f"[matmul_auto] Selected: {selected.config_family} "
        f"(backend={selected.backend}, score={selected.score:.3f}, "
        f"cache_hit={result.cache_hit}, time={result.selection_time_ms:.1f}ms)"
    )

    # Log suggestions
    for suggestion in result.suggestions:
        logger.warning(suggestion)

    # Benchmark mode: run all valid candidates and report timing
    if benchmark_mode:
        return _benchmark_all_candidates(
            input_tensor_a,
            input_tensor_b,
            result,
            bias=bias,
            activation=activation,
            dtype=dtype,
            memory_config=memory_config,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    # Extract features to check multi-device
    features = selector.extract_features(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        activation=activation,
        dtype=dtype,
        memory_config=memory_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )

    # Execute
    if features.get("is_multi_device", False):
        return _execute_multi_device(
            input_tensor_a,
            input_tensor_b,
            features,
            result,
            bias=bias,
            activation=activation,
            dtype=dtype,
            memory_config=memory_config,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    return _execute_with_config(
        input_tensor_a,
        input_tensor_b,
        result,
        bias=bias,
        activation=activation,
        dtype=dtype,
        memory_config=memory_config,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


def _benchmark_all_candidates(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    result: SelectionResult,
    **kwargs,
) -> ttnn.Tensor:
    """
    Run all valid candidates and report timing with a human-readable table.

    Prints a comparison table like:
        Config                                 | Latency (us) | Score | Selected
        MatmulMultiCoreReuseMultiCast1D        |          312 | 0.87  | *
        MatmulMultiCoreReuseMultiCastProgram   |          398 | 0.72  |
        MatmulMultiCoreReuseProgram            |          445 | 0.65  |
    """
    valid = [c for c in result.all_candidates if c.is_valid]
    best_time = float("inf")
    best_output = None
    best_candidate = None
    measured = []

    logger.info(f"[matmul_auto] Benchmark mode: testing {len(valid)} candidates")
    logger.info("")
    logger.info(f"{'Config':<45} | {'Latency (us)':>12} | {'Score':>6} | {'Status'}")
    logger.info(f"{'-'*45}-+-{'-'*12}-+-{'-'*6}-+-{'-'*10}")

    for i, candidate in enumerate(valid):
        try:
            # Warmup
            warmup_result = SelectionResult(
                selected_config=candidate,
                all_candidates=[candidate],
            )
            _ = _execute_with_config(input_tensor_a, input_tensor_b, warmup_result, **kwargs)
            ttnn.synchronize_device(input_tensor_a.device())

            # Timed runs (take median of 5)
            times = []
            for _ in range(5):
                ttnn.synchronize_device(input_tensor_a.device())
                start = time.perf_counter()
                output = _execute_with_config(input_tensor_a, input_tensor_b, warmup_result, **kwargs)
                ttnn.synchronize_device(input_tensor_a.device())
                elapsed = (time.perf_counter() - start) * 1e6  # microseconds
                times.append(elapsed)

            median_time = sorted(times)[len(times) // 2]
            candidate.measured_latency_us = median_time




            if median_time < best_time:
                best_time = median_time
                best_output = output
                best_candidate = candidate

            measured.append((candidate, median_time))

        except Exception as e:
            candidate.measured_latency_us = None
            logger.info(
                f"{candidate.config_family:<45} | {'FAILED':>12} | {candidate.score:>5.2f} | {str(e)[:30]}"
            )

    # Print sorted table
    if measured:
        measured.sort(key=lambda x: x[1])
        logger.info("")
        logger.info(f"{'Config':<45} | {'Latency (us)':>12} | {'Score':>6} | {'Selected'}")
        logger.info(f"{'-'*45}-+-{'-'*12}-+-{'-'*6}-+-{'-'*10}")
        for cand, lat in measured:
            marker = "  *" if cand is best_candidate else ""
            logger.info(f"{cand.config_family:<45} | {lat:>12.0f} | {cand.score:>5.2f} |{marker}")

    if best_candidate:
        logger.info("")
        logger.info(
            f"[matmul_auto] Benchmark winner: {best_candidate.config_family} "
            f"({best_time:.0f} us)"
        )
        # Update cache with the benchmark winner
        selector = MatmulAutoConfig()
        features = selector.extract_features(input_tensor_a, input_tensor_b, **kwargs)
        cache_key = selector.get_cache_key(features)
        _get_global_cache().put(cache_key, best_candidate)

    return best_output


# Convenience: make matmul_auto available at module level
__all__ = ["matmul_auto", "MatmulAutoConfig"]
