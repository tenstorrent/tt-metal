# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Base dataclasses for auto-config selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConfigCandidate:
    """A candidate configuration with its metadata."""

    config: Any
    config_family: str  # e.g., "MultiCast1D", "MultiCast2D", "DRAMSharded"
    backend: str  # "matmul" or "minimal_matmul"
    params: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    is_valid: bool = True
    validation_reason: str = "valid"
    measured_latency_us: Optional[float] = None
    math_fidelity: Optional[Any] = None

    def __repr__(self) -> str:
        status = "valid" if self.is_valid else f"invalid({self.validation_reason})"
        fid = f", fidelity={self.math_fidelity.name}" if self.math_fidelity is not None else ""
        return (
            f"ConfigCandidate(family={self.config_family}, backend={self.backend}, "
            f"score={self.score:.2f}, status={status}{fid})"
        )


@dataclass
class SelectionResult:
    """Result of the auto-config selection process."""

    selected_config: Optional[ConfigCandidate]
    all_candidates: List[ConfigCandidate]
    cache_hit: bool = False
    suggestions: List[str] = field(default_factory=list)
    selection_time_ms: float = 0.0

    @property
    def config(self) -> Any:
        if self.selected_config is None:
            return None
        return self.selected_config.config

    @property
    def backend(self) -> str:
        if self.selected_config is None:
            return "matmul"
        return self.selected_config.backend
