# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent configuration cache for auto-config selection."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from ttnn._experimental.auto_config.base import ConfigCandidate

logger = logging.getLogger(__name__)

SELECTOR_VERSION = "2.0.0"

DEFAULT_CACHE_DIR = os.environ.get(
    "TTNN_AUTO_CONFIG_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".ttnn", "auto_config_cache"),
)

_CACHE_HOME_DIR = os.path.abspath(os.path.join(os.path.expanduser("~"), ".ttnn"))
_CACHE_CWD_DIR = os.path.abspath(os.getcwd())
_CACHE_DEFAULT_DIR = os.path.abspath(DEFAULT_CACHE_DIR)
_CACHE_TEMP_DIR = os.path.abspath(tempfile.gettempdir())


def _sanitize_cache_dir(cache_dir: str) -> Path:
    """Resolve cache_dir and verify it lives under an allowed directory."""
    resolved = os.path.abspath(cache_dir)
    for base in (_CACHE_HOME_DIR, _CACHE_CWD_DIR, _CACHE_DEFAULT_DIR, _CACHE_TEMP_DIR):
        if resolved.startswith(base):
            return Path(resolved)
    raise ValueError(f"Cache directory '{resolved}' is outside allowed directories")


class ConfigCache:
    """Thread-safe, persistent cache for optimal matmul configurations."""

    def __init__(self, cache_dir: Optional[str] = None, max_entries: int = 10000):
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._max_entries = max_entries

        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        self._cache_dir = _sanitize_cache_dir(cache_dir)
        self._cache_file = self._cache_dir / "matmul_configs.json"

        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load cache entries from disk. Auto-deletes on corruption."""
        if not self._cache_file.exists():
            return
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            for key, entry in data.items():
                if entry.get("version") == SELECTOR_VERSION:
                    self._memory_cache[key] = entry
            logger.debug(f"Loaded {len(self._memory_cache)} cache entries from {self._cache_file}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load config cache (auto-deleting corrupt cache): {e}")
            try:
                self._cache_file.unlink()
            except OSError:
                pass

    def _save_to_disk(self) -> None:
        """Persist current cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(self._memory_cache, indent=2, default=str), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to save config cache: {e}")

    def _hash_key(self, key: str) -> str:
        """Create a compact hash of the cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, key: str, features: Optional[Dict[str, Any]] = None) -> Optional[ConfigCandidate]:
        """Look up a cached config by key. Returns None on miss or reconstruction failure."""
        hashed = self._hash_key(key)
        with self._lock:
            entry = self._memory_cache.get(hashed)
            if entry is None:
                return None
            if entry.get("version") != SELECTOR_VERSION:
                del self._memory_cache[hashed]
                return None

            if features is None:
                return None

            try:
                from ttnn._experimental.auto_config.candidate_generator import _build_config_from_params

                params = entry.get("params", {})
                params["config_family"] = entry.get("config_family", "MultiCast1D")
                candidate = _build_config_from_params(params, features)
                if candidate is not None:
                    candidate.score = entry.get("score", 0.0)
                    candidate.is_valid = True
                    candidate.validation_reason = "cached"
                return candidate
            except Exception as e:
                logger.debug("Failed to reconstruct cached config: %s", e)
                return None

    def put(self, key: str, candidate: ConfigCandidate) -> None:
        """Store a config candidate in the cache."""
        hashed = self._hash_key(key)
        with self._lock:
            if len(self._memory_cache) >= self._max_entries:
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]

            self._memory_cache[hashed] = {
                "version": SELECTOR_VERSION,
                "config_family": candidate.config_family,
                "backend": candidate.backend,
                "params": candidate.params,
                "score": candidate.score,
                "original_key": key[:200],
            }

            self._save_to_disk()

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._memory_cache.clear()
            if self._cache_file.exists():
                try:
                    self._cache_file.unlink()
                except OSError:
                    pass
            logger.info("Config cache cleared")

    def __len__(self) -> int:
        return len(self._memory_cache)
