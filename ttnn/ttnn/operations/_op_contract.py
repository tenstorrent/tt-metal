# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Typed support-refusals shared by every op written under the registry model.

An op's ``validate()`` gate raises one of these when a call falls outside the
op's declared support rectangle. Both derive from ``NotImplementedError`` so
the golden harness's ``xfail(strict=True, raises=NotImplementedError)`` marks
catch them, and so callers can distinguish "this op does not do that (yet)"
from "you passed me something malformed" (``ValueError`` / ``RuntimeError``).

See ``eval/REGISTRY_MODEL.md``.
"""

__all__ = [
    "OpContractError",
    "SupportRefusal",
    "UnsupportedAxisValue",
    "ExcludedCell",
]


class OpContractError(NotImplementedError):
    """Base class for support-refusals raised by an op's validate()."""


#: Canonical name used by the golden harness (`eval/golden_tests/conftest.py`
#: imports `SupportRefusal` and isinstance-matches it to convert a deliberate
#: refusal into a lenient xfail). Kept as an alias of the base class so the two
#: vocabularies name exactly one type — never two parallel hierarchies.
SupportRefusal = OpContractError


class UnsupportedAxisValue(OpContractError):
    """A single axis value is outside the op's SUPPORTED list for that axis."""


class ExcludedCell(OpContractError):
    """Every axis value is SUPPORTED, but the combination is in EXCLUSIONS."""
