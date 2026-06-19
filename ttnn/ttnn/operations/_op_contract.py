# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed support-refusal exceptions for the registry-model op contract.

A registry-model op's ``validate()`` is the runtime gate. When an input falls
outside the op's currently-supported rectangle, ``validate()`` raises one of
these typed refusals instead of running the kernel:

- ``UnsupportedAxisValue`` — some axis value is not in its ``SUPPORTED`` list.
- ``ExcludedCell``         — the cell is inside ``cartesian(SUPPORTED)`` but
                             listed in ``EXCLUSIONS`` (refused *for now*).

Both derive from ``SupportRefusal``, which in turn derives from
``NotImplementedError``. The two-level hierarchy lets the test harness
distinguish a *deliberate* refusal from a *genuine* op bug:

- ``eval/golden_harness.py`` marks unsupported / excluded cells
  ``xfail(strict=True, raises=NotImplementedError)`` at collection time — both
  refusal types match because they subclass ``NotImplementedError``.
- ``eval/golden_tests/conftest.py`` converts a refusal raised at *run* time
  (reference-shaped / translated suites, where the support-relevant axes are
  shape-derived and only known once ``validate()`` runs) into a *lenient*
  xfail — matched by ``isinstance(exc, SupportRefusal)``, so an unrelated
  ``NotImplementedError`` (a real bug) still surfaces as a failure.

Message text is free-form; the harness matches on type, never on wording.
"""

from __future__ import annotations

__all__ = ["SupportRefusal", "UnsupportedAxisValue", "ExcludedCell"]


class SupportRefusal(NotImplementedError):
    """Base for a deliberate op support refusal raised by ``validate()``.

    Subclasses ``NotImplementedError`` so the golden harness's
    ``xfail(strict=True, raises=NotImplementedError)`` matches it, while the
    distinct type lets the translated-suite conftest tell a deliberate refusal
    apart from an accidental ``NotImplementedError`` (a genuine bug).
    """


class UnsupportedAxisValue(SupportRefusal):
    """An axis value is outside the op's ``SUPPORTED`` list (per-axis miss)."""


class ExcludedCell(SupportRefusal):
    """A cell inside ``cartesian(SUPPORTED)`` the op refuses for now (``EXCLUSIONS``)."""
