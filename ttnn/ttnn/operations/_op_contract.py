# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Registry-model support-refusal exception types.

The op-template / verifier docs in this branch reference a small,
stable set of exception types that every registry-conformant op uses
to signal "this combination is not yet supported":

- :class:`SupportRefusal` — typed base class. The golden-test conftest
  ``isinstance(exc, SupportRefusal)`` checks scope translated tests'
  lenient xfail policy. Subclass of :class:`NotImplementedError` so
  the strict golden-test xfail decoration
  (``raises=NotImplementedError``) still matches without per-op
  retuning.
- :class:`UnsupportedAxisValue` — one axis value falls outside
  ``SUPPORTED[axis]``.
- :class:`ExcludedCell` — the cell as a whole is in ``EXCLUSIONS``
  (every axis value is individually inside ``SUPPORTED`` but the
  combination is refused for now).

The op file imports the two leaf classes and raises them in its
``validate()``. The conftest only ever does an ``isinstance(exc,
SupportRefusal)`` typed match — message wording is free.
"""

from __future__ import annotations


class SupportRefusal(NotImplementedError):
    """Base class for all registry-driven "this case is not supported" refusals.

    Subclass of :class:`NotImplementedError` so the existing
    ``pytest.mark.xfail(raises=NotImplementedError)`` decoration in the
    golden harness still matches without per-op tuning. The conftest's
    typed-match path keys off :class:`SupportRefusal` directly so that
    an unrelated ``NotImplementedError`` raised by the kernel (a genuine
    bug) does not get silently swallowed as an xfail.
    """


class UnsupportedAxisValue(SupportRefusal):
    """A single per-axis value falls outside the op's ``SUPPORTED`` list.

    Raised by ``validate()`` when one of the axis-wise lookups against
    ``SUPPORTED`` fails — i.e. the case is refused for axis-level
    reasons, not for any cross-axis combination.
    """


class ExcludedCell(SupportRefusal):
    """The (otherwise-supported) axis combination matches an ``EXCLUSIONS`` entry.

    Raised by ``validate()`` after every per-axis value passed its
    ``SUPPORTED`` check but the resulting cell dict matched one of the
    op's ``EXCLUSIONS`` entries — a refinement-candidate refusal.
    """


__all__ = [
    "ExcludedCell",
    "SupportRefusal",
    "UnsupportedAxisValue",
]
