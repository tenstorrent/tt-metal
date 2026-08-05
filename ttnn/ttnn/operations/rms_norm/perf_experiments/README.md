# rms_norm perf experiments

Durable artifacts from the op's perf tournaments. Each subdirectory is a
self-contained, on-device micro-benchmark of ONE kernel-level idea: a baseline
variant reconstructing what the op does today, one or more candidate variants,
a correctness gate against a torch reference (correctness is the only pass/fail
— perf is measured, never asserted), and the measured numbers stamped with the
box + arch they were taken on.

They are kept so a later round can re-measure a lever instead of re-deriving it,
and so a deferred WIN can be picked up with its numbers intact.

## Do NOT add an `__init__.py` to THIS directory

`ttnn/ttnn/operations/__init__.py` runs `pkgutil.walk_packages(__path__)`, which
imports and EXECUTES every module of every subpackage it can reach at
`import ttnn` time. With an `__init__.py` here, every scratch file in every
experiment dir runs on every `import ttnn` in the repo — during this tournament
that broke `import ttnn` repo-wide twice (a bench that parsed a profiler CSV at
module scope, and a work-in-progress package). Without it, this tree is
invisible to `walk_packages` and the experiments stay inert until run directly.

A marker `__init__.py` inside an individual leaf experiment dir is fine and is
in fact needed so pytest's `--import-mode=importlib` does not resolve a test as
`ttnn.ttnn....` and double-execute `ttnn/ttnn/__init__.py`.

Keep any module-level work behind `if __name__ == "__main__":`.
