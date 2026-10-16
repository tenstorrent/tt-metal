# Derived TTNN ops (bring-up forks)

When a model bring-up needs a change to an existing TTNN op, the op is forked here and changed here. The original op
is never edited. Each fork registers as `ttnn.bringup.<op>`, alongside the original.

Before forking, check this table. If the op is already here, reuse the fork, and put any new change behind an option
whose default keeps the fork's current behaviour. Each fork's `CHANGELOG.md` lists its source path and SHA, and every
change made to it (what, why, which model and task).

Python ops (a ProgramDescriptor op in a subfolder, no C++ build) are listed in `PYTHON_OPS` in `__init__.py`;
`ttnn.bringup.<name>` loads and registers them on first use. Their own unit suite lives in `<fork>/tests/unit/`; the
model cases (bring-up task O.1) are the top-level `tests/test_*.py`.

How to fork: `pipelines/prefill-stepwise/method/FORK_OP.md` in tt-agentic-bringup-sc. In short: `python
ttnn/ttnn/bringup/fork_op.py <source op folder> --model <model> --task <task>`, then make the change, fill the
changelog and this table, and run the fork's tests; the bring-up framework rebuilds tt-metal before a gate when a C++
fork changed. Each fork may carry a best-effort selection of its source op's own tests (`tests/source.yaml`, run in
place with `python -m bringup_fw.testing.fork_source`; baseline in `tests/source_baseline.json`). Every model that
calls a fork adds a random-input case to those tests for each call it makes (bring-up task O.1,
`pipelines/prefill-stepwise/method/FORK_TESTS.md`).

A person decides later whether a fork is ported back to the original op. When it is, the fork is deleted and its
users are switched back.

| Fork | Source @ SHA | Changes | Used by |
|---|---|---|---|
