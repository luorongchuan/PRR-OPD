# Third-Party Notices

The repository combines original PRR integration with modified upstream
research frameworks and evaluation sources. Preserve the following notices
when redistributing; the root MIT file does not override vendored Apache-2.0
licenses or external data/model terms.

| Component | Distributed location | Upstream version / license |
| --- | --- | --- |
| KDFlow and local training modifications | `training/kdflow/`, `training/LICENSE` | Base commit `3cfe26cdf284b4e117ee0479dc974235aec49b5a`; MIT, copyright 2026 Songming Zhang |
| EvalPlus | `evaluation/scorers/pinned14/vendor/test-code/vendor/evalplus/` | Commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`; Apache-2.0, bundled `LICENSE` |
| LiveCodeBench | `evaluation/scorers/pinned14/vendor/test-code/vendor/livecodebench/` | Commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`; MIT, bundled `LICENSE` |
| BigCodeBench | `evaluation/scorers/bigcodebench/` | Commit `09dd993f46c3fbf3a799465bb96d524edcb0b199`; Apache-2.0, bundled `LICENSE` |

`configs/third_party_commits.json` also retains the CRUXEval reference commit.
CRUXEval benchmark examples are not included. Mathematical verification uses
the separately installed Math-Verify and SymPy packages under their own terms.
The local mathematical extraction wrappers are preserved from the experiment.

EvalPlus and BigCodeBench changes consist of local integration, pinned data
selection, execution-resource guards, offline resources, and explicit
canonical-control validation. Package source headers and license files are
retained; this release is not endorsed by those upstream projects.

Qwen3 weights and all mathematical/code benchmarks must be downloaded
separately under their respective licenses. Resource metadata does not grant
redistribution rights to task assets. Dependencies installed into virtual
environments are not copied into the repository.
