# Contributing

Thanks for taking a look at the project. This is a cleaned dissertation codebase rather than a large maintained framework, so contributions are most useful when they keep the repository reproducible and easy to inspect.

Good contribution candidates:

- fixes to installation or environment recreation,
- small bug fixes in the shared SAC/objective code,
- clearer documentation for reproducing experiments,
- additional analysis scripts that consume generated CSV artifacts,
- narrowly scoped robustness conditions with recorded manifests.

Please keep generated outputs, large checkpoints, videos, raw TensorFlow event logs, generated caches, and private dissertation drafts out of Git. If a change affects reported results, include the command used to regenerate them and the relevant seed/configuration details.
