# GitHub source release

The anonymous review mirror is available at
[https://anonymous.4open.science/r/RW-Cite/](https://anonymous.4open.science/r/RW-Cite/).

Generate the code-only publication tree from a complete development checkout:

```bash
bash scripts/prepare_public_releases.sh --version 1.0 --code-only
```

The resulting tree contains the installable package, public scripts,
configurations, tests, comparison experiments, and documentation. It excludes
datasets, model checkpoints, logs, outputs, secrets, local merge inputs, and
development-only artifacts.

Before publication, verify the generated tree independently and regenerate
the archive so that the directory and archive contain identical files.
