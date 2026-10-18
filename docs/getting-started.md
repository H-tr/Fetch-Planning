# Getting Started

## Prerequisites

- **Linux** (x86_64, glibc 2.28+) on a CPU with AVX2 (Intel Haswell / AMD Zen or newer)
- **Python** 3.10–3.14

## Installation

Pre-built wheels are available for Python 3.10–3.14 on Linux x86_64. No local compilation required:

```bash
pip install fetch-planning
```

The wheels ship the `ikfast` and `pink` IK backends. The `trac_ik` backend
needs orocos-kdl and NLopt at build time, so it is only available in the
[development setup](#development-setup) below.

## Verify installation

```python
from fetch_planning.fetch import HOME_JOINTS
from fetch_planning.planning import create_planner

planner = create_planner("fetch")
goal = planner.sample_valid()
result = planner.plan(HOME_JOINTS.copy(), goal)
print(f"Planning {'succeeded' if result.success else 'failed'}")
```

## Building Wheels from Source

Wheels are built with [cibuildwheel](https://cibuildwheel.pypa.io) inside
the manylinux Docker images (configured under `[tool.cibuildwheel]` in
`pyproject.toml`). To build one locally:

```bash
pipx run cibuildwheel --platform linux --only cp312-manylinux_x86_64
```

The output goes to `wheelhouse/`. Requirements: Docker must be installed and running.

## Releasing

Pushing a `v*` tag runs `.github/workflows/release.yml`, which builds
manylinux_2_34 and manylinux_2_28 wheels for every supported Python and
publishes them to PyPI via Trusted Publishing:

```bash
# after bumping `version` in pyproject.toml on main
git tag v0.3.0
git push origin v0.3.0
```

## Development Setup

For contributing or rebuilding C++ dependencies from source, use [pixi](https://pixi.sh):

```bash
git clone --recursive https://github.com/H-tr/Fetch-Planning.git
cd Fetch-Planning
bash scripts/setup.sh
```

Or manually:

```bash
git clone --recursive https://github.com/H-tr/Fetch-Planning.git
cd Fetch-Planning
pixi install
pixi run cricket-build
pixi run foam-build
bash scripts/download_assets.sh
```
