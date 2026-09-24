# Installation

Everything here was developed and run on one configuration: **Ubuntu 24.04
LTS, Python 3.12, an RTX 5090** (sm_120, Blackwell) on driver 580. Other
distributions and GPUs are not claimed to work, they are simply untested.
Python 3.11 is the lowest version the package accepts, which rules out a
stock Ubuntu 22.04 (Python 3.10) unless you bring your own interpreter.

Isaac Sim 6.0 and PyTorch with CUDA 12.8 are required for the simulator
stages, which are recording, replay and evaluation. Trajectory extraction, fusion, training and reporting
run without a simulator.

:::{admonition} Quickstart
:class: tip

```bash
# adjust: the URL you clone from
REPO_URL=https://github.com/OWNER/so101-camera-multiview-bench.git

git clone "$REPO_URL"
cd so101-camera-multiview-bench

# 1. Fresh environment (any venv/conda works; Python >= 3.11)
python3 -m venv .venv && source .venv/bin/activate

# 2. Keep the machine's global packages out of this environment (see below)
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES

# 3. PyTorch with CUDA 12.8
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Isaac Sim + Isaac Lab (NVIDIA's index, not PyPI)
pip install --extra-index-url https://pypi.nvidia.com "isaacsim[all]==6.0.*"
pip install isaaclab

# 5. Pin torch back (Isaac Sim downgrades it)
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall --no-deps

# 6. This package
pip install -e '.[sim]'
```
:::

Skip steps 3 to 5 and install with plain `pip install -e .` if you only need
the simulator-free stages (trajectory extraction, training, reporting).

## The three environment lines, and what they keep out

They look like superstition, so here is what each one prevents. Set them in
every shell that runs a simulator command, not only during installation.

**`unset PYTHONPATH`.** Isaac Lab needs a clean interpreter, and the usual
reason it does not get one is ROS: sourcing a ROS setup script exports
`PYTHONPATH=/opt/ros/<distro>/lib/python3.X/site-packages`, and many setups do
that from `~/.bashrc` for every shell. The environment then imports ROS
packages built against a different Python, which surfaces as `rclpy` import
errors or as dependency conflicts that make no sense. Check yours with
`echo $PYTHONPATH`. If it is empty, this line does nothing, and you can keep
it anyway.

**`export PYTHONNOUSERSITE=1`.** Python reads the user-wide package directory
`~/.local/lib/python3.X/site-packages` even inside a virtual environment.
Anything you once installed there with `pip install --user` silently takes
precedence over what you installed here, and the resulting import error names
a package that `pip list` shows as present.

**`export OMNI_KIT_ACCEPT_EULA=YES`.** Records that you accept the Isaac Sim
licence agreement. Without it the simulator stops and waits for an answer,
which in a headless or scripted run looks like a hang.

## CUDA toolchain notes

- **cu128 only.** `cu130` ships unversioned NVIDIA wheels that crash Isaac
  Sim with `Bus Error`. Pin to `cu128`.
- **`torchcodec`.** Must match the PyTorch ABI. `torchcodec==0.10.0` is
  compatible with `torch==2.10+cu128`.

## Verifying the install

```bash
# Simulator-free: the package imports and the unit tests pass
python -c "import so101_mvbench; print(so101_mvbench.__file__)"
pip install -e '.[dev]' && pytest tests

# With Isaac Sim: the cheapest boot, prints the registered LiftCube-* tasks
python -m so101_mvbench.tools.list_envs
```

`list_envs` should print `LiftCube-Sim`, the OOD variants
(`-OOD-Light`, `-OOD-Position`, `-OOD-Visual`), and the multi-external
variants (`-2Ext`, `-3Ext`, `-4Ext`, `-6Cam`).

## Building this documentation

```bash
bash docs/serve_docs.sh    # bootstraps a small repo-local venv, then live-reloads
```

Before committing a documentation change, run the strict build:

```bash
cd docs && make check
```

It rebuilds from scratch (`-E`) and turns warnings into failures (`-W`). The
rebuild matters: an incremental build reports success for pages it never
re-read, which is how two broken docstrings once passed a green check.

## Next step

Continue with {doc}`first_steps` to record your first one-episode dataset.
