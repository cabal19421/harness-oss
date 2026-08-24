"""Setuptools configuration for the harness grounding gate + PRs pipeline."""

from pathlib import Path

from setuptools import find_packages, setup

# Read version from package __init__.py without importing the whole package.
_init = Path(__file__).resolve().parent / "harness" / "__init__.py"
_version = "0.1.0"
for line in _init.read_text(encoding="utf-8").splitlines():
    if line.startswith("__version__"):
        _version = line.split("=")[1].strip().strip("\"'")
        break

setup(
    name="harness",
    version=_version,
    description="Neuro-symbolic grounding gate + a design-docs → PRs pipeline with anti-hallucination gates.",
    long_description=(Path(__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
    if (Path(__file__).resolve().parent / "README.md").exists()
    else "",
    long_description_content_type="text/markdown",
    author="Harness Contributors",
    url="https://github.com/your-org/harness-oss",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    # Core install needs NOTHING beyond the stdlib. Extras unlock optional
    # features (see INSTALL.md). External tools (git/gh/tmux/go and an agent
    # CLI such as gemini) are installed via your OS package manager, not pip.
    install_requires=[],
    extras_require={
        # SMT backend for grounding: decides the 'call_binding' and
        # 'guard_exclusivity' constraint kinds (the builtin solver abstains on
        # both). Not a speed-up — a capability. See INSTALL.md.
        #
        # Deliberately UNCAPPED. Tested range: 4.12 … 5.1.0.0. The 4.x→5.x major
        # bump broke nothing this package uses (Int/Real/RealVal/BoolVal/And/Or/
        # Not/Solver/add/check/assertions/sat/unsat all predate 4.12; 5.0.0's one
        # documented break is Java-only), and `Z3Solver.__init__` now probes those
        # names at construction time — so an incompatible future z3 degrades to
        # the builtin solver (or raises under --require-z3) instead of being
        # pre-emptively refused at install time by a cap that guesses wrong.
        "z3": ["z3-solver>=4.12"],
        # Process reclamation inside a worktree (procutil.py). Optional ONLY on
        # Linux, where there is a /proc fallback — on macOS and Windows psutil is
        # the sole implementation and the fallback is a no-op, so unattended runs
        # there want this extra. Floor 7.1.3: 7.1.0 fixed the Process.cwd()
        # FileNotFoundError race on the exact path procutil walks (#2514/#2515),
        # and 7.1.2/7.1.3 fixed C-extension crashes — a SIGSEGV would blow
        # straight through procutil's "it never raises" contract.
        "proc": ["psutil>=7.1.3"],
        "dev": [                      # contributor toolchain (see CONTRIBUTING.md)
            # 9.1.1, not 7.0: <9.0.3 has a predictable /tmp/pytest-of-<user> root
            # (CVE-2025-71176), and 9.0.x silently IGNORES --strict-markers /
            # --strict-config passed through `addopts` — a silent weakening of the
            # oracle harness runs against gated repos. 9.1.0 is skipped on purpose:
            # it drops initial conftests under a `test*` directory on an
            # argument-less run, which is exactly this repo's layout.
            "pytest>=9.1.1",
            # Floors left alone on purpose: neither project's changelog states a
            # pytest-9 compatibility boundary, and both declare only a pytest lower
            # bound, so raising these would be a guess. (Note if you do revisit:
            # pytest-cov 7.0 dropped subprocess measurement and needs coverage
            # >=7.10.6.)
            "pytest-cov>=4.0",
            "pytest-mock>=3.10",
            # Capped. mypy 2.0 turned `local_partial_types` and `strict_bytes` ON
            # by default and rejects `--python-version 3.9` outright; the cap is
            # what stops a 3.0 doing the same again silently. mypy.ini pins every
            # flipped default explicitly so 1.x and 2.x agree on this codebase.
            "mypy>=2.0,<3",
            "ruff>=0.1",
            "pre-commit>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "harness=harness.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
