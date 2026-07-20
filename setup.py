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
    url="https://github.com/cabal19421/harness-oss",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    # Core install needs NOTHING beyond the stdlib. Extras unlock optional
    # features (see INSTALL.md). External tools (git/gh/tmux/claude/go) are
    # installed via your OS package manager, not pip.
    install_requires=[],
    extras_require={
        "z3": ["z3-solver>=4.12"],   # z3-accelerated grounding constraint solver
        "dev": [                      # contributor toolchain (see CONTRIBUTING.md)
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.10",
            "mypy>=1.0",
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
