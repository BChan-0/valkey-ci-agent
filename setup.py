"""Compatibility packaging entrypoint for older pip/setuptools builders."""

from setuptools import find_packages, setup


setup(
    name="valkey-ci-agent",
    version="0.1.0",
    description="CI automation agent for the Valkey project",
    python_requires=">=3.9",
    license="Apache-2.0",
    packages=find_packages(include=["scripts", "scripts.*"]),
    install_requires=[
        "PyGithub>=2.1.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "hypothesis>=6.82.0",
            "mypy>=1.5.0",
            "pytest-mock>=3.11.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
