"""Package metadata kept in setup.py for old-pip and offline compatibility."""

from setuptools import find_packages, setup


setup(
    name="longtraj-attribution",
    version="0.2.0",
    description="Critical-step attribution for long failed agent trajectories",
    packages=find_packages(include=("longtraj", "longtraj.*")),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.30.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "longtraj-attribution=longtraj.integration:main",
        ]
    },
)
