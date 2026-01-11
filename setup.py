from setuptools import find_packages, setup

setup(
    name="land_classifier",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.16.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "ruff",
            "isort",
            "pre-commit",
        ]
    },
)
