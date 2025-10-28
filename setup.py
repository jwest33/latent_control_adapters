from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="latent-control",
    version="1.0.0",
    description="Latent Control Adapters API for Language Models",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "latent-control=latent_control.cli:cli",
        ],
    },
    python_requires=">=3.11",
)
