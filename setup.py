from setuptools import find_packages, setup

setup(
    name='ddp_mnist',
            packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "mypy",
        ],
    },
    author="Gashon Hussein",
    description="DDP powered residual CNN for mnist classification",
    long_description=open("README.md").read(),
    url="https://github.com/gason/ddp-residual-cnn",
    python_requires=">=3.7",
)


