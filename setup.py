import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchshiftadd",
    version="0.0.1",
    author="TorchShiftAdd Team",
    author_email="haoran.you@gatech.edu",
    description="A PyTorch library for developing energy efficient multiplication-less models",
    license="Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GATECH-EIC/torchshiftadd",
    packages=setuptools.find_packages(),
    package_data={
        "torchshiftadd": [
            "layers/extension/adder_cuda.cpp",
            "layers/extension/adder_cuda_kernel.cu",
        ]
    },
    install_requires=[
        "torch>=1.7.0",
        "torchvision",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.2.0",
        "tqdm>=4.46.0",
        "ninja",
    ],
    python_requires=">=3.6,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)