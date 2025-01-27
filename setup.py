from setuptools import setup, find_packages

setup(
    name="quantum-vae",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Quantum Variational Autoencoder for reconstructing low-dimensional manifolds.",
    url="https://github.com/trabbani/quantum-vae",
    packages=find_packages(),
    install_requires=[
        "plotly==5.24.1",
        "nbformat==5.10.4",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
