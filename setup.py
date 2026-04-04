from setuptools import setup, find_packages

setup(
    name="cleanframe",
    version="0.3.0",
    packages=find_packages(),
    install_requires=["pandas"],
    python_requires=">=3.8",
)