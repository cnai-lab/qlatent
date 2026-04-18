from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8-sig")
setup(
    name="qlatent",
    version="1.0.24",
    description="A Python package for running psychometric on LLMs.",
    packages=find_packages(),
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
    "numpy",
    "pandas",
    "seaborn",
    "scipy",
    "scikit-learn",
    "transformers",
    "pingouin",
    "typeguard",
    "datasets",
    "evaluate",
    "matplotlib",
    "sentence_transformers",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)