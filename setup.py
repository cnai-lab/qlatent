from setuptools import setup, find_packages

setup(
name="qlatent",
version="1.0.7",
description="A Python package for running psychometric on LLMs.",
packages=find_packages(
        where=".",
        exclude=['data', 'qmnli','docs'],  # ['*'] by default
    ),
classifiers=[
"Programming Language :: Python :: 3",
"License :: OSI Approved :: Apache License 2.0",
"Operating System :: OS Independent",
],
include_package_data=True,
python_requires=">=3.8",
install_requires=parse_requirements('requirements.txt')
)