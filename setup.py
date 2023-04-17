from setuptools import find_packages, setup


# Package meta-data.
NAME = 'propinf'
DESCRIPTION = ''
URL = ''
AUTHOR = ''
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.0.1'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    license='MIT'
)


