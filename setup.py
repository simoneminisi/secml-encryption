from setuptools import setup, find_packages

import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: Implementation :: PyPy
Topic :: Software Development
Topic :: Scientific/Engineering
"""

setup(
    name="secml-encryption",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(
        where="src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    url="",
    license="MIT",
    author="Simone Minisi",
    author_email="simone.minisi@dibris.unige.it",
    description="SecML-Torch Encryption Plugin",
    include_package_data=True,
    install_requires=[
          'torch',
          'numpy',
          'scikit-learn',
          'tenseal'
    ],
)
