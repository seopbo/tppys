from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.resolve()

# Get the package version
version = (here / "version.txt").read_text(encoding="utf-8")
# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

with open(here / "requirements" / "requirements-dev.txt") as fp:
    dev_install_requires = fp.read().strip().split("\n")
dev_install_requires = [p.strip() for p in dev_install_requires]

with open(here / "requirements" / "requirements-prod.txt") as fp:
    prod_install_requires = fp.read().strip().split("\n")
prod_install_requires = [p.strip() for p in prod_install_requires]

setup(
    name="tppys",
    version=version,
    author="seopbo",
    author_email="bsk0130@gmail.com",
    description="Data tools for Large Language Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="NLP LLM",
    license="Apache",
    url="https://github.com/seopbo/tppys",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=prod_install_requires,  # External packages as dependencies
    extras_require={
        "dev": dev_install_requires,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
