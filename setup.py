import setuptools
import os
from pathlib import Path

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

module_path = Path(os.path.abspath(__file__)).parent.absolute()
package_name = "splitbregman"

try:
    pkg_version = version(package_name)
except Exception:
    pkg_version = "0.0.3"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=package_name,
    version=pkg_version,
    author="Kwang Eun Jang",
    author_email="kejang@stanford.edu",
    description="Implementation of Split Bregman",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kejang/splitbregman",
    project_urls={
        "Bug Tracker": "https://github.com/kejang/splitbregman/issues",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "splitbregman": [
            "thresholding/cuda/*.cu",
            "derivative/cuda/*.cu",
        ]
    },
)