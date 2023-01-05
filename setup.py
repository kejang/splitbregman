import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="splitbregman",
    version="0.0.0",
    author="Kwang Eun Jang",
    author_email="kejang@stanford.edu",
    description="Implementation of Split Bregman",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kejang/splitbregman",
    project_urls={
        "Bug Tracker": "https://github.com/kejang/splitbregman/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy",
        "cupy",
    ],
    include_package_data=True
)
