from setuptools import find_packages, setup

setup(
    name="halfspace",
    setuptools_git_versioning={"enabled": True},
    setup_requires=["setuptools-git-versioning"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Josh Ivanhoe",
    author_email="joshua.k.ivanhoe@gmail.com",
    url="https://github.com/joshivanhoe/halfspace.git",
    install_requires=[
        "mip>=1.15.0",
        "numpy>=1.25.2",
        "pandas>=2.0.3",
        "plotly>=5.15.0",
    ],
)
