from setuptools import find_packages, setup
from aml_dbx_cicd import __version__

setup(
    name="aml-databricks-mlops",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="Sample project",
    author="Sunil Sattiraju",
)
