try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    # pip install tomli
    import tomli as tomllib  # Python <3.11

from pathlib import Path
from setuptools import setup, find_packages

with open("pyproject.toml", "rb") as archivo:
    config_project = tomllib.load(archivo)

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=config_project["project"]["name"],  # Nombre del paquete en PyPI
    version=config_project["project"]["version"],
    packages=find_packages(),
    install_requires=[
        "loguru>=0.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Build Tools",
        "Intended Audience :: Developers",
    ],
    description=config_project["project"]["description"],
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/wisrovi/wutils",
    author="William Steve Rodriguez Villamizar",
    author_email="wisrovi.rodriguez@gmail.com",
    license="MIT",
    python_requires=">=3.6",  # Requiere Python >=3.6 y <3.10
)
