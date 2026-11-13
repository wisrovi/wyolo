"""Setup configuration for the wyolo package."""

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python <3.11

from setuptools import find_packages, setup

with open("pyproject.toml", "rb") as archivo:
    config_project = tomllib.load(archivo)

this_directory = Path(__file__).parent
readme_file = this_directory / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

setup(
    name=config_project["project"]["name"],
    version=config_project["project"]["version"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"wyolo": ["py.typed", "*.mermaid", "**/*.mermaid"]},
    install_requires=config_project["project"].get("dependencies", []),
    extras_require=config_project["project"].get("optional-dependencies", {}),
    classifiers=config_project["project"].get("classifiers", []),
    description=config_project["project"]["description"],
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/wisrovi/wyoloservice2_worker",
    author="William Steve Rodriguez Villamizar",
    author_email="wisrovi.rodriguez@gmail.com",
    license="MIT",
    python_requires=config_project["project"].get("requires-python", ">=3.8"),
)
