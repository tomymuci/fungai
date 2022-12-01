from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name = 'FungAI',
      version = "0.0.2",
      description = "FungAI Model (api_pred)",
      license = "MIT",
      author = "",
      author_email = "",
      url = "https://github.com/tomymuci/fungai",
      install_requires = requirements,
      packages = find_packages())
