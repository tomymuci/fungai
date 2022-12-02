from setuptools import find_packages
from setuptools import setup

<<<<<<< HEAD
with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name = 'FungAI',
      version = "0.0.3",
      description = "FungAI Model (api_pred)",
      license = "MIT",
      author = "",
      author_email = "",
      url = "https://github.com/tomymuci/fungai",
      install_requires = requirements,
      packages = find_packages())
=======
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='fungai',
      version="1.0",
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/fungai-run'],
      zip_safe=False)
>>>>>>> 18bb1ce ('initialcommit')
