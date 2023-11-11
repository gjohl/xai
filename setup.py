import pathlib
from setuptools import setup, find_packages


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='xai',
   version='0.0.1',
   description='Explainable AI research',
   long_description=long_description,
   author='Gurpreet Johl',
   author_email='gurpreetjohl@gmail.com',
   packages=find_packages(),
   install_requires=pathlib.Path('requirements.txt').read_text(),  # external packages as dependencies
   scripts=[],
)
