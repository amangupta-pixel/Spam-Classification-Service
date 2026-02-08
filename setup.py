'''
The setup.py is an essential part of packaging and distributing the Python Projects. 
It is used by setup tools (or distutils in python versions) to define the configuration of your projects, 
such as its metadata, dependencies and more
'''

from setuptools import find_packages, setup
import os
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements

    """
    requirement_lst:List[str] = []
    file_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(file_path,'r') as file:
            # Read lines from the line
            lines = file.readlines()
            ## Process each line
            for line in lines:
                requirement = line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
    
    except FileNotFoundError:
        print("requirements.txt file not found")
        
    return requirement_lst

setup(
    name="Spam-Ham-Classifier",
    version="0.0.1",
    author="Aman Gupta",
    author_email="guptaaman89630@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()
)
