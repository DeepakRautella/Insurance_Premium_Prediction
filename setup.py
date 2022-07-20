from setuptools import setup,find_packages
from typing import List

#Declaring variables from setup functions
PROJECT_NAME="Insurace_premium_prediction"
VERSION="0.0.3"
AUTHOR="DEEPAK RAUTELLA"
DESCRIPTION="This is my first self made project"
REQUIREMENT_FILE_NAME="requirements.txt"

HYPEN_E_DOT="-e ."


def get_requirements_list()->List[str]:
    """
    Description : This Function is going to return list of requirements
    mention in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file :
        requirement_list=requirement_file.readlines()
        requirement_list=[requirement_name.replace("\n","") for requirement_name in requirement_list]
        if HYPEN_E_DOT in requirement_list:
            requirement_list.remove(HYPEN_E_DOT)

        return requirement_list


setup(
name=PROJECT_NAME,
version=VERSION,
author=AUTHOR,
description=DESCRIPTION,
packages=find_packages(),
install_requires=get_requirements_list()
)