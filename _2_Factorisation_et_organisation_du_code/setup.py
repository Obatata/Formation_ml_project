#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'heart-classification-model'
DESCRIPTION = "A remplir selon le contenu de votre package ! ."
URL = "https://github.com/Obatata/Formation_ml_project"
EMAIL = "batata.oussama@gmail.com"
AUTHOR = "Obatata"
REQUIRES_PYTHON = ">=3.7.0"

long_description = DESCRIPTION

setup(
    name=NAME,
    version="1.0",
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("test_prediction_model")),
)
