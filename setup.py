#! /usr/bin/python3

from setuptools import setup

setup(
    install_requires=["encore", "glue",], #"pycairo/cairocffi"
    name="glance",
    version = "0.5",
    license = 'MIT',
    author="Daniel Filonik",
    author_email="d.filonik@hdr.qut.edu.au",
    description = 'High-level graphics abstractions library.',
    url = 'http://github.com/filonik/glance',
    packages = [
        'glance',
        'glance.colors',
        'glance.graphics',
        'glance.mathematics',
        'glance.mathematics.geometries',
        'glance.painters',
        'glance.palettes',
        'glance.scenes',
        'glance.shapes',
        'glance.tiles',
        'glance.visualizations',
    ],
    package_data = {
        "glance.colors": [ "*.json" ],
        "glance.palettes": [ "*.json" ],
        "glance.shapes": [ "*.json" ],
    }
)