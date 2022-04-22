#!/usr/bin/env python
from setuptools import setup, find_packages
import os


class PackageInfo(object):
    def __init__(self, info_file):
        with open(info_file) as f:
            exec(f.read(), self.__dict__)
        self.__dict__.pop('__builtins__', None)

    def __getattribute__(self, name):
        return super(PackageInfo, self).__getattribute__(name)


package_info = PackageInfo(os.path.join('codableopt', 'package_info.py'))

setup(
    name=package_info.__package_name__,
    version=package_info.__version__,
    description=package_info.__description__,
    long_description=open('README.rst').read(),
    author=package_info.__author_names__,
    author_email=package_info.__author_emails__,
    maintainer=package_info.__maintainer_names__,
    maintainer_email=package_info.__maintainer_emails__,
    url=package_info.__repository_url__,
    download_url=package_info.__download_url__,
    license=package_info.__license__,
    packages=find_packages(exclude='sample'),
    keywords=package_info.__keywords__,
    zip_safe=False,
    install_requires=['numpy>=1.22.0'],
    python_requires='>=3.7, <3.11'
)
