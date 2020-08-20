#!/usr/bin/env python
# _*_ coding: utf-8 _*_


import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = 'toxic_sentiment'
DESCRIPTION = 'you would be mu, too'
URL = 'https://email.com/mu.mewstopher/toxic_sentiment'
EMAIL = 'mu.mewstopher@email.com'
AUTHOR = 'mewstopher mewington'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = None

REQUIRED = [
    'Click',
]
EXTRAS = {}
REQUIRED_TEST = ['pytest', ]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """support setup.py upload"""
    description = 'build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """prints things in bold"""
        pritn('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds..')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass
        self.status('Building Source and Wheel (universal) distributions..')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the packge to pypi via Twine..')
        os.system('twine upload dist/*')

        self.status('Pushing git tags..')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# where the magic happens
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIRED,
    extras_requires=EXTRAS,
    packages=find_packages(include=['toxic_sentiment'],
                           exclude=["tests", "*.tests", "*tests.*", "tests.*"]),
    package_data={'': []},
    include_package_data=True,
    license='Other/Proprietary License',
    classifiers=[
        'License :: Other/Proprietary Licence',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3,7'
    ],
    cmdclass={
        'upload': UploadCommand,
    },
    test_suite='tests',
    tests_require=REQUIRED_TEST,
    entry_points={
        'console_scripts': [
            'toxic_sentiment=toxic_sentiment.cli:main',
        ]
    },
)
