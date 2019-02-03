import os
import shutil
import sys

from setuptools import setup, find_packages, Command
from os import path

# To upload:
# pip install --upgrade twine wheel setuptools
# python setup.py upload

NAME = 'incal'
DESCRIPTION = 'Learning SMT(LRA) formulas'
URL = 'https://github.com/smtlearning/incal'
EMAIL = 'samuel.kolb@me.com'
AUTHOR = 'Samuel Kolb'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = "0.1.1"

# What packages are required for this module to be executed?
REQUIRED = [
    'pywmi', 'numpy', 'typing', 'pysmt', 'matplotlib', 'scikit-learn', 'pickledb'
]

# What packages are optional?
EXTRAS = {
        'sdd': ["pysdd"]
}

here = os.path.abspath(os.path.dirname(__file__))

with open(path.join(here, "README.md")) as ref:
    long_description = ref.read()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            shutil.rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        # self.status('Pushing git tags…')
        # os.system('git tag v{0}'.format(about['__version__']))
        # os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=('tests',)),
    zip_safe=False,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    setup_requires=['pytest-runner'],
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "incal-experiments = incal.experiments.cli:main",
            "incal-track = incal.experiments.learn:track",
        ]
    },
    cmdclass={
        'upload': UploadCommand,
    },
)
