from setuptools import setup, find_packages

# Distribute: python setup.py sdist upload

setup(
    name='incal',
    version='0.0.1',
    description='Learning SMT(LRA) formulas',
    url='https://github.com/smtlearning/incal',
    author='Samuel Kolb',
    author_email='samuel.kolb@me.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['pywmi', 'numpy', 'typing', 'pysmt', 'matplotlib', 'scikit-learn'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "incal-experiments = incal.data.cli:main",
            "incal-track = incal.data.learn:track",
            "incal = incal.cli:main"
        ]
    }
)
