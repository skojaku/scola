from setuptools import setup, find_packages
from codecs import open

setup(
    name='scola',
    version='0.0.1',
    description='Python code for the Scola algorithm for constructing networks from correlation matrices',
    long_description='Python code for the Scola algorithm for constructing networks from correlation matrices',
    url='https://github.com/skojaku/scola',
    author='Sadamori Kojaku',
    author_email='sadamori.koujaku@bristol.ac.uk',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='correlation matrix filtering lasso',
    packages=find_packages(),
    install_requires=['scipy>=1.0', 'numpy', 'configcorr', 'tdm', 'pickle'],
)
