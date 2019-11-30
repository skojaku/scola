from setuptools import setup, find_packages
from codecs import open

setup(
    name='scola',
    version='0.1.0',
    description='Python code for the Scola algorithm',
    long_description='Python code for the Scola algorithm',
    url='https://github.com/skojaku/scola',
    author='Sadamori Kojaku',
    author_email='freesailing@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='correlation matrix network lasso',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'tqdm'],
)
