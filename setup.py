'''
Created on Mar 24, 2012

@author: tin@byg.dtu.dk

Run from command line:

python setup.py sdist
python setup.py bdist_wininst

This will generate a distribution zip file and a windows executable installer
Can be installed by running from the unzipped temporary directory:

python setup.py install

Or from development directory, in development mode - will reflect changes made
in the original development directory instantly automatically.

python setup.py develop
'''

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
    name = "pypf",
    version = "0.1.0",
    description = "one-line description here",
    author = "Thomas Ingeman-Nielsen",
    author_email = "tin@byg.dtu.dk",
    url = "http://???/",
    keywords = ["some","key","words"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Fortran",
        "Development Status :: 3 - Alpha",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",      
        ],
    packages=find_packages(),
    include_package_data = True,
    package_data = {
        'pypf': ['resources/brown1997/*']},
    long_description = """\
pypf
----------------

Put extensive description of package here!
""", install_requires=['matplotlib'])
