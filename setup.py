import sys
from setuptools import setup

from dl_helper import VERSION
PKGNAME = "dl-helper"
MIN_PYTHON_VERSION = (3, 5)



# --------------------------------------------------------------------

if sys.version_info < MIN_PYTHON_VERSION:
    sys.exit('**** Sorry, {} {} needs at least Python {}'.format(
        PKGNAME, VERSION, '.'.join(map(str, MIN_PYTHON_VERSION))))


setup_args = dict(name=PKGNAME,
                  version=VERSION,
                  author="Paulo Villegas",
                  author_email="paulo.vllgs@gmail.com",

                  description="Miscellaneous tiny utils to help working with ML/DL tasks in an IPython Notebook context",
                  url="https://github.com/paulovn/dl-helper",
                  platforms=["any"],
                  classifiers=["Development Status :: 4 - Beta",
                               "Environment :: Console",
                               "Intended Audience :: Developers",
                               "Programming Language :: Python",
                               "Programming Language :: Python :: 3",
                               "License :: OSI Approved :: BSD License",
                               "Operating System :: OS Independent"
                  ],

                  install_requires=['setuptools'],

                  packages=["dl_helper", "dl_helper.krs"],

                  include_package_data=False,       # otherwise package_data is not used
                  )


if __name__ == '__main__':
    setup(**setup_args)
