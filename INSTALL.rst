Installation
=============

Minimum supported python version is  3.6

Install libraries
------------------
In your preferred python environment run ::

  >> pip install emva1288


Development
------------
If you want to contribute to this project, clone it and install it in development mode::

  >> git clone https://github.com/EMVA1288/emva1288.git (or from a fork)
  >> pip install -e .


Tests
^^^^^^
To add and run the tests, first install the test dependencies::

  >> pip install -e .[tests]

Run the tests to make sure nothing important is broken::

  >> pytest


Documentation
^^^^^^^^^^^^^
To rebuild the documentation you must first install the documentation packages::

  >> pip install -e .[doc]

Go to the documentation folder and rebuild::

  >> cd doc
  >> make clean
  >> make
