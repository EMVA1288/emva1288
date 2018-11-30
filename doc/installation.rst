Installation
============

As a basic requirement is python > 3.5

Windows
-------
.. todo::
    Update to use winpython


Install for normal users
------------------------
Once the python environment is installed just run ::

  >> pip install emva1288


Development
------------
If you want to contribute to this project, clone it and install it::

  >> git clone https://github.com/EMVA1288/emva1288.git (or from a fork)
  >> pip install -e .[doc]

If you make code modifications, you can run the unittests to make sure
nothing important is broken::

  >> python tests.py

This will run the test suite and print the code coverage in the console.

Documentation
-------------
To rebuild the documentation::

  >> cd doc
  >> make clean
  >> make html


