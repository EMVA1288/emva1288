Installation
=============

As a basic requirement is python > 3.5

Windows
--------

WinPython
^^^^^^^^^^
Download WinPython from: http://winpython.github.io

| Install it to any directory.
| Run the **WinPython Command Prompt.exe** from this directory.
| Within this command prompt procceed with the steps in the sections below.


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
  >> make 