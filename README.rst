EMVA1288
========

The EMVA1288 module is a Python package to process and print results of an
EMVA1288 tests.

This is the reference implementation for the EMVA1288 Standard for
Measurement and Presentation of Specifications for Machine Vision
Sensors and Cameras.

Please visit `Emva1288
Website <http://www.emva.org/standards-technology/emva-1288/>`__ for information
and latest releases of the standard.

Documentation
-------------
For more information visit `the documentation page
<http://emva1288.readthedocs.io/en/latest/>`__

Testing with Pytest
-------------
Install Requirements : *pip install -r requirements.txt*

execute a subset of smoke tests:

- **pytest -s -m smoke**

Execute the entire suite of regression tests, with a html report:

- **pytest -m regression -s -v --durations --html=emva1288/tests/reports/pytest_report.html**

-Other available py marks to scope testing: camera , filters , tile

-refer to additional tests logs : *logs/tests.log*