EMVA1288
========

This is the reference implementation for the EMVA1288 Standard for
Measurement and Presentation of Specifications for Machine Vision Sensors and Cameras.

Please visit [Emva1288 Website](http://www.emva.org/cms/index.php?idcat=26) for information and latest releases of the standard.


Requirements
============
To use the emva1288 package you need to have installed
- numpy
- opencv
- matplotlib
- lxml


Installation
============
For inexperienced windows users it is recommended to use a prepackaged python distribution that
contains the required packages. One excellent option for Windows users is [Anaconda](http://continuum.io/)

For the time being there is no installer for the package.
Just copy the full repository and add it's path to your `PYTHONPATH` environment variable

Description
===========
If you are in a rush there is one utility class that allows to do the full
process in a very simple way
```
from emva1288 import Emva1288

# Load your data descriptor file
e = Emva1288(filename)
# Print the results
e.results()
# Plot the graphics
e.plot()
```


If you are interested keep reading.

The code is dvidided in 7 parts

parser.ParseEmvaDescriptorFile
------------------------------
This class takes an EMVA1288 descriptor file and loads its content into a python dictionary.

An EMVA1288 descriptor file is a file that contains the description
of an EMVA1288 test including exposure times, photon count and corresponding images

[Example of a descriptor file] (examples/EMVA1288_Descriptor_File.txt)


loader.LoadImageData
--------------------
This class takes a dictionary (product of `parser.ParseEmvaDescriptorFile`). Load the
related images and reduce it's data to the minimum possible, **preserving all relevant image data in
as integers**. The resulting data is a Python dictionary.

[Example of the reduced data](examples/EMVA1288_image_data.txt)

data.Data1288
-------------
This class takes a dictionary with image data (product of `loader.LoadImageData`), and transforms it
into data that can be used for the EMVA1288 computations.
It is important to note, that this is separate from `LoadImageData` because this step, produces float values
that are not easily transportable (db, json, etc...) without loosing accuracy.

results.Results1288
-------------------
This class takes the data from `data.Data1288` and compute the actual EMVA1288 values.

plotting.Plotting1288
---------------------
This class takes a `results.Results1288` object and produces all the plots needed to create
a reference datasheet of the EMVA1288 test


Usage
=====
To use the code, you need to have a set of images that correspond to an EMVA1288 test.
There are some sample image sets provided by the standard development group.
[Example datasets](https://emva1288.plan.io/projects/emva1288-standard-public/files).

Download one or all of these datasets, extract its content, and use them as input
in the examples shown below.


Examples
========
* [A simple example](examples/simple_emva_process.py)
* [Step by step example](examples/full_emva_process.py)
