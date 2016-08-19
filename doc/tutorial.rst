Tutorial
########

Below is a basic tutorial on how to operate the EMVA1288 module.

Description
===========

If you are in a rush there is one utility class that allows to do the
full process in a very simple way

::

    from emva1288 import Emva1288

    # Load your data descriptor file
    e = Emva1288(filename)
    # Print the results
    e.results()
    # Plot the graphics
    e.plot()

If you are interested keep reading.

The main EMVA1288 processing code is divided into 6 parts.
In the package, there is also a :class:`~emva1288.camera.dataset_generator.DatasetGenerator`
class that can generate automatically a set of data that can be analysed. The data are
generated with a :class:`~emva1288.camera.camera.Camera` simulator that can be run
as a stand-alone object.

1) Parser
---------

This class takes an EMVA1288 descriptor file and loads its content into
a python dictionary.

API Reference : :class:`~emva1288.process.parser.ParseEmvaDescriptorFile`

An EMVA1288 descriptor file is a file that contains the description of
an EMVA1288 test including exposure times, photon count and
corresponding images relative path to the descriptor.

`Example of a descriptor file <https://github.com/EMVA1288/datasets/blob/master/EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt>`__

The images needs to be loaded after the descriptor file have been parsed.
Usage example ::

  from emva1288.process.parser import ParseEmvaDescriptorFile

  # Absolute path to descriptor file
  path = "Path/To/Datasets/descriptor_file.txt"
  
  # Create the parser object
  parser = ParseEmvaDescriptorFile(path)
  
  # We can access all the images path with the images attribute
  images = parser.images


2) Loader
---------

This class takes a dictionary (product of a
:class:`~emva1288.process.parser.ParseEmvaDescriptorFile` object).
It loads the related images and reduce
it's data to the minimum possible, **preserving all relevant image data
in as integers**. The resulting data is a Python dictionary.

API Reference : :class:`~emva1288.process.loader.LoadImageData`

`Example of the reduced data <https://github.com/EMVA1288/emva1288/blob/master/examples/EMVA1288_image_data.txt>`__

Usage example ::

  from emva1288.process.loader import LoadImageData

  # Assuming we have a parser object that had parsed the descriptor file
  images = parser.images

  # Create the loader object
  loader = LoadImageData(images)

  # We can access the images' data with the data attribute
  image_raw_data = loader.data


3) Data
-------

This class takes a dictionary with image data (product of a
:class:`~emva1288.process.loader.LoadImageData` object),
and transforms it into data that can be used
for the EMVA1288 computations. It is important to note, that this is
separate from :class:`~emva1288.process.loader.LoadImageData` because this step
produces float values
that are not easily transportable (db, json, etc...) without losing
accuracy.

API Reference: :class:`~emva1288.process.data.Data1288`

Usage example ::

  from emva1288.process.data import Data1288

  # Assuming a image raw data have been loaded by a loader
  image_raw_data = loader.data

  # Extract data from images
  data = Data1288(image_raw_data)

  # Extracted data can be accessed with the data attribute
  extracted_data = data.data


4) Results
----------

This class takes the data from :class:`~emva1288.process.data.Data1288`
and compute the actual
EMVA1288 values.

API Reference : :class:`~emva1288.process.results.Results1288`

Usage example ::
  
  from emva1288.process.results import Results1288

  # Assuming image data have been extracted by a Data1288 object
  extracted_data = data.data

  # Compute the results from data
  results = Results1288(extracted_data)

  # Results can be printed in the console with the print_results method
  results.print_results()


5) Plotting
-----------

This class takes a :class:`~emva1288.process.results.Results1288`
object and produces all the
plots needed to create a reference datasheet of the EMVA1288 test

API Reference : :class:`~emva1288.process.plotting.Plotting1288`

Usage example ::

  from emva1288.process.plotting import Plotting1288

  # Assuming results have been computed
  # Create the plot object
  plot = Plotting1288(results)

  # Show the plots with the plot method
  plot.plot()


6) Report
---------

This class creates a directory with all the files needed to compile a latex
report.

API Reference : :class:`~emva1288.report.report.Report1288`

Here is an example of how the report generator works::

  from emva1288.report import Report1288

  outdir = "path/to/output/directory"
  marketing = {dictionary containing marketing infos}
  basic = {dictionary containing basic infos}
  setup = {dictionary containing setup infos}
  cover_page = "path/to/the/cover/page.tex"
  op = {Dictionary containing the operation point infos to
        publish in the report}
  data = {dictionary containing the op data}

  # create report object
  report = Report1288(outdir, marketing=marketing, basic=basic,
                      setup=setup, cover_page=cover_page)

  # add operation points
  report.add(op, data)

  # create report tex files
  report.latex()

  # next, compile the files somehow.

There is four functions that can be used to add custom informations to the report.

- First, there is the :func:`~emva1288.report.report.info_marketing` function.
  This is a function that returns a dictionary to fill with the marketing data
  needed for the report.
- Second, there is the :func:`~emva1288.report.report.info_op` function.
  This function returns a dictionary serving as a place holder for all the data
  needed for an operation point in the report
- Third, the :func:`~emva1288.report.report.info_setup` creates a dictionary
  containing the experimental setup informations.
- And fourth, the :func:`~emva1288.report.report.info_basic` does the same but
  for basic informations common to all operation points.

Usage
=====

To use the code, you need to have a set of images that correspond to an
EMVA1288 test. There are some sample image sets provided by the standard
development group. `Example
datasets <https://github.com/EMVA1288/datasets>`__.

Download one or all of these datasets, extract its content, and use them
as input in the examples shown below.

Examples
========

-  `A simple example <https://github.com/EMVA1288/emva1288/blob/master/examples/simple_emva_process.py>`__
-  `Step by step example <https://github.com/EMVA1288/emva1288/blob/master/examples/full_emva_process.py>`__
-  `Experimental report generation module <https://github.com/EMVA1288/emva1288/blob/master/examples/sample_report.py>`__
