Image Comparison
=================

| To easily exchange images and perform inter-implementation tests, we have defined a set of rules.
| ANYBODY who wants somebody else to test a set of images has to follow this convention.

Store the image stack for comparison
-------------------------------------

Measurements
^^^^^^^^^^^^^

| All images to perform an EMVA1288 measurement are stored as 8bit or 16bit images. 
| The used exposure times and number of collected photons per pixel are reported.

 **Temporal noise**

  * Bright: 2 Images per illumination step
  * Dark: 2 Images per exposure time

 **Spatial Noise Measurements**

  * Bright: N Images; Exposure time for 50% of the saturation capacity
  * Dark: N Images; Exposure time for 50% of the saturation capacity

File Format
^^^^^^^^^^^^

| One plain text file organize the path to the images and holds the information about the used exposure times and number of photons.
| There are some key characters to organize the file:

The char '#' at the beginning of the line starts a comment::

  # Starts a comment line


The char 'n' starts a line with some information about the images::

  # n bits/pixel=12 width/pixel=656 height/pixel=494
  n 12 656 494

The char 'b' starts a bright measurement with some following images 'i'::

  # b exposureTime/ns=40000.0 ns numberPhotons/photons=120.0 photons 
  b 40000.0 120.0
  # i relativePathToTheImage
  i b_000_snap_001.tif
  i b_000_snap_002.tif

The char 'd' starts a dark measurement with some following images 'i'::

  # d exposureTime/ns=40000.0 ns
  d 40000.0
  # i relativePathToTheImage
  i d_049_snap_001.tif
  i d_049_snap_002.tif

The char 'i' starts a section with images. Each line has a relative path to one image of this section::

  # i relativePathToTheImage
  i d_049_snap_001.tif
  i d_049_snap_002.tif

The char 'v' gives the version of the EMVA1288 Standard used to generate the measurement::

  # Measurement for EMVA1288 Standard Release 3.0
  v 3.0

The char 'l' passes information for the latex file to use with the template::

  # l variablename value
  l InterfaceType Gigabit Ethernet

Complete Example File
^^^^^^^^^^^^^^^^^^^^^^

`Example of a descriptor file <https://github.com/EMVA1288/datasets/blob/master/EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt>`__
