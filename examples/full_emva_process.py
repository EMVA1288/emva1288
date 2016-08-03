"""
Calling one by one the classes that compose the reference implementation
process a descriptor file, print the results and plot the graphics
"""

import os
from emva1288 import process
from emva1288.camera.dataset_generator import DatasetGenerator

# dir_ = '/home/work/1288/datasets/'
# fname = 'EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt'

dataset_generator = DatasetGenerator(img_x=100, img_y=50, bit_depth=8)

# Parse the descriptor file
# parser = process.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
parser = process.ParseEmvaDescriptorFile(dataset_generator.descriptor_path)

# Load images
imgs = process.LoadImageData(parser.images)

# Extract data from images
dat = process.Data1288(imgs.data)

# Compute the results
res = process.Results1288(dat.data)
res.print_results()

# Plot the graphics
plot = process.Plotting1288(res)
plot.plot()
