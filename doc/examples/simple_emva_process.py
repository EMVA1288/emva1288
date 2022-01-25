"""
Using the utility class Emva1288, process a descriptor file
print results and plot graphics
"""

import os
from emva1288.process import Emva1288
from emva1288.camera.dataset_generator import DatasetGenerator

# dir_ = '/home/work/1288/datasets/'
# fname = 'EMVA1288_ReferenceSet_001_CCD_12Bit/EMVA1288_Data.txt'
# fname = os.path.join(dir_, fname)

dataset_generator = DatasetGenerator(width=100,
                                     height=50,
                                     bit_depth=8,
                                     dark_current_ref=30)
fname = dataset_generator.descriptor_path

e = Emva1288(fname)
e.results()
e.plot()
