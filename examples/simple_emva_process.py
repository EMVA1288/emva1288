"""
Using the utility class Emva1288, process a descriptor file
print results and plot graphics
"""

import os
from emva1288 import Emva1288

dir_ = '/home/work/1288/datasets/'
fname = 'EMVA1288_ReferenceSet_001_CCD_12Bit/EMVA1288_Data.txt'

e = Emva1288(os.path.join(dir_, fname))
e.results()
e.plot()
