"""
Calling one by one the classes that compose the reference implementation
process a descriptor file, print the results and plot the graphics
"""

import os
from emva1288 import process

dir_ = '/home/work/1288/datasets/'
fname = 'EMVA1288_ReferenceSet_001_CCD_12Bit/EMVA1288_Data.txt'

info = process.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
imgs = process.LoadImageData(info.info)
dat = process.Data1288(imgs.data)
res = process.Results1288(dat)
res.print_results()
plot = process.Plotting1288(res)
plot.plot()
