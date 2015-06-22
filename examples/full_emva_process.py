"""
Calling one by one the classes that compose the reference implementation
process a descriptor file, print the results and plot the graphics
"""

import os
import emva1288

dir_ = '/home/work/1288/datasets/'
fname = 'EMVA1288_ReferenceSet_001_CCD_12Bit/EMVA1288_Data.txt'

info = emva1288.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
imgs = emva1288.LoadImageData(info.info)
dat = emva1288.Data1288(imgs.data)
res = emva1288.Results1288(dat)
res.print_results()
plot = emva1288.Plotting1288(res)
plot.plot()
