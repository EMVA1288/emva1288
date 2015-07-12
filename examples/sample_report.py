import emva1288
from emva1288.report.report import Report1288, op, marketing
import os

dir_ = '/home/work/1288/datasets/'
fname = 'EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt'

info = emva1288.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
imgs = emva1288.LoadImageData(info.info)
dat = emva1288.Data1288(imgs.data)

mark = marketing()
mark.vendor = 'Sample vendor'
mark.model = 'Unknown model'
mark.interface = 'simulation'
mark.resolution_x = 640
mark.resolution_y = 480

report = Report1288(mark)

op1 = op()
op1.gain = 3
op1.offset = 20
op1.wavelength = 525
op1.extra = True

report.add(op1, dat)
report.latex('ajaytuque')

#     op2 = op()
#     op2.gain = 111
#     op2.offset = 2222
#
#
#     r.add(op2)
    # r.pdf()
