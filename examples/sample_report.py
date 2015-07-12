import emva1288
from emva1288.report.report import Report1288, op, marketing
import os

# Load one test to add it as operation point
dir_ = '/home/work/1288/datasets/'
fname = 'EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt'

info = emva1288.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
imgs = emva1288.LoadImageData(info.info)
dat = emva1288.Data1288(imgs.data)

# Setup the marketing values
# marketing has several parameters
# logo, vendor, model, serial, sensor_type, sensor_name,
# resolution_x, resolution_y, sensor_diagonal, lens_mount,
# shutter, overlap, readout_rate, dark_current_compensation,
# interface, watermark, qe_plot

mark = marketing()
mark.vendor = 'Sample vendor'
mark.model = 'Unknown model'
mark.interface = 'simulation'
mark.resolution_x = 640
mark.resolution_y = 480

# Initialize the report with the marketing data
report = Report1288(mark)

# Operation point has several parameters
# bit_depth, gain, offset, exposure_time, wavelength,
# temperature, housing_temperature, fpn_correction,
# results, plots, extra

op1 = op()
op1.gain = 3
op1.offset = 20
op1.wavelength = 525
op1.extra = True

# Add the operation point to the report
# we can add as many operation points as we want
# we pass the emva1288.Data1288 object to extract automatically all the results
# and graphics
report.add(op1, dat)

# Generate the report directory with all the latex files and figures
report.latex('myreport')
