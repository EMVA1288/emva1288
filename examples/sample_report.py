import emva1288
import emva1288.report as report
import os

# Load one test to add it as operation point
dir_ = '/home/work/1288/datasets/'
fname = 'EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt'

info = emva1288.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
imgs = emva1288.LoadImageData(info.info)
dat = emva1288.Data1288(imgs.data)


# Description of the setup
setup = report.info_setup()
setup.standard_version = 3.1

# Basic information
basic = report.info_basic()
basic.vendor = 'Simulation'
basic.data_type = 'Single'
basic.sensor_type = 'simulated sensor'
basic.resolution = '640x480'


# Marketing information
marketing = report.info_marketing()


# Initialize the report with the marketing data
# Provide a non existent name for the output directory
myreport = report.Report1288('myreport',
                             marketing=marketing,
                             setup=setup,
                             basic=basic)

# Operation point
# bit_depth, gain, black_level, exposure_time, wavelength,
# temperature, housing_temperature, fpn_correction,
# summary_only

op1 = report.info_op()
op1.gain = 0.1
op1.offset = 29.4
op1.bit_depth = '12 bits'
op1.summary_only = False
op1.illumination = 'Variable with constant exposure time'
op1.irradiation_steps = 50

# Add the operation point to the report
# we can add as many operation points as we want
# we pass the emva1288.Data1288 object to extract automatically all the results
# and graphics
myreport.add(op1, dat)

# Generate the latex files
myreport.latex()
