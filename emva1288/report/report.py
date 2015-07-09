import jinja2
import os
from collections import namedtuple

marketing = namedtuple('marketing',
                       'logo, vendor, model, '
                       'serial, sensor_type, sensor_name, '
                       'resolution_x, resolution_y, '
                       'sensor_diagonal, lens_mount, '
                       'shutter, overlap, readout_rate, '
                       'dark_current_compensation, interface, '
                       'watermark, qe_plot')
marketing.logo = False
marketing.vendor = 'Dude copmpany'
marketing.model = 'Cam dude'
marketing.serial = '-'
marketing.sensor_type = 'CMOS'
marketing.sensor_name = 'dude sensor'
marketing.resolution_x = 1024
marketing.resolution_y = 768
marketing.sensor_diagonal = '8mm'
marketing.lens_mount = 'cmount'
marketing.shutter = 'Global'
marketing.overlap = '-'
marketing.readout_rate = '-'
marketing.dark_current_compensation = 'None'
marketing.interface = '-'
marketing.watermark = False
marketing.qe_plot = False


def op():
    o = namedtuple('op', 'bit_depth, gain,'
                   'offset, exposure_time, wavelength, '
                   'temperature, housing_temperature, '
                   'fpn_correction, results, plots, extra')
    o.bit_depth = '-'
    o.gain = '-'
    o.offset = '-'
    o.exposure_time = '-'
    o.wavelength = '-'
    o.temperature = 'room'
    o.housing_temperature = '-'
    o.fpn_correction = '-'
    o.extra = False
    return o


class Report1288(object):
    def __init__(self, marketing):
        self.renderer = jinja2.Environment(
            block_start_string='%{',
            block_end_string='%}',
            variable_start_string='%{{',
            variable_end_string='%}}',
            comment_start_string='%{#',
            comment_end_string='%#}',
            loader=jinja2.FileSystemLoader(os.path.abspath('templates'))
        )
        self.ops = []
        self.marketing = marketing

    def stylesheet(self):
        stylesheet = self.renderer.get_template('emvadatasheet.sty')
        return stylesheet.render(marketing=self.marketing)

    def report(self):
        report = self.renderer.get_template('report.tex')
        return report.render(marketing=self.marketing,
                             operation_points=self.ops)

    def add(self, op):
        self.ops.append(op)

if __name__ == '__main__':
    r = Report1288(marketing)
#     print(r.report())

    import os
    import emva1288

    dir_ = '/home/work/1288/datasets/'
    fname = 'EMVA1288_ReferenceSet_003_Simulation_12Bit/EMVA1288_Data.txt'

    info = emva1288.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
    imgs = emva1288.LoadImageData(info.info)
    dat = emva1288.Data1288(imgs.data)
    res = emva1288.Results1288(dat)

    op1 = op()
    op1.gain = 333
    op1.offset = 444
    op1.results = res.results
    op1.plots = emva1288.Plotting1288(res)
    r.add(op1)

    op2 = op()
    op2.gain = 111
    op2.offset = 2222
    op2.results = res.results
    op2.plots = emva1288.Plotting1288(res)

    r.add(op2)
    print(r.report())
