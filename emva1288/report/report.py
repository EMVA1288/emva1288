import jinja2
import os
import shutil
from distutils.dir_util import copy_tree
from collections import namedtuple
from tempfile import TemporaryDirectory

from .. results import Results1288
from .. plotting import Plotting1288, EVMA1288plots


def marketing(**kwargs):
    m = namedtuple('marketing',
                   'logo, vendor, model, '
                   'serial, sensor_type, sensor_name, '
                   'resolution_x, resolution_y, '
                   'sensor_diagonal, lens_mount, '
                   'shutter, overlap, readout_rate, '
                   'dark_current_compensation, interface, '
                   'watermark, qe_plot, '
                   'emva1288_logo, missingplot, missinglogo')

    # For these attributes default is False
    # for the rest is '-'
    kwargs.setdefault('logo', False)
    kwargs.setdefault('watermark', False)
    kwargs.setdefault('qe_plot', False)
    kwargs.setdefault('emva1288_logo',
                      os.path.join('files', 'EMVA1288Logo.pdf'))
    kwargs.setdefault('missinglogo', os.path.join('files', 'missinglogo.pdf'))
    kwargs.setdefault('missingplot', os.path.join('files', 'missingplot.pdf'))
    for field in m._fields:
        v = kwargs.pop(field, '-')
        setattr(m, field, v)
    return m


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
    o.results = None
    o.plots = None
    o.id = None
    return o

_CURRDIR = os.path.abspath(os.path.dirname(__file__))


class Report1288(object):
    def __init__(self, mark):
        self._tmpdir = None

        self.renderer = jinja2.Environment(
            block_start_string='%{',
            block_end_string='%}',
            variable_start_string='%{{',
            variable_end_string='%}}',
            comment_start_string='%{#',
            comment_end_string='%#}',
            loader=jinja2.FileSystemLoader(os.path.join(_CURRDIR, 'templates'))
        )
        self.ops = []
        self.marketing = mark
        self._temp_dirs()

    def _temp_dirs(self):
        self._tmpdir = TemporaryDirectory()
        tmpfiles = os.path.join(self._tmpdir.name, 'files')
        os.makedirs(tmpfiles)
        currfiles = os.path.join(_CURRDIR, 'files')
        copy_tree(currfiles, tmpfiles)
        markfiles = os.path.join(self._tmpdir.name, 'marketing')
        os.makedirs(markfiles)

        if self.marketing.logo:
            fname = os.path.basename(self.marketing.logo)
            if not fname:
                print('invalid logo file', fname)
            else:
                shutil.copy(os.path.abspath(self.marketing.logo),
                            os.path.join(markfiles, fname))
                self.marketing.logo = os.path.join('marketing', fname)

        if self.marketing.qe_plot:
            fname = os.path.basename(self.marketing.qe_plot)
            if not fname:
                print('invalid qe_plot file', fname)
            else:
                shutil.copy(os.path.abspath(self.marketing.qe_plot),
                            os.path.join(markfiles, fname))
                self.marketing.qe_plot = os.path.join('marketing', fname)

    def _write_file(self, name, content):
        fname = os.path.join(self._tmpdir.name, name)
        with open(fname, 'w') as f:
            f.write(content)
        return fname

    def _stylesheet(self):
        stylesheet = self.renderer.get_template('emvadatasheet.sty')
        return stylesheet.render(marketing=self.marketing)

    def _report(self):
        report = self.renderer.get_template('report.tex')
        return report.render(marketing=self.marketing,
                             operation_points=self.ops)

    def latex(self, dir_):
        '''Generate report latex files in a given directory'''

        self._write_file('emvadatasheet.sty', self._stylesheet())
        self._write_file('report.tex', self._report())

        outdir = os.path.abspath(dir_)
        try:
            os.makedirs(outdir)
        except FileExistsError:
            pass
        copy_tree(self._tmpdir.name, outdir)
        print('Report files found in:', outdir)

    def _results(self, data):
        return Results1288(data)

    def _plots(self, data, id_):
        res = self._results(data)
        plots = Plotting1288(res)
        savedir = os.path.join(self._tmpdir.name, id_)
        os.mkdir(savedir)
        plots.plot(savedir=savedir, show=False)
        names = {}
        for cls in EVMA1288plots:
            names[cls.__name__] = os.path.join(id_, cls.__name__ + '.pdf')
        return names

    def add(self, op_, data=None):
        if not op_.id:
            op_.id = 'OP%d' % (len(self.ops) + 1)
        if not op_.results and data:
            op_.results = self._results(data).results
        if not op_.plots and data:
            op_.plots = self._plots(data, op_.id)
        self.ops.append(op_)
