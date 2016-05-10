import jinja2
import os
import shutil
from distutils.dir_util import copy_tree
from collections import namedtuple
import posixpath
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvas
import numpy as np

from .. results import Results1288
from .. plotting import EVMA1288plots


def _none_tuple(t, **kwargs):
    '''Making default None for all fields'''
    for field in t._fields:
        v = kwargs.pop(field, None)
        setattr(t, field, v)


def info_setup(**kwargs):
    '''Container for setup information'''
    s = namedtuple('setup',
                   ['light_source',  # Integrating sphere, ...
                    'light_source_non_uniformity',  # Eq. 27
                    'irradiation_calibration_accuracy',
                    'irradiation_measurement_error',
                    'standard_version'])
    _none_tuple(s, **kwargs)
    return s


def info_basic(**kwargs):
    '''Container for basic information'''
    b = namedtuple('basic',
                   ['vendor',
                    'model',
                    'data_type',
                    'sensor_type',
                    'sensor_diagonal',
                    'lens_category',
                    'resolution',
                    'pixel_size',
                    #########
                    # For CCD
                    'readout_type', 'transfer_type',
                    # For CMOS
                    'shutter_type', 'overlap_capabilities',
                    #########
                    'maximum_readout_rate',
                    'dark_current_compensation',
                    'interface_type',
                    'qe_plot'])
    _none_tuple(b, **kwargs)
    return b


def info_marketing(**kwargs):
    m = namedtuple('marketing',
                   ['logo',
                    'watermark',
                    'missingplot',
                    'cover_page'])

    _none_tuple(m, **kwargs)
    return m


def info_op(**kwargs):
    o = namedtuple('op',
                   ['name',
                    'id',
                    # Camera Setup
                    'bit_depth',
                    'gain',
                    'exposure_time',
                    'black_level',
                    'roi',
                    'fpn_correction',
                    # Operation point parameters
                    'illumination',  # Constant with variable exposure time
                                     # Variable with constant exposure time
                                     # Pulsed with constant exposure time
                    'irradiation_steps',  # Number of irradiation steps
                    # External conditions
                    'wavelength',
                    'fwhm',
                    'temperature',
                    'housing_temperature',
                    # Options
                    'summary_only'])
    _none_tuple(o, **kwargs)

    return o

_CURRDIR = os.path.abspath(os.path.dirname(__file__))


class Report1288(object):
    def __init__(self,
                 outdir,
                 setup=None,
                 basic=None,
                 marketing=None,
                 cover_page=False):

        self._outdir = os.path.abspath(outdir)

        self.renderer = self.template_renderer()
        self.ops = []
        self.marketing = marketing or info_marketing()
        self.basic = basic or info_basic()
        self.setup = setup or info_setup()
        self.cover_page = cover_page
        self._make_dirs(outdir)

    @staticmethod
    def template_renderer(dirname=None):
        if not dirname:
            dirname = os.path.join(_CURRDIR, 'templates')
        renderer = jinja2.Environment(
            block_start_string='%{',
            block_end_string='%}',
            variable_start_string='%{{',
            variable_end_string='%}}',
            comment_start_string='%{#',
            comment_end_string='%#}',
            loader=jinja2.FileSystemLoader(dirname))

        def missingnumber(value, precision):
            if value in (None, np.nan):
                return '-'
            t = '{:.%df}' % precision
            return t.format(value)

        def missingfilter(value, default='-'):
            if value in (None, np.nan):
                return default
            return value

        renderer.filters['missing'] = missingfilter
        renderer.filters['missingnumber'] = missingnumber
        return renderer

    def _make_dirs(self, outdir):
        '''Create the directory structure for the report
        If the directory exist, raise an error
        '''
        try:
            os.makedirs(self._outdir)
        except FileExistsError:
            pass
        print('Output Dir: ', self._outdir)

        files_dir = os.path.join(self._outdir, 'files')
        try:
            os.makedirs(files_dir)
        except FileExistsError:
            pass
        currfiles = os.path.join(_CURRDIR, 'files')
        copy_tree(currfiles, files_dir)

        upload_dir = os.path.join(self._outdir, 'upload')
        try:
            os.makedirs(upload_dir)
        except FileExistsError:
            pass

        def uploaded_file(fname, default):
            if fname:
                shutil.copy(os.path.abspath(fname), upload_dir)
                v = posixpath.join(
                    'upload',
                    os.path.basename(fname))
            else:
                v = posixpath.join('files', default)
            return v
        self.marketing.logo = uploaded_file(self.marketing.logo,
                                            'missinglogo.pdf')
        self.marketing.missingplot = uploaded_file(self.marketing.missingplot,
                                                   'missingplot.pdf')
        self.basic.qe_plot = uploaded_file(self.basic.qe_plot,
                                           'missingplot.pdf')

    def _write_file(self, name, content):
        fname = os.path.join(self._outdir, name)
        with open(fname, 'w') as f:
            f.write(content)
        return fname

    def _stylesheet(self):
        stylesheet = self.renderer.get_template('emvadatasheet.sty')
        return stylesheet.render(marketing=self.marketing,
                                 basic=self.basic)

    def _report(self):
        report = self.renderer.get_template('report.tex')
        return report.render(marketing=self.marketing,
                             basic=self.basic,
                             setup=self.setup,
                             operation_points=self.ops,
                             cover_page=self.cover_page)

    def latex(self):
        '''Generate report latex files'''

        self._write_file('emvadatasheet.sty', self._stylesheet())
        self._write_file('report.tex', self._report())

    def _results(self, data):
        return Results1288(data)

    def _plots(self, results, id_):
        names = {}
        savedir = os.path.join(self._outdir, id_)
        try:
            os.mkdir(savedir)
        except FileExistsError:
            pass
        for plt_cls in EVMA1288plots:
            figure = Figure()
            _canvas = FigureCanvas(figure)
            plot = plt_cls(figure)
            plot.plot(results)
            plot.rearrange()
            fname = plt_cls.__name__ + '.pdf'
            figure.savefig(os.path.join(savedir, fname))
            names[plt_cls.__name__] = posixpath.join(id_, fname)
        return names

    def add(self, op, data):
        n = len(self.ops) + 1
        op.id = 'OP%d' % (n)
        if not op.name:
            op.name = op.id
        results = self._results(data)
        op.results = results.results
        results.id = n
        op.plots = self._plots(results, op.id)
        self.ops.append(op)
