import jinja2
import os
import shutil
from distutils.dir_util import copy_tree
from collections import namedtuple
from tempfile import TemporaryDirectory
import posixpath

from .. results import Results1288
from .. plotting import Plotting1288, EVMA1288plots


def _none_tuple(t, **kwargs):
    '''Making default None for all fields'''
    for field in t._fields:
        v = kwargs.pop(field, None)
        setattr(t, field, v)


def info_setup(**kwargs):
    '''Container for setup information'''
    s = namedtuple('setup',
                   ['light_source',
                    'standard_version'])
    _none_tuple(s)
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
                    'missingplot'])

    _none_tuple(m, **kwargs)
    return m


def info_op(**kwargs):
    o = namedtuple('op',
                   ['bit_depth',
                    'gain',
                    'exposure_time',
                    'black_level',
                    'fpn_correction'
                    # External conditions
                    'wavelength',
                    'temperature',
                    'housing_temperature',
                    # Options
                    'summary_only'])
    _none_tuple(o, **kwargs)

    return o

_CURRDIR = os.path.abspath(os.path.dirname(__file__))


class Report1288(object):
    def __init__(self, setup=None, basic=None, marketing=None):
        self._tmpdir = None

        self.renderer = self._template_renderer()
        self.ops = []
        self.marketing = marketing or info_marketing()
        self.basic = basic or info_basic()
        self.setup = setup or info_setup()
        self._temp_dirs()

    def _template_renderer(self):
        renderer = jinja2.Environment(
            block_start_string='%{',
            block_end_string='%}',
            variable_start_string='%{{',
            variable_end_string='%}}',
            comment_start_string='%{#',
            comment_end_string='%#}',
            loader=jinja2.FileSystemLoader(os.path.join(_CURRDIR,
                                                        'templates')))

        def missingnumber(value, precision):
            if value is None:
                return '-'
            t = '{:.%df}' % precision
            return t.format(value)

        def missingfilter(value, default='-'):
            if value is None:
                return default
            return value

        renderer.filters['missing'] = missingfilter
        renderer.filters['missingnumber'] = missingnumber
        return renderer

    def _temp_dirs(self):
        self._tmpdir = TemporaryDirectory()
        tmpfiles = os.path.join(self._tmpdir.name, 'files')
        os.makedirs(tmpfiles)
        currfiles = os.path.join(_CURRDIR, 'files')
        copy_tree(currfiles, tmpfiles)
        markfiles = os.path.join(self._tmpdir.name, 'marketing')
        os.makedirs(markfiles)

        def default_image(attr, default):
            img = getattr(self.marketing, attr)
            if img:
                shutil.copy(os.path.abspath(img), markfiles)
                v = posixpath.join(
                    'marketing',
                    os.path.basename(img))
            else:
                v = posixpath.join('files', default)
            setattr(self.marketing, attr, v)

        default_image('logo', 'missinglogo.pdf')
        default_image('missingplot', 'missingplot.pdf')

    def _write_file(self, name, content):
        fname = os.path.join(self._tmpdir.name, name)
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

    def _plots(self, results, id_):
        plots = Plotting1288(results)
        savedir = os.path.join(self._tmpdir.name, id_)
        os.mkdir(savedir)
        plots.plot(savedir=savedir, show=False)
        names = {}
        for cls in EVMA1288plots:
            names[cls.__name__] = posixpath.join(id_, cls.__name__ + '.pdf')
        return names

    def add(self, op, data):
        op.id = 'OP%d' % (len(self.ops) + 1)
        results = self._results(data)
        op.results = results.results
        op.plots = self._plots(results, op.id)
        self.ops.append(op)
