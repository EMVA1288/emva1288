import jinja2
import os
import shutil
from distutils.dir_util import copy_tree
from collections import OrderedDict
import posixpath
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvas
import numpy as np

from emva1288.process import Results1288
from emva1288.process.plotting import EVMA1288plots


def info_setup(**kwargs):
    """Container for setup information.

    All kwargs are used to update the setup information dictionary.

    Returns
    -------
    dict : A dictionary containing setup informations.
    The keys are:

      - *'Light source'* : The light source type (e.g. integrating sphere).
      - *'Light source non uniformity'* : The light source
        introducing non uniformity.
      - *'Irradiation calibration accuracy'* : The irradiation calibration
        incertainty.
      - *'Irradiation measurement error'* : The irradiation measurement
        incertainty.
      - *'Standard version'* : The EMVA1288 standard version number used.
    """
    s = OrderedDict()
    s['Light source'] = None
    s['Light source non uniformity'] = None
    s['Irradiation calibration accuracy'] = None
    s['Irradiation measurement error'] = None
    s['Standard version'] = None
    s.update(kwargs)
    return s


def info_basic(**kwargs):
    """Container for basic information.

    All kwargs are used to update the basic information dictionary for the
    report.

    Returns
    -------
    dict : A dictionary containing basic informations for the report.
    The keys are:

      - *'vendor'* : The vendor name that manufactures the camera.
      - *'model'* : The model of the tested camera.
      - *'data_type'* : The label given to the data used for the test.
      - *'sensor_type'*: The type of the tested sensor within the camera.
      - *'sensor_diagonal'* : The number of pixel in the sensor diagonal.
      - *'lens_category'* : The lens category used for the test.
      - *'resolution'* : The camera's resolution.
      - *'pixel_size'* : The sensor's pixel size.
      - *'readout_type'* : The readout type of the sensor (for CCD sensors).
      - *'transfer_type'* : The transfer type of the sensor (for CCDs).
      - *'shutter_type'* : The shutter type of the sensor (for CMOS sensors).
      - *'overlap_capabilities'* : The overlap capabilities of the sensor
        (for CMOS sensors).
      - *'maximum_readout_rate'* : The camera's maximal readout rate.
      - *'dark_current_compensation'* : If the camera support dark current
        compensation, specify it in this entry.
      - *'interface_type'* : The camera's interface type.
      - *'qe_plot'* : The sensor's quantum efficency plots.
    """
    b = {'vendor': None,
         'model': None,
         'data_type': None,
         'sensor_type': None,
         'sensor_diagonal': None,
         'lens_category': None,
         'resolution': None,
         'pixel_size': None,
         #########
         # For CCD
         'readout_type': None,
         'transfer_type': None,
         # For CMOS
         'shutter_type': None,
         'overlap_capabilities': None,
         #########
         'maximum_readout_rate': None,
         'dark_current_compensation': None,
         'interface_type': None,
         'qe_plot': None
         }
    b.update(kwargs)
    return b


def info_marketing(**kwargs):
    """Container for marketing informations.

    All kwargs are used to update the returned dictionary containing the
    marketing informations.

    Returns
    -------
    dict : A dictionary containing the marketing informations.
    The keys are:

      - *'logo'* : The path to the logo icon.
      - *'watermark'* : A text that will be printed on every page of the
        report in the background in transparent red.
      - *'missingplot'* : The path to a missing plot icon.
      - *'cover_page'* : The path to a custom cover page for the report.
    """
    m = {'logo': None,
         'watermark': None,
         'missingplot': None,
         'cover_page': None
         }
    m.update(kwargs)
    return m


def info_op():
    """Container for operation points informations.

    The returned dictionary must be filled after calling this function.

    Returns
    -------
    dict : An empty dictionary with the following keys:

      - *'name'* : The test name.
      - *'id'* : The test id.
      - *'summary_only'* : True or False, tells if, for a specific OP, the
        report should do a summary of the test instead of a full description.
      - *'results'* : The results of the test.
      - *'camera_settings'* : (OrderedDict) the dictionary of the camera's
        settings for the test.
      - *'test_parameters'* : (OrderedDict) the dictionary of the other test
        parameters.
    """
    d = {'name': None,
         'id': None,
         'summary_only': None,
         'results': None,
         'camera_settings': OrderedDict(),
         'test_parameters': OrderedDict()}
    return d


_CURRDIR = os.path.abspath(os.path.dirname(__file__))


class Report1288(object):
    """Class that has the purpose of creating a pdf report of one or more
    optical tests. This class only creates the report TeX files using the
    templates. TeX files must be compiled afterwards to generate the
    pdf files.
    """
    def __init__(self,
                 outdir,
                 setup=None,
                 basic=None,
                 marketing=None,
                 cover_page=False):
        """Report generator init method.

        Informations stored in the report can be specified by the kwargs
        passed to the object. It can be useful to use the
        :func:`info_marketing`, :func:`info_setup` and :func:`info_basic`
        functions to generate the corresponding dictionaries.
        The report generator uses `jinja2` to render the templates. Upon init,
        it calls the :meth:`template_renderer` method to get the `jinja2`
        object that will interact with the templates. Then it creates the
        output directories and files that will contain the output files.
        To create the report for different operating point/tests, one must
        call the :meth:`add` method to add each test he wants to publish in
        the report. Then, to conclude the report, he must call the
        :meth:`latex` method to generate the TeX files.

        Parameters
        ----------
        outdir : str
                 The path to the directory that will contain the report files.
        setup : dict, optional
                A dictionary containing the setup informations. If None, the
                report uses the dictionary of the :func:`info_setup`
                function.
        basic : dict, optional
                A dictionary containing basic informations about the test. If
                None, the report generator takes the dictionary from the
                :func:`info_basic` function.
        marketing : dict, optional
                    A dictionary containing
        cover_page : str, optional
                     The path to the cover page for the report. If False,
                     no cover page will be included in the report.
        """
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
        """Method that creates the renderer for the TeX report file.

        Uses the :class:`jinja2:jinja2.Environment` object to create
        the renderer. Also defines some filters for the environment
        for the missing numbers and general missings.

        Parameters
        ----------
        dirname : str, optional
                  The path to the template directory containing the TeX
                  templates. If None, it will get the templates from the
                  `./templates/` directory.

        Returns
        -------
        The renderer.
        """
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
            # Filter for missing numbers
            if value in (None, np.nan):
                return '-'
            t = '{:.%df}' % precision
            return t.format(value)

        def missingfilter(value, default='-'):
            # General filter for missing objects that are not numbers
            if value in (None, np.nan):
                return default
            return value

        renderer.filters['missing'] = missingfilter
        renderer.filters['missingnumber'] = missingnumber
        return renderer

    def _make_dirs(self, outdir):
        """Create the directory structure for the report
        If the directory exist, raise an error
        """
        try:
            os.makedirs(self._outdir)
        except FileExistsError:  # pragma: no cover
            pass
        print('Output Dir: ', self._outdir)

        files_dir = os.path.join(self._outdir, 'files')
        try:
            os.makedirs(files_dir)
        except FileExistsError:  # pragma: no cover
            pass
        currfiles = os.path.join(_CURRDIR, 'files')
        copy_tree(currfiles, files_dir)

        upload_dir = os.path.join(self._outdir, 'upload')
        try:
            os.makedirs(upload_dir)
        except FileExistsError:  # pragma: no cover
            pass

        def uploaded_file(fname, default):
            if fname:  # pragma: no cover
                shutil.copy(os.path.abspath(fname), upload_dir)
                v = posixpath.join(
                    'upload',
                    os.path.basename(fname))
            else:
                v = posixpath.join('files', default)
            return v

        self.marketing['logo'] = uploaded_file(self.marketing['logo'],
                                               'missinglogo.pdf')

        self.marketing['missingplot'] = uploaded_file(
            self.marketing['missingplot'],
            'missingplot.pdf')

        self.basic['qe_plot'] = uploaded_file(self.basic['qe_plot'],
                                              'missingplot.pdf')

    def _write_file(self, name, content):
        # write content into a file
        fname = os.path.join(self._outdir, name)
        with open(fname, 'w') as f:
            f.write(content)
        return fname

    def _stylesheet(self):
        # generate the stylesheet content
        stylesheet = self.renderer.get_template('emvadatasheet.sty')
        return stylesheet.render(marketing=self.marketing,
                                 basic=self.basic)

    def _report(self):
        # Generate the report contents
        report = self.renderer.get_template('report.tex')
        return report.render(marketing=self.marketing,
                             basic=self.basic,
                             setup=self.setup,
                             operation_points=self.ops,
                             cover_page=self.cover_page)

    def latex(self):
        """Generate report latex files.
        """

        self._write_file('emvadatasheet.sty', self._stylesheet())
        self._write_file('report.tex', self._report())

    def _results(self, data):
        return Results1288(data)

    def _plots(self, results, id_):
        """Create the plots for the report.

        The report will include all the plots contained in the
        `~emva1288.process.plotting.EVMA1288plots` list. All plots
        will be saved in pdf format in the output directory.
        """
        names = {}
        savedir = os.path.join(self._outdir, id_)
        try:
            os.mkdir(savedir)
        except FileExistsError:  # pragma: no cover
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

    def add(self, op, data, results=None):
        """Method that adds an operation point to the report.

        The data supplied are passed through a
        :class:`~emva1288.process.results.Results1288` object to be processed
        for the report. Also creates the plots
        which will appears in the report.

        Parameters
        ----------
        op : dict
             The dictionary containing the operation point informations.
             This dictionary must absolutely contain a 'name' key.
             See the :func:`info_op` function to get an idea to what
             keys to give.
        data : dict
               The corresponding operation point data. It must be able to
               be processed by an instance of the
               :class:`~emva1288.process.results.Results1288` class.
        """
        n = len(self.ops) + 1
        op['id'] = 'OP%d' % (n)
        if not op['name']:
            op['name'] = op['id']
        if not results:
            results = self._results(data)
        op['results'] = results.results_by_section
        results.id = n
        op['plots'] = self._plots(results, op['id'])
        self.ops.append(op)
