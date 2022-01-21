import unittest
import tempfile
import os
from emva1288.process import ParseEmvaDescriptorFile, LoadImageData, Data1288
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.report import info_op, Report1288
from emva1288.process.plotting import EVMA1288plots


class TestReportGenerator(unittest.TestCase):
    _height = 50
    _width = 100
    _bit_depth = 8
    _L = 50
    _steps = 10
    _radiance_min = None
    _exposure_max = 50000000

    def setUp(self):
        # create dataset
        self.dataset = DatasetGenerator(height=self._height,
                                        width=self._width,
                                        bit_depth=self._bit_depth,
                                        L=self._L,
                                        steps=self._steps,
                                        radiance_min=self._radiance_min,
                                        exposure_max=self._exposure_max)

        # parse dataset
        self.parser = ParseEmvaDescriptorFile(self.dataset.descriptor_path)
        # load images
        self.loader = LoadImageData(self.parser.images)
        # create data
        self.data = Data1288(self.loader.data)
        # create operation point dict
        self.op = info_op()

    def tearDown(self):
        del self.dataset
        del self.parser
        del self.loader
        del self.op
        del self.data

    def test_report_generation(self):
        with tempfile.TemporaryDirectory() as outdir:
            # create report
            report = Report1288(outdir)
            report.add(self.op, self.data.data)
            # check that output directory has been created
            for directory in ('files', 'OP1', 'upload'):
                self.assertTrue(os.path.isdir(os.path.join(outdir, directory)))
            # check that plots have been created and saved
            for plt in EVMA1288plots:
                name = plt.__name__
                path = os.path.join(outdir, 'OP1', name + '.pdf')
                self.assertTrue(os.path.isfile(path))
            # generate tex files
            report.latex()
            # check that tex files have been created
            for fil in ('emvadatasheet.sty', 'report.tex'):
                path = os.path.join(outdir, fil)
                self.assertTrue(os.path.isfile(path))
