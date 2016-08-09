import unittest
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.loader import LoadImageData
from emva1288.process.data import Data1288
from emva1288.unittests.test_routines import del_obj


class TestData(unittest.TestCase):
    _height = 50
    _width = 100
    _bit_depth = 8
    _L = 50
    _steps = 10
    _radiance_min = None
    _exposure_max = 50000000

    def _init(self):
        # create dataset
        dataset = DatasetGenerator(height=self._height,
                                   width=self._width,
                                   bit_depth=self._bit_depth,
                                   L=self._L,
                                   steps=self._steps,
                                   radiance_min=self._radiance_min,
                                   exposure_max=self._exposure_max)
        # parse dataset
        parser = ParseEmvaDescriptorFile(dataset.descriptor_path)
        # load images
        loader = LoadImageData(parser.images)
        # create data
        data = Data1288(loader.data)
        return dataset, parser, loader, data

    def test_data(self):
        """Test that data1288 retrieves information."""
        ds, p, l, d = self._init()
        self.dataset = ds
        self.parser = p
        self.loader = l
        self.data = d

        # test number of pixels
        self.assertEqual(self.data.pixels, self._height * self._width)

        # test data attribute
        #####################
        data = self.data.data

        # Test spatial
        self.assertEqual(data['spatial']['L'], self._L)
        self.assertEqual(data['spatial']['L_dark'], self._L)  # same L for dark
        # spatial exposure time
        texp = list(self.dataset.points['spatial'].keys())[0]
        self.assertEqual(data['spatial']['texp'], texp)
        # spatial photons
        radiance = self.dataset.points['spatial'][texp][0]
        photons = round(self.dataset.cam.get_photons(radiance), 3)
        self.assertEqual(data['spatial']['u_p'], photons)
        # spatial data are images
        for typ in ('avg', 'avg_dark', 'pvar', 'pvar_dark', 'sum',
                    'sum_dark', 'var', 'var_dark'):
            self.assertTrue(typ in data['spatial'].keys())
            self.assertEqual(data['spatial'][typ].shape, (self._height,
                                                          self._width))

        # test temporal
        # all temporal data are arrays of length steps
        for typ in ('s2_y', 's2_ydark', 'texp', 'u_p', 'u_y', 'u_ydark'):
            self.assertTrue(typ in data['temporal'].keys())
            self.assertEqual(len(data['temporal'][typ]), self._steps)

        # test exposure times and photons have well be retrieved
        times = list(self.dataset.points['temporal'].keys())
        for i, (exp, photons) in enumerate(zip(data['temporal']['texp'],
                                               data['temporal']['u_p'])):
            time = times[i]
            radiance = self.dataset.points['temporal'][time][0]
            photon = round(self.dataset.cam.get_photons(radiance, time), 3)
            self.assertEqual(exp, times[i])
            self.assertEqual(photons, photon)

        # delete objects
        del_obj(self.dataset, self.parser, self.loader, self.data)

    def test_1exposure(self):
        """Test that when there is only one exposure time, the temporal data
        dictionary has same length than the number of photons."""
        self._radiance_min = 0.1
        self._exposure_max = 1000000
        ds, p, l, d = self._init()
        self.dataset = ds
        self.parser = p
        self.loader = l
        self.data = d
        temporal = self.data.data['temporal']
        l = len(temporal['u_p'])
        # test that all temporal data arrays have same length
        self.assertEqual(len(temporal['texp']), l)
        self.assertEqual(len(temporal['u_ydark']), l)
        self.assertEqual(len(temporal['s2_ydark']), l)

        del_obj(self.dataset, self.parser, self.loader, self.data)

    def test_data_errors(self):
        # Test that given an incomplete data dictionary, it will raise errors
        # if there is no dark data in temporal
        with self.assertRaises(ValueError):
            dat = {'width': 1, 'height': 1,
                   'temporal': {0: {0.1: None}},
                   'spatial': {0: {0.1: None}}}
            d = Data1288(dat)

        # if no dark data in spatial
        with self.assertRaises(ValueError):
            dat = {'width': 1, 'height': 1,
                   'temporal': {0: {0.0: {'sum': 0, 'pvar': 0},
                                    0.1: {'sum': 0, 'pvar': 0}}},
                   'spatial': {0: {0.1: {'sum': 0, 'pvar': 0}}}}
            d = Data1288(dat)

        # if there is no bright image for each dark
        with self.assertRaises(ValueError):
            dat = {'width': 1, 'height': 1,
                   'temporal': {0: {0.0: None}},
                   'spatial': {0: {0.0: None}}}
            d = Data1288(dat)

        # If there is no bright image for spatial
        with self.assertRaises(ValueError):
            dat = {'width': 1, 'height': 1,
                   'temporal': {0: {0.0: {'sum': 0, 'pvar': 0},
                                    0.1: {'sum': 0, 'pvar': 0}}},
                   'spatial': {0: {0.0: {'sum': 0, 'pvar': 0}}}}
            d = Data1288(dat)

        # If there is more than 1 exposure time with spatial data
        with self.assertRaises(ValueError):
            dat = {'width': 1, 'height': 1,
                   'temporal': {0: {0.0: {'sum': 0, 'pvar': 0},
                                    0.1: {'sum': 0, 'pvar': 0}}},
                   'spatial': {0: {0.0: {'sum': 0, 'pvar': 0},
                                   0.1: {'sum': 0, 'pvar': 0}},
                               1: {0.0: {'sum': 0, 'pvar': 0},
                                   0.1: {'sum': 0, 'pvar': 0}}}}
            d = Data1288(dat)
