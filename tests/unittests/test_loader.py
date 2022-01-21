import unittest
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.loader import LoadImageData


class TestLoader(unittest.TestCase):
    _height = 50
    _width = 100
    _bit_depth = 8
    _L = 50
    _steps = 10

    def _init(self):
        # Create dataset to load
        dataset = DatasetGenerator(height=self._height,
                                   width=self._width,
                                   bit_depth=self._bit_depth,
                                   L=self._L,
                                   steps=self._steps)
        descriptor_path = dataset.descriptor_path
        # create the parser
        parser = ParseEmvaDescriptorFile(descriptor_path)
        # create loader
        loader = LoadImageData(parser.images)
        return dataset, parser, loader

    def test_loader(self):
        d, p, l = self._init()
        self.dataset = d
        self.parser = p
        self.loader = l
        # Test that checks if loader actually loads data from images given
        # by the parser

        # test that the data attribute contains the good infos
        data = self.loader.data
        self.assertEqual(data['height'], self._height)
        self.assertEqual(data['width'], self._width)
        first_exp_time = self.dataset.cam.exposure_min
        # temporal data should contain 2 datasets (one bright one dark)
        temporal_data = data['temporal'][first_exp_time]
        self.assertEqual(len(temporal_data), 2)
        self.assertTrue(0.0 in temporal_data.keys())
        # there should be steps data sets for temporal
        self.assertEqual(len(self.loader.data['temporal']), self._steps)

        spatial_texp = list(self.dataset.points['spatial'].keys())[0]

        # spatial data should contain 2 sets (one dark and one bright)
        spatial_data = data['spatial'][spatial_texp]
        self.assertEqual(len(spatial_data), 2)
        self.assertTrue(0.0 in spatial_data.keys())
        # data should be made of L images
        self.assertEqual(spatial_data[0.0]['L'], self._L)

        # check data type and format
        for typ in ('sum', 'pvar'):
            # data is sum and pvar
            self.assertTrue(typ in spatial_data[0.0].keys())
            self.assertTrue(typ in temporal_data[0.0].keys())
            # spatial data is sum images and pvar
            self.assertEqual(spatial_data[0.0][typ].shape, (self._height,
                                                            self._width))

        del self.parser
        del self.dataset
        del self.loader

    def test_loader_errors(self):
        # check that images with no dark images raise ValueError
        with self.assertRaises(ValueError):
            images = {'temporal': {0: {0.1: ""}},
                      'spatial': {0: {0.1: ""}}}
            l = LoadImageData(images)
        # check that one image for temporal instead of 2 raise valueerror
        with self.assertRaises(ValueError):
            images = {'temporal': {0: {0.0: ""}},
                      'spatial': {0: {0.0: ""}}}
            l = LoadImageData(images)

        # Check that an image that does not exist raise an IOError
        with self.assertRaises(IOError):
            images = {'temporal': {0: {0.0: ["."], 0.1: ["."]}},
                      'spatial': {0: {0.0: ["."], 0.1: ["."]}}}
            l = LoadImageData(images)
