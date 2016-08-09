import unittest
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.camera.dataset_generator import DatasetGenerator


class TestParser(unittest.TestCase):
    # attrbutes for dataset generator
    _bit_depth = 8
    _version = '3.0'
    _height = 50
    _width = 100
    _L = 50
    _steps = 10

    # Don't test error raised because those errors appears when descriptor file
    # is not well formatted and not because of a code failure.

    def setUp(self):
        # create data descriptor file for parser
        self.d_generator = DatasetGenerator(bit_depth=self._bit_depth,
                                            height=self._height,
                                            width=self._width,
                                            L=self._L,
                                            version=self._version,
                                            steps=self._steps)

    def tearDown(self):
        # delete generator to delete all the generated files
        del self.d_generator

    def test_good_descriptorfile(self):
        # test that the parser actually parses the file with the generated file
        descriptor_file = self.d_generator.descriptor_path
        parser = ParseEmvaDescriptorFile(descriptor_file)
        # data manually taken from the file:
        bits = self._bit_depth
        height = self._height
        width = self._width
        times = self.d_generator.points['temporal'].keys()
        first_exp_time = list(times)[0]
        first_rad = self.d_generator.points['temporal'][first_exp_time][0]
        first_pcount = round(self.d_generator.cam.get_photons(first_rad,
                                                              first_exp_time),
                             3)

        # check data have correctly been parsed
        self.assertEqual(parser.version, self._version)
        self.assertEqual(parser.format['bits'], bits)
        self.assertEqual(parser.format['height'], height)
        self.assertEqual(parser.format['width'], width)
        # for this expTime and pcount, there is only 2 images thus temporal
        im = parser.images['temporal'][first_exp_time][first_pcount]
        # The length of this dict should be 2
        self.assertEqual(len(im), 2)

        # For spatial data
        points = self.d_generator.points['spatial']
        spatial_texp = list(points.keys())[0]
        spatial_rad = list(points.values())[0][0]
        # round here because pcount are rounded in descriptor file
        spatial_pcount = round(self.d_generator.cam.get_photons(spatial_rad),
                               3)
        im_spatial = parser.images['spatial'][spatial_texp][spatial_pcount]
        # the length of this dict should be greater than 2
        self.assertGreater(len(im_spatial), 2)

        # For dark images, pcount should be 0
        # for this time, there is a dark image
        # dark images are normal images with 0.0 photon count
        keys = parser.images['temporal'][first_exp_time].keys()
        self.assertTrue(0.0 in keys)
