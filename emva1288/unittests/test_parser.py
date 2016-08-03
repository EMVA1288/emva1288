import unittest
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.camera.dataset_generator import DatasetGenerator


class TestParser(unittest.TestCase):
    # attrbutes for dataset generator
    _bit_depth = 8
    _version = '3.0'
    _img_x = 100
    _img_y = 50
    _L = 50
    _steps = 10

    # Don't test error raised because those errors appears when descriptor file
    # is not well formatted and not because of a code failure.

    def setUp(self):
        # create data descriptor file for parser
        self.dataset_generator = DatasetGenerator(bit_depth=self._bit_depth,
                                                  img_x=self._img_x,
                                                  img_y=self._img_y,
                                                  L=self._L,
                                                  version=self._version,
                                                  steps=self._steps)

    def tearDown(self):
        # delete generator to delete all the generated files
        del self.dataset_generator

    def test_good_descriptorfile(self):
        # test that the parser actually parses the file with the generated file
        descriptor_file = self.dataset_generator.descriptor_path
        parser = ParseEmvaDescriptorFile(descriptor_file)
        # data manually taken from the file:
        bits = self._bit_depth
        height = self._img_y
        width = self._img_x
        first_exp_time = self.dataset_generator.points[0]['exposure']
        first_pcount = self.dataset_generator.points[0]['photons']

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
        points = self.dataset_generator.points[self._steps // 2]
        # this index is spatial by definition of generator
        spatial_texp = points['exposure']
        spatial_pcount = points['photons']
        im_spatial = parser.images['spatial'][spatial_texp][spatial_pcount]
        # the length of this dict should be greater than 2
        self.assertGreater(len(im_spatial), 2)

        # For dark images, pcount should be 0
        dark_texp = self.dataset_generator.points[self._steps]['exposure']
        # for this time, there is a dark image
        # dark images are normal images with 0.0 photon count
        self.assertTrue(0.0 in parser.images['temporal'][dark_texp].keys())
