import unittest
from os import path
from emva1288.process.parser import ParseEmvaDescriptorFile


class TestParser(unittest.TestCase):

    # Don't test error raised because those errors appears when descriptor file
    # is not well formatted and not because of a code failure.

    def test_good_descriptorfile(self):
        # test that the parser actually parses the file with the example file
        dir_tests = path.abspath(path.dirname(__file__))
        descriptor_file = path.join(dir_tests,
                                    '../../examples/'
                                    'EMVA1288_Descriptor_File.txt')
        parser = ParseEmvaDescriptorFile(descriptor_file)
        # data manually taken from the file:
        bits = 12
        height = 480
        width = 640
        version = '3.0'
        first_exp_time = 40000.0
        first_pcount = 120.0
        self.assertEqual(parser.version, version)
        self.assertEqual(parser.format['bits'], bits)
        self.assertEqual(parser.format['height'], height)
        self.assertEqual(parser.format['width'], width)
        # for this expTime and pcount, there is only 2 images thus temporal
        im = parser.images['temporal'][first_exp_time][first_pcount]
        # The length of this dict should be 2
        self.assertEqual(len(im), 2)

        # For spatial data
        spatial_texp = 5160000.0
        spatial_pcount = 15508.0
        im_spatial = parser.images['spatial'][spatial_texp][spatial_pcount]
        # the length of this dict should be greater than 2
        self.assertGreater(len(im_spatial), 2)

        # For dark images, pcount should be 0
        dark_texp = 40000.0  # for this time, there is a dark image
        # dark images are normal images with 0.0 photon count
        self.assertTrue(0.0 in parser.images['temporal'][dark_texp].keys())
