import numpy as np
from emva1288.process.parser import ParseEmvaDescriptorFile


def test_good_descriptorfile(dataset):
    # test that the parser actually parses the file with the generated file
    descriptor_file = dataset.descriptor_path
    parser = ParseEmvaDescriptorFile(descriptor_file)
    # data manually taken from the file:
    times = dataset.points['temporal'].keys()
    first_exp_time = list(times)[0]
    first_rad = dataset.points['temporal'][first_exp_time][0]
    first_pcount = np.round(np.sum(dataset.cam.get_photons(
                            first_rad, first_exp_time), axis=2).mean(), 3)

    # check data have correctly been parsed
    # assert parser.version == VERSION)
    assert parser.format['bits'] == dataset.cam.bit_depth
    assert parser.format['height'] == dataset.cam.height
    assert parser.format['width'] == dataset.cam.width
    # # for this expTime and pcount, there is only 2 images thus temporal
    im = parser.images['temporal'][first_exp_time][first_pcount]
    # The length of this dict should be 2
    assert len(im) == 2

    # For spatial data
    points = dataset.points['spatial']
    spatial_texp = list(points.keys())[0]
    spatial_rad = list(points.values())[0][0]
    # round here because pcount are rounded in descriptor file
    spatial_pcount = np.round(np.sum(dataset.cam.get_photons(
                              spatial_rad), axis=2).mean(), 3)
    im_spatial = parser.images['spatial'][spatial_texp][spatial_pcount]
    # the length of this dict should be greater than 2
    assert len(im_spatial) > 2

    # For dark images, pcount should be 0
    # for this time, there is a dark image
    # dark images are normal images with 0.0 photon count
    keys = parser.images['temporal'][first_exp_time].keys()
    assert 0.0 in keys
