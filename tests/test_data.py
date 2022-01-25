import pytest
import numpy as np
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.loader import LoadImageData
from emva1288.process.data import Data1288


def get_dataset(radiance_min=None, exposure_max=50000000):
    dataset = DatasetGenerator(height=50,
                               width=100,
                               bit_depth=8,
                               L=50,
                               steps=10,
                               radiance_min=radiance_min,
                               exposure_max=exposure_max)
    # parse dataset
    parser = ParseEmvaDescriptorFile(dataset.descriptor_path)
    # load images
    loader = LoadImageData(parser.images)
    # create data
    data = Data1288(loader.data)
    return dataset, parser, loader, data


def test_data():
    """Test that data1288 retrieves information."""
    dataset, parser, loader, data = get_dataset()

    # test number of pixels
    assert data.pixels == dataset.cam.height * dataset.cam.width

    # test data attribute
    #####################

    # Test spatial
    assert data.data['spatial']['L'] == dataset._L
    assert data.data['spatial']['L_dark'] == dataset._L  # same L for dark
    # spatial exposure time
    texp = list(dataset.points['spatial'].keys())[0]
    assert data.data['spatial']['texp'] == texp
    # spatial photons
    radiance = dataset.points['spatial'][texp][0]
    photons = np.round(np.sum(dataset.cam.get_photons(radiance), axis=2).mean(), 3)
    assert data.data['spatial']['u_p'] == photons
    # spatial data are images
    for typ in ('sum_dark', 'sum'):
        assert typ in data.data['spatial'].keys()
        assert data.data['spatial'][typ].shape == (dataset.cam.height, dataset.cam.width)

    # test temporal
    # all temporal data are arrays of length steps
    for typ in ('s2_y', 's2_ydark', 'texp', 'u_p', 'u_y', 'u_ydark'):
        assert typ in data.data['temporal'].keys()
        assert len(data.data['temporal'][typ]) == dataset._steps

    # test exposure times and photons have well be retrieved
    times = list(dataset.points['temporal'].keys())
    for i, (exp, photons) in enumerate(zip(data.data['temporal']['texp'],
                                           data.data['temporal']['u_p'])):
        time = times[i]
        radiance = dataset.points['temporal'][time][0]
        photon = np.round(np.sum(dataset.cam.get_photons(radiance, time), axis=2).mean(), 3)
        assert exp == times[i]
        assert photons, photon


def test_1exposure():
    """Test that when there is only one exposure time, the temporal data
    dictionary has same length than the number of photons."""
    dataset, parser, loader, data = get_dataset(radiance_min=0.1, exposure_max=1000000)
    temporal = data.data['temporal']
    l = len(temporal['u_p'])
    # test that all temporal data arrays have same length
    assert len(temporal['texp']) == l
    assert len(temporal['u_ydark']) == l
    assert len(temporal['s2_ydark']) == l


def test_data_errors():
    # Test that given an incomplete data dictionary, it will raise errors
    # if there is no dark data in temporal
    with pytest.raises(ValueError):
        dat = {'width': 1, 'height': 1,
               'temporal': {0: {0.1: None}},
               'spatial': {0: {0.1: None}}}
        d = Data1288(dat)

    # if no dark data in spatial
    with pytest.raises(ValueError):
        dat = {'width': 1, 'height': 1,
               'temporal': {0: {0.0: {'sum': 0, 'pvar': 0},
                                0.1: {'sum': 0, 'pvar': 0}}},
               'spatial': {0: {0.1: {'sum': 0, 'pvar': 0}}}}
        d = Data1288(dat)

    # if there is no bright image for each dark
    with pytest.raises(ValueError):
        dat = {'width': 1, 'height': 1,
               'temporal': {0: {0.0: None}},
               'spatial': {0: {0.0: None}}}
        d = Data1288(dat)

    # If there is no bright image for spatial
    with pytest.raises(ValueError):
        dat = {'width': 1, 'height': 1,
               'temporal': {0: {0.0: {'sum': 0, 'pvar': 0},
                                0.1: {'sum': 0, 'pvar': 0}}},
               'spatial': {0: {0.0: {'sum': 0, 'pvar': 0}}}}
        d = Data1288(dat)

    # If there is more than 1 exposure time with spatial data
    with pytest.raises(ValueError):
        dat = {'width': 1, 'height': 1,
               'temporal': {0: {0.0: {'sum': 0, 'pvar': 0},
                                0.1: {'sum': 0, 'pvar': 0}}},
               'spatial': {0: {0.0: {'sum': 0, 'pvar': 0},
                               0.1: {'sum': 0, 'pvar': 0}},
                           1: {0.0: {'sum': 0, 'pvar': 0},
                               0.1: {'sum': 0, 'pvar': 0}}}}
        d = Data1288(dat)
