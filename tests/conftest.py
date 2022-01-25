import pytest
import numpy as np
from emva1288.camera.camera import Camera
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.loader import LoadImageData
from emva1288.process.data import Data1288
from emva1288.camera.routines import Qe


@pytest.fixture
def camera():
    return Camera()


@pytest.fixture(params=['single_exposure', 'multi_exposure'])
def dataset(request):
    height=50
    width=100
    bit_depth=8
    L=50
    steps=10
    kwargs = {}
    dsnu = np.zeros((height, width))
    dsnu[0, :] += 5
    prnu = np.ones((height, width))
    prnu[-1, :] += 1.5
    kwargs['qe'] = Qe(width=width, height=height)
    kwargs['radiance_min'] = None
    kwargs['exposure_max'] = 5000000000
    kwargs['dark_current_ref'] = 30

    kwargs['temperature'] = 20
    kwargs['temperature_ref'] = 20
    kwargs['K'] = 0.5
    kwargs['exposure_min'] = 50000
    kwargs['dsnu'] = dsnu
    kwargs['prnu'] = prnu

    # TODO: replace this fishy selection of values with something more understandable
    if request.param == 'single_exposure':
        kwargs['exposure_fixed'] = 10000000

    elif request.param == 'multi_exposure':
        kwargs['exposure_fixed'] = None
    else:
        raise ValueError("invalid internal test config")
    dataset = DatasetGenerator(height=height,
                               width=width,
                               bit_depth=bit_depth,
                               L=L,
                               steps=steps,
                               **kwargs
                               )
    return dataset


@pytest.fixture
def parser(dataset):
    parser = ParseEmvaDescriptorFile(dataset.descriptor_path)
    return dataset, parser


@pytest.fixture
def loader(parser):
    dataset, parser = parser
    loader = LoadImageData(parser.images)
    return dataset, parser, loader


@pytest.fixture
def data(loader):
    dataset, parser, loader = loader
    data = Data1288(loader.data)
    return dataset, parser, loader, data
