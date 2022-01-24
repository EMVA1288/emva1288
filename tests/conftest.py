import pytest
import numpy as np
from emva1288.camera.camera import Camera
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.loader import LoadImageData
from emva1288.process.data import Data1288
from emva1288.camera.routines import Qe


# def pytest_generate_tests(metafunc):
#     if "dataset" in metafunc.fixturenames:
#         metafunc.parametrize("dataset", ["single_exposure", None], indirect=True)


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


# #parser
# # attrbutes for dataset generator
# _version = '3.0'
#
#
# #loader
# _bit_depth = 8
# _height = 50
# _width = 100
# _L = 50
# _steps = 10
#
# #Report
# _radiance_min = None
# _exposure_max = 50000000
#
# #results
# _qe = Qe(width=_width, height=_height)
# _radiance_min = None
# _exposure_max = 5000000000
# _dark_current_ref = 30
# _exposure_fixed = 10000000
# _temperature = 20
# _temperature_ref = 20
# _K = 0.5
# _exposure_min = 50000
# _dsnu = np.zeros((_height, _width))
# _dsnu[0, :] += 5
# _prnu = np.ones((_height, _width))
# _prnu[-1, :] += 1.5
#
#
#
# def setUp(self):
#     # create dataset
#     self.dataset = DatasetGenerator(height=self._height,
#                                     width=self._width,
#                                     bit_depth=self._bit_depth,
#                                     L=self._L,
#                                     steps=self._steps,
#                                     radiance_min=self._radiance_min,
#                                     exposure_max=self._exposure_max)
#
#
# # loader
#     dataset = DatasetGenerator(height=50,
#                                width=100,
#                                bit_depth=8,
#                                L=50,
#                                steps=10)
#     descriptor_path = dataset.descriptor_path
#     # create the parser
#     parser = ParseEmvaDescriptorFile(descriptor_path)
#     # create loader
#     loader = LoadImageData(parser.images)
#     return dataset, parser, loader
#
#
# #results
#     _height = 50
#     _width = 100
#     _bit_depth = 8
#     _L = 50
#     _qe = Qe(width=_width, height=_height)
#     _steps = 10
#     _radiance_min = None
#     _exposure_max = 5000000000
#     _dark_current_ref = 30
#     _exposure_fixed = 10000000
#     _temperature = 20
#     _temperature_ref = 20
#     _K = 0.5
#     _exposure_min = 50000
#     _dsnu = np.zeros((_height, _width))
#     _dsnu[0, :] += 5
#     _prnu = np.ones((_height, _width))
#     _prnu[-1, :] += 1.5
#
#     def test_results_exposure_variation(self):
#         dt, p, l, da, r = _init(height=self._height,
#                                 width=self._width,
#                                 bit_depth=self._bit_depth,
#                                 L=self._L,
#                                 qe=self._qe,
#                                 steps=self._steps,
#                                 radiance_min=self._radiance_min,
#                                 exposure_max=self._exposure_max,
#                                 exposure_min=self._exposure_min,
#                                 K=self._K,
#                                 dark_current_ref=self._dark_current_ref,
#                                 temperature=self._temperature,
#                                 temperature_ref=self._temperature_ref,
#                                 dsnu=self._dsnu,
#                                 prnu=self._prnu)
#
# #parser
#    # attrbutes for dataset generator
#     _bit_depth = 8
#     _version = '3.0'
#     _height = 50
#     _width = 100
#     _L = 50
#     _steps = 10
#
#     # Don't test error raised because those errors appears when descriptor file
#     # is not well formatted and not because of a code failure.
#
#     def setUp(self):
#         # create data descriptor file for parser
#         self.d_generator = DatasetGenerator(bit_depth=self._bit_depth,
#                                             height=self._height,
#                                             width=self._width,
#                                             L=self._L,
#                                             version=self._version,
#                                             steps=self._steps)