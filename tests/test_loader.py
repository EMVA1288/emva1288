import pytest
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.loader import LoadImageData


# @pytest.fixture
# def dataset():
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


@pytest.mark.parametrize("dataset", ['multi_exposure'], indirect=True)
def test_loader(loader):
    dataset, parser, loader = loader
    # Test that checks if loader actually loads data from images given
    # by the parser

    # test that the data attribute contains the good infos
    data = loader.data
    assert data['height'] == dataset.cam.height
    assert data['width'] == dataset.cam.width

    first_exp_time = dataset.cam.exposure_min
    # temporal data should contain 2 datasets (one bright one dark)
    temporal_data = data['temporal'][first_exp_time]
    assert len(temporal_data) == 2
    assert 0.0 in temporal_data.keys()

    # there should be steps data sets for temporal
    assert len(loader.data['temporal']) == dataset._steps

    spatial_texp = list(dataset.points['spatial'].keys())[0]

    # spatial data should contain 2 sets (one dark and one bright)
    spatial_data = data['spatial'][spatial_texp]
    assert len(spatial_data) == 2
    assert 0.0 in spatial_data.keys()
    # data should be made of L images
    assert spatial_data[0.0]['L'] == dataset._L

    # check data type and format
    for typ in ('sum', 'pvar'):
        # data is sum and pvar
        assert typ in spatial_data[0.0].keys()
        assert typ in temporal_data[0.0].keys()
        # spatial data is sum images and pvar
        assert spatial_data[0.0][typ].shape == (dataset.cam.height, dataset.cam.width)


def test_loader_errors():
    # check that images with no dark images raise ValueError
    with pytest.raises(ValueError):
        images = {'temporal': {0: {0.1: ""}},
                  'spatial': {0: {0.1: ""}}}
        l = LoadImageData(images)
    # check that one image for temporal instead of 2 raise valueerror
    with pytest.raises(ValueError):
        images = {'temporal': {0: {0.0: ""}},
                  'spatial': {0: {0.0: ""}}}
        l = LoadImageData(images)

    # Check that an image that does not exist raise an IOError
    with pytest.raises(IOError):
        images = {'temporal': {0: {0.0: ["."], 0.1: ["."]}},
                  'spatial': {0: {0.0: ["."], 0.1: ["."]}}}
        l = LoadImageData(images)
