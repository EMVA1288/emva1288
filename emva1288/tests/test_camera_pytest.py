import pytest
import inspect
from emva1288.camera.camera import Camera
import numpy as np
from emva1288.camera import routines
from emva1288.tests.utils.logger import Logger


logger = Logger('camera_tests.log')

@pytest.fixture
def camera():
    return Camera()


def bayer():
    h, w = [480, 640]
    wavelength = np.linspace(400, 800, 100)
    transmission_red = 670
    transmission_blue = 450
    transmission_green = 550
    b_layer = routines.get_bayer_filter(transmission_green,
                                        transmission_red,
                                        transmission_blue,
                                        transmission_green,
                                        w, h, wavelength)
    qe = routines.Qe(filter=b_layer)
    cam = Camera(width=w, height=h, qe=qe)
    cam_d = Camera(width=w, height=h)
    target = cam.img_max / 2
    radiance = cam_d.get_radiance_for(mean=target)
    img = cam.grab(radiance)
    green_filter = np.tile([[0, 1], [1, 0]], (int(h/2), int(w/2)))
    blue_filter = np.tile([[1, 0], [1, 1]], (int(h/2), int(w/2)))
    red_filter = np.tile([[1, 1], [0, 1]], (int(h/2), int(w/2)))

    gf = b_layer[0, 0, :].mean()
    rf = b_layer[0, 1, :].mean()
    bf = b_layer[1, 0, :].mean()

    values = {"img": img, "green_filter": green_filter, "blue_filter": blue_filter, "red_filter": red_filter,
              "gf": gf, "rf": rf, "bf": bf, "target": target}
    logger.log(values)
    return values


@pytest.mark.order(1)
@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.camera
def test_setup_camera(camera):
    logger.log(f'\n The {inspect.stack()[0][3]} test has passed successfully.')
    return camera


@pytest.mark.order(1)
@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.camera
def test_teardown_camera(camera):
    del camera
    logger.log(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


@pytest.mark.order(1)
@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.camera
def test_img(camera):
    img = camera.grab(0)
    assert (camera.height, camera.width) == np.shape(img)
    logger.log(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


@pytest.mark.order(1)
@pytest.mark.regression
@pytest.mark.camera
def test_radiance(camera):
    img1 = camera.grab(0)
    img2 = camera.grab(camera.get_radiance_for(mean=250))
    assert (img1.mean() < img2.mean()),\
        pytest.fail(f' The mean value of img1 is expected to be less than the mean value of img2 . \n'
                    f'img1::: \n {img1}\n img2::: \n {img2}', False)
    logger.log(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


@pytest.mark.order(2)
@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.tile
def test_get_tile_1d():
    h, w = [1, 24]
    dim1 = np.zeros((8))
    res_array = routines.get_tile(dim1, h, w)
    res_dim1 = np.zeros((24))
    assert w == res_array.shape[0]
    assert res_dim1.tolist() == res_array.tolist()
    logger.log(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


@pytest.mark.order(2)
@pytest.mark.regression
@pytest.mark.tile
def test_get_tile_2d():
    h, w = [5, 7]
    dim2 = np.zeros((3))
    res_array = routines.get_tile(dim2, h, w)
    res_dim2 = np.zeros((5, 7))
    assert (h, w) == res_array.shape
    assert res_dim2.tolist() == res_array.tolist()
    logger.log(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


@pytest.mark.order(3)
@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.filters
def test_bayer_green_filter_mean():
    values = bayer()
    green_filter_mean = (np.ma.masked_array(values['img'], mask=values['green_filter']).mean())
    expected = 32.32
    assert (np.ma.masked_array(values['img'], mask=values['green_filter']).mean()) == pytest.approx(expected, abs=.1), \
        pytest.fail(f' The actual green filter mean value: {green_filter_mean} is not within the expected range')
    logger.log(f' The {inspect.stack()[0][3]} test has completed successfully.'
               f'\n The green filter mean value of: {green_filter_mean} \n is within the expected range of: {expected}')


@pytest.mark.order(3)
@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.filters
def test_bayer_green_filter():
    values = bayer()
    expected = 25.115
    green_filter = (values['target'] * values['rf'])
    assert (values['target'] * values['gf']) == pytest.approx(expected, abs=.01), \
        pytest.fail(f' The actual green filter value: {green_filter} is not within the expected range')
    logger.log(f' \n The {inspect.stack()[0][3]} test has completed successfully.'
               f'\n The green filter value of: {green_filter} \n is within the expected range of: {expected}')


@pytest.mark.order(3)
@pytest.mark.regression
@pytest.mark.filters
def test_bayer_red_filter_mean():
    values = bayer()
    expected = 34.68
    red_filter_mean = (np.ma.masked_array(values['img'], mask=values['red_filter']).mean())
    assert (np.ma.masked_array(values['img'], mask=values['red_filter']).mean()) == pytest.approx(expected, abs=.1), \
        pytest.fail(f' The actual red filter mean value: {red_filter_mean} is not within the expected range')
    logger.log(f' The {inspect.stack()[0][3]} test has completed successfully.'
               f'\n The red filter mean value of: {red_filter_mean} \n is within the expected range of: {expected}')


@pytest.mark.order(3)
@pytest.mark.regression
@pytest.mark.filters
def test_bayer_red_filter():
    values = bayer()
    expected = 25.03
    red_filter = (values['target'] * values['rf'])
    assert (values['target'] * values['rf']) == pytest.approx(expected, abs=.01), \
        pytest.fail(f' The actual red filter value: {red_filter} is not within the expected range')
    logger.log(f' The {inspect.stack()[0][3]} test has completed successfully.'
               f'\n The red filter value of: {red_filter} \n is within the expected range of: {expected}')


@pytest.mark.order(3)
@pytest.mark.regression
@pytest.mark.filters
def test_bayer_blue_filter_mean():
    values = bayer()
    expected = 20.88
    blue_filter_mean = (np.ma.masked_array(values['img'], mask=values['blue_filter']).mean())
    assert (np.ma.masked_array(values['img'], mask=values['blue_filter']).mean()) == pytest.approx(expected, abs=.1), \
        pytest.fail(f' The actual blue filter mean value: {blue_filter_mean} is not within the expected range')
    logger.log(f' The {inspect.stack()[0][3]} test has completed successfully.'
               f'\n The red filter mean value of: {blue_filter_mean} \n is within the expected range of: {expected}')


@pytest.mark.order(3)
@pytest.mark.regression
@pytest.mark.filters
def test_bayer_blue_filter():
    values = bayer()
    expected = 25.03
    blue_filter = (values['target'] * values['rf'])
    assert (values['target'] * values['rf']) == pytest.approx(expected, abs=.01), \
        pytest.fail(f' The actual blue filter value: {blue_filter} is not within the expected range')
    logger.log(f' The {inspect.stack()[0][3]} test has completed successfully.'
               f'\n The blue filter value of: {blue_filter} \n is within the expected range of: {expected}')
