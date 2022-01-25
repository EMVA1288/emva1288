import pytest
import inspect
from emva1288.camera.camera import Camera
import numpy as np
from emva1288.camera import routines

import logging
logger = logging.getLogger(__name__)


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
    # Get the radiance to grab from the second cam. The output radiance
    # is affected by qe, so the bayer_filter as well
    radiance = cam_d.get_radiance_for(mean=target)
    img = cam.grab(radiance)
    green_filter = np.tile([[0, 1], [1, 0]], (int(h/2), int(w/2)))
    blue_filter = np.tile([[1, 0], [1, 1]], (int(h/2), int(w/2)))
    red_filter = np.tile([[1, 1], [0, 1]], (int(h/2), int(w/2)))

    gf = b_layer[0, 0, :].mean()
    rf = b_layer[0, 1, :].mean()
    bf = b_layer[1, 0, :].mean()

    values = {"img": img, "green_filter": green_filter, "blue_filter": blue_filter, "red_filter": red_filter,
              "green_bayer": gf, "red_bayer": rf, "blue_bayer": bf, "target": target}
    logger.info(values)
    return values


def test_setup_camera(camera):
    logger.info(f'\n The {inspect.stack()[0][3]} test has passed successfully.')
    return camera


def test_teardown_camera(camera):
    del camera
    logger.info(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


def test_img(camera):
    img = camera.grab(0)
    assert (camera.height, camera.width) == np.shape(img)
    logger.info(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


def test_radiance(camera):
    img1 = camera.grab(0)
    img2 = camera.grab(camera.get_radiance_for(mean=250))
    assert (img1.mean() < img2.mean()),\
        pytest.fail(f' The mean value of img1 is expected to be less than the mean value of img2 . \n'
                    f'img1::: \n {img1}\n img2::: \n {img2}', False)
    logger.info(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


def test_get_tile_1d():
    h, w = [1, 24]
    dim1 = np.zeros((8))
    res_array = routines.get_tile(dim1, h, w)
    res_dim1 = np.zeros((24))
    assert w == res_array.shape[0]
    assert (res_dim1 == res_array).all()
    logger.info(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


def test_get_tile_2d():
    h, w = [5, 7]
    dim2 = np.zeros((3))
    res_array = routines.get_tile(dim2, h, w)
    res_dim2 = np.zeros((5, 7))
    assert (h, w) == res_array.shape
    assert (res_dim2 == res_array).all()
    logger.info(f'\n The {inspect.stack()[0][3]} test has passed successfully.')


@pytest.mark.parametrize('colour', ['red', 'green', 'blue'])
def test_bayer_filters(colour):
    values = bayer()
    filtr = values['target'] * values['red' +'_bayer']

    # TODO: the xxx_filter are too close to one another, so setting a fixed red_filter pass the test.
    #  Make the colors diverge more
    filter_mean = (np.ma.masked_array(values['img'], mask=values[colour +'_filter']).mean())
    test_name = inspect.stack()[0][3]
    assert filter_mean == pytest.approx(filtr, abs=10), \
        pytest.fail(f'The {colour} filter mean value: {filter_mean}\n is not within the target range: {filtr}')
    logger.info(f' The {test_name} test has completed successfully.')


def test_prnu():
    # Init the parameters
    h, w = [480, 640]
    rep = 200
    value8 = 3
    # create the pattern of the prnu
    prnu_array = np.ones((8))
    prnu_array[-1] = value8
    prnu = routines.get_tile(prnu_array, h, w)
    # Set the camera for testing the prnu
    cam = Camera(width=w, height=h, prnu=prnu)
    var = np.sqrt(cam._sigma2_dark_0)
    target = cam.img_max / 2
    # The target (top_target) is the multiplication of the target
    # (what we expect without prnu) and the value8(prnu). We can do it
    # because if we look at the _u_e function in emva1288.camera.camera
    # the prnu affect the QE with a multiplication. So if we just
    # multiplied the target by the prnu it's the same thing.
    # But this value can go over the maximal value for one pixel, this
    # is why we use the min() function to take the maximal value than the
    # camera can take.
    top_target = min(target * value8, cam.img_max)
    radiance = cam.get_radiance_for(mean=target)
    img = cam.grab(radiance)
    # create the mask
    prnu_mask = np.zeros((8))
    prnu_mask[-1] = 1
    prnu_mask_resize = routines.get_tile(prnu_mask, h, w)
    prnu_non_mask = np.ones((8))
    prnu_non_mask[-1] = 0
    prnu_non_mask_resize = routines.get_tile(prnu_non_mask, h, w)
    # Test if the mean it's 100% of the target +/- variance
    assert np.ma.masked_array(img, mask=prnu_mask_resize).mean() == pytest.approx(target, abs=var)
    # Test if the mean of the 8th value it's value8
    # multiplied be the target +/- variance
    assert np.ma.masked_array(img, mask=prnu_non_mask_resize).mean() == pytest.approx(top_target, abs=var)


def test_dsnu():
    # Init the parameters
    h, w = [480, 640]
    value8 = 5
    rep = 200
    # create the pattern of the dsnu
    dsnu_array = np.ones((8))
    dsnu_array[-1] = value8
    dsnu = routines.get_tile(dsnu_array, h, w)
    # Set the camera for testing the dsnu
    cam = Camera(width=w, height=h, dsnu=dsnu)
    var = np.sqrt(cam._sigma2_dark_0)
    # The target is the number of electrons who are not affected
    # by the dsnu. To resume, we suppose to observe is a combinaison of
    # electrons from the dark signal and the temperature. The total need
    # to be multiplied by the gain of the system (K).
    # for more eplination see the grab function in emva1288.camera.camera
    target = cam.K * (cam._dark_signal_0 + cam._u_therm())
    # Here the target (top_target) is the part who is affected by
    # the dsnu. Physicaly, the same phenomen append but this time the
    # dark signal is NonUniform so thw value who represent the dsnu is
    # added to the dark signal befor the multiplication of the gain.
    top_target = cam.K * (cam._dark_signal_0 + cam._u_therm() + value8)
    img = cam.grab(0)
    # create the mask
    dsnu_mask = np.zeros((8))
    dsnu_mask[-1] = 1
    dsnu_mask_resize = routines.get_tile(dsnu_mask, h, w)
    dsnu_non_mask = np.ones((8))
    dsnu_non_mask[-1] = 0
    dsnu_non_mask_resize = routines.get_tile(dsnu_non_mask, h, w)
    # Test if the mean it's 100% of the target +/- variance
    assert np.ma.masked_array(img, mask=dsnu_mask_resize).mean() == pytest.approx(target, abs=var)
    # Test if the mean of the 8th value it's value8
    # multiplied be the target +/- variance
    assert np.ma.masked_array(img, mask=dsnu_non_mask_resize).mean() == pytest.approx(top_target, abs=var)
