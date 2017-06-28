import unittest
from emva1288.camera.camera import Camera
import numpy as np
from emva1288.camera import routines


class CameraTestCase(unittest.TestCase):
    def setUp(self):
        self.cam = Camera()

    def tearDown(self):
        del self.cam

    def test_img(self):
        img = self.cam.grab(0)
        self.assertEqual((self.cam.height, self.cam.width), np.shape(img))

    def test_radiance(self):
        img1 = self.cam.grab(0)
        img2 = self.cam.grab(self.cam.get_radiance_for(mean=250))
        self.assertLess(img1.mean(), img2.mean())


class CameraTestBayer(unittest.TestCase):
    def test_get_bayer(self):
        # Init the parameters
        h, w = [7, 5]

        transmition_pixel_1 = 1
        transmition_pixel_2 = 2
        transmition_pixel_3 = 3
        transmition_pixel_4 = 4
        b_layer = routines.get_bayer_filter(transmition_pixel_1,
                                            transmition_pixel_2,
                                            transmition_pixel_3,
                                            transmition_pixel_4, w, h)
        # Supposed Results
        lines = [1, 2, 1, 2, 1]
        columns = [1, 3, 1, 3, 1, 3, 1]
        # Test to see if the layer come right
        self.assertEqual(lines, b_layer[0].tolist())
        self.assertEqual(columns, b_layer[:, 0].tolist())

    def test_bayer_layer(self):
        # Init the parameters
        h, w = [480, 640]
        transmition_red = 0.15
        transmition_blue = 0.02
        transmition_green = 1.
        b_layer = routines.get_bayer_filter(transmition_green,
                                            transmition_red,
                                            transmition_blue,
                                            transmition_green, w, h)
        # Test if the b_layer have the same shape than what we give it
        self.assertEqual((h, w), b_layer.shape)
        # Set the camera for testing the layer
        cam = Camera(width=w, height=h, radiance_factor=b_layer)
        target = cam.img_max / 2
        radiance = cam.get_radiance_for(mean=target)
        img = cam.grab(radiance)
        green_filter = routines.get_bayer_filter(0, 1, 1, 0, w, h)
        blue_filter = routines.get_bayer_filter(1, 1, 0, 1, w, h)
        red_filter = routines.get_bayer_filter(1, 0, 1, 1, w, h)
        # Test if the mean of the green it's 100% of the target +/- 5
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=green_filter).mean(),
            target, delta=5.0,
            msg="green not in range")
        # Test if the mean of the red it's 15% of the target +/- 5
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=red_filter).mean(),
            target*transmition_red, delta=5.0,
            msg="red not in range")
        # Test if the mean of the blue it's 2% of the target +/- 5
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=blue_filter).mean(),
            target*transmition_blue, delta=5.0,
            msg="blue not in range")


class CameraTestPrnuDsnu(unittest.TestCase):
    # This function replace: np.tile(array, (h, w))[:h, :w]
    # and manage to save execution time
    def get_tile(self, arr, height, width):
        # To reduce the execution time, we will reduce the width of the array
        # to the shape expected "+1" to be sure than the number is not to
        # short.
        w_r = int(np.floor(width / arr.shape[0]) + 1)
        # create a array with the dimension given and the array given
        tile = np.tile(arr, (height, w_r))[:height, :width]
        return tile

    def test_prnu(self):
        # Init the parameters
        h, w = [480, 640]
        rep = 200
        value8 = 3
        # create the pattern of the prnu
        prnu_array = np.ones((8))
        prnu_array[-1] = value8
        prnu = self.get_tile(prnu_array, h, w)
        # Set the camera for testing the prnu
        cam = Camera(width=w, height=h, prnu=prnu)
        var = np.sqrt(cam._sigma2_dark_0)
        target = cam.img_max / 2
        top_target = target * value8
        if top_target >= cam.img_max:
            top_target = cam.img_max
        radiance = cam.get_radiance_for(mean=target)
        img = cam.grab(radiance)
        # create the mask
        prnu_mask = np.zeros((8))
        prnu_mask[-1] = 1
        prnu_mask_resize = self.get_tile(prnu_mask, h, w)
        prnu_non_mask = np.ones((8))
        prnu_non_mask[-1] = 0
        prnu_non_mask_resize = self.get_tile(prnu_non_mask, h, w)
        # Test if the mean it's 100% of the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=prnu_mask_resize).mean(),
            target, delta=var,
            msg="values are not in range")
        # Test if the mean of the 8th value it's value8
        # multiplied be the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=prnu_non_mask_resize).mean(),
            top_target, delta=var,
            msg="8th value it's not in range")

    def test_dsnu(self):
        # Init the parameters
        h, w = [480, 640]
        value8 = 5
        rep = 200
        # create the pattern of the dsnu
        dsnu_array = np.ones((8))
        dsnu_array[-1] = value8
        dsnu = self.get_tile(dsnu_array, h, w)
        # Set the camera for testing the dsnu
        cam = Camera(width=w, height=h, dsnu=dsnu)
        var = np.sqrt(cam._sigma2_dark_0)
        target = cam.K * (cam._dark_signal_0 + cam._u_therm())
        top_target = cam.K * (cam._dark_signal_0 + cam._u_therm() + value8)
        img = cam.grab(0)
        # create the mask
        dsnu_mask = np.zeros((8))
        dsnu_mask[-1] = 1
        dsnu_mask_resize = self.get_tile(dsnu_mask, h, w)
        dsnu_non_mask = np.ones((8))
        dsnu_non_mask[-1] = 0
        dsnu_non_mask_resize = self.get_tile(dsnu_non_mask, h, w)
        # Test if the mean it's 100% of the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=dsnu_mask_resize).mean(),
            target, delta=var,
            msg="values are not in range")
        # Test if the mean of the 8th value it's value8
        # multiplied be the target +/- variance
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=dsnu_non_mask_resize).mean(),
            top_target, delta=var,
            msg="8th value it's not in range")
