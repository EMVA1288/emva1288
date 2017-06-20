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
        h = 7
        w = 5
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
        h = 480
        w = 640
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
        # Test if the mean of the green it's 100% of the target +/- 5%
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=green_filter).mean(),
            target, delta=5.0,
            msg="green not in range")
        # Test if the mean of the red it's 15% of the target +/- 5%
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=red_filter).mean(),
            target*transmition_red, delta=5.0,
            msg="red not in range")
        # Test if the mean of the blue it's 2% of the target +/- 5%
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=blue_filter).mean(),
            target*transmition_blue, delta=5.0,
            msg="blue not in range")


class CameraTestPrnuDsnu(unittest.TestCase):
    def test_prnu(self):
        # Init the parameters
        h = 480
        w = 640
        value8 = 3
        prnu_array = np.array([1, 1, 1, 1, 1, 1, 1, value8])
        w_r = int(np.floor(w / prnu_array.shape[0]) + 1)
        prnu = np.tile(prnu_array, (h, w_r))[:h, :w]
        # Test if the prnu have the same shape than what we give it
        self.assertEqual((h, w), prnu.shape)
        # Set the camera for testing the prnu
        cam = Camera(width=w, height=h, prnu=prnu)
        target = cam.img_max / 2
        radiance = cam.get_radiance_for(mean=target)
        img = cam.grab(radiance)
        prnu_array_test_1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        prnu_test_1 = np.tile(prnu_array_test_1, (h, w_r))[:h, :w]
        prnu_array_test_8 = np.array([1, 1, 1, 1, 1, 1, 1, 0])
        prnu_test_8 = np.tile(prnu_array_test_8, (h, w_r))[:h, :w]
        # Test if the mean it's 100% of the target +/- 5%
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=prnu_test_1).mean(),
            target, delta=5.0,
            msg="1 it's not in range")
        # Test if the mean of the 8th value it's 3x of the target +/- 5%
        top_target = target * value8
        if top_target >= 255:
            top_target = 255.
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=prnu_test_8).mean(),
            top_target, delta=5.0,
            msg="8 value it's not in range")

    def test_dsnu(self):
        # Init the parameters
        h = 480
        w = 640
        rep = 200
        value8 = 5
        dsnu_array = np.array([0, 0, 0, 0, 0, 0, 0, value8])
        w_r = int(np.floor(w / dsnu_array.shape[0]) + 1)
        dsnu = np.tile(dsnu_array, (h, w_r))[:h, :w]
        # Test if the dsnu have the same shape than what we give it
        self.assertEqual((h, w), dsnu.shape)
        # Set the camera for testing the dsnu
        cam = Camera(width=w, height=h, dsnu=dsnu)
        target = cam.dark_signal_0 / 10  # offset
        mean_img = []
        for i in range(1, rep):
            mean_img.append(cam.grab(0))
            tmp_img = np.mean(mean_img, axis=0)
            mean_img = [tmp_img]
        img = mean_img
        dsnu_array_test_1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        dsnu_test_1 = np.tile(dsnu_array_test_1, (h, w_r))[:h, :w]
        dsnu_array_test_8 = np.array([1, 1, 1, 1, 1, 1, 1, 0])
        dsnu_test_8 = np.tile(dsnu_array_test_8, (h, w_r))[:h, :w]
        # Test if the mean it's 100% of the target +/- 1
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=dsnu_test_1).mean(),
            target, delta=1.0,
            msg="1 it's not in range")
        # Test if the mean of the 8th value +/- 1
        top_target = target + value8 / 10
        self.assertAlmostEqual(np.ma.masked_array(
            img,
            mask=dsnu_test_8).mean(),
            top_target, delta=1.0,
            msg="8 value it's not in range")
