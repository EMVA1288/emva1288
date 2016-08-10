# This module contains a generator that generates a descriptor file and the
# corresponding images using the implemented camera.

from emva1288.camera.camera import Camera as Cam
from emva1288.camera.points_generator import PointsGenerator
from collections import OrderedDict
import numpy as np
import tempfile
import os
from PIL import Image


def _get_emva_gain(cam):
    """Find the gain to satisfy EMVA1288 requirements"""
    gini = cam.K
    # Find gain with a minum temporal noise of 0.5DN
    g = cam.Ks[0]
    for gain in cam.Ks:
        cam.K = gain
        g = gain
        img1 = cam.grab(0).astype(np.int64)
        img2 = cam.grab(0).astype(np.int64)
        if (img1 - img2).std() > 0.5:
            break
    cam.K = gini
    return g


def _get_emva_blackoffset(cam):
    """Find the blackoffset to satifsfy EMVA1288 requirements"""
    bini = cam.blackoffset
    # Find black offset with a maximum of 0.5% of values at Zero
    bo = cam.blackoffsets[0]
    pixels = cam.width * cam.height
    for i in cam.blackoffsets:
        cam.blackoffset = i
        img = cam.grab(0)
        bo = i
        if np.count_nonzero(img) > pixels * .995:
            break
    cam.blackoffset = bini
    return bo


class DatasetGenerator:
    """Dataset generator.

    Creates a descriptor file and the corresponding linked images for a
    a exposure variant test example according to the emva1288 standart.
    The images are created using the implemented camera in the emva module.
    """

    def __init__(self,
                 steps=100,
                 L=50,
                 version='3.0',
                 image_format='png',  # best memory consumption
                 outdir=None,  # directory where to save the dataset
                 radiance_min=None,
                 radiance_max=None,
                 exposure_fixed=None,
                 **kwargs
                 ):
        """Dataset generator init method.

        The generator uses a
        :class:`~emva1288.camera.points_generator.PointsGenerator`
        object to create the operation points. It then grabs the images
        for these points using a
        :class:`~emva1288.camera.camera.Camera` simulator object.
        The camera is intialized according to the
        given kwargs. Then, after getting the test points,
        it :meth:`makes <run_test>` the images
        with it by changing its
        exposure time, or the radiation and :meth:`saves <save_images>`
        the images and the descriptor file.

        Parameters
        ----------
        L : int, optional
            The number of image taken during a spatial test point.
        version : str, optional
                  Data version to add in descriptor file.
        image_format : str, optional
                       The image's format when they are saved.
        outdir : str, optional
                 The output directory where the descriptor file and the images
                 will be saved. If None, it will create a tempory directory
                 that will be deleted (and its contents) when the dataset
                 generator object is deleted.
        radiance_min : float, optional
                       Same as in
                       :class:`~emva1288.camera.points_generator.PointsGenerator`.
        radiance_max : float, optional
                       Same as in
                       :class:`~emva1288.camera.points_generator.PointsGenerator`.
        exposure_fixed : float, optional
                         Same as in
                         :class:`~emva1288.camera.points_generator.PointsGenerator`.
        kwargs : All other kwargs are passed to the camera.
        """
        self._steps = steps  # number of points to take
        self.cam = Cam(**kwargs)
        # set the camera parameters for the test
        self.cam.exposure = self.cam.exposure_min

        # If no blackoffset/gain are specified find them according to standard
        if 'blackoffset' not in kwargs:
            self.cam.blackoffset = _get_emva_blackoffset(self.cam)
        if 'K' not in kwargs:
            self.cam.K = _get_emva_gain(self.cam)

        # create test points
        points = PointsGenerator(self.cam,
                                 radiance_min=radiance_min,
                                 radiance_max=radiance_max,
                                 exposure_fixed=exposure_fixed,
                                 steps=self._steps)
        self._points = points.points

        self._L = L  # number of images to take for a spatial test
        self._version = version  # data version
        # store image format
        self._image_format = image_format

        # images will be saved one at a time during the generation into outdir
        self.outdir = outdir
        # create temporary directory to store the dataset
        if outdir is None:
            self.tempdir = tempfile.TemporaryDirectory()
            self.outdir = self.tempdir.name
        # create dir where images will be saved
        os.makedirs(os.path.join(self.outdir, 'images'))

        # run test
        self._descriptor_path = self.run_test()

    @property
    def points(self):
        """The test points suite."""
        return self._points

    @property
    def descriptor_path(self):
        """The absolute path to the descriptor file."""
        return self._descriptor_path

    def _is_point_spatial_test(self, i):
        """Check if a point index should be a spatial test.

        Spatial points are done at midpoint of bright and dark series.
        """
        v = self._steps // 2
        if i in (v, self._steps + v):
            return True
        return False

    def _get_descriptor_line(self, exposure, radiance):
        """Create the line introducing a test point images in descriptor."""
        if radiance == 0.0:
            # dark image
            return "d %.1f" % exposure
        # bright image
        # round photons count to three decimals
        return "b %.1f %.3f" % (exposure,
                                round(self.cam.get_photons(radiance), 3))

    def _get_image_names(self, number, L):
        """Create an image filename."""
        names = []
        for l in range(L):
            names.append("img_%04d.%s" % (number, self._image_format))
            number += 1
        return names, number

    def _get_imgs(self, radiance, L):
        """Create a list of image from the given radiances.
        """
        # computes an array of dict whose keys are the name of the file
        # and the data of the image to save
        imgs = []
        for l in range(L):
            imgs.append(self.cam.grab(radiance))
        return imgs

    def run_test(self):
        """Run the test points, save the images and generate descriptor."""
        descriptor_text = OrderedDict()
        image_number = 0
        # descriptor file path
        path = os.path.join(self.outdir, "EMVA1288descriptor.txt")
        # open descriptor file to write images in it
        with open(path, "w") as f:
            # write version
            f.write("v %s\n" % self._version)
            # wtite camera's properties
            f.write("n %i %i %i\n" % (self.cam.bit_depth,
                                      self.cam.width,
                                      self.cam.height))
            for kind in ('temporal', 'spatial'):

                # number of image to take
                L = 2
                if kind == 'spatial':
                    L = self._L
                for texp, radiances in self.points[kind].items():
                    # set camera
                    self.cam.exposure = texp
                    # Grab all images for these radiances
                    for radiance in radiances:
                        # Get descriptor line introducting the images
                        f.write("%s\n"
                                % self._get_descriptor_line(texp, radiance))
                        for l in range(L):
                            # grab
                            img = self.cam.grab(radiance)
                            # write the name in descriptor
                            name = "image%i.%s" % (image_number,
                                                   self._image_format)
                            f.write("i images\\%s\n" % name)
                            image_number += 1

                            # save image
                            self.save_image(img, name)

        # return descriptor path
        return path

    def save_image(self, img, name):
        """Save the image.
        """
        # save the images contained in d
        dtype = np.uint32
        mode = 'I'
        if self.cam.bit_depth <= 8:
            # 8 bit images have special format for PIL
            dtype = np.uint8
            mode = 'L'
        im = Image.fromarray(img.astype(dtype), mode)
        path = os.path.join(self.outdir, 'images', name)
        # (filename already contains image format)
        im.save(path)
        # erase image from dict to reduce memory consuption
