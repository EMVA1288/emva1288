# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""Load image data
This class takes a dictionary (product of parser.ParseEmvaDescriptorFile).
Load the related images and reduce it's data to the minimum possible,
preserving all relevant image data in as integers
"""

import os
import numpy as np
from . import routines
import logging


class LoadImageData(object):
    """Using an info dictionary, load the images and fill data with
    the relevant information.
    """

    def __init__(self,
                 images,
                 path='',
                 loglevel=logging.INFO,
                 fload=routines.load_image,
                 fload_args=[],
                 fload_kwargs={}):
        """Loader init method.

        Uses a :class:`python:logging.Logger` object to print infos for the
        user. In the initialization, it :meth:`loads <_load_data>` the infos
        from the images into the data dictionary.

        Parameters
        ----------
        images : dict
                 The
                 :attr:`~emva1288.process.parser.ParseEmvaDescriptorFile.images`
                 attribute of a
                 :class:`~emva1288.process.parser.ParseEmvaDescriptorFile`
                 object that have parsed the images of a descriptor file.
        path : str, optional
               The path to the directory containing the images.
        loglevel : int, optional
                   The information level for the
                   :class:`~python:logging.Logger`.
        fload : func. optional
                The function that will load the images (one at a time).
        fload_args : list, optional
                     The list of args for the fload function.
        fload_kwargs : dict, optional
                       The kwargs dictionary for the fload function.
        """
        self.data = {'temporal': {}, 'spatial': {}}
        self._fload = fload
        self._fload_args = fload_args
        self._fload_kwargs = fload_kwargs
        self._shape = set()

        logging.basicConfig()
        self.log = logging.getLogger('Loader')
        self.log.setLevel(loglevel)

        self._path = path
        # load the images
        self._load_data(images)

    def _load_data(self, images):
        """Using the information in images dict fill self.data when loading
        the images and fills the temporal and spatial dicts
        """
        # iterate over the kind
        for kind in ('temporal', 'spatial'):
            exposures = set(images[kind].keys())

            # iterate over the exposure times for each kind
            for exposure in exposures:
                # for this exposure time, get the list of photons counts
                photon_counts = list(images[kind][exposure].keys())

                if 0.0 not in photon_counts:
                    # Every exposure time should have at least one dark
                    # image with a 0 photon count.
                    raise ValueError('Every exposure must have a 0.0 photons'
                                     ' for dark information')
                if len(photon_counts) < 2:
                    # Every exposure time should have at least one dark image
                    # and one bright image.
                    raise ValueError('All exposure must have at least 2 points'
                                     ' one dark and one bright')

                # Iterate over the images for this exposure time and this
                # photon count
                for photons, fnames in images[kind][exposure].items():
                    # get the image data
                    data_imgs = self._get_imgs_data(fnames, kind)
                    # fill data dict
                    self.data[kind].setdefault(exposure, {})
                    self.data[kind][exposure][photons] = data_imgs

        shape = self._shape.pop()
        self.data['height'] = shape[0]
        self.data['width'] = shape[1]

    def _get_imgs_data(self, fnames, kind):
        """Returns the desired image data
        This depends on the kind of data
        """
        # Load a list containing the images
        arr_imgs = self._load_imgs(fnames)
        # Get the sum and the pseudo-variance of this list of images
        imgs = routines.get_int_imgs(arr_imgs)
        # For spatial we only want the sum and pseudo-var of the images
        if kind == 'spatial':
            return imgs
        # For temporal, we only want the basic statistics (we don't need
        # the whole sum and pvar images)
        d = {}
        d['sum'] = np.sum(imgs['sum'])  # sum of the sum of images
        d['pvar'] = np.sum(imgs['pvar'])  # sum of the pvar image
        return d

    def _load_imgs(self, fnames):
        """For a list of path to images, load and append them to a returned list.
        """
        imgs = []
        # Iterate over all images
        for fname in fnames:
            # Get the path to the specific image.
            filename = os.path.join(self._path, fname)
            self.log.debug('Loading ' + fname)
            if not os.path.isfile(filename):
                # If the path is not good, raise an error.
                raise IOError('Not such file: ' + filename)

            # If path is good, load the image using the fload function
            img = self._fload(filename,
                              *self._fload_args,
                              **self._fload_kwargs)
            # Append the loaded image to the list
            imgs.append(img)

            # Add the image shape to the set
            self._shape.add(img.shape)
            if len(self._shape) > 1:  # pragma: no cover
                # If the shape set contains more than 1 element, this means
                # not all images have the same shape.
                raise ValueError('All images must have the same shape')

        return imgs
