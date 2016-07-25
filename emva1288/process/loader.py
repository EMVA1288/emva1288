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
    '''Using a info dictionary, load the images and fill data with
    the relevant information

    Parameters
    ----------
    images: Dict with images to load, result of ParseEmvaDescriptorFile
    path: optional path to load the images
    loglevel: Logging level
    fload: function used to load the images
    fload_args: arguments passed to fload
    fload_kwargs: keyword arguments passed to fload
    '''

    def __init__(self,
                 images,
                 path='',
                 loglevel=logging.INFO,
                 fload=routines.load_image,
                 fload_args=[],
                 fload_kwargs={}):

        self.data = {'temporal': {}, 'spatial': {}}
        self._fload = fload
        self._fload_args = fload_args
        self._fload_kwargs = fload_kwargs
        self._shape = set()

        logging.basicConfig()
        self.log = logging.getLogger('Loader')
        self.log.setLevel(loglevel)

        self._path = path
        self._load_data(images)

    def _load_data(self, images):
        '''Using the information in images dict fill self.data loading the
        images and filling the temporal and spatial dicts
        '''

        for kind in ('temporal', 'spatial'):
            exposures = set(images[kind].keys())

            for exposure in exposures:
                photon_counts = list(images[kind][exposure].keys())
                if 0.0 not in photon_counts:
                    raise ValueError('Every exposure must have a 0.0 photons'
                                     ' for dark information')
                if len(photon_counts) < 2:
                    raise ValueError('All exposure must have at least 2 points'
                                     ' one dark and one bright')

                for photons, fnames in images[kind][exposure].items():
                    data_imgs = self._get_imgs_data(fnames, kind)
                    self.data[kind].setdefault(exposure, {})
                    self.data[kind][exposure][photons] = data_imgs

        shape = self._shape.pop()
        self.data['height'] = shape[0]
        self.data['width'] = shape[1]

    def _get_imgs_data(self, fnames, kind):
        '''Return the desired image data
        This depends on the kind of data
        '''
        arr_imgs = self._load_imgs(fnames)
        imgs = routines.get_int_imgs(arr_imgs)
        # For spatial we want the images
        if kind != 'temporal':
            return imgs
        # For temporal, we want numbers
        d = {}
        d['sum'] = np.sum(imgs['sum'])
        d['pvar'] = np.sum(imgs['pvar'])
        return d

    def _load_imgs(self, fnames):
        '''For a list of images, load and append them to the return array
        '''
        imgs = []
        for fname in fnames:
            filename = os.path.join(self._path, fname)
            self.log.debug('Loading ' + fname)
            if not os.path.isfile(filename):
                raise IOError('Not such file: ' + filename)
            img = self._fload(filename,
                              *self._fload_args,
                              **self._fload_kwargs)
            imgs.append(img)

            self._shape.add(img.shape)
            if len(self._shape) > 1:
                raise ValueError('All images must have the same shape')

        return imgs
