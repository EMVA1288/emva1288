# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""Transform image data into data
This class takes a dictionary with image data
(product of :class:`emva1288.process.loader.LoadImageData`)
and transforms it into data that can be used for the EMVA1288 computations.
It is important to note, that this is separate from LoadImageData because this
step, produces float values that are not easily transportable (db, json,
etc...) without loosing accuracy.

"""

import numpy as np
import logging


class Data1288(object):
    """Take data from parsed images (descriptor file)
    and expose it as temporal and spatial dicts
    This dicts have the appropriate form for the processing
    """

    def __init__(self,
                 data,
                 loglevel=logging.INFO):
        """Data processing object init method.

        It sets the logging configuration and creates a
        :class:`python:logging.Logger` object using
        :func:`python:logging.getLogger` and sets the loglevel.
        Creates the data dictionaries with the call of
        :meth:`_get_temporal` for temporal data and :meth:`_get_spatial`
        for spatial data.

        Parameters
        ----------
        data : The data descriptor file.
        loglevel : int, optional
                   The loglevel for the Logger object.
        """

        logging.basicConfig()
        self.log = logging.getLogger('Data')
        self.log.setLevel(loglevel)

        self.pixels = data['width'] * data['height']

        self.data = {}
        self.data['temporal'] = self._get_temporal(data['temporal'])
        self.data['spatial'] = self._get_spatial(data['spatial'])

    def _get_float_keys(self, d, depth=2):
        """Method to get the exposure times from the data sets and
        the corresponding data.

        Parameters
        ----------
        d : dict
            The data dictionary
        depth : int, optional
                The number of layers of data before reaching the exposure time.
                This is different for bright data and dark data: bright data
                has an extra layer between the exposure time and the data that
                corresponds to the photon count.

        Returns
        -------
        dict : Dictionary of the data with an additional layer.
               The keys of this dict is the exposure time (float)
               and they map to the corresponding data dict.
        """
        r = {}
        for k in d.keys():
            if depth > 1:
                r[float(k)] = self._get_float_keys(d[k], depth - 1)
            else:
                r[float(k)] = d[k]
        return r

    def _get_temporal(self, data):
        """Fill the temporal dict, with the stuff that we need.
        Compute the averages and variances from the sums (sum and pvar)

        Uses the :meth:`_get_float_keys` method to work out the
        mapping of exposure times and :meth:`_get_spatial_data`
        to process data.
        If there is only one exposure time,
        the arrays in the returned dict will all have the same length as the
        photon count array. For this case, the exposure times and the dark
        value data array elements will all be the same.

        Parameters
        ----------
        data : The data dictionary containing the temporal data sets.

        Returns
        -------
        dict : A dict containing all temporal test data.
               The keys are the following:
               *'texp'*: the array of the
               exposure times used for the test,
               *'u_p'*: the array of
               photon count in a pixel for each exposure time and photon count,
               *'u_y'*: the
               array of the mean digital value for each exposure
               time and photon count,
               *'s2_y'*:
               the array of the digital value variance for each exposure
               time and photon count,
               *'u_ydark'*: the array of the mean digital dark value
               for each exposure time
               and *'s2_ydark'*: the
               array of the digital dark value variance for each exposure time.

        Raises
        ------
        AssertionError
            If bright and dark data don't have the same exposures.
        """
        temporal = {}

        # Get dict with exposure time as keys
        bright = self._get_float_keys(data['bright'])
        dark = self._get_float_keys(data['dark'], depth=1)

        assert not set(bright.keys()) - set(dark.keys()), \
            'Dark and bright must have same exposures'

        # texp is now an array of exposure times
        texp = np.asarray(sorted(bright.keys()))
        temporal['texp'] = texp

        u_p = []
        u_y = []
        s2_y = []
        u_ydark = []
        s2_ydark = []

        for t in texp:
            # photons is a list of photon counts for bright
            # images for each exposure time
            photons = sorted(bright[t].keys())
            for p in photons:
                # For each photon count, get the data
                u_p.append(p)
                d = self._get_temporal_data(bright[t][p])
                u_y.append(d['mean'])
                s2_y.append(d['var'])

            # get data for dark image
            d = self._get_temporal_data(dark[t])
            u_ydark.append(d['mean'])
            s2_ydark.append(d['var'])

        # Append all data to temporal dict
        temporal['u_p'] = np.asarray(u_p)
        temporal['u_y'] = np.asarray(u_y)
        temporal['s2_y'] = np.asarray(s2_y)
        temporal['u_ydark'] = np.asarray(u_ydark)
        temporal['s2_ydark'] = np.asarray(s2_ydark)

        # In case we have only one exposure, we need arrays with the
        # same length as the up
        # we just repeat the same value over and over
        if len(temporal['texp']) == 1:
            l = len(temporal['u_p'])

            v = temporal['texp'][0]
            temporal['texp'] = np.asarray([v for _i in range(l)])

            v = temporal['u_ydark'][0]
            temporal['u_ydark'] = np.asarray([v for _i in range(l)])

            v = temporal['s2_ydark'][0]
            temporal['s2_ydark'] = np.asarray([v for _i in range(l)])

        return temporal

    def _get_temporal_data(self, d):
        """Convert temporal image data to mean and variance

        The mean is the sum of the pixels of the two images divided by
        (2 * self.pixels)

        The variance is the pseudo variance(integer), divided by
        (4 * self.pixels)

        Parameters
        ----------
        d : dict
            The data dictionary that contains the sum and pvar of the
            pixels of a sum of two consecutive images with
            the same photon count
            and exposure time.

        Returns
        -------
        dict : A data dictionary with the following keys:
               *'mean'*: the mean as described above and
               *'var'*: the variance as described above.
        """
        mean_ = d['sum'] / (2.0 * self.pixels)
        var_ = d['pvar'] / (4.0 * self.pixels)
        return {'mean': mean_, 'var': var_}

    def _get_spatial(self, data):
        """Fill the spatial dictionary.

        The images (sum and pvar) are preserved,
        they are needed for processing.
        Uses :meth:`_get_float_keys` to work out the exposure times and
        :meth:`_get_spatial_data` to process data.

        Parameters
        ----------
        data : The data dictionary to take data from.

        Returns
        -------
        dict : A dict containing all spatial test data.
               The keys are the following:
               - *'texp'*: the array of exposure times for spatial tests,
               - *'u_p'*: the array of photon count average for
                 each exposure times.
               - *'sum'*: the array of the image sum for each
                 photon count and exposure time,
               - *'pvar'*: the array of the pvar image for each
                 photon count and exposure time,
               - *'L'*: the number of image taken to make the sum and
                 pvar images for each photon count and each exposure time,
               - *'avg'*: the average computed from the sum image for
                 each photon count and each exposure time,
               - *'var'*: the varianve computed from the pvar image for each
                 photon count and each exposure time,
               - *'sum_dark'*: the sum image in the dark for
                 each exposure time,
               - *'pvar_dark'*: the pvar image in the dark
                 for each exposure time,
               - *'L_dark'*: the number of image taken in the dark to
                 compose the sum and pvar images for each exposure time,
               - *'avg_dark'*: the average computed from the dark
                 sum image for each exposure time,
               - *'var_dark'*: the variance computed from the dark
                 pvar image for each exposure time.

        Raises
        ------
        AssertionError
            If the exposure times of dark images and bright
            images are different.
        """
        spatial = {}

        # Get dicts with exposure times as keys
        bright = self._get_float_keys(data['bright'])
        dark = self._get_float_keys(data['dark'], depth=1)

        assert not set(bright.keys()) - set(dark.keys()), \
            'Dark and bright must have same exposures'

        # texp contains the exposure times
        texp = np.asarray(sorted(bright.keys()))
        spatial['texp'] = texp

        u_p = []
        _sum = []
        _pvar = []
        _L = []
        _avg = []
        _var = []

        _sum_dark = []
        _pvar_dark = []
        _L_dark = []
        _avg_dark = []
        _var_dark = []

        for t in texp:
            # photons is the list of photon counts for one exposure time
            photons = sorted(bright[t].keys())

            for p in photons:
                d = self._get_spatial_data(bright[t][p])

                u_p.append(p)
                _sum.append(d['sum'])
                _pvar.append(d['pvar'])
                _L.append(d['L'])
                _avg.append(d['avg'])
                _var.append(d['var'])

            d = self._get_spatial_data(dark[t])

            _sum_dark.append(d['sum'])
            _pvar_dark.append(d['pvar'])
            _L_dark.append(d['L'])
            _avg_dark.append(d['avg'])
            _var_dark.append(d['var'])

        spatial['u_p'] = np.asarray(u_p)
        spatial['sum'] = np.asarray(_sum)
        spatial['pvar'] = np.asarray(_pvar)
        spatial['L'] = np.asarray(_L)
        spatial['avg'] = np.asarray(_avg)
        spatial['var'] = np.asarray(_var)

        spatial['sum_dark'] = np.asarray(_sum_dark)
        spatial['pvar_dark'] = np.asarray(_pvar_dark)
        spatial['L_dark'] = np.asarray(_L_dark)
        spatial['avg_dark'] = np.asarray(_avg_dark)
        spatial['var_dark'] = np.asarray(_var_dark)

        return spatial

    def _get_spatial_data(self, d):
        """Add the mean and variance to the spatial image data.

        The mean is the sum of the images divided by L,
        which is the number of images for the spatial test.

        The variance is the pseudovariance divided by
        (L^2 * (L-1)).

        Parameters
        ----------
        d : dict
            The data dictionary containing the sum and pvar of the images.

        Returns
        -------
        dict : A data dictionary processed from the input. The keys are:
               - *'sum'*: the sum image preserved from input,
               - *'pvar'*: the pvar image preserved from input,
               - *'L'*: the number of image summed,
               - *'avg'*: the average of the sum image as described above and
               - *'var'*: the variance as described above computed
                 from the pvar image.
        """
        sum_ = np.asarray(d['sum'], dtype=np.int64)
        pvar_ = np.asarray(d['pvar'], dtype=np.int64)
        L = int(d['L'])
        avg_ = sum_ / (1.0 * L)
        var_ = pvar_ / (1.0 * np.square(L) * (L - 1))

        return {'sum': sum_,
                'pvar': pvar_,
                'L': L,
                'avg': avg_,
                'var': var_}
