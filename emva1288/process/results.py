# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""Compute EMVA1288 values from data
This class takes the data from data.Data1288 and compute the actual EMVA1288
values.
"""

from __future__ import print_function
import logging
import numpy as np
from emva1288.process import routines
from scipy.ndimage import convolve


class Results1288(object):
    """Class used to process data and to generate pdf report using LaTeX.

    When properties from this class are computed, their docstring are also
    parsed using the :func:`~emva1288.process.routines.cls_1288_info`
    function to retrieve more data informations (like units or full name).
    """
    #########################################################################
    # Docstrings with .. emva1288:: flags are used when computing the       #
    # properties to add relevant information to data for further processing #
    # like plotting or creating a report                                    #
    #########################################################################
    def __init__(self,
                 data,
                 pixel_area=None,
                 index_u_ysat=None,
                 loglevel=logging.DEBUG):
        """Results computation init method.

        This class uses a :class:`python:logging.Logger` object to display
        informations for users.

        Parameters
        ----------
        data : dict
               The data dictionary to compute the results from.
        pixel_area : float, optional
                     The area of one pixel in um^2.
        index_u_ysat : int, optional
                       The index of the u_y array at which we consider
                       that the camera saturates. This is used if forcing
                       the saturation point is necessary.
        loglevel : int, optional
                   The level for the :class:`python:logging.Logger` object.
        """
        logging.basicConfig()
        self.log = logging.getLogger('Results')
        self.log.setLevel(loglevel)

        self.temporal = data['temporal']
        self.spatial = data['spatial']

        self.pixel_area = pixel_area
        self._s2q = 1.0 / 12.0
        self._index_start = 0
        self._index_sensitivity_min = 0

        self._histogram_Qmax = 256  # Maximum number of bins in histograms
        # Convolution kernel for high pass filter in defect pixel
        self._histogram_high_pass_box = (-1) * np.ones((5, 5), dtype=np.int64)
        self._histogram_high_pass_box[2, 2] = 24

        # Sometimes we need to force the saturation point
        # in those cases pass the index in the initialization of Results1288
        self._index_u_ysat = index_u_ysat

    @property
    def s2q(self):
        """Quantification noise."""
        return self._s2q

    @property
    def index_start(self):
        """The array's starting index.

        .. emva1288::
            :Section: info
            :Short: Start array index
        """
        return self._index_start

    @property
    def index_u_ysat(self):
        """Index of saturation.

        .. emva1288::
            :Section: sensitivity
            :Short: Saturation index
        """

        if self._index_u_ysat:
            return self._index_u_ysat

        max_ = 0
        max_i = 0

        # we have to loop backwards because sometimes we have some
        # noise pics that really bother the computation
        s2_y = self.temporal['s2_y']
        for i in range(len(s2_y) - 1, -1, -1):
            # Check that is not a local max
            if (s2_y[i] >= max_) or \
               (s2_y[abs(i - 1)] >= max_):
                max_ = s2_y[i]
                max_i = i
            elif (s2_y[i] < max_) and \
                 (s2_y[abs(i - 1)] < max_):
                break

        self._index_u_ysat = max_i
        return self._index_u_ysat

    @property
    def index_sensitivity_max(self):
        """Index for linear fits in sensitivity part of the standard
        (70% of saturation).

        .. emva1288::
            :Section: sensitivity
            :Short: Sensitivity fit maximum index
        """

        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        m = 0.7 * (Y[self.index_u_ysat])
        return max(np.argwhere(Y <= m))[0]

    @property
    def index_sensitivity_min(self):
        """Sensitivity minimum index.

        Index for linear fits in sensitivity part of the standard
        (70% of saturation)

        .. emva1288::
            :Section: sensitivity
            :Short: Sensitivity fit minimum index
        """
        return self._index_sensitivity_min

    @property
    def R(self):
        """Responsivity.

        Slope of the (u_y - u_y_dark) Vs u_p. Fit with offset = 0
        Uses the :func:`~emva1288.process.routines.LinearB0` function
        to make the fit.

        .. emva1288::
            :Section: sensitivity
            :Short: Responsivity
            :Symbol: R
            :Unit: DN/p
        """

        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        X = self.temporal['u_p']

        val, _error = routines.LinearB0(X[self.index_sensitivity_min:
                                          self.index_sensitivity_max + 1],
                                        Y[self.index_sensitivity_min:
                                          self.index_sensitivity_max + 1])

        return val[0]

    @property
    def K(self):
        """Overall system gain.

        Slope of (s2_y - s2_y_dark) Vs (u_y - u_y_dark). Fit with
        offset = 0. Uses the :func:`~emva1288.process.routines.LinearB0`
        to make the fit.

        .. emva1288::
            :Section: sensitivity
            :Short: System gain
            :Symbol: K
            :Unit: $DN/e^-$
            :LatexName: K
        """

        X = self.temporal['u_y'] - self.temporal['u_ydark']
        Y = self.temporal['s2_y'] - self.temporal['s2_ydark']

        val, _error = routines.LinearB0(X[self.index_sensitivity_min:
                                          self.index_sensitivity_max + 1],
                                        Y[self.index_sensitivity_min:
                                          self.index_sensitivity_max + 1])

        return val[0]

    def inverse_K(self):
        """Inverse of overall system gain.

        .. emva1288::
            :Section: sensitivity
            :Short: Inverse of overall system gain
            :Symbol: 1/K
            :Unit: $e^-/DN$
            :LatexName: InvK
        """

        return 1. / self.K

    @property
    def QE(self):
        """Quantum efficiency.

        It is retrieved as the ratio of the responsivity to the overall
        system gain.

        .. emva1288::
            :Section: sensitivity
            :Short: Quantum efficiency
            :Symbol: $\eta$
            :Unit: \%
            :LatexName: QE
        """

        return 100.0 * self.R / self.K

    @property
    def sigma_y_dark(self):
        """Temporal Dark Noise.

        Uses :func:`~emva1288.process.routines.LinearB` to make the fit.

        .. emva1288::
            :Section: sensitivity
            :Short: Temporal Dark Noise
            :Symbol: $\sigma_{y.dark}$
            :Unit: DN
            :LatexName: SigmaYDark
        """

        if len(np.unique(self.temporal['texp'])) <= 2:
            s2_ydark = self.temporal['s2_ydark'][0]
        else:
            fit, _error = routines.LinearB(self.temporal['texp'],
                                           self.temporal['s2_ydark'])
            s2_ydark = fit[1]

        # Lower limit for the temporal dark noise
        # The temporal dark noise in this range is dominated by the
        # quantization noise
        if s2_ydark < 0.24:
            s2_ydark = 0.24

        return np.sqrt(s2_ydark)

    @property
    def sigma_d(self):
        """Temporal Dark Noise.

        .. emva1288::
            :Section: sensitivity
            :Short: Temporal Dark Noise
            :Symbol: $\sigma_d$
            :Unit: $e^-$
            :LatexName: SigmaDark
        """

        return np.sqrt((self.sigma_y_dark ** 2) - self._s2q) / self.K

    @property
    def u_p_min(self):
        """Absolute sensitivity threshold.

        .. emva1288::
            :Section: sensitivity
            :Short: Absolute sensitivity threshold
            :Symbol: $\mu_{p.min}$
            :Unit: $p$
            :LatexName: UPMin
        """

        return (100.0 / self.QE) * ((self.sigma_y_dark / self.K) + 0.5)

    @property
    def u_p_min_area(self):
        """Sensitivity threshold per pixel area.

        Returns None if pixel area is not defined or 0.

        .. emva1288::
            :Section: sensitivity
            :Short: Sensitivity threshold
            :Symbol: $\mu_{p.min.area}$
            :Unit: $p/\mu m^2$
            :LatexName: UPMin
        """

        if not self.pixel_area:
            return None

        return self.u_p_min / self.pixel_area

    @property
    def u_e_min(self):
        """Sensitivity threshold.

        .. emva1288::
            :Section: sensitivity
            :Short: Sensitivity threshold
            :Symbol: $\mu_{e.min}$
            :Unit: $e^-$
            :LatexName: UEMin
        """
        return self.QE * self.u_p_min / 100.0

    @property
    def u_e_min_area(self):
        """Sensitivity threshold per pixel area.

        Returns None if the pixel area is not defined or 0.

        .. emva1288::
            :Section: sensitivity
            :Short: Sensitivity threshold
            :Symbol: $\mu_{e.min.area}$
            :Unit: $e^-/\mu m^2$
            :LatexName: UEMin
        """
        if not self.pixel_area:
            return None

        return self.u_e_min / self.pixel_area

    @property
    def u_p_sat(self):
        """Saturation Capacity.

        .. emva1288::
            :Section: sensitivity
            :Short: Saturation Capacity
            :Symbol: $\mu_{p.sat}$
            :Unit: $p$
            :LatexName: UPSat
        """

        return self.temporal['u_p'][self.index_u_ysat]

    @property
    def u_p_sat_area(self):
        """Saturation Capacity per pixel area.

        Returns None if pixel area is not defined or 0.

        .. emva1288::
            :Section: sensitivity
            :Short: Saturation Capacity
            :Symbol: $\mu_{p.sat.area}$
            :Unit: $p/\mu m^2$
            :LatexName: UPSat
        """

        if not self.pixel_area:
            return None

        return self.u_p_sat / self.pixel_area

    @property
    def u_e_sat(self):
        """Saturation Capacity.

        Number of electrons at saturation.

        .. emva1288::
            :Section: sensitivity
            :Short: Saturation Capacity
            :Symbol: $\mu_{e.sat}$
            :Unit: $e^-$
            :LatexName: UESat
        """

        return self.QE * self.u_p_sat / 100.0

    @property
    def u_e_sat_area(self):
        """Saturation Capacity per pixel area.

        Returns None if pixel area is not defined or 0.

        .. emva1288::
            :Section: sensitivity
            :Short: Saturation Capacity
            :Symbol: $\mu_{e.sat.area}$
            :Unit: $e^-/\mu m^2$
            :LatexName: UESat
        """

        if not self.pixel_area:
            return None

        return self.u_e_sat / self.pixel_area

    @property
    def SNR_max(self):
        """Maximum Signal-to-Noise Ratio.

        .. emva1288::
            :Section: sensitivity
            :Short: Signal-to-Noise Ratio
            :Symbol: $SNR_{max}$
            :LatexName: SNRMax
        """

        return np.sqrt(self.u_e_sat)

    def SNR_max_dB(self):
        """Maximum Signal to Noise Ratio in Db.

        .. emva1288::
            :Section: sensitivity
            :Short: Maximum Signal to Noise Ratio in Db
            :Symbol: $SNR_{max.dB}$
            :Unit: dB
            :LatexName: SNRMaxDB
        """

        return 20. * np.log10(self.SNR_max)

    def SNR_max_bit(self):
        """Maximum Signal to Noise Ratio in Bits.

        .. emva1288::
            :Section: sensitivity
            :Short: Maximum Signal to Noise Ratio in Bits
            :Symbol: $SNR_{max.bit}$
            :Unit: bit
            :LatexName: SNRMaxBit
        """

        return np.log2(self.SNR_max)

    def inverse_SNR_max(self):
        """Inverse Maximum Signal to Noise Ratio.

        .. emva1288::
            :Section: sensitivity
            :Short: Inverse Maximum Signal to Noise Ratio
            :Symbol: $SNR_{max}^{-1}$
            :Unit: \%
            :LatexName: InvSNRMax
        """

        return 100. / self.SNR_max

    @property
    def DR(self):
        """Dynamic Range.

        Defined as the saturation capacity devided by the absolute sensitivity
        threshold. The greater this number is, the greater the operational
        range of a camera (between the dark noise level and the saturation
        level).

        .. emva1288::
            :Section: sensitivity
            :Short: Dynamic Range
            :Symbol: DR
            :LatexName: DR
        """

        return self.u_p_sat / self.u_p_min

    def DR_dB(self):
        """Dynamic Range in deciBels.

        It is defined as 20 * log_10 ( Dynamic Range ).

        .. emva1288::
            :Section: sensitivity
            :Short: Dynamic Range
            :Symbol: $DR_{dB}$
            :Unit: dB
            :LatexName: DRDB
        """

        return 20. * np.log10(self.DR)

    def DR_bit(self):
        """Dynamic Range in bits.

        It is defined as log_2 ( Dynamic Range ).

        .. emva1288::
            :Section: sensitivity
            :Short: Dynamic Range
            :Symbol: $DR_{bit}$
            :Unit: bit
            :LatexName: DRBit
        """

        return np.log2(self.DR)

    @property
    def index_linearity_min(self):
        """Linearity fit minimun index.

        Minimum index for linear fit (5% of saturation).

        .. emva1288::
            :Section: linearity
            :Short: Linearity fit minimun index
        """
        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        vmin = 0.05 * (Y[self.index_u_ysat])

        return min(np.argwhere(Y >= vmin))[0]

    @property
    def index_linearity_max(self):
        """Linearity fit maximum index.

        Maximum index for linear fit (95% of saturation).

        .. emva1288::
            :Section: linearity
            :Short: Linearity fit maximum index
        """

        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        vmax = 0.95 * (Y[self.index_u_ysat])
        return max(np.argwhere(Y <= vmax))[0]

    def linearity(self):
        """Returns a dictionary containing linearity information.

        It fits the mean digital signal in function of the mean photon count
        (Linear fit) using the EMVA1288 standard for linear fit.

        Returns
        -------
        dict : Linearity dictionary.
               The keys are:

               - *'fit_slope'* : The slope of the linear fit.
               - *'fit_offset'* : The offset of the fit.
               - *'relative_deviation'* : The relative deviation of the real
                 data from the fit (in %) for the whole array.
               - *'linearity_error_min'* : The minimal value of the
                 relative deviation.
               - *'linearity_error_max'* : The maximal value of the
                 relative deviation.
        """
        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        X = self.temporal['u_p']

        # The maximum index has to be included, this is the reason for the +1
        imax = self.index_linearity_max + 1
        imin = self.index_linearity_min

        ##################################################################
        # Following the emva1288 standart for the computation of the fit #
        ##################################################################

        X_ = X[imin: imax]
        Y_ = Y[imin: imax]

        xy = np.sum(X_ / Y_)
        xy2 = np.sum(X_ / (Y_ ** 2))
        x2y2 = np.sum((X_ / Y_) ** 2)
        _y = np.sum(1. / Y_)
        _y2 = np.sum(1. / (Y_ ** 2))

        b = ((xy * xy2) - (x2y2 * _y)) / ((xy2 ** 2) - (x2y2 * _y2))
        a = (xy - (b * xy2)) / x2y2

        # The equivalent numpy polyfit is
        # a, b = np.polynomial.polynomial.polyfit(X_, Y_, 1, w=1 / Y_)

        dev = 100. * (Y - (a * X + b)) / (a * X + b)

        lin = {}
        lin['fit_slope'] = a
        lin['fit_offset'] = b
        lin['relative_deviation'] = dev
        lin['linearity_error_min'] = np.min(dev[imin: imax])
        lin['linearity_error_max'] = np.max(dev[imin: imax])

        return lin

    @property
    def LE_min(self):
        """ Min Linearity error.

        .. emva1288::
            :Section: linearity
            :Short: Min Linearity error
            :Symbol: $LE_{min}$
            :Unit: \%
            :LatexName: LEMin
        """

        return self.linearity()['linearity_error_min']

    @property
    def LE_max(self):
        """Max Linearity error.

        .. emva1288::
            :Section: linearity
            :Short: Max Linearity error
            :Symbol: $LE_{max}$
            :Unit: \%
            :LatexName: LEMax
        """

        return self.linearity()['linearity_error_max']

    @property
    def u_I_var_DN(self):
        """Dark Current from variance.

        The dark current from variance is the square root of the slope of the
        dark signal variance in function of the exposure time.
        Returns NaN if u_I_var is imaginary (if the fit slope is negative).

        .. emva1288::
            :Section: dark_current
            :Short: Dark Current from variance
            :Symbol: $\mu_{I.var.DN}$
            :Unit: $DN/s$
            :LatexName: UIVar
        """
        if len(np.unique(self.temporal['texp'])) <= 2:
            return np.nan

        fit, _error = routines.LinearB(self.temporal['texp'],
                                       self.temporal['s2_ydark'])

        if fit[0] < 0:
            return np.nan
        # Multiply by 10^9 because exposure times are in nanoseconds
        return fit[0] * (10 ** 9)

    @property
    def u_I_var(self):
        """Dark Current from variance.

        The dark current from variance is the square root of the slope of the
        dark signal variance in function of the exposure times divided
        by the overall system gain.
        Returns NaN if u_I_var is imaginary (if the fit slope is negative).

        .. emva1288::
            :Section: dark_current
            :Short: Dark Current from variance
            :Symbol: $\mu_{I.var}$
            :Unit: $e^-/s$
            :LatexName: UIVar
        """
        ui = self.u_I_var_DN
        if ui is np.nan:
            return np.nan

        return ui / (self.K ** 2)

    @property
    def u_I_mean_DN(self):
        """Dark Current from mean.

        The dark current from mean is the slope of the dark signal mean in
        function of the exposure time.
        Returns NaN if the number of different exposure times is less than 3.

        .. emva1288::
            :Section: dark_current
            :Short: Dark Current from mean
            :Symbol: $\mu_{I.mean.DN}$
            :Unit: $DN/s$
        """

        if len(np.unique(self.temporal['texp'])) <= 2:
            return np.nan

        fit, _error = routines.LinearB(self.temporal['texp'],
                                       self.temporal['u_ydark'])
        # Multiply by 10 ^ 9 because exposure time in nanoseconds
        return fit[0] * (10 ** 9)

    @property
    def u_I_mean(self):
        """Dark Current from mean.

        The dark current from mean is the slope of the dark signal mean in
        function of the exposure times divided by the overall system gain.
        Returns NaN if the number of different exposure times is less than 3.

        .. emva1288::
            :Section: dark_current
            :Short: Dark Current from mean
            :Symbol: $\mu_{I.mean}$
            :Unit: $e^-/s$
        """
        ui = self.u_I_mean_DN
        if ui is np.nan:
            return np.nan
        return ui / self.K

    @property
    def sigma_2_y_stack(self):
        """Temporal variance stack.

        Mean value of the bright variance image.

        .. emva1288::
            :Section: spatial
            :Short: Temporal variance stack
            :Symbol: $\sigma^2_{y.stack}$
            :Unit: DN2
        """

        return np.mean(self.spatial['var'])

    @property
    def sigma_2_y_stack_dark(self):
        """Temporal variance stack dark.

        Mean value of the dark variance image.

        .. emva1288::
            :Section: spatial
            :Short: Temporal variance stack dark
            :Symbol: $\sigma^2_{y.stack.dark}$
            :Unit: DN2
        """

        return np.mean(self.spatial['var_dark'])

    @property
    def s_2_y_measured(self):
        """Spatial variance measure.

        Variance value of the bright variance image.

        .. emva1288::
            :Section: spatial
            :Short: Spatial variance measure
            :Symbol: $s^2_{y.measured}$
            :Unit: DN2
        """
        # ddof = 1 (delta degrees of freedom) accounts for the minus 1 in the
        # divisor for the calculation of variance
        return np.var(self.spatial['avg'], ddof=1)

    @property
    def s_2_y_measured_dark(self):
        """Spatial variance measured dark.

        Variance value of the dark variance image.

        .. emva1288::
            :Section: spatial
            :Short: Spatial variance measured dark
            :Symbol: $s^2_{y.measured.dark}$
            :Unit: DN2
        """
        # ddof = 1 (delta degrees of freedom) accounts for the minus 1 in the
        # divisor for the calculation of variance
        return np.var(self.spatial['avg_dark'], ddof=1)

    @property
    def s_2_y(self):
        """Spatial variance from image.

        .. emva1288::
            :Section: spatial
            :Short: Spatial variance from image
            :Symbol: $s^2_{y}$
            :Unit: DN2
        """
        return self.s_2_y_measured - (self.sigma_2_y_stack /
                                      self.spatial['L'])

    @property
    def s_2_y_dark(self):
        """Spatial variance from image,

        .. emva1288::
            :Section: spatial
            :Short: Spatial variance from image
            :Symbol: $s^2_{y}$
            :Unit: DN2
        """
        return self.s_2_y_measured_dark - (self.sigma_2_y_stack_dark /
                                           self.spatial['L_dark'])

    @property
    def DSNU1288(self):
        """DSNU.

        Dark Signal NonUniformity (in e^-) is defined as the deviation
        standard of the dark signal devided by the overall system gain.
        If the variance is negative, it will return NaN instead of an
        imaginary number.

        .. emva1288::
            :Section: spatial
            :Short: DSNU
            :Symbol: $DSNU_{1288}$
            :Unit: $e^-$
            :LatexName: DSNU
        """

        if self.s_2_y_dark < 0.:
            return np.nan
        return np.sqrt(self.s_2_y_dark) / self.K

    def DSNU1288_DN(self):
        """DSNU in DN.

        Defined as the DSNU in e^- multiplied by the overall system gain.
        Returns NaN if the dark signal variance is negative.

        Returns
        -------
        float : The DSNU in DN.


        .. emva1288::
            :Section: spatial
            :Short: DSNU in DN
            :Symbol: $DSNU_{1288.DN}$
            :Unit: DN
            :LatexName: DSNUDN
        """

        if self.s_2_y_dark < 0:
            return np.nan
        return np.sqrt(self.s_2_y_dark)

    @property
    def PRNU1288(self):
        """PRNU.

        Photo Response NonUniformity (in %) is defined as the square root of
        the difference between the spatial variance of a bright image (or from
        an average of bright images to remove temporal difformities) and the
        spatial variance of dark signal, divided by the difference between the
        mean of a bright image and the mean of a dark image.

        .. emva1288::
            :Section: spatial
            :Short: PRNU
            :Symbol: $PRNU_{1288}$
            :Unit: \%
            :LatexName: PRNU
        """

        return (np.sqrt(self.s_2_y - self.s_2_y_dark) * 100 /
                (np.mean(self.spatial['avg']) -
                 np.mean(self.spatial['avg_dark'])))

    @property
    def histogram_PRNU(self):
        """PRNU histogram.

        Uses the :func:`~emva1288.process.routines.Histogram1288` function
        to make the histogram.

        .. emva1288::
            :Section: defect_pixel
            :Short: PRNU histogram
        """

        # For prnu, perform the convolution
        y = self.spatial['sum'] - self.spatial['sum_dark']
        # Slicing at the end is to remove boundary effects.
        y = convolve(y, self._histogram_high_pass_box)[2:-2, 2:-2]

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins
        h['bins'] /= (self.spatial['L'] * 25.)

        return h

    @property
    def histogram_PRNU_accumulated(self):
        """Accumulated PRNU histogram.

        Uses the :func:`~emva1288.process.routines.Histogram1288` function
        to make the histogram.

        .. emva1288::
            :Section: defect_pixel
            :Short: accumulated PRNU histogram
        """

        # For prnu, perform the convolution
        y = self.spatial['sum'] - self.spatial['sum_dark']
        y = convolve(y, self._histogram_high_pass_box)[2:-2, 2:-2]

        # For the accumulated histogram substract the mean
        y = np.abs(y - int(np.mean(y)))

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins
        h['bins'] /= (self.spatial['L'] * 25.)

        # Perform the cumulative summation
        h['values'] = np.cumsum(h['values'][::-1])[::-1]
        # we want it as percentage of pixels
        h['values'] = 100 * h['values'] / y.size

        return h

    @property
    def histogram_DSNU(self):
        """DSNU histogram.

        Uses the :func:`~emva1288.process.routines.Histogram1288` function
        to make the histogram.

        .. emva1288::
            :Section: defect_pixel
            :Short: DSNU histogram
        """

        # For dsnu, the image is just the dark image, upscaled to have
        # only integers
        y = self.spatial['sum_dark']

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins, this is due to upscaling the average image to have
        # only integers
        scale = self.spatial['L_dark'] * 25.
        h['bins'] /= scale
        # The image is not centered around zero, so we shift the bins
        h['bins'] -= (y.mean() / scale)
        return h

    @property
    def histogram_DSNU_accumulated(self):
        """Accumulated DSNU histogram.

        Uses the :func:`~emva1288.process.routines.Histogram1288` function
        to make the histogram.

        .. emva1288::
            :Section: defect_pixel
            :Short: accumulated DSNU histogram
        """

        # Dark image upscaled to have only integers
        y = self.spatial['sum_dark']
        # For the accumulated dsnu histogram, substract the mean from the image
        y = np.abs(y - int(np.mean(y)))

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins
        h['bins'] /= (self.spatial['L_dark'] * 25.)

        # Perform the cumulative summation (the ::-1 means backwards)
        # because the cumsum function is performed contrary to what we need
        h['values'] = np.cumsum(h['values'][::-1])[::-1]

        # we want it as percentage of pixels
        h['values'] = 100 * h['values'] / y.size

        return h

    def xml(self, filename=None):  # pragma: no cover
        """Method that writes the results in xml format to a file.

        Parameters
        ----------
        filename : str, optional
                   The file to write the results. If None, the xml string
                   won't be written but will be returned instead.

        Returns
        -------
        str : If the xml string is not written into a file, it is returned.
        """
        results = self.results_by_section
        if not filename:
            return routines.dict_to_xml(results)
        routines.dict_to_xml(results, filename=filename)

    @property
    def results(self):
        """Dictionnary with all the values and metadata for EMVA1288 values.

        It uses the :func:`~emva1288.process.routines.obj_to_dict` to compute
        all the results at once.
        """
        return routines.obj_to_dict(self)

    @property
    def results_by_section(self):  # pragma: no cover
        """Results ordered by section."""
        return routines._sections_first(self.results)

    def print_results(self):  # pragma: no cover
        """Print results to the screen."""
        results = self.results_by_section

        for section, attributes in results.items():
            print('*' * 50)
            print(section)
            print('-' * 50)

            for attribute, info in attributes.items():
                if 'value' not in info:
                    continue

                print('{:<50}{:<30}{:>10}'.format(info.get('short'),
                                                  str(info.get('symbol')),
                                                  str(info.get('value'))))
        print('*' * 50)
        print(' ')
