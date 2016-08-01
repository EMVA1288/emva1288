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
    """Class used to process data and to generate pdf report using LaTeX."""
    def __init__(self,
                 data,
                 pixel_area=None,
                 index_u_ysat=None,
                 loglevel=logging.DEBUG):

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
        """Quantification noise"""
        return self._s2q

    @property
    def index_start(self):
        """The array indexes start at

        .. emva1288::
            :Section: info
            :Short: Start array index
        """
        return self._index_start

    @property
    def index_u_ysat(self):
        """Index of saturation

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
        (70% of saturation)

        .. emva1288::
            :Section: sensitivity
            :Short: Sensitivity fit maximum index
        """

        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        m = 0.7 * (Y[self.index_u_ysat])
        return max(np.argwhere(Y <= m))[0]

    @property
    def index_sensitivity_min(self):
        """Sensitivity minimum index

        Index for linear fits in sensitivity part of the standard
        (70% of saturation)

        .. emva1288::
            :Section: sensitivity
            :Short: Sensitivity fit minimum index
        """
        return self._index_sensitivity_min

    @property
    def R(self):
        """Responsivity

        Slope of the (u_y - u_y_dark) Vs u_p. Fit with offset = 0

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
        """Overall system gain

        Slope of (s2_y - s2_y_dark) Vs (u_y - u_y_dark). Fit with
        offset = 0

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
        """Inverse of overall system gain

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
        """Quantum efficiency

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
        """Temporal Dark Noise

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
        """Temporal Dark Noise

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
        """Absolute sensitivity threshold

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
        """Sensitivity threshold per area

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
        """Sensitivity threshold

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
        """Sensitivity threshold

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
        """Saturation Capacity

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
        """Saturation Capacity per pixel area

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
        """Saturation Capacity

        Number of electrons at saturation

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
        """Saturation Capacity

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
        """Maximum Signal-to-Noise Ratio

        .. emva1288::
            :Section: sensitivity
            :Short: Signal-to-Noise Ratio
            :Symbol: $SNR_{max}$
            :LatexName: SNRMax
        """

        return np.sqrt(self.u_e_sat)

    def SNR_max_dB(self):
        """Maximum Signal to Noise Ratio in Db

        .. emva1288::
            :Section: sensitivity
            :Short: Maximum Signal to Noise Ratio in Db
            :Symbol: $SNR_{max.dB}$
            :Unit: dB
            :LatexName: SNRMaxDB
        """

        return 20. * np.log10(self.SNR_max)

    def SNR_max_bit(self):
        """Maximum Signal to Noise Ratio in Bits

        .. emva1288::
            :Section: sensitivity
            :Short: Maximum Signal to Noise Ratio in Bits
            :Symbol: $SNR_{max.bit}$
            :Unit: bit
            :LatexName: SNRMaxBit
        """

        return np.log2(self.SNR_max)

    def inverse_SNR_max(self):
        """Inverse Maximum Signal to Noise Ratio

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
        """Dynamic Range

        .. emva1288::
            :Section: sensitivity
            :Short: Dynamic Range
            :Symbol: DR
            :LatexName: DR
        """

        return self.u_p_sat / self.u_p_min

    def DR_dB(self):
        """Dynamic Range

        .. emva1288::
            :Section: sensitivity
            :Short: Dynamic Range
            :Symbol: $DR_{dB}$
            :Unit: dB
            :LatexName: DRDB
        """

        return 20. * np.log10(self.DR)

    def DR_bit(self):
        """Dynamic Range

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
        """Linearity fit minimun index

        Minimum index for linear fit in (5% of saturation)

        .. emva1288:
            :Section: linearity
            :Short: Linearity fit minimun index
        """
        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        vmin = 0.05 * (Y[self.index_u_ysat])

        return min(np.argwhere(Y >= vmin))[0]

    @property
    def index_linearity_max(self):
        """Linearity fit maximum index

        .. emva1288::
            :Section: linearity
            :Short: Linearity fit maximum index
        """

        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        vmax = 0.95 * (Y[self.index_u_ysat])
        return max(np.argwhere(Y <= vmax))[0]

    def linearity(self):
        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        X = self.temporal['u_p']

        # The maximum index has to be included, this is the reason for the +1
        imax = self.index_linearity_max + 1
        imin = self.index_linearity_min

        X_ = X[imin: imax]
        Y_ = Y[imin: imax]
        xy = np.sum(X_ / Y_)
        xy2 = np.sum(X_ / (Y_ ** 2))
        x2y2 = np.sum((X_ / Y_) ** 2)
        _y = np.sum(1. / Y_)
        _y2 = np.sum(1. / (Y_ ** 2))

        b = ((xy * xy2) - (x2y2 * _y)) / ((xy2 ** 2) - (x2y2 * _y2))
        a = (xy - (b * xy2)) / x2y2

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
        """ Min Linearity error

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
        """Max Linearity error

        .. emva1288::
            :Section: linearity
            :Short: Max Linearity error
            :Symbol: $LE_{max}$
            :Unit: \%
            :LatexName: LEMax
        """

        return self.linearity()['linearity_error_max']

    @property
    def u_I_var(self):
        """Dark Current from variance

        .. emva1288::
            :Section: dark_current
            :Short: Dark Current from variance
            :Symbol: $\mu_{I.var}$
            :Unit: $e^-/s$
            :LatexName: UIVar
        """

        fit, _error = routines.LinearB(self.temporal['texp'],
                                       self.temporal['s2_ydark'])

        if fit[0] < 0:
            return np.nan
        return np.sqrt(fit[0] * (10 ** 9)) / self.K

    @property
    def u_I_mean(self):
        """Dark Current from mean

        .. emva1288::
            :Section: dark_current
            :Short: Dark Current from mean
            :Symbol: $\mu_{I.mean}$
            :Unit: e/s
        """

        if len(np.unique(self.temporal['texp'])) <= 2:
            return np.nan

        fit, _error = routines.LinearB(self.temporal['texp'],
                                       self.temporal['u_ydark'])
        return fit[0] * (10 ** 9) / self.K

    @property
    def sigma_2_y_stack(self):
        """Temporal variance stack

        Mean value of the bright variance image

        .. emva1288::
            :Section: spatial
            :Short: Temporal variance stack
            :Symbol: $\sigma^2_{y.stack}$
            :Unit: DN2
        """

        return np.mean(self.spatial['var'])

    @property
    def sigma_2_y_stack_dark(self):
        """Temporal variance stack dark

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
        """Spatial variance measure

        Variance value of the bright variance image

        .. emva1288::
            :Section: spatial
            :Short: Spatial variance measure
            :Symbol: $s^2_{y.measured}$
            :Unit: DN2
        """

        return np.var(self.spatial['avg'], ddof=1)

    @property
    def s_2_y_measured_dark(self):
        """Spatial variance measured dark

        Variance value of the dark variance image

        .. emva1288::
            :Section: spatial
            :Short: Spatial variance measured dark
            :Symbol: $s^2_{y.measured.dark}$
            :Unit: DN2
        """

        return np.var(self.spatial['avg_dark'], ddof=1)

    @property
    def s_2_y(self):
        """Spatial variance from image


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
        """Spatial variance from image

        .. emva1288:
            :Section: spatial
            :Short: Spatial variance from image
            :Symbol: $s^2_{y}$
            :Unit: DN2
        """

        return self.s_2_y_measured_dark - (self.sigma_2_y_stack_dark /
                                           self.spatial['L_dark'])

    @property
    def DSNU1288(self):
        """DSNU

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
        """DSNU in DN

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
        """PRNU

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
        """PRNU histogram

        .. emva1288::
            :Section: defect_pixel
            :Short: PRNU histogram
        """

        # For prnu, perform the convolution
        y = self.spatial['sum'] - self.spatial['sum_dark']
        y = convolve(y, self._histogram_high_pass_box)[2:-2, 2:-2]

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins
        h['bins'] /= (self.spatial['L'] * 25.)

        return h

    @property
    def histogram_PRNU_accumulated(self):
        """Accumulated PRNU histogram

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

        return h

    @property
    def histogram_DSNU(self):
        """DSNU histogram

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
        h['bins'] /= (self.spatial['L_dark'] * 25.)

        return h

    @property
    def histogram_DSNU_accumulated(self):
        """Accumulated DSNU histogram

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

        # Perform the cumulative summation (the ::-1 means backwars)
        # because the cumsum function is performed contrary to what we need
        h['values'] = np.cumsum(h['values'][::-1])[::-1]

        return h

    def xml(self, filename=None):
        d = routines.obj_to_dict(self)
        if not filename:
            return routines.dict_to_xml(d)
        routines.dict_to_xml(d, filename=filename)

    @property
    def results(self):
        d = routines.obj_to_dict(self)
        results = {}

        for section in sorted(d.keys()):
            results[section] = {}
            for method in sorted(d[section].keys()):
                s = d[section][method]
                if s.get('value', False) is not False:
                    results[section][method] = {
                        'short': s.get('short'),
                        'symbol': s.get('symbol'),
                        'value': s['value'],
                        'unit': s.get('unit'),
                        'latexname': s.get('latexname')}

        return results

    def print_results(self):
        d = self.results

        for section in sorted(d.keys()):
            print('*' * 50)
            print(section)
            print('-' * 50)

            for method in sorted(d[section].keys()):
                s = d[section][method]
                print('{:<50}{:<30}{:>10}'.format(s.get('short'),
                                                  str(s.get('symbol')),
                                                  str(s.get('value'))))
        print('*' * 50)
        print(' ')
