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
from emva1288 import routines
from scipy.ndimage import convolve


class Results1288(object):
    def __init__(self,
                 data,
                 index_u_ysat=None,
                 loglevel=logging.DEBUG):

        logging.basicConfig()
        self.log = logging.getLogger('Results')
        self.log.setLevel(loglevel)

        self.temporal = data['temporal']
        self.spatial = data['spatial']
        self.name = data['name']
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
        return self._s2q

    @property
    def index_start(self):
        """
        **Section: info
        **Short: The array indexes start at
        **Symbol:
        **Comment:
        **Unit:
        """
        return self._index_start

    @property
    def index_u_ysat(self):
        """
        **Section: sensitivity
        **Short: Saturation Point
        **Symbol:
        **Comment: Index of saturation
        **Unit:
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
        **Section: sensitivity
        **Symbol:
        **Unit:
        **Short:Sensitivity fit maximum index
        """

        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        m = 0.7 * (Y[self.index_u_ysat])
        return max(np.argwhere(Y <= m))[0]

    @property
    def index_sensitivity_min(self):
        """
        **Section: sensitivity
        **Short:Sensitivity fit minimum index
        **Symbol:
        **Comment:Index for linear fits in sensitivity part of the standard (70% of saturation)
        **Unit:
        """
        return self._index_sensitivity_min

    @property
    def R(self):
        """
        **Section: sensitivity
        **Short:Responsivity
        **Symbol:R
        **Comment:Slope of the (u_y - u_y_dark) Vs u_p. Fit with offset = 0
        **Unit:DN/p
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
        """
        **Section: sensitivity
        **Short:System gain
        **Symbol:K
        **Comment:Slope of (s2_y - s2_y_dark) Vs (u_y - u_y_dark). Fit with offset = 0
        **Unit: $DN/e^-$
        **LatexName:K
        """

        X = self.temporal['u_y'] - self.temporal['u_ydark']
        Y = self.temporal['s2_y'] - self.temporal['s2_ydark']

        val, _error = routines.LinearB0(X[self.index_sensitivity_min:
                                          self.index_sensitivity_max + 1],
                                        Y[self.index_sensitivity_min:
                                          self.index_sensitivity_max + 1])

        return val[0]

    def inverse_K(self):
        """
        **Section: sensitivity
        **Short:Inverse system gain
        **Symbol: 1/K
        **Comment:
        **Unit: $e^-/DN$
        **LatexName: InvK
        """

        return 1. / self.K

    @property
    def QE(self):
        """
        **Section: sensitivity
        **Short: Quantum efficiency
        **Symbol: $\eta$
        **Comment:
        **Unit: \%
        **LatexName: QE
        """

        return 100.0 * self.R / self.K

    @property
    def sigma_y_dark(self):
        """
        **Section: sensitivity
        **Short: Temporal Dark Noise
        **Symbol: $\sigma_{y.dark}$
        **Comment:
        **Unit: DN
        **LatexName: SigmaYDark
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
        """
        **Section: sensitivity
        **Short: Temporal Dark Noise
        **Symbol:$\sigma_d$
        **Comment:
        **Unit: $e^-$
        **LatexName: SigmaDark
        """

        return np.sqrt((self.sigma_y_dark ** 2) - self._s2q) / self.K

    @property
    def u_p_min(self):
        """
        **Section: sensitivity
        **Short: Sensitivity threshold
        **Symbol:$\mu_{p.min}$
        **Comment:
        **Unit: $p$
        **LatexName: UPMin
        """

        return (100.0 / self.QE) * ((self.sigma_y_dark / self.K) + 0.5)

    def u_e_min(self):
        """
        **Section: sensitivity
        **Short: Sensitivity threshold
        **Symbol:$\mu_{e.min}$
        **Comment:
        **Unit: $e^-$
        **LatexName: UEMin
        """
        return self.QE * self.u_p_min / 100.0

    @property
    def u_p_sat(self):
        """
        **Section: sensitivity
        **Short:Saturation Capacity
        **Symbol:$\mu_{p.sat}$
        **Comment:
        **Unit: $p$
        **LatexName: UPSat
        """

        return self.temporal['u_p'][self.index_u_ysat]

    @property
    def u_e_sat(self):
        """
        **Section: sensitivity
        **Short:Saturation Capacity
        **Symbol:$\mu_{e.sat}$
        **Comment:Number of electrons at saturation
        **Unit: $e^-$
        **LatexName: UESat
        """

        return self.QE * self.u_p_sat / 100.0

    @property
    def SNR_max(self):
        """
        **Section: sensitivity
        **Short: Maximum Signal to Noise Ratio
        **Symbol: $SNR_{max}$
        **Comment:
        **Unit:
        **LatexName: SNRMax
        """

        return np.sqrt(self.u_e_sat)

    def SNR_max_dB(self):
        """
        **Section: sensitivity
        **Short:Maximum Signal to Noise Ratio in Db
        **Symbol: $SNR_{max.dB}$
        **Comment:
        **Unit: dB
        **LatexName: SNRMaxDB
        """

        return 20. * np.log(self.SNR_max)

    def SNR_max_bit(self):
        """
        **Section: sensitivity
        **Short: Maximum Signal to Noise Ratio in Bits
        **Symbol: $SNR_{max.bit}$
        **Comment:
        **Unit: bit
        **LatexName: SNRMaxBit
        """

        return np.log2(self.SNR_max)

    def inverse_SNR_max(self):
        """
        **Section: sensitivity
        **Short: Inverse Maximum Signal to Noise Ratio
        **Symbol: $SNR_{max}^{-1}$
        **Comment:
        **Unit: \%
        **LatexName: InvSNRMax
        """

        return 100. / self.SNR_max

    @property
    def DR(self):
        """
        **Section: sensitivity
        **Short: Dynamic Range
        **Symbol: DR
        **Comment:
        **Unit:
        **LatexName: DR
        """

        return self.u_p_sat / self.u_p_min

    def DR_dB(self):
        """
        **Section: sensitivity
        **Short: Dynamic Range
        **Symbol: $DR_{dB}$
        **Comment:
        **Unit: dB
        **LatexName: DRDB
        """

        return 20. * np.log(self.DR)

    def DR_bit(self):
        """
        **Section: sensitivity
        **Short: Dynamic Range
        **Symbol: $DR_{bit}$
        **Comment:
        **Unit: bit
        **LatexName: DRBit
        """

        return np.log2(self.DR)

    @property
    def index_linearity_min(self):
        """
        **Section: linearity
        **Short:Linearity fit minimun index
        **Symbol:
        **Comment:Minimum index for linear fit in (5% of saturation)
        **Unit:
        """
        Y = self.temporal['u_y'] - self.temporal['u_ydark']
        vmin = 0.05 * (Y[self.index_u_ysat])

        return min(np.argwhere(Y >= vmin))[0]

    @property
    def index_linearity_max(self):
        """
        **Section: linearity
        **Short: Linearity fit maximum index
        **Symbol:
        **Comment:
        **Unit:
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
        """
        **Section: linearity
        **Short: Min Linearity error
        **Symbol: $LE_{min}$
        **Comment:
        **Unit: \%
        **LatexName: LEMin
        """

        return self.linearity()['linearity_error_min']

    @property
    def LE_max(self):
        """
        **Section: linearity
        **Short: Max Linearity error
        **Symbol: $LE_{max}$
        **Comment:
        **Unit: \%
        **LatexName: LEMax
        """

        return self.linearity()['linearity_error_max']

    @property
    def u_I_var(self):
        """
        **Section: dark_current
        **Short:Dark Current from variance
        **Symbol:$\mu_{I.var}$
        **Comment:
        **Unit: $e^-/s$
        **LatexName: UIVar
        """

        fit, _error = routines.LinearB(self.temporal['texp'],
                                       self.temporal['s2_ydark'])

        if fit[0] < 0:
            return np.nan
        return np.sqrt(fit[0] * (10 ** 9)) / self.K

    @property
    def u_I_mean(self):
        """
        **Section: dark_current
        **Short:Dark Current from mean
        **Symbol:$\mu_{I.mean}$
        **Comment:
        **Unit: e/s
        """

        if len(np.unique(self.temporal['texp'])) <= 2:
            return np.nan

        fit, _error = routines.LinearB(self.temporal['texp'],
                                       self.temporal['u_ydark'])
        return fit[0] * (10 ** 9) / self.K

    @property
    def sigma_2_y_stack(self):
        """
        **Section: spatial
        **Short: Temporal variance stack
        **Symbol: $\sigma^2_{y.stack}$
        **Comment: Mean value of the bright variance image
        **Unit: DN2
        """

        return np.mean(self.spatial['var'][0])

    @property
    def sigma_2_y_stack_dark(self):
        """
        **Section: spatial
        **Short: Temporal variance stack dark
        **Symbol:$\sigma^2_{y.stack.dark}$
        **Comment:Mean value of the dark variance image.
        **Unit:DN2
        """

        return np.mean(self.spatial['var_dark'][0])

    @property
    def s_2_y_measured(self):
        """
        **Section: spatial
        **Short: Spatial variance measure
        **Symbol:$s^2_{y.measured}$
        **Comment:Variance value of the bright variance image
        **Unit:DN2
        """

        return np.var(self.spatial['avg'][0], ddof=1)

    @property
    def s_2_y_measured_dark(self):
        """
        **Section: spatial
        **Short: Spatial variance measured dark
        **Symbol:$s^2_{y.measured.dark}$
        **Comment:Variance value of the dark variance image
        **Unit:DN2
        """

        return np.var(self.spatial['avg_dark'][0], ddof=1)

    @property
    def s_2_y(self):
        """
        **Section: spatial
        **Short: Spatial variance from image
        **Symbol:$s^2_{y}$
        **Comment:s_2_y_measured - sigma_2_y_stack / number_images
        **Unit:DN2
        """

        return self.s_2_y_measured - (self.sigma_2_y_stack /
                                      self.spatial['L'][0])

    @property
    def s_2_y_dark(self):
        """
        **Section: spatial
        **Short: Spatial variance from image
        **Symbol:$s^2_{y}$
        **Comment:s_2_y_measured - sigma_2_y_stack / number_images
        **Unit:DN2
        """

        return self.s_2_y_measured_dark - (self.sigma_2_y_stack_dark /
                                           self.spatial['L_dark'][0])

    @property
    def s_2_y_measured_spectrogram(self):
        """
        **Section: spatial
        **Short: Spatial variance measured from spectrogram
        **Symbol:$s^2_{y.measured.spectrogram}$
        **Comment:
        **Unit:DN2
        """

        return np.mean(routines.FFT1288(self.spatial['avg'][0])[8: -7])

    @property
    def s_2_y_measured_spectrogram_dark(self):
        """
        **Section: spatial
        **Short: Spatial variance measured dark from spectrogram
        **Symbol:$s^2_{y.measured.spectrogram.dark}$
        **Comment:
        **Unit:DN2
        """

        return np.mean(routines.FFT1288(self.spatial['avg_dark'][0]))

    @property
    def s_2_y_spectrogram(self):
        """
        **Section: spatial
        **Short: Spatial variance from spectrogram
        **Symbol:$s^2_{y.spectrogram}$
        **Comment:
        **Unit:DN2
        """
        return (self.s_2_y_measured_spectrogram -
                (self.sigma_2_y_stack / self.spatial['L'][0]))

    @property
    def s_2_y_spectrogram_dark(self):
        """
        **Section: spatial
        **Short: Spatial variance from spectrogram dark
        **Symbol:$s^2_{y.spectrogram.dark}$
        **Comment:
        **Unit:DN2
        """

        return (self.s_2_y_measured_spectrogram_dark -
                (self.sigma_2_y_stack_dark / self.spatial['L_dark'][0]))

    @property
    def DSNU1288(self):
        """
        **Section: spatial
        **Short: DSNU
        **Symbol: $DSNU_{1288}$
        **Comment:
        **Unit: $e^-$
        **LatexName: DSNU
        """

        if self.s_2_y_dark < 0.:
            return np.nan
        return np.sqrt(self.s_2_y_dark) / self.K

    def DSNU1288_DN(self):
        """
        **Section: spatial
        **Short: DSNU in DN
        **Symbol: $DSNU_{1288.DN}$
        **Comment:
        **Unit: DN
        **LatexName: DSNUDN
        """

        if self.s_2_y_dark < 0:
            return np.nan
        return np.sqrt(self.s_2_y_dark)

    @property
    def PRNU1288(self):
        """
        **Section: spatial
        **Short: PRNU
        **Symbol: $PRNU_{1288}$
        **Comment:
        **Unit: \%
        **LatexName: PRNU
        """

        return (np.sqrt(self.s_2_y - self.s_2_y_dark) * 100 /
                (np.mean(self.spatial['avg'][0]) -
                 np.mean(self.spatial['avg_dark'][0])))

    @property
    def F_50(self):
        """
        **Section: spatial
        **Short: Non Whiteness factor 50%
        **Symbol: $F_{50\%}$
        **Comment:
        **Unit:
        """

        return (self.s_2_y_measured_spectrogram /
                np.median(routines.FFT1288(self.spatial['avg'][0])[8: -7]))

    @property
    def F_dark(self):
        """
        **Section: spatial
        **Short: Non Whiteness factor Dark
        **Symbol: $F_{dark}$
        **Comment:
        **Unit:
        """

        return (self.s_2_y_measured_spectrogram_dark /
                np.median(routines.FFT1288(self.spatial['avg_dark'][0])))

    @property
    def histogram_PRNU(self):
        """
        **Section: defect_pixel
        **Short: PRNU histogram
        **Symbol:
        **Comment:
        **Unit:
        """

        # For prnu, perform the convolution
        y = self.spatial['sum'][0] - self.spatial['sum_dark'][0]
        y = convolve(y, self._histogram_high_pass_box)[2:-2, 2:-2]

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins
        h['bins'] /= (self.spatial['L'][0] * 25.)

        return h

    @property
    def histogram_PRNU_accumulated(self):
        """
        **Section: defect_pixel
        **Short: accumulated PRNU histogram
        **Symbol:
        **Comment:
        **Unit:
        """

        # For prnu, perform the convolution
        y = self.spatial['sum'][0] - self.spatial['sum_dark'][0]
        y = convolve(y, self._histogram_high_pass_box)[2:-2, 2:-2]

        # For the accumulated histogram substract the mean
        y = np.abs(y - int(np.mean(y)))

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins
        h['bins'] /= (self.spatial['L'][0] * 25.)

        # Perform the cumulative summation
        h['values'] = np.cumsum(h['values'][::-1])[::-1]

        return h

    @property
    def histogram_DSNU(self):
        """
        **Section: defect_pixel
        **Short: DSNU histogram
        **Symbol:
        **Comment:
        **Unit:
        """

        # For dsnu, the image is just the dark image, upscaled to have
        # only integers
        y = self.spatial['sum_dark'][0]

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins, this is due to upscaling the average image to have
        # only integers
        h['bins'] /= (self.spatial['L_dark'][0] * 25.)

        return h

    @property
    def histogram_DSNU_accumulated(self):
        """
        **Section: defect_pixel
        **Short: accumulated DSNU histogram
        **Symbol:
        **Comment:
        **Unit:
        """

        # Dark image upscaled to have only integers
        y = self.spatial['sum_dark'][0]
        # For the accumulated dsnu histogram, substract the mean from the image
        y = np.abs(y - int(np.mean(y)))

        h = routines.Histogram1288(y, self._histogram_Qmax)
        # Rescale the bins
        h['bins'] /= (self.spatial['L_dark'][0] * 25.)

        # Perform the cumulative summation (the ::-1 means backwars)
        # because the cumsum function is performed contrary to what we need
        h['values'] = np.cumsum(h['values'][::-1])[::-1]

        return h

    def xml(self, filename=None):
        d = routines.obj_to_dict(self)
        if not filename:
            return routines.dict_to_xml(d)
        routines.dict_to_xml(d, filename=filename)
#
#     def latex(self):
#         # TODO: Merge this with generate_latex
#
#         # The two template tex sections
#         r = {'operation_point': '', 'camera': ''}
#
#         # The operation point data for the tex files, comes from the
#         # Results1288 object
#         d = routines.obj_to_dict(self)
#         t = ''
#         for section, v in d.items():
#             t += '\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
#             t += '% Section: ' + section + '\n%\n'
#             for methodname, content in v.items():
#                 if 'LatexName' in content:
#                     t += '% Method: ' + methodname + '\n'
#                     t += '% Desc: ' + content['Short'] + '\n'
#                     t += '\\def\\%s{%s}\n' % (content['LatexName'],
#                                               str(content['Value']))
#                     t += '\\def\\U%s{%s}\n' % (content['LatexName'],
#                                                content['Unit'])
#                     t += '\\def\\S%s{%s}\n' % (content['LatexName'],
#                                                content['Symbol'])
#                     t += '\n'
#         r['operation_point'] += t
#
#         # Some data to be included in the tex content, comes from
#         # the descriptor file
#         # For camera data
#         #
#         # c varname varcontent
#         #
#         # For operation point data
#         # o varname varcontent
#         #
#         # This data was loaded by ProcessEmvaDescriptorFile
#         # and is already in self.data.data['datasheet']
#         for section in r.keys():
#             if self.data.data['datasheet'][section]:
#                 t = ''
#                 t += '\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
#                 t += '% Latex variables from input\n\n'
#                 for varname, value in self.data.data['datasheet'][section].items():
#                     t += '\\def\\' + str(varname) + '{' + str(value) + '}\n'
#                 r[section] += t
#
#         return r

    @property
    def results(self):
        d = routines.obj_to_dict(self)
        results = {}
        for section in d.keys():
            results[section] = {}
            for method in d[section].keys():
                s = d[section][method]
                if s.get('Value', False):
                    results[section][method] = {'short': s['Short'],
                                                'symbol': s['Symbol'],
                                                'value': s['Value'],
                                                'unit': s['Unit'],
                                                'latexname': s.get('LatexName')}
        return results

    def print_results(self):
        d = self.results
        for section in d.keys():
            print('*' * 50)
            print(section)
            print('*' * 50)

            for method in d[section].keys():
                s = d[section][method]
                print('{:<50}{:<30}{:>10}'.format(s['short'],
                                                  s['symbol'],
                                                  s['value']))
        print('*' * 50)
        print(' ')
