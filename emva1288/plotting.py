# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""Plot the results
This class takes a results.Results1288 object and produces all the plots
needed to create a reference datasheet of the EMVA1288 test

"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from emva1288 import routines


class Plotting1288(object):
    plots = ['sensitivity',
             'u_ydark',
             'PTC',
             'SNR',
             'linearity',
             'deviation_linearity',
             'horizontal_spectrogram_PRNU',
             'horizontal_spectrogram_DSNU',
             'vertical_spectrogram_PRNU',
             'vertical_spectrogram_DSNU',
             'accumulated_log_histogram_DSNU',
             'accumulated_log_histogram_PRNU',
             'logarithmic_histogram_DSNU',
             'logarithmic_histogram_PRNU',
             'horizontal_profile',
             'vertical_profile']

    def __init__(self, *args, **kwargs):
        '''
        Creates and shows all plots necessary to prepare a camera or sensor
        descriptive report compliant with EMVA Standard 1288 version 3.1.
        '''
        self.figures = {}
        # Constants
        # Number of bins for Defect Pixel Characterization's histograms
        self.Q = 25

        self._titles = kwargs.pop('titles', True)

        # Get data
        self.tests = args

        ##########################
        # Make sure each test have a reference to be able to identify it
        # visually.
        for i in range(len(self.tests)):
            if not self.tests[i].name:
                self.tests[i].name = i
        ##########################

    def show(self):
        # Show plots
        plt.show()

    def plot(self, *plots):

        # Create plots
        if not plots:
            plots = range(len(self.plots))

        for i in plots:
            if i not in range(len(self.plots)):
                print('Error ', i, 'is not valid index')
                print('Plot has to be integer in ', range(len(self.plots)))
            else:
                fig = getattr(self, 'plot_' + self.plots[i])()
                self.figures[self.plots[i]] = fig
                if not self._titles:
                    fig.canvas.set_window_title('Fig %d' % (i + 1))

    def set_legend(self, ax):
        '''
        This function is just to prevent a very long legend when there are
        several datasets
        If more than one test, only label the first plot as Dataset #
        if not then use the labels setted in the different plots
        '''
        handles, labels = ax.get_legend_handles_labels()
        if len(self.tests) > 1:
            d = int(len(labels) / len(self.tests))
            for j in range(len(labels)):
                t = int(j / d)
                labels[j] = 'Dataset: ' + str(self.tests[t].name)
            ax.legend(handles[::d], labels[::d], loc='upper left')
        else:
            ax.legend(loc='upper left')

    def plot_sensitivity(self):
        '''
        Create Sensitivity plot with all instances of Processing in self.tests.
        '''
        fig = plt.figure()

        if self._titles:
            fig.canvas.set_window_title('Sensitivity')
        ax = fig.add_subplot(111)

        for test in self.tests:

            ax.plot(test.temporal['u_p'],
                    test.temporal['u_y'] - test.temporal['u_ydark'],
                    label='Data',
                    gid='dataset/%s:data/raw' % test.name)

            ax.plot(test.temporal['u_p'],
                    test.R * test.temporal['u_p'], '--',
                    label='Fit',
                    gid='dataset/%s:data/fits/data' % test.name)

            xi = test.temporal['u_p'][test.index_sensitivity_min]
            xf = test.temporal['u_p'][test.index_sensitivity_max]
            yi = (test.temporal['u_y'] -
                  test.temporal['u_ydark'])[test.index_sensitivity_min]
            yf = (test.temporal['u_y'] -
                  test.temporal['u_ydark'])[test.index_sensitivity_max]

            ax.plot((xi, xf), (yi, yf),
                    label='Fit range',
                    linestyle='None',
                    marker='o',
                    gid='dataset/%s:data/fits/range' % test.name)

            # To do : Standard EMVA3 asks to print on graph $\mu_{y.dark}$.

        self.set_legend(ax)

        ax.set_title('Sensitivity')
        ax.set_xlabel('$\mu_p$ [mean number of photons/pixel]')
        ax.set_ylabel('$\mu_y - \mu_{y.dark}$ [DN]')

        return fig

    def plot_u_ydark(self):
        '''
        Create $\mu_{y.dark}$ plot with all instances of Processing in
        self.tests.
        '''

        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Mean gray value in dark')
        ax = fig.add_subplot(111)

        for test in self.tests:
            if np.mean(test.temporal['texp']) == test.temporal['texp'][0]:
                ax.plot(test.temporal['texp'],
                        test.temporal['u_ydark'],
                        'o',
                        markersize=5,
                        label='Data',
                        gid='dataset.%s' % test.name)
            else:
                ax.plot(test.temporal['texp'],
                        test.temporal['u_ydark'],
                        label='Data',
                        gid='dataset.%s' % test.name)

        self.set_legend(ax)

        ax.set_title('$\mu_{y.dark}$')
        ax.set_xlabel('exposure time [ns]')
        ax.set_ylabel('$\mu_{y.dark}$ [DN]')

        return fig

    def plot_PTC(self):
        '''
        Create Photon Transfer plot with all instances of Processing in
        self.tests.
        '''
        fig = plt.figure(facecolor='white')
        if self._titles:
            fig.canvas.set_window_title('Photon Transfer')
        ax = fig.add_subplot(111)

        for test in self.tests:
            X = test.temporal['u_y'] - test.temporal['u_ydark']
            Y = test.temporal['s2_y'] - test.temporal['s2_ydark']
            ax.plot(X, Y,
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.plot(X, test.K * X,
                    linestyle='--',
                    label='Fit',
                    gid='dataset.%s|fits.data' % test.name)

            ax.plot((X[test.index_u_ysat], ), (Y[test.index_u_ysat], ),
                    marker='o',
                    linestyle='None',
                    label='Saturation',
                    gid='dataset.%s|other.saturation' % test.name)

            ax.plot((X[test.index_sensitivity_min],
                     X[test.index_sensitivity_max]),
                    (Y[test.index_sensitivity_min],
                     Y[test.index_sensitivity_max]),
                    linestyle='None',
                    marker='o',
                    label='Fit range',
                    gid='dataset.%s|fits.range' % test.name)

            # Todo : Standard EMVA3 asks to print on graph
            # $\sigma^2_{y.dark}$ and K with its one-sigma statistical
            # uncertainty.
        self.set_legend(ax)
        ax.set_title('Photon Transfer')
        ax.set_xlabel('$\mu_y - \mu_{y.dark}$ [DN]')
        ax.set_ylabel('$\sigma^2_y - \sigma^2_{y.dark}$ [DN$^2$]')

        return fig

    def plot_SNR(self):
        '''
        Create SNR plot with all instances of Processing in self.tests.
        '''
        fig = plt.figure(facecolor='white')
        if self._titles:
            fig.canvas.set_window_title('SNR')
        ax = fig.add_subplot(111)

        max_ideal = []

        for test in self.tests:

            X = np.arange(test.u_p_min, test.u_p_sat,
                          (test.u_p_sat - test.u_p_min) / 100.0)

            # remove the zeros on the denominator, at saturation the temporal
            # noise is zero!
            nz = np.nonzero(test.temporal['s2_y'])
            ax.plot(test.temporal['u_p'][nz],
                    (test.temporal['u_y'] - test.temporal['u_ydark'])[nz] /
                    np.sqrt(test.temporal['s2_y'][nz]),
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.plot(X,
                    ((test.QE / 100) * X) /
                    np.sqrt((test.sigma_d) ** 2 +
                            (test.s2q / (test.K) ** 2) +
                            ((test.QE / 100) * X)),
                    linestyle='--',
                    label='Theoretical',
                    gid='dataset.%s|other.theoretical' % test.name)

            ideal = np.sqrt(X)
            max_ideal.append(ideal[-1])
            ax.plot((X),
                    ideal,
                    linestyle='--',
                    label='Ideal',
                    gid='dataset.%s|other.ideal' % test.name)

            ax.axvline(test.u_p_min,
                       linestyle='--',
                       label='$\mu_{p.min} = %.1f[p]$' % test.u_p_min,
                       gid='dataset.%s|other.threshold' % test.name)

            ax.axvline(test.u_p_sat,
                       linestyle='--',
                       label='$\mu_{p.sat} = %.1f[p]$' % test.u_p_sat,
                       gid='dataset.%s|other.saturation' % test.name)

            ax.plot(X,
                    ((test.QE / 100) * X) /
                    np.sqrt((test.sigma_d) ** 2 +
                            (test.s2q / (test.K) ** 2) +
                            ((test.QE / 100) * X) +
                            (test.DSNU1288 ** 2) +
                            (((test.PRNU1288 / 100) *
                              (test.QE / 100.) * X) ** 2)),
                    linestyle='--',
                    label='Total SNR',
                    gid='dataset.%s|other.theoretical' % test.name)

        self.set_legend(ax)
        ax.loglog()
        ax.set_ylim(1, max(max_ideal))
        ax.set_title('SNR')
        ax.set_xlabel('$\mu_{p}$ [mean number of photons/pixel]')
        ax.set_ylabel('SNR')

        return fig

    def plot_linearity(self):
        '''
        Create Linearity plot with all instances of Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Linearity')
        ax = fig.add_subplot(111)

        for test in self.tests:
            X = test.temporal['u_p']
            Y = test.temporal['u_y'] - test.temporal['u_ydark']
            ax.plot(X, Y,
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.plot(X,
                    test.linearity()['fit_slope'] *
                    X + test.linearity()['fit_offset'],
                    linestyle='--',
                    label='Fit',
                    gid='dataset.%s|fits.data' % test.name)

            ax.plot((X[test.index_linearity_min], X[test.index_linearity_max]),
                    (Y[test.index_linearity_min], Y[test.index_linearity_max]),
                    label='Fit range',
                    linestyle='None',
                    marker='o',
                    gid='dataset.%s|fits.range' % test.name)

        self.set_legend(ax)
        ax.set_title('Linearity')
        ax.set_xlabel('$\mu_{p}$ [mean number of photons/pixel]')
        ax.set_ylabel('$\mu_y - \mu_{y.dark}$ [DN]')

        return fig

    def plot_deviation_linearity(self):
        '''
        Create Deviation Linearity plot with all instances of Processing in
        self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Deviation linearity')
        ax = fig.add_subplot(111)

        for test in self.tests:
            X = test.temporal['u_p'][test.index_linearity_min:
                                     test.index_linearity_max]
            deviation = test.linearity()['relative_deviation']
            Y = deviation[test.index_linearity_min: test.index_linearity_max]
            ax.plot(X, Y,
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.plot((X[0], X[-1]),
                    (Y[0], Y[-1]),
                    label='Fit range',
                    linestyle='None',
                    marker='o',
                    gid='dataset.%s|fits.range' % test.name)

        self.set_legend(ax)
        ax.set_title('Deviation linearity')
        ax.set_xlabel('$\mu_{p}$ [mean number of photons/pixel]')
        ax.set_ylabel('Linearity error LE [%]')

        return fig

    def plot_horizontal_spectrogram_PRNU(self):
        '''
        Create Horizontal spectrogram PRNU plot with all instances of
        Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Horizontal spectrogram PRNU')
        ax = fig.add_subplot(111)

        for test in self.tests:
            # In Release 3.2, there is no subtraction of the residue.
            spectrogram = routines.FFT1288(test.spatial['avg'][0])

            ax.plot(routines.GetFrecs(spectrogram[:(np.shape(spectrogram)[0] /
                                                    2)]),
                    (np.sqrt(spectrogram[:(np.shape(spectrogram)[0] / 2)])),
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.axhline(np.sqrt(test.sigma_2_y_stack),
                       label='$\sigma^2_{y.stack}$',
                       linestyle='--',
                       gid='dataset.%s|other.variance' % test.name)

            # TODO: Standard EMVA3 asks to print on graph s_w and F.

        self.set_legend(ax)
        ax.set_yscale('log')
        ax.set_title('Horizontal spectrogram PRNU')
        ax.set_xlabel('cycles [periods/pixel]')
        ax.set_ylabel('standard deviation and relative presence of each '
                      'cycle [DN]')
        # TODO: shorthen the ylabel

        return fig

    def plot_horizontal_spectrogram_DSNU(self):
        '''
        Create Horizontal spectrogram DSNU plot with all instances of
        Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Horizontal spectrogram DSNU')
        ax = fig.add_subplot(111)

        for test in self.tests:
            # In release 3.2, there is no subtraction of the residue.
            spectrogram = routines.FFT1288(test.spatial['avg_dark'][0])
            ax.plot(routines.GetFrecs(spectrogram[:(np.shape(spectrogram)[0] /
                                                    2)]),
                    np.sqrt(spectrogram[:(np.shape(spectrogram)[0] / 2)]),
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.axhline(np.sqrt(test.sigma_2_y_stack_dark),
                       label='$\sigma^2_{y.stack.dark}$',
                       linestyle='--',
                       gid='dataset.%s|other.variance' % test.name)

        self.set_legend(ax)
        ax.set_yscale('log')
        ax.set_title('Horizontal spectrogram DSNU')
        ax.set_xlabel('cycles [periods/pixel]')
        ax.set_ylabel('standard deviation and relative presence of each '
                      'cycle [DN]')
        # TODO: shorthen the ylabel
        return fig

    def plot_vertical_spectrogram_PRNU(self):
        '''
        Create Vertical spectrogram PRNU plot with all instances of Processing
        in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Vertical spectrogram PRNU')
        ax = fig.add_subplot(111)

        for test in self.tests:
            # In release 3.2, there is no subtraction of the residue.
            spectrogram = routines.FFT1288(test.spatial['avg'][0], rotate=True)

            ax.plot((routines.GetFrecs(spectrogram[:(np.shape(spectrogram)[0] /
                                                     2)])),
                    (np.sqrt(spectrogram[:(np.shape(spectrogram)[0] / 2)])),
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.axhline(np.sqrt(test.sigma_2_y_stack),
                       label='$\sigma^2_{y.stack}$',
                       linestyle='--',
                       gid='dataset.%s|other.variance' % test.name)

        self.set_legend(ax)
        ax.set_yscale('log')
        ax.set_title('Vertical spectrogram PRNU')
        ax.set_xlabel('cycles [periods/pixel]')
        ax.set_ylabel('standard deviation and relative presence of each '
                      'cycle [DN]')
        # TODO: shorthen the ylabel
        return fig

    def plot_vertical_spectrogram_DSNU(self):
        '''
        Create Vertical spectrogram DSNU plot with all instances of
        Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Vertical spectrogram DSNU')
        ax = fig.add_subplot(111)

        for test in self.tests:
            # In release 3.2, there is no subtraction of the residue.
            spectrogram = routines.FFT1288(test.spatial['avg_dark'][0],
                                           rotate=True)
            ax.plot(routines.GetFrecs(spectrogram[:(np.shape(spectrogram)[0] /
                                                    2)]),
                    np.sqrt(spectrogram[:(np.shape(spectrogram)[0] / 2)]),
                    label='Data',
                    gid='dataset.%s' % test.name)

            ax.axhline(np.sqrt(test.sigma_2_y_stack_dark),
                       label='$\sigma^2_{y.stack.dark}$',
                       linestyle='--',
                       gid='dataset.%s|other.variance' % test.name)

        self.set_legend(ax)
        ax.set_yscale('log')
        ax.set_title('Vertical spectrogram DSNU')
        ax.set_xlabel('cycles [periods/pixel]')
        ax.set_ylabel('standard deviation and relative presence of each '
                      'cycle [DN]')
        # TODO: shorthen the ylabel
        return fig

    def plot_logarithmic_histogram_DSNU(self):
        '''
        Create Logarithmic histogram DSNU plot with all instances of
        Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Logarithmic histogram DSNU')
        ax = fig.add_subplot(111)

        for test in self.tests:
            hist = test.histogram_DSNU

            ax.plot(hist['bins'], hist['values'],
                    gid='dataset.%s' % test.name,
                    label='Data')
            ax.plot(hist['bins'], hist['model'],
                    '--',
                    gid='dataset.%s|other.normal' % test.name,
                    label='Model')

        self.set_legend(ax)

        ax.set_yscale('log')
        plt.axis(ymin=1.0)
        ax.set_title('Logarithmic histogram DSNU')
        ax.set_xlabel('Deviation from the mean [DN]')
        ax.set_ylabel('Number of pixels')

        return fig

    def plot_logarithmic_histogram_PRNU(self):
        '''
        Create Logarithmic histogram PRNU plot with all instances of
        Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Logarithmic histogram PRNU')
        ax = fig.add_subplot(111)

        for test in self.tests:
            hist = test.histogram_PRNU

            ax.plot(hist['bins'], hist['values'],
                    gid='dataset.%s' % test.name,
                    label='Data')
            ax.plot(hist['bins'], hist['model'], '--',
                    gid='dataset.%s|other.normal' % test.name,
                    label='Model')

        self.set_legend(ax)

        ax.set_yscale('log')
        plt.axis(ymin=0.5)
        ax.set_title('Logarithmic histogram PRNU')
        ax.set_xlabel('Deviation from the mean [%]')
        ax.set_ylabel('Number of pixels')

        return fig

    def plot_accumulated_log_histogram_DSNU(self):
        '''
        Create Accumulated log histogram DSNU plot with all instances of
        Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Accumulated log histogram DSNU')
        ax = fig.add_subplot(111)

        for test in self.tests:
            hist = test.histogram_DSNU_accumulated

            ax.plot(hist['bins'], hist['values'],
                    gid='dataset.%s' % test.name,
                    label='Data')

        self.set_legend(ax)

        ax.set_yscale('log')
        ax.set_title('Accumulated log histogram DSNU')
        ax.set_xlabel('Minimal deviation from the mean [DN]')
        ax.set_ylabel('Percentage of pixels deviating from the mean at '
                      'least of : ')
        # TODO: shorthen the ylabel
        return fig

    def plot_accumulated_log_histogram_PRNU(self):
        '''
        Create Accumulated log histogram PRNU plot with all instances of
        Processing in self.tests.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Accumulated log histogram PRNU')

        ax = fig.add_subplot(111)

        for test in self.tests:
            hist = test.histogram_PRNU_accumulated

            ax.plot(hist['bins'], hist['values'],
                    gid='dataset.%s' % test.name,
                    label='Data')

        self.set_legend(ax)

        ax.set_yscale('log')
        ax.set_title('Accumulated log histogram PRNU')
        ax.set_xlabel('Minimal deviation from the mean [%]')
        ax.set_ylabel('Percentage of pixels deviating from the mean at '
                      'least of : ')
        # TODO: shorthen the ylabel
        return fig

    def plot_horizontal_profile(self):
        '''
        Create Horizontal profile plot with all instances of Processing in
        self.tests.
        Profile is done with spatial images.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Horizontal profile')
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        bmin = []
        bmax = []
        dmin = []
        dmax = []
        length = []

        for test in self.tests:
            img = test.spatial['avg'][0]
            profile = np.mean(img, axis=0)
            profile_min = np.min(img, axis=0)
            profile_max = np.max(img, axis=0)
            x = np.arange(np.shape(profile)[0])
            l = np.shape(img)[0]
            profile_mid = img[l / 2, :]

            length.append(np.shape(profile)[0])

            avg_ = np.mean(img)
            bmax.append(1.1 * avg_)
            bmin.append(0.9 * avg_)

            ax.plot(x, profile_mid,
                    label='50% mid',
                    gid='dataset.%s|illumination.50.mid' % test.name)
            ax.plot(x, profile_min,
                    label='50% min',
                    gid='dataset.%s|illumination.50.min' % test.name)
            ax.plot(x, profile_max,
                    label='50% max',
                    gid='dataset.%s|illumination.50.max' % test.name)
            ax.plot(x, profile,
                    label='50% mean',
                    gid='dataset.%s|illumination.50.mean' % test.name)

            img = test.spatial['avg_dark'][0]
            profile_dark = np.mean(img, axis=0)
            profile_dark_min = np.min(img, axis=0)
            profile_dark_max = np.max(img, axis=0)
            x_dark = np.arange(np.shape(profile_dark)[0])

            l = np.shape(img)[0]
            profile_dark_mid = img[l / 2, :]

            avg_ = np.mean(profile_dark_max)
            dmax.append(1.1 * avg_)

            avg_ = np.mean(profile_dark_min)
            dmin.append(0.9 * avg_)

            ax2.plot(x_dark, profile_dark_mid,
                     label='Dark mid',
                     gid='dataset.%s|illumination.dark.mid' % test.name)
            ax2.plot(x_dark, profile_dark_min,
                     label='Dark min',
                     gid='dataset.%s|illumination.dark.min' % test.name)
            ax2.plot(x_dark, profile_dark_max,
                     label='Dark max',
                     gid='dataset.%s|illumination.dark.max' % test.name)
            ax2.plot(x_dark, profile_dark,
                     label='Dark mean',
                     gid='dataset.%s|illumination.dark.mean' % test.name)

        fig.suptitle('Horizontal profile')
        ax.set_title('PRNU')
        ax.set_ylabel('Vertical line [DN]')
        ax2.set_title('DSNU')
        ax2.set_xlabel('Index of the line')
        ax2.set_ylabel('Vertical line [DN]')
        ax.axis(ymin=min(bmin), ymax=max(bmax), xmax=max(length))
        ax2.axis(ymin=min(dmin), ymax=max(dmax), xmax=max(length))

        return fig

    def plot_vertical_profile(self):
        '''
        Create Vertical profile plot with all instances of Processing in
        self.tests.
        Profile is done with spatial images.
        '''
        fig = plt.figure()
        if self._titles:
            fig.canvas.set_window_title('Vertical profile')

        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        bmin = []
        bmax = []
        dmin = []
        dmax = []
        length = []

        for test in self.tests:
            img = test.spatial['avg'][0] - test.spatial['avg_dark'][0]
            avg_ = 100. / np.mean(img)
            profile = np.mean(img, axis=1)
            profile_min = np.min(img, axis=1)
            profile_max = np.max(img, axis=1)
            l = np.shape(img)[1]
            profile_mid = img[:, l / 2]
            y = np.arange(np.shape(profile)[0])

            length.append(np.shape(profile)[0])
            avg_ = np.mean(img)
            bmax.append(1.1 * avg_)
            bmin.append(0.9 * avg_)

            ax2.plot(profile_mid, y,
                     label='50% mid',
                     gid='dataset.%s|illumination.50.mid' % test.name)
            ax2.plot(profile_min, y,
                     label='50% min',
                     gid='dataset.%s|illumination.50.min' % test.name)
            ax2.plot(profile_max, y,
                     label='50% max',
                     gid='dataset.%s|illumination.50.max' % test.name)
            ax2.plot(profile, y,
                     label='50% mean',
                     gid='dataset.%s|illumination.50.mean' % test.name)

            img = test.spatial['avg_dark'][0]
            profile_dark = np.mean(img, axis=1)
            profile_dark_min = np.min(img, axis=1)
            profile_dark_max = np.max(img, axis=1)
            y_dark = np.arange(np.shape(profile_dark)[0])

            l = np.shape(img)[1]
            profile_dark_mid = img[:, l / 2]

            avg_ = np.mean(profile_dark_max)
            dmax.append(1.1 * avg_)

            avg_ = np.mean(profile_dark_min)
            dmin.append(0.9 * avg_)

            ax.plot(profile_dark_mid, y_dark,
                    label='Dark mid',
                    gid='dataset.%s|illumination.dark.mid' % test.name)
            ax.plot(profile_dark_min, y_dark,
                    label='Dark min',
                    gid='dataset.%s|illumination.dark.min' % test.name)
            ax.plot(profile_dark_max, y_dark,
                    label='Dark max',
                    gid='dataset.%s|illumination.dark.max' % test.name)
            ax.plot(profile_dark, y_dark,
                    label='Dark mean',
                    gid='dataset.%s|illumination.dark.mean' % test.name)

        fig.suptitle('Vertical profile')
        ax.set_title('DSNU')
        ax.set_xlabel('Vertical line [DN]')
        ax.set_ylabel('Index of the line')
        ax2.set_title('PRNU')
        ax2.set_xlabel('Vertical line [DN]')
        ax2.axis(xmin=min(bmin), xmax=max(bmax), ymax=max(length))
        ax.axis(xmin=min(dmin), xmax=max(dmax), ymax=max(length))

        return fig
