# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""Plot the results
This class takes a results.Results1288 object and produces all the plots
needed to create a reference datasheet of the EMVA1288 test

"""

from __future__ import print_function
import numpy as np

from . import routines


class Emva1288Plot(object):
    """Base class for emva plots."""
    name = ""
    """The figure's name (used as title if title is none)."""

    title = None
    """The figure's title."""

    xlabel = None
    """The x axis label."""

    ylabel = None
    """The y axis label."""

    xscale = None
    """The x axis scale."""

    yscale = None
    """The y axis scale."""

    def __init__(self, figure):
        """Base class for emva plots init function.

        The only mandatory attribute is the name, the rest are for use
        in the :meth:`setup_figure` method.

        Parameters
        ----------
        figure : The :class:`matplotlib:matplotlib.figure.Figure`
                 object to plot.
        """
        self.figure = figure
        self.setup_figure()

    def setup_figure(self):
        """Simple wrapper for one plot per figure

        Takes the name, xlabel, ylabel, xscale and yscale
        for one plot case.

        If more than one plot, just overwrite as you wish.
        """
        ax = self.figure.add_subplot(111)
        if self.title:
            ax.set_title(self.title)
        else:
            ax.set_title(self.name)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
        if self.xscale:
            ax.set_xscale(self.xscale)
        if self.yscale:
            ax.set_yscale(self.yscale)

        self.ax = ax

    def plot(self, test):
        """Method to show the figures.

        Parameters
        ----------
        test : Do nothing for this method but can be
               used for the subclass method.

        Raises
        ------
        NotImplementedError
            If this method is not overridden.

        Notes
        -----
        Must be overridden in subclasses.
        """
        raise NotImplementedError

    def set_legend(self, ax):
        """Shortcut to add legend.

        Parameters
        ----------
        ax : The :class:`matplotlib:matplotlib.axes.Axes` object to which
             the legend will be added.
        """
        ax.legend(loc='best')
        legend = ax.get_legend()
        if legend is not None:
            if getattr(legend, 'draggable', False):
                legend.draggable(True)

    def rearrange(self):
        """Opportunity to change axis or limits after all the tests have
        been plotted.

        Uses :meth:`matplotlib:matplotlib.figure.Figure.tight_layout` method.
        """
        self.figure.tight_layout()

    def reduce_ticks(self, ax, axis, n=4):
        """Reduce the number of ticks in ax.axis

        Uses the :meth:`matplotlib:matplotlib.axes.Axes.locator_params` method.

        Parameters
        ----------
        ax : The :class:`matplotlib:matplotlib.axes.Axes` object to which
             the number of ticks will be changed.
        axis : str, {'x', 'y', 'both'}
               Axis on which to operate.
        n : int, optional
            Number of bins between ticks to be left.
        """
        ax.locator_params(axis=axis, nbins=n)


class PlotSensitivity(Emva1288Plot):
    name = 'Sensitivity'
    xlabel = '$\mu_p$ [mean number of photons/pixel]'
    ylabel = '$\mu_y - \mu_{y.dark}$ [DN]'

    def plot(self, test):
        ax = self.ax
        ax.plot(test.temporal['u_p'],
                test.temporal['u_y'] - test.temporal['u_ydark'],
                label='Data',
                gid='%d:data' % test.id)

        ax.plot(test.temporal['u_p'],
                test.R * test.temporal['u_p'], '--',
                label='Fit',
                gid='%d:fit' % test.id)

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
                gid='%d:marker' % test.id)

        self.set_legend(ax)

    def rearrange(self):
        self.ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
        self.figure.tight_layout()


class PlotUyDark(Emva1288Plot):
    '''Create $\mu_{y.dark}$ plot'''

    name = 'Mean gray value in dark'
    title = '$\mu_{y.dark}$'
    xlabel = 'exposure time [ns]'
    ylabel = '$\mu_{y.dark}$ [DN]'

    def plot(self, test):
        ax = self.ax

        if np.mean(test.temporal['texp']) == test.temporal['texp'][0]:
            ax.plot(test.temporal['texp'],
                    test.temporal['u_ydark'],
                    'o',
                    markersize=5,
                    label='Data',
                    gid='%d:data' % test.id)
        else:
            ax.plot(test.temporal['texp'],
                    test.temporal['u_ydark'],
                    label='Data',
                    gid='%d:data' % test.id)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))
        self.set_legend(ax)


class PlotPTC(Emva1288Plot):
    '''Create Photon Transfer plot'''

    name = 'Photon Transfer'
    xlabel = '$\mu_y - \mu_{y.dark}$ [DN]'
    ylabel = '$\sigma^2_y - \sigma^2_{y.dark}$ [DN$^2$]'

    def plot(self, test):
        ax = self.ax

        X = test.temporal['u_y'] - test.temporal['u_ydark']
        Y = test.temporal['s2_y'] - test.temporal['s2_ydark']
        ax.plot(X, Y,
                label='Data',
                gid='%d:data' % test.id)

        ax.plot(X, test.K * X,
                linestyle='--',
                label='Fit',
                gid='%d:fit' % test.id)

        ax.plot((X[test.index_u_ysat], ), (Y[test.index_u_ysat], ),
                marker='s',
                linestyle='None',
                label='Saturation',
                gid='%d:marker' % test.id)

        ax.plot((X[test.index_sensitivity_min],
                 X[test.index_sensitivity_max]),
                (Y[test.index_sensitivity_min],
                 Y[test.index_sensitivity_max]),
                linestyle='None',
                marker='o',
                label='Fit range',
                gid='%d:marker' % test.id)

        self.set_legend(ax)


class PlotSNR(Emva1288Plot):
    '''Create SNR plot '''

    name = 'Signal to Noise Ratio'
    xlabel = '$\mu_{p}$ [mean number of photons/pixel]'
    ylabel = 'SNR'

    def setup_figure(self):
        super(PlotSNR, self).setup_figure()
        self.ax.loglog()
        max_ideal = []
        self.max_ideal = max_ideal

    def plot(self, test):
        ax = self.ax

        X = np.arange(test.u_p_min, test.u_p_sat,
                      (test.u_p_sat - test.u_p_min) / 100.0)

        # remove the zeros on the denominator, at saturation the temporal
        # noise is zero!
        nz = np.nonzero(test.temporal['s2_y'])
        ax.plot(test.temporal['u_p'][nz],
                (test.temporal['u_y'] - test.temporal['u_ydark'])[nz] /
                np.sqrt(test.temporal['s2_y'][nz]),
                marker='.',
                ls=' ',
                label='Data',
                gid='%d:data' % test.id)

        ax.plot(X,
                ((test.QE / 100) * X) /
                np.sqrt((test.sigma_d) ** 2 +
                        (test.s2q / (test.K) ** 2) +
                        ((test.QE / 100) * X)),
                linestyle=':',
                label='Theoretical',
                gid='%d:fit' % test.id)

        ideal = np.sqrt(X)
        self.max_ideal.append(ideal[-1])
        ax.plot((X),
                ideal,
                linestyle='-.',
                label='Ideal',
                gid='%d:fit' % test.id)

        ax.axvline(test.u_p_min,
                   label='$\mu_{p.min} = %.1f[p]$' % test.u_p_min,
                   gid='%d:marker' % test.id)

        ax.axvline(test.u_p_sat,
                   label='$\mu_{p.sat} = %.1f[p]$' % test.u_p_sat,
                   gid='%d:marker' % test.id)

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
                gid='%d:fit' % test.id)

        ax.set_ylim(1, max(self.max_ideal))
        self.set_legend(ax)


class PlotLinearity(Emva1288Plot):
    '''Create Linearity plot'''

    name = 'Linearity'
    xlabel = '$\mu_{p}$ [mean number of photons/pixel]'
    ylabel = '$\mu_y - \mu_{y.dark}$ [DN]'

    def plot(self, test):
        ax = self.ax

        X = test.temporal['u_p']
        Y = test.temporal['u_y'] - test.temporal['u_ydark']
        ax.plot(X, Y,
                label='Data',
                gid='%d:data' % test.id)

        ax.plot(X,
                test.linearity()['fit_slope'] *
                X + test.linearity()['fit_offset'],
                linestyle='--',
                label='Fit',
                gid='%d:fit' % test.id)

        ax.plot((X[test.index_linearity_min], X[test.index_linearity_max]),
                (Y[test.index_linearity_min], Y[test.index_linearity_max]),
                label='Fit range',
                linestyle='None',
                marker='o',
                gid='%d:marker' % test.id)

        self.set_legend(ax)
        self.ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))


class PlotDeviationLinearity(Emva1288Plot):
    '''Create Deviation Linearity plot'''

    name = 'Deviation linearity'
    xlabel = '$\mu_{p}$ [mean number of photons/pixel]'
    ylabel = 'Linearity error LE [%]'

    def plot(self, test):
        ax = self.ax

        X = test.temporal['u_p'][test.index_linearity_min:
                                 test.index_linearity_max]
        deviation = test.linearity()['relative_deviation']
        Y = deviation[test.index_linearity_min: test.index_linearity_max]
        ax.plot(X, Y,
                label='Data',
                gid='%d:data' % test.id)

        ax.plot((X[0], X[-1]),
                (Y[0], Y[-1]),
                label='Fit range',
                linestyle='None',
                marker='o',
                gid='%d:marker' % test.id)

        self.set_legend(ax)
        self.ax.ticklabel_format(axis='x', style='sci', scilimits=(1, 4))


class PlotHorizontalSpectrogramPRNU(Emva1288Plot):
    '''Create Horizontal spectrogram PRNU plot'''

    name = 'Horizontal spectrogram PRNU'
    xlabel = 'cycles [periods/pixel]'
    ylabel = 'Standard deviation and\nrelative presence of each cycle [%]'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax

        data = test.spatial['avg'] - test.spatial['avg_dark']
        data_mean = np.mean(data)

        spectrogram = routines.FFT1288(data) / data_mean

        ax.plot(routines.GetFrecs(spectrogram),
                (np.sqrt(spectrogram)),
                label='Data',
                gid='%d:data' % test.id)

        ax.axhline(test.PRNU1288,
                   label='$PRNU_{1288}$',
                   linestyle='--',
                   color='r',
                   gid='%d:marker' % test.id)

        ax.axhline(100 * np.sqrt(test.sigma_2_y_stack) / data_mean,
                   label='$\sigma^2_{y.stack}$',
                   linestyle='-.',
                   color='g',
                   gid='%d:marker' % test.id)

        self.set_legend(ax)


class PlotHorizontalSpectrogramDSNU(Emva1288Plot):
    '''Create Horizontal spectrogram DSNU plot'''

    name = 'Horizontal spectrogram DSNU'
    xlabel = 'cycles [periods/pixel]'
    ylabel = 'Standard deviation and\nrelative presence of each cycle [DN]'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax

        spectrogram = routines.FFT1288(test.spatial['avg_dark'])
        ax.plot(routines.GetFrecs(spectrogram),
                np.sqrt(spectrogram),
                label='Data',
                gid='%d:data' % test.id)

        ax.axhline(test.DSNU1288_DN(),
                   label='$DSNU_{1288.DN}$',
                   linestyle='--',
                   color='r',
                   gid='%d:marker' % test.id)

        ax.axhline(np.sqrt(test.sigma_2_y_stack_dark),
                   label='$\sigma^2_{y.stack.dark}$',
                   linestyle='-.',
                   color='g',
                   gid='%d:marker' % test.id)

        self.set_legend(ax)


class PlotVerticalSpectrogramPRNU(Emva1288Plot):
    '''Create Vertical spectrogram PRNU plot'''

    name = 'Vertical spectrogram PRNU'
    xlabel = 'cycles [periods/pixel]'
    ylabel = 'Standard deviation and\nrelative presence of each cycle [%]'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax
        data = test.spatial['avg'] - test.spatial['avg_dark']
        data_mean = np.mean(data)
        spectrogram = routines.FFT1288(data, rotate=True) / data_mean

        ax.plot((routines.GetFrecs(spectrogram)),
                (np.sqrt(spectrogram)),
                label='Data',
                gid='%d:data' % test.id)

        ax.axhline(test.PRNU1288,
                   label='$PRNU_{1288}$',
                   linestyle='--',
                   color='r',
                   gid='%d:marker' % test.id)

        ax.axhline(100 * np.sqrt(test.sigma_2_y_stack) / data_mean,
                   label='$\sigma^2_{y.stack}$',
                   linestyle='-.',
                   color='g',
                   gid='%d:marker' % test.id)

        self.set_legend(ax)


class PlotVerticalSpectrogramDSNU(Emva1288Plot):
    '''Create Vertical spectrogram DSNU plot'''

    name = 'Vertical spectrogram DSNU'
    xlabel = 'Cycles [periods/pixel]'
    ylabel = 'Standard deviation and\nrelative presence of each cycle [DN]'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax

        spectrogram = routines.FFT1288(test.spatial['avg_dark'],
                                       rotate=True)
        ax.plot(routines.GetFrecs(spectrogram),
                np.sqrt(spectrogram),
                label='Data',
                gid='%d:data' % test.id)

        ax.axhline(test.DSNU1288_DN(),
                   label='$DSNU_{1288.DN}$',
                   linestyle='--',
                   color='r',
                   gid='%d:marker' % test.id)

        ax.axhline(np.sqrt(test.sigma_2_y_stack_dark),
                   label='$\sigma^2_{y.stack.dark}$',
                   linestyle='-.',
                   color='g',
                   gid='%d:marker' % test.id)

        self.set_legend(ax)


class PlotLogarithmicHistogramDSNU(Emva1288Plot):
    '''Create Logarithmic histogram DSNU plot'''

    name = 'Logarithmic histogram DSNU'
    xlabel = 'Deviation from the mean [DN]'
    ylabel = 'Number of pixels'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax

        hist = test.histogram_DSNU

        ax.plot(hist['bins'], hist['values'],
                gid='%d:data' % test.id,
                label='Data')
        ax.plot(hist['bins'], hist['model'],
                '--',
                gid='%d:fit' % test.id,
                label='Model')

        self.set_legend(ax)

        ax.axis(ymin=1.0, ymax=np.max(hist['values']) * 2)


class PlotLogarithmicHistogramPRNU(Emva1288Plot):
    '''Create Logarithmic histogram PRNU plot'''

    name = 'Logarithmic histogram PRNU'
    xlabel = 'Deviation from the mean [%]'
    ylabel = 'Number of pixels'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax
        hist = test.histogram_PRNU

        ax.plot(hist['bins'], hist['values'],
                gid='%d:data' % test.id,
                label='Data')
        ax.plot(hist['bins'], hist['model'], '--',
                gid='%d:fit' % test.id,
                label='Model')

        self.set_legend(ax)
        ax.axis(ymin=0.5, ymax=np.max(hist['values']) * 2)


class PlotAccumulatedLogHistogramDSNU(Emva1288Plot):
    '''Create Accumulated log histogram DSNU plot'''

    name = 'Accumulated log histogram DSNU'
    xlabel = 'Minimal deviation from the mean [DN]'
    ylabel = 'Percentage of pixels/bin'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax
        hist = test.histogram_DSNU_accumulated

        ax.plot(hist['bins'], hist['values'],
                gid='%d:data' % test.id,
                label='Data')

        self.set_legend(ax)
        self.figure.tight_layout()


class PlotAccumulatedLogHistogramPRNU(Emva1288Plot):
    '''Create Accumulated log histogram PRNU plot'''

    name = 'Accumulated log histogram PRNU'
    xlabel = 'Minimal deviation from the mean [%]'
    ylabel = 'Percentage of pixels/bin'
    yscale = 'log'

    def plot(self, test):
        ax = self.ax

        hist = test.histogram_PRNU_accumulated

        ax.plot(hist['bins'], hist['values'],
                gid='%d:data' % test.id,
                label='Data')

        self.set_legend(ax)


class ProfileBase(Emva1288Plot):
    def __init__(self, *args, **kwargs):
        Emva1288Plot.__init__(self, *args, **kwargs)
        # Dict to keep track of max and min values to adjust the
        # limit of the axis
        self.axis_limits = {'bright': {'max': [], 'min': [], 'length': []},
                            'dark': {'max': [], 'min': [], 'length': []}}

    def _get_extremes(self, mean_, min_, max_):
        min_min_i = np.argmax(mean_ - min_)
        min_min = min_[min_min_i]
        min_perc = np.abs(100. - (min_min * 100. / mean_[min_min_i]))
        min_label = 'Min ({:.1f} {:.1f}%)'.format(min_min,
                                                  min_perc)

        max_max_i = np.argmax(max_ - min_)
        max_max = max_[max_max_i]
        max_perc = np.abs(100. - (max_max * 100. / mean_[max_max_i]))
        max_label = 'Max ({:.1f} {:.1f}%)'.format(max_max,
                                                  max_perc)

        return {'min_deviation': min_min, 'min_precentage': min_perc,
                'min_label': min_label, 'max_label': max_label,
                'max_deviation': max_max, 'max_percentage': max_perc}

    def _get_image_profiles(self, image):
        """Get the images profiles. Supports the masked arrays.

        Masked arrays cannot be shown directly with continuous lines.
        To fix this, we only return the profiles excluding the masks
        constants.

        Also returns the x arrays corresponding the the profile.
        """
        img = image
        if self.vertical:
            img = np.transpose(image)

        profile = np.mean(img, axis=0)
        profile_min = np.min(img, axis=0)
        profile_max = np.max(img, axis=0)

        mid_i = np.shape(img)[0]
        profile_mid = img[mid_i // 2, :]

        # Verifications for profile_mid
        if isinstance(profile_mid, np.ma.core.MaskedArray):
            if profile_mid.mask.all():
                # If by chance mid is a column of masked constant,
                # take the next one
                profile_mid = img[mid_i // 2 + 1, :]

        # use _get_x_y to get the x and profile of unmasked values
        d = {'mean': self._get_x_y(profile),
             'min': self._get_x_y(profile_min),
             'max': self._get_x_y(profile_max),
             'mid': self._get_x_y(profile_mid),
             }
        return d

    def _get_x_y(self, profile):
        x = np.arange(len(profile))
        if isinstance(profile, np.ma.MaskedArray):
            masks = profile.mask
            # Index lists of where there is masked constants
            index = [i for i in range(len(masks)) if not masks[i]]
            # chop out masked values
            x = x[index]
            profile = profile[index]
        return (x, profile)

    def get_profiles(self, bright, dark):
        b_p = self._get_image_profiles(bright)
        # index 1 is the profile (0 is x-axis)
        b_mean = np.mean(b_p['mean'][1])
        self.axis_limits['bright']['min'].append(0.9 * b_mean)
        self.axis_limits['bright']['max'].append(1.1 * b_mean)

        d_p = self._get_image_profiles(dark)
        self.axis_limits['dark']['min'].append(0.9 * np.mean(d_p['min'][1]))
        self.axis_limits['dark']['max'].append(1.1 * np.mean(d_p['max'][1]))
        return {'bright': b_p, 'dark': d_p}

    def plot(self, test):
        # index of profiles of what will be the x and y axis
        x = 0
        y = 1
        legend_loc = 'upper right'
        if self.vertical:
            x = 1
            y = 0
            legend_loc = (0.8, 0.65)

        ax = self.ax
        ax2 = self.ax2

        bimg = test.spatial['avg'] - test.spatial['avg_dark']
        dimg = test.spatial['avg_dark']
        profiles = self.get_profiles(bimg, dimg)

        # to keep the lines number for legend
        bright_plots = []
        labels = []

        for typ, color in (('mid', 'green'),
                           ('min', 'blue'),
                           ('max', 'orange'),
                           ('mean', 'black')):
            # label has first letter capital
            label = typ.capitalize()
            labels.append(label)

            # bright plot
            l = ax.plot(profiles['bright'][typ][x],
                        profiles['bright'][typ][y],
                        label=label,
                        color=color,
                        gid='%d:marker' % test.id)[0]
            bright_plots.append(l)

            # dark plot
            ax2.plot(profiles['dark'][typ][x],
                     profiles['dark'][typ][y],
                     label=label,
                     color=color,
                     gid='%d:data' % test.id)

        # Place legend
        self.figure.legend(bright_plots,
                           labels,
                           loc=legend_loc)


class PlotHorizontalProfile(ProfileBase):
    '''Create Horizontal profile plot
    Profile is done with spatial images.
    '''

    name = 'Horizontal profile'
    vertical = False

    def setup_figure(self):
        self.ax = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.figure.suptitle(self.name)
        self.ax.set_title('PRNU')
        self.ax.set_ylabel('Vertical line [DN]')
        self.ax2.set_title('DSNU')
        self.ax2.set_xlabel('Index of the line')
        self.ax2.set_ylabel('Vertical line [DN]')

    def rearrange(self):
        self.ax.set_xticks([])
        self.ax.axis(ymin=min(self.axis_limits['bright']['min']),
                     ymax=max(self.axis_limits['bright']['max']))
        self.ax2.axis(ymin=min(self.axis_limits['dark']['min']),
                      ymax=max(self.axis_limits['dark']['max']))

        self.reduce_ticks(self.ax2, 'y')
        self.reduce_ticks(self.ax, 'y')
        self.figure.tight_layout(pad=2)


class PlotVerticalProfile(ProfileBase):
    '''Create Vertical profile plot.
    Profile is done with spatial images.
    '''

    name = 'Vertical profile'
    vertical = True

    def setup_figure(self):
        self.ax2 = self.figure.add_subplot(121)
        self.ax = self.figure.add_subplot(122)
        self.figure.suptitle(self.name)
        self.ax2.set_title('DSNU')
        self.ax2.set_xlabel('Vertical line [DN]')
        self.ax2.set_ylabel('Index of the line')
        self.ax.set_title('PRNU')
        self.ax.set_xlabel('Vertical line [DN]')

    def rearrange(self):
        self.ax.set_yticks([])
        self.ax.axis(xmin=min(self.axis_limits['bright']['min']),
                     xmax=max(self.axis_limits['bright']['max']))
        self.ax2.axis(xmin=min(self.axis_limits['dark']['min']),
                      xmax=max(self.axis_limits['dark']['max']))
        self.ax2.invert_yaxis()
        self.ax.invert_yaxis()
        self.reduce_ticks(self.ax2, 'x')
        self.reduce_ticks(self.ax, 'x')
        self.figure.tight_layout()


EVMA1288plots = [PlotPTC,
                 PlotSNR,
                 PlotSensitivity,
                 PlotUyDark,
                 PlotLinearity,
                 PlotDeviationLinearity,
                 PlotHorizontalSpectrogramPRNU,
                 PlotHorizontalSpectrogramDSNU,
                 PlotVerticalSpectrogramPRNU,
                 PlotVerticalSpectrogramDSNU,
                 PlotLogarithmicHistogramDSNU,
                 PlotLogarithmicHistogramPRNU,
                 PlotAccumulatedLogHistogramDSNU,
                 PlotAccumulatedLogHistogramPRNU,
                 PlotHorizontalProfile,
                 PlotVerticalProfile]
"""
    This list is quite exhaustive. There are the links
    to corresponding documentation:

    - :class:`~emva1288.process.plotting.PlotPTC`
    - :class:`~emva1288.process.plotting.PlotSNR`
    - :class:`~emva1288.process.plotting.PlotSensitivity`
    - :class:`~emva1288.process.plotting.PlotUyDark`
    - :class:`~emva1288.process.plotting.PlotLinearity`
    - :class:`~emva1288.process.plotting.PlotDeviationLinearity`
    - :class:`~emva1288.process.plotting.PlotHorizontalSpectrogramPRNU`
    - :class:`~emva1288.process.plotting.PlotHorizontalSpectrogramDSNU`
    - :class:`~emva1288.process.plotting.PlotVerticalSpectrogramPRNU`
    - :class:`~emva1288.process.plotting.PlotVerticalSpectrogramDSNU`
    - :class:`~emva1288.process.plotting.PlotLogarithmicHistogramDSNU`
    - :class:`~emva1288.process.plotting.PlotLogarithmicHistogramPRNU`
    - :class:`~emva1288.process.plotting.PlotAccumulatedLogHistogramPRNU`
    - :class:`~emva1288.process.plotting.PlotAccumulatedLogHistogramDSNU`
    - :class:`~emva1288:emva1288.process.plotting.PlotHorizontalProfile`
    - :class:`~emva1288.process.plotting.PlotVerticalProfile`
"""


class Plotting1288(object):
    """EMVA1288 plots

    Creates and shows all plots necessary to prepare a camera or sensor
    descriptive report compliant with EMVA Standard 1288 version 3.1.
    """

    def __init__(self, *tests):
        '''
        Parameters
        ----------
        tests: list
            List of tests to Plot
        '''

        self.tests = []
        for test in tests:
            if not getattr(test, 'id', False):
                test.id = id(test)
        self.tests.append(test)

    def plot(self, *plots):
        """Plot EMVA1288 plots

        Parameters
        ----------
        plots: list
            List of plots to plot

        """
        import matplotlib.pyplot as plt
        if not plots:
            plots = EVMA1288plots

        for i, plot_cls in enumerate(plots):
            figure = plt.figure(i)
            plot = plot_cls(figure)
            for test in self.tests:
                plot.plot(test)
            plot.rearrange()
            figure.canvas.set_window_title(plot.name)
        plt.show()
