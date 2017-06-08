import unittest
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.loader import LoadImageData
from emva1288.process.data import Data1288
from emva1288.process.results import Results1288
from emva1288.camera.dataset_generator import DatasetGenerator
from emva1288.unittests.test_routines import del_obj
import numpy as np


def _init(pixel_area=0, **kwargs):
    # create dataset
    dataset = DatasetGenerator(**kwargs)

    # parse dataset
    parser = ParseEmvaDescriptorFile(dataset.descriptor_path)
    # load image data
    loader = LoadImageData(parser.images)
    # create data
    data = Data1288(loader.data)
    # Make results object
    px = pixel_area
    if pixel_area == 0:
        px = dataset.cam.pixel_area
    results = Results1288(data.data, pixel_area=px)
    return dataset, parser, loader, data, results


class TestResults(unittest.TestCase):
    _height = 50
    _width = 100
    _bit_depth = 8
    _L = 50
    _qe = 0.5
    _steps = 10
    _radiance_min = None
    _exposure_max = 5000000000
    _dark_current_ref = 30
    _exposure_fixed = 10000000
    _temperature = 20
    _temperature_ref = 20
    _K = 0.5
    _exposure_min = 50000
    _dsnu = np.zeros((_height, _width))
    _dsnu[0, :] += 5
    _prnu = np.ones((_height, _width))
    _prnu[-1, :] += 1.5

    def test_results_exposure_variation(self):
        dt, p, l, da, r = _init(height=self._height,
                                width=self._width,
                                bit_depth=self._bit_depth,
                                L=self._L,
                                qe=self._qe,
                                steps=self._steps,
                                radiance_min=self._radiance_min,
                                exposure_max=self._exposure_max,
                                exposure_min=self._exposure_min,
                                K=self._K,
                                dark_current_ref=self._dark_current_ref,
                                temperature=self._temperature,
                                temperature_ref=self._temperature_ref,
                                dsnu=self._dsnu,
                                prnu=self._prnu)

        self.dataset = dt
        self.parser = p
        self.loader = l
        self.data = da
        self.results = r

        ###########################
        # Test results properties #
        ###########################

        # test that quantification noise is really 1/12
        self.assertEqual(self.results.s2q, 1.0 / 12.0)

        # test that indexes are integers and in good range
        for attr in ('index_start', 'index_u_ysat', 'index_sensitivity_max',
                     'index_sensitivity_min', 'index_linearity_min',
                     'index_linearity_max'):
            value = getattr(self.results, attr)
            self.assertTrue(type(value) is int or type(value) is np.int64,
                            msg="%s is not an integer but should be!" % attr)
            self.assertTrue(value < len(self.data.data['temporal']['u_y']))
            self.assertTrue(value >= 0)
            self.assertTrue(attr in self.results.results.keys(),
                            msg="%s does not appear in the results!" % attr)

        # Test that EMVA values are float and positive
        for a in ('s2q', 'R', 'K', 'QE', 'sigma_y_dark', 'sigma_d', 'u_p_min',
                  'u_p_min_area', 'u_e_min', 'u_e_min_area', 'u_p_sat',
                  'u_p_sat_area', 'u_e_sat', 'SNR_max', 'DR', 'LE_min',
                  'LE_max', 'u_I_var', 'u_I_mean', 'sigma_2_y_stack',
                  'sigma_2_y_stack_dark', 's_2_y_measured', 's_2_y',
                  's_2_y_dark', 'DSNU1288', 'PRNU1288'):
            value = getattr(self.results, a)
            self.assertTrue(isinstance(value, float),
                            msg="%s is not a float but should be!" % a)
            if not a == 's2q':
                self.assertTrue(a in self.results.results,
                                msg="%s does not appear in the results!" % a)
            # except for linearity errors and dark currents,
            # everything always should be positive
            if (a not in ('LE_min', 'LE_max') and
               value is not np.nan):
                self.assertGreaterEqual(value, 0.0,
                                        msg="%s is negative but"
                                            " should be positive!" % a)

        ###############################################################
        # The following deltas are purely guesstimates and are prone to
        # errors in the future if they are not really significant
        ###############################################################

        # Test quantum efficiency is retrieved with a +/- 5% incertainty
        self.assertAlmostEqual(self._qe * 100, self.results.QE, delta=5.0,
                               msg="The difference between the expected QE and"
                                   "the retrieved one is greater than 5%!")

        # Test that overall system gain
        # is retrieved with a +/- 0.01 incertainty
        self.assertAlmostEqual(self.dataset.cam.K, self.results.K, delta=0.1,
                               msg="The difference between expected"
                                   "system gain"
                                   "and the retrieved one"
                                   "is greater than 0.01!")

        self.assertEqual(self.results.inverse_K(), 1 / self.results.K)

        # Test that responsivity is coherent with QE and system gain
        self.assertAlmostEqual(self.results.R,
                               self.results.QE * self.results.K / 100,
                               delta=0.001)  # division errors compensation

        # Test that dark current is actually retrieved from both methods
        self.assertAlmostEqual(self._dark_current_ref, self.results.u_I_mean,
                               delta=5.0,
                               msg="Dark current is not well retrieved from"
                                   " mean dark signal.")
        self.assertAlmostEqual(self._dark_current_ref, self.results.u_I_var,
                               delta=10.0,
                               msg="Dark current is not well retrieved from"
                                   " dark signal variance.")

        # Test that u_e_sat_area = u_e_sat / area
        self.assertAlmostEqual(self.results.u_e_sat_area,
                               self.results.u_e_sat /
                               self.dataset.cam.pixel_area,
                               delta=0.01)

        # Test that SNR_max is sqrt of u_e_sat
        self.assertAlmostEqual(self.results.SNR_max,
                               np.sqrt(self.results.u_e_sat),
                               delta=0.01)

        # Test that SNR_max_db is 20log_10(SNR_max)
        self.assertAlmostEqual(self.results.SNR_max_dB(),
                               20 * np.log10(self.results.SNR_max),
                               delta=0.01)

        # Test that SNR_max_bit is log_2(SNR_max)
        self.assertAlmostEqual(self.results.SNR_max_bit(),
                               np.log2(self.results.SNR_max),
                               delta=0.01)

        # Test that SNR_max inverse is 100 / SNR_max
        self.assertAlmostEqual(self.results.inverse_SNR_max(),
                               100 / self.results.SNR_max,
                               delta=0.01)

        # Test that DR is u_p_sat / u_p_min
        self.assertAlmostEqual(self.results.DR,
                               self.results.u_p_sat / self.results.u_p_min,
                               delta=0.01)

        # Test that DR_dB is 20log_10(DR)
        self.assertAlmostEqual(self.results.DR_dB(),
                               20 * np.log10(self.results.DR),
                               delta=0.01)

        # Test that DR_bit is log_2(DR)
        self.assertAlmostEqual(self.results.DR_bit(),
                               np.log2(self.results.DR),
                               delta=0.01)

        # Test that DSNU is sqrt(s2_ydark) / gain
        self.assertAlmostEqual(self.results.DSNU1288,
                               np.sqrt(self.results.s_2_y_dark) /
                               self.results.K,
                               delta=0.01)

        # Test that DSNU in DN is DSNU * K
        self.assertAlmostEqual(self.results.DSNU1288_DN(),
                               self.results.DSNU1288 * self.results.K,
                               delta=0.01)

        # Test that PRNU is the same as defined in EMVA1288 standard
        self.assertAlmostEqual(self.results.PRNU1288,
                               np.sqrt(self.results.s_2_y -
                                       self.results.s_2_y_dark) * 100 /
                               (np.mean(self.data.data['spatial']['avg']) -
                                np.mean(self.data.data['spatial']['avg_'
                                                                  'dark'])))

        # Test that histograms contains relevant keys and are numpy arrays
        hists = ('histogram_PRNU', 'histogram_PRNU_accumulated',
                 'histogram_DSNU', 'histogram_DSNU_accumulated')
        keys = ('bins', 'model', 'values')
        for hist in hists:
            h = getattr(self.results, hist)
            for key in keys:
                self.assertTrue(key in h.keys())
                self.assertTrue(isinstance(h[key], np.ndarray))

        # delete objects
        del_obj(self.dataset, self.parser, self.loader, self.data,
                self.results)

    def test_results_current_variation(self):
        dt, p, l, da, r = _init(height=self._height,
                                width=self._width,
                                bit_depth=self._bit_depth,
                                L=self._L,
                                steps=self._steps,
                                dark_current_ref=self._dark_current_ref,
                                exposure_fixed=self._exposure_fixed,
                                radiance_min=self._radiance_min,
                                exposure_max=self._exposure_max,
                                exposure_min=self._exposure_min,
                                temperature=self._temperature,
                                temperature_ref=self._temperature_ref,
                                K=self._K,
                                dsnu=self._dsnu,
                                prnu=self._prnu)
        self.dataset = dt
        self.parser = p
        self.loader = l
        self.data = da
        self.results = r

        data = self.data.data
        # Test that s_ydark is not a fit because only 1 texp
        self.assertAlmostEqual(self.results.sigma_y_dark,
                               np.sqrt(data['temporal']['s2_ydark'][0]),
                               delta=0.1)

        del_obj(self.dataset, self.parser, self.loader, self.data,
                self.results)

    def test_results_without_pixel_area(self):
        dt, p, l, da, r = _init(pixel_area=None,
                                height=self._height,
                                width=self._width,
                                bit_depth=self._bit_depth,
                                L=self._L,
                                steps=self._steps,
                                dark_current_ref=self._dark_current_ref,
                                exposure_max=self._exposure_max,
                                exposure_min=self._exposure_min,
                                temperature=self._temperature,
                                temperature_ref=self._temperature_ref,
                                K=self._K,
                                dsnu=self._dsnu,
                                prnu=self._prnu)
        self.dataset = dt
        self.parser = p
        self.loader = l
        self.data = da
        self.results = r

        # Test relevant properties are None
        self.assertIs(self.results.u_p_min_area, None)
        self.assertIs(self.results.u_e_min_area, None)
        self.assertIs(self.results.u_p_sat_area, None)
        self.assertIs(self.results.u_e_sat_area, None)

        del_obj(self.dataset, self.parser, self.loader, self.data,
                self.results)

    def test_nans(self):
        # Test that less than 2 texp will yield a NaN for u_I_mean
        data = {'temporal': {'texp': [0, 1]},
                'spatial': {}}
        r = Results1288(data)
        self.assertIs(r.u_I_mean, np.nan)

        # Test that a negative slope for t vs s2_ydark will yield Nan for
        # u_I_var
        data['temporal']['s2_ydark'] = [1, 0]
        r = Results1288(data)
        self.assertIs(r.u_I_var, np.nan)

        # Test that a negative s2y_dark will yield a Nan for DSNU1288
        data['spatial'] = {'avg_dark': [0, 0, 0],
                           'var_dark': [1, 1, 1],
                           'L_dark': 3}
        r = Results1288(data)
        self.assertIs(r.DSNU1288, np.nan)
        self.assertIs(r.DSNU1288_DN(), np.nan)

        del r
