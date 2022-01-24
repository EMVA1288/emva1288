import pytest
from emva1288.process.results import Results1288
from emva1288.process.routines import high_pass_filter
import numpy as np


@pytest.fixture(scope="function")
def results(data):
    dataset, parser, loader, data = data
    results = Results1288(data.data, pixel_area=dataset.cam.pixel_area)
    return dataset, parser, loader, data, results


def test_properties(results):
    dataset, parser, loader, data, results = results
    # test that quantification noise is really 1/12
    assert results.s2q == 1.0 / 12.0

    # test that indexes are integers and in good range
    for attr in ('index_start', 'index_u_ysat', 'index_sensitivity_max',
                 'index_sensitivity_min', 'index_linearity_min',
                 'index_linearity_max'):
        value = getattr(results, attr)
        assert isinstance(value, (int, np.int64))
        assert value < len(data.data['temporal']['u_y'])
        assert value >= 0
        assert attr in results.results.keys()

    # Test that EMVA values are float and positive
    for a in ('s2q', 'R', 'K', 'QE', 'sigma_y_dark', 'sigma_d', 'u_p_min',
              'u_p_min_area', 'u_e_min', 'u_e_min_area', 'u_p_sat',
              'u_p_sat_area', 'u_e_sat', 'SNR_max', 'DR', 'LE_min',
              'LE_max', 'u_I_var', 'u_I_mean', 'sigma_2_y_stack',
              'sigma_2_y_stack_dark', 's_2_y_measured', 's_2_y',
              's_2_y_dark', 'DSNU1288', 'PRNU1288'):
        value = getattr(results, a)
        assert isinstance(value, float)
        if not a == 's2q':
            assert a in results.results
        # except for linearity errors and dark currents,
        # everything always should be positive
        if a not in ('LE_min', 'LE_max') and value is not np.nan:
            assert value >= 0.0


def test_sensitivity(results):
    dataset, parser, loader, data, results = results
    ###############################################################
    # The following deltas are purely guesstimates and are prone to
    # errors in the future if they are not really significant
    ###############################################################

    # Test quantum efficiency is retrieved with a +/- 5% incertainty
    assert dataset.cam._qe.qe.mean() * 100 == pytest.approx(results.QE, abs=10)
    # Test that overall system gain
    # is retrieved with a +/- 0.01 incertainty
    assert dataset.cam.K == pytest.approx(results.K, abs=0.1)
    assert results.inverse_K() == 1 / results.K
    # # Test that responsivity is coherent with QE and system gain
    assert results.R == pytest.approx(results.QE * results.K / 100, abs=0.001)
    # division errors compensation


@pytest.mark.parametrize("dataset", ['multi_exposure'], indirect=True, scope='function')
def test_dark_current(results):
    dataset, parser, loader, data, results = results
    # Test that dark current is actually retrieved from both methods
    assert dataset.cam._dark_current_ref == pytest.approx(results.u_I_mean, abs=5)
    assert dataset.cam._dark_current_ref == pytest.approx(results.u_I_var, abs=10)


def test_saturation(results):
    dataset, parser, loader, data, results = results
    # Test that u_e_sat_area = u_e_sat / area
    assert results.u_e_sat_area == results.u_e_sat / dataset.cam.pixel_area


@pytest.mark.parametrize("dataset", ['single_exposure'], indirect=True)
def test_SNR(results):
    dataset, parser, loader, data, results = results
    # Test that u_e_sat_area = u_e_sat / area
    assert results.u_e_sat_area == results.u_e_sat / dataset.cam.pixel_area

    # Test that SNR_max is sqrt of u_e_sat
    assert results.SNR_max == np.sqrt(results.u_e_sat)

    # Test that SNR_max_db is 20log_10(SNR_max)
    assert results.SNR_max_dB() == 20 * np.log10(results.SNR_max)

    # Test that SNR_max_bit is log_2(SNR_max)
    assert results.SNR_max_bit() == np.log2(results.SNR_max)

    # Test that SNR_max inverse is 100 / SNR_max
    assert results.inverse_SNR_max() == 100 / results.SNR_max


def test_DR(results):
    dataset, parser, loader, data, results = results
    # Test that DR is u_p_sat / u_p_min
    assert results.DR == results.u_p_sat / results.u_p_min

    # Test that DR_dB is 20log_10(DR)
    assert results.DR_dB() == 20 * np.log10(results.DR)

    # Test that DR_bit is log_2(DR)
    assert results.DR_bit() == np.log2(results.DR)


@pytest.mark.parametrize("dataset", ['single_exposure'], indirect=True)
def test_DSNU(results):
    dataset, parser, loader, data, results = results
    # Test that DSNU is sqrt(s2_ydark) / gain
    assert results.DSNU1288 == np.sqrt(results.s_2_y_dark) / results.K

    # Test that DSNU in DN is DSNU * K
    assert results.DSNU1288_DN() == results.DSNU1288 * results.K


@pytest.mark.parametrize("dataset", ['single_exposure'], indirect=True)
def test_PRNU(results):
    dataset, parser, loader, data, results = results
    # Test that PRNU is the same as defined in EMVA1288 standard
    print(results.s_2_y - results.s_2_y_dark)
    assert results.PRNU1288 == (np.sqrt(results.s_2_y - results.s_2_y_dark) * 100 / (data.data['spatial']['avg_mean'] - data.data['spatial']['avg_mean_dark']))


def test_histograms(results):
    dataset, parser, loader, data, results = results
    # Test that histograms contains relevant keys and are numpy arrays
    hists = ('histogram_PRNU', 'histogram_PRNU_accumulated',
             'histogram_DSNU', 'histogram_DSNU_accumulated')
    keys = ('bins', 'model', 'values')
    for hist in hists:
        h = getattr(results, hist)
        for key in keys:
            assert key in h.keys()
            assert isinstance(h[key], np.ndarray)


@pytest.mark.parametrize("dataset", ['single_exposure'], indirect=True)
def test_results_current_variation(results):
    dataset, parser, loader, data, results = results
    # Test that s_ydark is not a fit because only 1 texp
    assert results.sigma_y_dark == pytest.approx(np.sqrt(data.data['temporal']['s2_ydark'][0]), abs=0.1)


def test_results_without_pixel_area(data):
    dataset, parser, loader, data = data
    results = Results1288(data.data, pixel_area=None)

    assert results.u_p_min_area is None
    assert results.u_e_min_area is None
    assert results.u_p_sat_area is None
    assert results.u_e_sat_area is None


def test_nans():
    # Test that less than 2 texp will yield a NaN for u_I_mean
    data = {'temporal': {'texp': [0, 1]},
            'spatial': {}}
    r = Results1288(data)
    assert r.u_I_mean is np.nan

    # Test that a negative slope for t vs s2_ydark will yield Nan for
    # u_I_var
    data['temporal']['s2_ydark'] = [1, 0]
    r = Results1288(data)
    assert r.u_I_var is np.nan

    # Test that a negative s2y_dark will yield a Nan for DSNU1288
    data['spatial'] = {'avg_var_dark': 0.,
                       'var_mean_dark': 1.,
                       'L_dark': 3}
    r = Results1288(data)
    assert r.DSNU1288 is np.nan
    assert r.DSNU1288_DN() is np.nan


def test_highpass_filter():
    # instantiate a bayer filter and a wrong filter
    bayer_filter = np.tile([[0, 1], [1, 0]], (500, 500))
    wrong_filter = np.tile([[1, 1, 1, 0], [1, 0, 0, 1]], (500, 250))

    # Source images. imgc is for color image (image with a bayer filter)
    img = np.abs(np.random.random([1000, 1000]))
    imgc = np.ma.array(img, mask=bayer_filter)

    # Image wrong's purpose is to assert that a ValueError is raised
    img_wrong = np.ma.array(img, mask=wrong_filter)

    # Pass the highpass filter on the source images
    res = high_pass_filter(img, 5)
    img_filt = res['img']/res['multiplicator']
    res = high_pass_filter(imgc, 5)
    imgc_filt = res['img']/res['multiplicator']

    # To assert that the highpass filter does its job, a low frequency
    # signal will be introduced in the image
    signal = 10 * np.sin(np.linspace(0, np.pi, 1000))
    sig_img = [line + val for line, val in zip(img, signal)]
    sig_imgc = np.ma.array(sig_img, mask=bayer_filter)

    # filtering the images with a low frequency signal
    res = high_pass_filter(sig_img, 5)
    sig_img_filt = res['img']/res['multiplicator']
    res = high_pass_filter(sig_imgc, 5)
    sig_imgc_filt = res['img']/res['multiplicator']

    # Assert raises in case of bad kernel size or bad mask
    with pytest.raises(ValueError):
        high_pass_filter(img_wrong, 5)
    with pytest.raises(ValueError):
        high_pass_filter(imgc, 6)

    # assert that the means of the filtered images are almost 0
    assert 0 == pytest.approx(np.mean(imgc_filt), abs=3)
    assert 0 == pytest.approx(np.mean(img_filt), abs=3)

    # assert that the means of the filtered images are almost 0 even with
    # a low frequency signal
    assert 0 == pytest.approx(np.mean(sig_img_filt), abs=3)
    assert 0 == pytest.approx(np.mean(sig_imgc_filt), abs=3)

    # assert that the filtered images with and without a signal are almost
    # equal
    assert np.allclose(sig_img_filt, img_filt, rtol=0, atol=10e-4)
    assert np.allclose(sig_imgc_filt, imgc_filt, rtol=0, atol=10e-4)
