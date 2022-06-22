import pytest
import numpy as np

from emva1288.process import routines


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
    res = routines.high_pass_filter(img, 5)
    img_filt = res['img']/res['multiplicator']
    res = routines.high_pass_filter(imgc, 5)
    imgc_filt = res['img']/res['multiplicator']

    # To assert that the highpass filter does its job, a low frequency
    # signal will be introduced in the image
    signal = 10 * np.sin(np.linspace(0, np.pi, 1000))
    sig_img = [line + val for line, val in zip(img, signal)]
    sig_imgc = np.ma.array(sig_img, mask=bayer_filter)

    # filtering the images with a low frequency signal
    res = routines.high_pass_filter(sig_img, 5)
    sig_img_filt = res['img']/res['multiplicator']
    res = routines.high_pass_filter(sig_imgc, 5)
    sig_imgc_filt = res['img']/res['multiplicator']

    # Assert raises in case of bad kernel size or bad mask
    with pytest.raises(ValueError):
        routines.high_pass_filter(img_wrong, 5)
    with pytest.raises(ValueError):
        routines.high_pass_filter(imgc, 6)

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


def test_FFT1288_masked():
    """ Make sure the masking happens properly in the fft calculation """
    rows = 512
    cols = 128
    bayer_filter = np.tile([[0, 1], [1, 0]], (rows//2, cols//2))
    img = np.abs(np.random.random([rows, cols]))
    # apply column offset to some bottom rows only and make sure we see it in the fft
    img[128::, ::8] += 10
    fft = routines.FFT1288(img_=np.ma.array(img, mask=bayer_filter))
    assert (fft > 1).any()
