# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""Utils functions

"""
from __future__ import print_function
import numpy as np
import os
from scipy.optimize import leastsq
from lxml import etree
from PIL import Image
from collections import OrderedDict
# import cv2


SIGNIFICANT_DIGITS = 7


def load_image(fname):
    img = Image.open(fname)
    img = np.asarray(img.split()[0])
    # img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_UNCHANGED)
    # img = cv2.split(img)[0]
    return img


def get_int_imgs(imgs):
    '''
    Returns the sum and pseudo-variance from list of images
    sum is just the image resulting on the addition of all the images
    pvar is the pseudo-variance, this is
    pvar = SUM((Li - SUM(i))^2)
    to get variance from pseudo-variance
    var = (1/(L^2) * 1/(L - 1)) * pvar
    '''
    L = len(imgs)

    sum_ = 0
    sq_ = 0
    for img in imgs:
        # we force the images as int64 to make sure we do not clip
        i = img.astype(np.int64)
        sum_ += i
        sq_ += np.square(i)

    # the pseudo variance can be computed from the sum image and the sum of
    # the square images
    var_ = L * (L * sq_ - np.square(sum_))

    return {'L': L, 'sum': sum_, 'pvar': var_}


def LinearB0(Xi, Yi):
    X = np.asfarray(Xi)
    Y = np.asfarray(Yi)

    # we want a function y = m * x
    def fp(v, x):
        return x * v[0]

    # the error of the function e = x - y
    def e(v, x, y):
        return (fp(v, x) - y)

    # the initial value of m, we choose 1, because we thought YODA would
    # have chosen 1
    v0 = [1.0]

    vr, _success = leastsq(e, v0, args=(X, Y))

    # compute the R**2 (sqrt of the mean of the squares of the errors)
    err = np.sqrt(sum(np.square(e([vr], X, Y))) / (len(X) * len(X)))

    # Some versions of leastsq returns an array, other a scalar, so here we
    # make sure
    # it is an array
    val = np.array([vr]).flatten()

    return val, err


def LinearB(Xi, Yi):
    X = np.asfarray(Xi)
    Y = np.asfarray(Yi)

    # we want a function y = m * x + b
    def fp(v, x):
        return x * v[0] + v[1]

    # the error of the function e = x - y
    def e(v, x, y):
        return (fp(v, x) - y)

    # the initial value of m, we choose 1, because we thought YODA would
    # have chosen 1
    v0 = np.array([1.0, 1.0])

    vr, _success = leastsq(e, v0, args=(X, Y))

    # compute the R**2 (sqrt of the mean of the squares of the errors)
    err = np.sqrt(sum(np.square(e(vr, X, Y))) / (len(X) * len(X)))

#    print vr, success, err
    return vr, err


def GetImgShape(img):
    rows = 1

    if img.ndim == 1:
        cols, = img.shape
    else:
        rows, cols = img.shape
    return rows, cols


def FFT1288(m, rotate=False):
    mm = np.asfarray(np.copy(m))

    if rotate is True:
        mm = mm.transpose()

    _rows, cols = GetImgShape(mm)

    # This is just in case we are talking about really small or really
    # big arrays
    if (cols < 10) or (cols > 50000):
        return []

    # Substract the mean of the image
    mm = mm - np.mean(mm)

    # perform the fft in the x direction
    fft = np.fft.fft(mm, axis=1)

    fft = fft / np.sqrt(cols)

    fabs = np.real(fft * np.conjugate(fft))

    # extract the mean of each column of the fft
    r = np.mean(fabs, axis=0)

    # Return only half of the spectrogram (it is symemtrical)
    return r[: cols // 2]


def GetFrecs(fft):
    n = len(fft)
    x = np.arange(n)
    x = x * 1.0 / (2 * n)
    return x


def Histogram1288(img, Qmax):
    y = np.ravel(img)

    ymin = np.min(y)
    ymax = np.max(y)

    # Because we are working with integers, minimum binwidth is 1
    W = 1
    q = ymax - ymin
    Q = q + 1

    # When too many bins, create a new integer binwidth
    if Q > Qmax:
        # We want the number of bins as close as possible to Qmax (256)
        W = int(np.ceil(1. * q / (Qmax - 1)))
        Q = int(np.floor(1. * q / W)) + 1

    # The bins
    # we need one more value for the numpy histogram computation
    # numpy used bin limits
    # in our interpretation we use the lower limit of the bin
    B = [ymin + (i * W) for i in range(Q + 1)]

    # Normal distribution with the original sigma, and mean
    mu = np.mean(y)
    sigma = np.std(y)
    normal = ((1. * (ymax - ymin) / Q) *
              np.size(y) / (np.sqrt(2 * np.pi) * sigma) *
              np.exp(-0.5 * (1. / sigma * (B[:-1] - mu)) ** 2))


#############################################
#
#   # Reference algorithm, it's pretty slow
#   # the numpy version gives the same results
#
#     #The histogram container
#     H = np.zeros((Q,), dtype=np.int64)
#
#     #Histogram computation
#     for yi in y:
#         q = (yi - ymin) / W
#         H[q] += 1
#############################################

    H, _b = np.histogram(y, B, range=(ymin, ymax))

    return {'bins': np.asfarray(B[:-1]), 'values': H, 'model': normal}


def cls_1288_info(cls):
    """Dictionnary that represents results.

    Parameters
    ----------
    cls : Class from wich to extract the information.

    Returns
    -------
    dict :
        Dictionnary extracted using the format defined by
        a custom sphinx directive
        with the following format::

          {attribute1: {'section': section name,
                        'units': attribute units,
                        'short': attribute short description,
                        'latexname': latex name for the attribute,
                        'symbol': symbol to represent the value}}

    """

    d = OrderedDict()
    items = [name for name in sorted(cls.__dict__.keys())]
    for attribute_name in items:
        # Extract the doc from the Processing methods
        doc = getattr(cls, attribute_name).__doc__
        if not doc:
            continue

        # All the lines in the docstring
        lines = [s.strip() for s in doc.splitlines()]

        # to store the relevant tag lines
        tag_lines = []
        n = len(lines)

        flag = False
        for line in lines:
            # Get only those that are relevant (start with .. emva1288::)
            if line.startswith('.. emva1288::'):
                flag = True
                continue
            if flag:
                if not line.strip():
                    flag = False
                    continue
                tag_lines.append(line)
        # if there are not relevant lines skip and go to next method
        if not tag_lines:
            continue

        # To store the info from the doc
        attribute_info = {}
        for line in tag_lines:
            tags = [x.strip() for x in line.split(':', 2) if x.strip()]
            # Each valid tag has to be xx:yy
            if len(tags) != 2:
                continue
            # Fill the dict
            attribute_info[tags[0].lower()] = tags[1]

        # If there is no section set it as other
        attribute_info.setdefault('section', 'other')
        d[attribute_name] = attribute_info
    return d


def _sections_first(dct):
    """For backwards compatibility where we use to have
    results with section as first keys
    """
    d = OrderedDict()

    sections = sorted({k['section'] for k in dct.values()})
    for section in sections:
        d[section] = OrderedDict()
        for k, v in dct.items():
            if v['section'] != section:
                continue
            d[section][k] = v
    return d


def obj_to_dict(obj):
    '''
    Get the info dict from the object class
    Add the values or Data to this dict

    for each method if the return value is a dict, it is inserted as
    d[SectionName][MethodName][Data] = ReturnValue
    if not
    d[SectionName][MethodName][Value] = ReturnValue
    '''
    d = cls_1288_info(obj.__class__)
    for attribute in d.keys():
        # Get the value for the given attribute
        val = getattr(obj, attribute)
        if callable(val):
            val = val()

        if isinstance(val, dict):
            d[attribute]['data'] = val
        else:
            d[attribute]['value'] = val
    return d


def dict_to_xml(d, root='results', filename=None):
    '''
    Takes a dict and return a well formed xml string
    '''

    def key_to_xml(d, r):
        '''
        Recursive call to add the key/value from dict to the r element tree
        If the value is an array joint the values casted as strings
        with whitespaces separator
        if the value is something else, it is casted as string
        '''
        for k in d.keys():
            e = etree.SubElement(r, k)
            # when the value for the key is a dict, call the function again
            if isinstance(d[k], dict):
                r.append(key_to_xml(d[k], e))

            # if the value is an array
            # add the values of the array as a string separated by whitespaces
            # Note to self: add other array types as needed
            elif isinstance(d[k], np.ndarray):
                a = [str(x) for x in d[k]]
                e.text = ' '.join(a)

            # if something else, just add the corresponding string value
            else:
                e.text = str(d[k])
        return r

    tree = etree.Element(root)
    xml = key_to_xml(d, tree)

    t = etree.tostring(xml, pretty_print=True)
    if filename is None:
        return t
    with open(filename, 'w') as f:
        f.write(t.decode('utf-8'))


def xml_to_dict(xml):
    '''
    If xml is a file, opens and parse, if string, parse it from string
    Convert the xml to a dict using element_to_dict
    Process the resulting dict:
    Cast Data to numpy float arrays (split the string by whitespaces)
    Cast Value to float
    '''
    try:
        if os.path.isfile(xml):
            tree = etree.parse(xml)
        else:
            tree = etree.ElementTree.fromstring(xml)
    except:
        print('Problems loading XML')
        return None

    def element_to_dict(r):
        '''
        Recursive call to add dictionnary elements from the r xml element
        '''
        dout = {}
        for child in r:
            # The resulting keys are forced to lowercase to compoensate
            # for some windows versions that use CamelCase
            if list(child):
                # if the element has children call the function again with
                # children as r
                dout[child.tag.lower()] = element_to_dict(child)
            else:
                dout[child.tag.lower()] = child.text
        return dout

    root = tree.getroot()
    d = element_to_dict(root)
    # loop to reconstruct arrays from strings in Data elements
    for section, method in d.items():
        for methodname, value in method.items():
            if 'data' in value:
                for data in value['data']:
                    v = value['data'][data]
                    v = v.strip()
                    v = v.split()
                    d[section][methodname]['data'][data] = np.asfarray(v)
            else:
                v = value['value']
                if v in ('None', None, 'none'):
                    v = None
                else:
                    # sometimes the decimal point is written with , instead of.
                    v = v.replace(',', '.')
                    v = float(v)
                d[section][methodname]['value'] = v
    return d


def round_significant(v, sig=SIGNIFICANT_DIGITS):
    '''
    Round up to the given significant digits, used for comparison
    '''
    if v == 0.0:
        return 0.0
    return round(v, sig - np.int(np.floor(np.log10(np.abs(v)))) - 1)


round_array = np.vectorize(round_significant)


def compare_xml(x1, x2, filename=None):
    # load the xml into dicts
    f1 = xml_to_dict(x1)
    f2 = xml_to_dict(x2)
    s = ''

    # if something is wrong abort
    if f1 is None or f2 is None:
        return s

    c1 = list(f1.keys())
    c2 = list(f2.keys())

    # loop throught the combined categories
    categories = set(c1) | set(c2)
    for category in sorted(categories):
        s += '*' * 70 + '\n'
        s += category + '\n'
        s += '*' * 70 + '\n'
        # check if missing category in one of the dicts
        if category not in c1 or category not in c2:
            t1 = category in c1
            t2 = category in c2
            s += '{0:<35}'.format('PRESENT')
            s += '{0:<20}{1:<20}FAIL'.format(str(t1), str(t2)) + '\n'
            continue

        m1 = f1[category].keys()
        m2 = f2[category].keys()
        # loop throught the combined methodnames
        methodnames = set(m1) | set(m2)
        for methodname in sorted(methodnames):
            s += '{0:<35}'.format(methodname)

            # check if methodname in dict
            if methodname not in m1:
                v1 = None
                a1 = None
            # get the value and the data
            else:
                v1 = f1[category][methodname].get('value', None)
                a1 = f1[category][methodname].get('data', None)

            if methodname not in m2:
                v2 = None
                a2 = None
            else:
                v2 = f2[category][methodname].get('value', None)
                a2 = f2[category][methodname].get('data', None)

            # first we check for values and then for data
            # if both present only values will be taken in count for comparison
            # If both are values
            if v1 is not None and v2 is not None:
                try:
                    r = (round_significant(v1) - round_significant(v2)) == 0.0
                except:
                    r = False
                t1 = v1
                t2 = v2

            # if both are arrays
            elif a1 is not None and a2 is not None:
                k1 = a1.keys()
                k2 = a2.keys()
                t1 = 'Array'
                t2 = 'Array'

                # if different keys, is invalid
                if (set(k1) ^ set(k2)):
                    r = False
                else:
                    # loop throught the keys
                    for k in k1:
                        try:
                            r = np.max(np.abs(round_array(a1[k]) -
                                              round_array(a2[k]))) == 0.0
                        except:
                            r = False

                        if not r:
                            break
            else:
                t1 = str(v1)
                t2 = str(v2)
                r = False

            s += '{0:<20}{1:<20}'.format(t1, t2)
            if r:
                s += 'OK' + '\n'
            else:
                s += 'FAIL' + '\n'

        s += '\n'

    if filename is None:
        return s
    with open(filename, 'w') as f:
        f.write(s)
        return s
