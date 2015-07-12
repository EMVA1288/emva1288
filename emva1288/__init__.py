# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""EMVA 1288 Package initialisation file"""

from .data import Data1288
from .loader import LoadImageData
from .parser import ParseEmvaDescriptorFile
from .results import Results1288
from .plotting import Plotting1288
from .report import report


class Emva1288(object):
    def __init__(self, fname):
        info = ParseEmvaDescriptorFile(fname)
        imgs = LoadImageData(info.info)
        dat = Data1288(imgs.data)
        self._results = Results1288(dat)

    def results(self):
        return self._results.print_results()

    def plot(self):
        plot = Plotting1288(self._results)
        plot.plot()
