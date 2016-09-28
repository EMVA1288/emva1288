# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""EMVA 1288 Package initialisation file"""

from emva1288.process.data import Data1288
from emva1288.process.loader import LoadImageData
from emva1288.process.parser import ParseEmvaDescriptorFile
from emva1288.process.results import Results1288
from emva1288.process.plotting import Plotting1288


class Emva1288(object):
    def __init__(self, fname):
        parser = ParseEmvaDescriptorFile(fname)
        imgs = LoadImageData(parser.images)
        dat = Data1288(imgs.data)
        self._results = Results1288(dat.data)

    def results(self):
        self._results.print_results()

    def plot(self, *plots):
        plot = Plotting1288(self._results)
        plot.plot(*plots)

    def xml(self, filename=None):
        return self._results.xml(filename)
