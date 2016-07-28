# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can
# be found in the LICENSE file.

"""EMVA 1288 descriptor parser
This class takes an EMVA1288 descriptor file and loads its content into a
python dictionary.

An EMVA1288 descriptor file is a file that contains the description of an
EMVA1288 test including exposure times, photon count and corresponding images
"""

import numpy as np
import os
import pprint
import logging


class ParseEmvaDescriptorFile(object):
    """Take an image descriptor file and transform it into
    an usable directory
    """

    def __init__(self, filename, path=None, loglevel=logging.INFO):
        """Parser init method.

        Uses a :class:`python:logging.Logger` object to print infos of the
        parcing process. This method :meth:`loads <_load_file>` the file
        and :meth:`gets <_fill_info>` the information from it.

        Parameters
        ----------
        filename : str
                   The descriptor file's name or the complete path to it.
        path : str, optional
               The path to the descriptor file.
        loglevel : int, optional
                   The logger level.
        """
        # The items are in the form of
        # exposure:{photons:[fname1, fname2,...]}, photons....}
        # for dark, the number of photons is 0.0

        # If no path is given, the filename path will be used to fill
        # the images dict
        self._path = path

        self.format = {}  # bits, witdth, height
        self.version = None
        self.images = {'temporal': {},
                       'spatial': {}}

        logging.basicConfig()
        self.log = logging.getLogger('Parser')
        self.log.setLevel(loglevel)

        self._load_file(filename)
        self._fill_info()
        self.log.debug(pprint.pformat(self.images))

    def _get_images_filenames(self):
        """
        From the current line in self._lines array
        get all the consecutive "i filename"
        if less than 2 consecutive, raise an error
        """
        fnames = []
        while self._lines:
            line = self._lines.pop()
            l = self._split_line(line)
            if l[0] != 'i':
                # Ups, to the end of images, reappend last line that is not an
                # image line
                self._lines.append(line)
                break
            if len(l) != 2:  # pragma: no cover
                raise SyntaxError('Wrong format: "%s" should be "i filename"'
                                  % line)
                break
            # append image path to fnames
            npath = os.path.normpath(l[1])
            path = os.path.join(self._path, *npath.split('\\'))
            fnames.append(path)

        if len(fnames) < 2:  # pragma: no cover
            raise SyntaxError('Each image series, has to '
                              'have at least two images')

        return fnames

    def _get_kind(self, fnames):
        """
        Guess what kind of data based on the number of images
        Temporal = 2 images for each measurement point
        Spatial = >2 images for each measurement point
        """
        L = len(fnames)
        if L == 2:
            kind = 'temporal'
        else:
            kind = 'spatial'

        return kind

    def _add_pcount(self, exposure, photons, fnames):
        """Add images to a given exposure/phton

        For a given exposure and photon count
        add the appropiate image filenames to the self.images dict
        """
        # is it temporal or spatial data
        kind = self._get_kind(fnames)
        # create the exposure time dictionary for this exposure time
        # if it is not already existing
        self.images[kind].setdefault(exposure, {})

        # if this dict for this exposure time and this
        # photons count already existed, raise an error in order to not
        # overwrite existing data.
        if photons in self.images[kind][exposure]:  # pragma: no cover
            raise SyntaxError('Only one set of images exp %.3f photons %.3f'
                              % (exposure, photons))

        # append the images path to a dict whose key is the photons count
        # inside the exposure time dict
        self.images[kind][exposure][photons] = fnames

    def _fill_info(self):
        """
        Iterate through all the lines in the descriptor file
        and parse them by their first character. Fill self.images
        """

        # Start at the end of the file
        self._lines.reverse()
        while self._lines:
            # pop it such that other methods know the current processed line
            line = self._lines.pop()
            # check line is good format and split it
            l = self._split_line(line)

            # descriptor file supposed format ##
            # n bits width height
            # b exposureTime(ns) numberPhotons  (bright image)
            # i relativePathToTheImage
            # d exposureTime(ns)                (dark image)

            if l[0] == 'v':
                # if line starts with 'v', this is the version
                self.version = l[1]
                self.log.info('Version ' + l[1])
                continue

            if l[0] == 'n':
                # for lines that starts with n, there is always 4 elements
                # n + bits + width + height
                # There should be only one of this line in the file
                if len(l) != 4:  # pragma: no cover
                    raise SyntaxError('Wrong format: "%s" should be "n bits '
                                      'width height"' % line)

                if self.format:  # pragma: no cover
                    # if it is the second line found
                    # of this type raise error
                    raise SyntaxError('Only one "n bits width height" is '
                                      'allowed per file')

                self.format['bits'] = int(l[1])
                self.format['width'] = int(l[2])
                self.format['height'] = int(l[3])
                continue

            if l[0] == 'b':
                # For lines that starts with b. there is always 3 elements
                # b + exposureTime + numberPhotons (bright images)
                if len(l) != 3:  # pragma: no cover
                    raise SyntaxError('Wrong format: "%s" should be "b '
                                      'exposure photons"' % line)

                # Replace floating point representation if wrong format.
                exposure = np.float(l[1].replace(',', '.'))
                photons = np.float(l[2].replace(',', '.'))
                # For this settings, get all the corresponding images
                fnames = self._get_images_filenames()
                # Add the images path to the images[kind][exposure][photons]
                # dictionary where kind = temporal or spatial
                self._add_pcount(exposure, photons, fnames)

                continue

            if l[0] == 'd':
                # For lines that starts with d, there is always 2 elements
                # d + exposureTime (dark images)
                if len(l) != 2:  # pragma: no cover
                    raise SyntaxError('Wrong format: "%s" should be "d '
                                      'exposure"' % line)

                # replace floating point representation if wring format
                exposure = np.float(l[1].replace(',', '.'))
                # For this exposure, get all the corresponding images
                fnames = self._get_images_filenames()
                # Add the images path to the images dict.
                self._add_pcount(exposure, np.float(0.0), fnames)

                continue

            # If line is of an unknown format, warn user.
            self.log.warning('Unknown command ' + line)  # pragma: no cover

    def _split_line(self, line):
        """
        For every line of descriptorfile
        check that it has at least two arguments
        split by white spaces and strip white spaces from elements
        """
        l = [x.strip() for x in line.split()]
        if (not l) or (len(l) < 2):  # pragma: no cover
            raise SyntaxError('Wrong format line: %s' % line)
        return l

    def _load_file(self, filename):
        """
        Load a file, split by lines removing the comments (starts with #)
        """
        self.log.info('Opening ' + filename)
        f = open(filename, 'r')

        # To add location when opening images
        # If no path was passed as kwarg, set it to the filename path
        if self._path is None:
            self._path = os.path.dirname(filename)

        # get the lines and strip them if they are not comments
        try:
            self._lines = [x.strip() for x in f.readlines() if x.strip() and
                           not x.strip().startswith('#')]
        except UnicodeDecodeError:  # pragma: no cover
            # If there is an unknown character in the file, speak it!
            raise UnicodeDecodeError("File: '%s', has non-utf8 characters."
                                     "Find them and kill them!" % filename)
