# -*- coding: utf-8 -*-
# Copyright (c) 2014 The EMVA1288 Authors. All rights reserved.
# Use of this source code is governed by a GNU GENERAL PUBLIC LICENSE that can be
# found in the LICENSE file.

"""EMVA 1288 descriptor parser
This class takes an EMVA1288 descriptor file and loads its content into a python dictionary.

An EMVA1288 descriptor file is a file that contains the description of an EMVA1288 test
including exposure times, photon count and corresponding images

"""

import numpy as np
import os
import pprint
import logging


class ParseEmvaDescriptorFile(object):
    '''Take an image descriptor file and transform it into
    an usable directory
    '''

    def __init__(self, filename, loglevel=logging.DEBUG):
        #For dark the items are in the form of
        #exposure:{[fname1, fname2, ...]}
        #for bright the items are in the form of
        #exposure:{photons:[fname1, fname2,...]}, photons....}

        self.info = {'filename': None,
                     'path': None,
                     'version': None,
                     'format': {},  # bits, witdth, height
                     'temporal': {'dark': {}, 'bright': {}},
                     'spatial': {'dark': {}, 'bright': {}},
                     'camera_info': {},
                     'operation_point_info': {},
                     }

        logging.basicConfig()
        self.log = logging.getLogger('Parser')
        self.log.setLevel(loglevel)

        self._load_file(filename)
        self._fill_info()
        self.log.debug(pprint.pformat(self.info))

    def _get_images_filenames(self):
        '''
        From the current line in self._lines array
        get all the consecutive "i filename"
        if less than 2 consecutive, raise an error
        '''
        fnames = []
        while self._lines:
            line = self._lines.pop()
            l = self._split_line(line)
            if l[0] != 'i':
                #Ups, to the end of images, reappend last line that is not an
                #image line
                self._lines.append(line)
                break
            if len(l) != 2:
                raise SyntaxError('Wrong format: "%s" should be "i filename"'
                                  % line)
                break

            npath = os.path.normpath(l[1])
            path = os.path.join(*npath.split('\\'))
            fnames.append(path)

        if len(fnames) < 2:
            raise SyntaxError('Each image series, has to '
                              'have at least two images')

        return fnames

    def _get_kind(self, fnames):
        '''
        Guess what kind of data based on the number of images
        '''
        L = len(fnames)
        if L == 2:
            kind = 'temporal'
        else:
            kind = 'spatial'

        return kind

    def _add_bright_pcount(self, exposure, photons, fnames):
        '''
        For a given exposure and photon count
        add the appropiate image filenames to the self.info dict
        '''
        kind = self._get_kind(fnames)
        step = self.info[kind]['bright'].setdefault(exposure, {})

        if photons in step:
            raise SyntaxError('Only one set of bright images per photon count '
                              '%.3f' % (photons))

        step[photons] = fnames

    def _add_dark_pcount(self, exposure, fnames):
        kind = self._get_kind(fnames)

        if exposure in self.info[kind]['dark']:
            raise SyntaxError('Only one set of images per dark exposure %.3f'
                              % (exposure))
        self.info[kind]['dark'][exposure] = fnames

    def _fill_info(self):
        '''
        Walk trhought all the lines in the descriptor file
        by the first character fill self.info
        This info is what should be the descriptor
        (too late to change to this?)
        '''

        self._lines.reverse()
        while self._lines:
            line = self._lines.pop()
            l = self._split_line(line)

            if l[0] == 'v':
                self.info['version'] = l[1]
                self.log.info('Version ' + l[1])
                continue

            if l[0] == 'n':
                if len(l) != 4:
                    raise SyntaxError('Wrong format: "%s" should be "n bits '
                                      'width height"' % line)

                if self.info.get('format'):
                    raise SyntaxError('Only one "n bits width height" is '
                                      'allowed per file')

                self.info['format']['bits'] = int(l[1])
                self.info['format']['width'] = int(l[2])
                self.info['format']['height'] = int(l[3])
                continue

            if l[0] == 'b':
                if len(l) != 3:
                    raise SyntaxError('Wrong format: "%s" should be "b '
                                      'exposure photons"' % line)

                exposure = np.float(l[1].replace(',', '.'))
                photons = np.float(l[2].replace(',', '.'))
                fnames = self._get_images_filenames()
                self._add_bright_pcount(exposure, photons, fnames)

                continue

            if l[0] == 'd':
                if len(l) != 2:
                    raise SyntaxError('Wrong format: "%s" should be "d '
                                      'exposure"' % line)

                exposure = np.float(l[1].replace(',', '.'))
                fnames = self._get_images_filenames()
                self._add_dark_pcount(exposure, fnames)

                continue

            #Data for datasheet operation point starts with o
            if l[0] == 'o':
                if len(l) < 2:
                    raise SyntaxError('Wrong format: %s should be "l varname '
                                      'values"' % line)
                self.info['operation_point_info'][l[1]] = ' '.join(l[2:])
                continue

            #Data for datasheet camera starts with c
            if l[0] == 'c':
                if len(l) < 2:
                    raise SyntaxError('Wrong format: %s should be "l varname '
                                      'values"' % line)
                self.info['camera_info'][l[1]] = ' '.join(l[2:])
                continue

            self.log.warning('Unknown command ' + line)

    def _split_line(self, line):
        '''
        For every line of descriptorfile
        check that it has at least two arguments
        split by white spaces and strip white spaces from elements
        '''
        l = [x.strip() for x in line.split()]
        if (not l) or (len(l) < 2):
            raise SyntaxError('Wrong format line: %s' % line)
        return l

    def _load_file(self, filename):
        '''
        Load a file, split by lines removing the comments (starts with #)
        '''
        self.log.debug('Opening ' + filename)
        f = open(filename, 'r')

        #To add location when opening images
        path = os.path.realpath(filename)
        self.info['path'] = os.path.dirname(path)
        self.info['filename'] = os.path.basename(path)

        self._lines = [x.strip() for x in f.readlines() if x.strip() and
                       not x.strip().startswith('#')]
