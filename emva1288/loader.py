import os
import numpy as np
from emva1288 import routines
import logging


class LoadImageData(object):
    '''Using a info dictionary, load the images and fill data with
    the relevant information

    Parameters
    ----------
    info: Dict result of ParseEmvaDescriptorFile
    loglevel: Logging level
    fload: function used to load the images
    fload_args: arguments passed to fload
    fload_kwargs: keyword arguments passed to fload
    path: path to load the images from, if None, the path in info is used
    '''

    def __init__(self,
                 info,
                 loglevel=logging.DEBUG,
                 fload=routines.load_image,
                 fload_args=[],
                 fload_kwargs={},
                 path=None):

        self.data = {'version': None,
                    'format': {},  # bits, witdth, height
                    'name': None,
                    'info': {},
                    'temporal': {'dark': {}, 'bright': {}},
                    'spatial': {'dark': {}, 'bright': {}},
                    }
        self._fload = fload
        self._fload_args = fload_args
        self._fload_kwargs = fload_kwargs

        logging.basicConfig()
        self.log = logging.getLogger('Loader')
        self.log.setLevel(loglevel)

        if path is not None:
            self._path = path
        else:
            self._path = info['path']
        self._load_data(info)

    def _load_data(self, info):
        '''Using the information in self.info fill self.data loading the images
        and filling the temporal and spatial dicts
        '''

        self.data['version'] = info['version']
        self.data['format'] = info['format']
        self.data['name'] = info['filename']
        self.data['camera_info'] = info['camera_info']
        self.data['operation_point_info'] = info['operation_point_info']

        for kind in ('temporal', 'spatial'):
            b_exp = set(info[kind]['bright'].keys())
            d_exp = set(info[kind]['dark'].keys())

            if  b_exp - d_exp:
                raise SyntaxError('%s Bright and dark must have the '
                                  'same exposures' % kind)

            for exposure in b_exp:
                for photons in info[kind]['bright'][exposure]:
                    fnames = info[kind]['bright'][exposure][photons]
                    data_imgs = self._get_imgs_data(fnames, kind)
                    self.data[kind]['bright'].setdefault(exposure, {})
                    self.data[kind]['bright'][exposure][photons] = data_imgs

                fnames = info[kind]['dark'][exposure]
                data_imgs = self._get_imgs_data(fnames, kind)
                self.data[kind]['dark'][exposure] = data_imgs

    def _get_imgs_data(self, fnames, kind):
        '''Return the desired image data
        This depends on the kind of data
        '''
        arr_imgs = self._load_imgs(fnames)
        imgs = routines.get_int_imgs(arr_imgs)
        #For spatial we want the images
        if kind != 'temporal':
            return imgs
        #For temporal, we want numbers
        d = {}
        d['sum'] = np.sum(imgs['sum'])
        d['pvar'] = np.sum(imgs['pvar'])
        return d

    def _load_imgs(self, fnames):
        '''For a list of images, load and append them to the return array
        '''
        imgs = []
        for fname in fnames:
            filename = os.path.join(self._path, fname)
            self.log.debug('Loading ' + fname)
            if not os.path.isfile(filename):
                raise IOError('Not such file: ' + filename)
            imgs.append(self._fload(filename,
                                    *self._fload_args,
                                    **self._fload_kwargs))

        return imgs
