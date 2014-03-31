import numpy as np
import logging


class Data1288(object):
    '''
    Take data from parsed images (descriptor file)
    and expose it as temporal and spatial dicts
    This dicts have the form appropiate for the processing
    '''

    def __init__(self,
                 data,
                 loglevel=logging.DEBUG):

        self.pixels = data['format']['width'] * data['format']['height']

        logging.basicConfig()
        self.log = logging.getLogger('Data')
        self.log.setLevel(loglevel)

        self.data = {}
        self.data['name'] = data.get('name', 'Unkown')
        self.data['format'] = data['format']
        self.data['temporal'] = self._get_temporal(data['temporal'])
        self.data['spatial'] = self._get_spatial(data['spatial'])

    def _get_float_keys(self, d, depth=2):
        r = {}
        for k in d.keys():
            if depth > 1:
                r[float(k)] = self._get_float_keys(d[k], depth - 1)
            else:
                r[float(k)] = d[k]
        return r

    def _get_temporal(self, data):
        '''
        Fill the temporal dict, with the stuff that we need.
        Compute the averages and variances from the sums (sum and pvar)
        '''
        temporal = {}

        bright = self._get_float_keys(data['bright'])
        dark = self._get_float_keys(data['dark'], depth=1)

        assert not set(bright.keys()) - set(dark.keys()), \
            'Dark and bright must have same exposures'

        texp = np.asarray(sorted(bright.keys()))
        temporal['texp'] = texp

        u_p = []
        u_y = []
        s2_y = []
        u_ydark = []
        s2_ydark = []

        for t in texp:
            photons = sorted(bright[t].keys())
            for p in photons:
                u_p.append(p)
                d = self._get_temporal_data(bright[t][p])
                u_y.append(d['mean'])
                s2_y.append(d['var'])

            d = self._get_temporal_data(dark[t])
            u_ydark.append(d['mean'])
            s2_ydark.append(d['var'])

        temporal['u_p'] = np.asarray(u_p)
        temporal['u_y'] = np.asarray(u_y)
        temporal['s2_y'] = np.asarray(s2_y)
        temporal['u_ydark'] = np.asarray(u_ydark)
        temporal['s2_ydark'] = np.asarray(s2_ydark)

        #In case we have only one exposure, we need arrays with the
        #same length as the up
        #we just repeat the same value over and over
        if len(temporal['texp']) == 1:
            l = len(temporal['u_p'])

            v = temporal['texp'][0]
            temporal['texp'] = np.asarray([v for _i in range(l)])

            v = temporal['u_ydark'][0]
            temporal['u_ydark'] = np.asarray([v for _i in range(l)])

            v = temporal['s2_ydark'][0]
            temporal['s2_ydark'] = np.asarray([v for _i in range(l)])

        return temporal

    def _get_temporal_data(self, d):
        '''Convert temporal image data to mean and variance

        The mean is the sum of the pixles of the two images divided by
        (2 * self.pixels)

        The variance is the pseudo variance(integer), divided by
        (4 * self.pixels)
        '''
        mean_ = d['sum'] / (2.0 * self.pixels)
        var_ = d['pvar'] / (4.0 * self.pixels)
        return {'mean': mean_, 'var': var_}

    def _get_spatial(self, data):
        '''Fill the spatial dict
        The images (sum and pvar) are preserved, they are needed for processing
        '''
        spatial = {}

        bright = self._get_float_keys(data['bright'])
        dark = self._get_float_keys(data['dark'], depth=1)

        assert not set(bright.keys()) - set(dark.keys()), \
            'Dark and bright must have same exposures'

        texp = np.asarray(sorted(bright.keys()))
        spatial['texp'] = texp

        u_p = []
        _sum = []
        _pvar = []
        _L = []
        _avg = []
        _var = []

        _sum_dark = []
        _pvar_dark = []
        _L_dark = []
        _avg_dark = []
        _var_dark = []

        for t in texp:
            photons = sorted(bright[t].keys())

            for p in photons:
                d = self._get_spatial_data(bright[t][p])

                u_p.append(p)
                _sum.append(d['sum'])
                _pvar.append(d['pvar'])
                _L.append(d['L'])
                _avg.append(d['avg'])
                _var.append(d['var'])

            d = self._get_spatial_data(dark[t])

            _sum_dark.append(d['sum'])
            _pvar_dark.append(d['pvar'])
            _L_dark.append(d['L'])
            _avg_dark.append(d['avg'])
            _var_dark.append(d['var'])

        spatial['u_p'] = np.asarray(u_p)
        spatial['sum'] = np.asarray(_sum)
        spatial['pvar'] = np.asarray(_pvar)
        spatial['L'] = np.asarray(_L)
        spatial['avg'] = np.asarray(_avg)
        spatial['var'] = np.asarray(_var)

        spatial['sum_dark'] = np.asarray(_sum_dark)
        spatial['pvar_dark'] = np.asarray(_pvar_dark)
        spatial['L_dark'] = np.asarray(_L_dark)
        spatial['avg_dark'] = np.asarray(_avg_dark)
        spatial['var_dark'] = np.asarray(_var_dark)

        return spatial

    def _get_spatial_data(self, d):
        '''Add the mean and variance to the spatial image data

        The mean is the sum of the images divided by L

        The variance is the pseudovariance divided by
        (L^2 * (L-1))
        '''
        sum_ = np.asarray(d['sum'], dtype=np.int64)
        pvar_ = np.asarray(d['pvar'], dtype=np.int64)
        L = int(d['L'])
        avg_ = sum_ / (1.0 * L)
        var_ = pvar_ / (1.0 * np.square(L) * (L - 1))

        return {'sum': sum_,
                'pvar': pvar_,
                'L': L,
                'avg': avg_,
                'var': var_}
