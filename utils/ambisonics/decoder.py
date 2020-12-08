from utils.ambisonics.common import spherical_harmonics_matrix
from utils.ambisonics.position import Position
from utils.ambisonics.common import AmbiFormat
import numpy as np

DECODING_METHODS = ['projection', 'pseudoinv']
DEFAULT_DECODING = 'projection'


class AmbiDecoder(object):
    def __init__(self, speakers_pos, ambi_format=AmbiFormat(), method=DEFAULT_DECODING):
        assert method in DECODING_METHODS
        if isinstance(speakers_pos, Position):
            speakers_pos = [speakers_pos]
        assert isinstance(speakers_pos, list) and all([isinstance(p, Position) for p in speakers_pos])
        self.speakers_pos = speakers_pos
        self.sph_mat = spherical_harmonics_matrix(speakers_pos,
                                                  ambi_format.order,
                                                  ambi_format.ordering,
                                                  ambi_format.normalization)
        self.method = method
        if self.method == 'pseudoinv':
            self.pinv = np.linalg.pinv(self.sph_mat)

    def decode(self, ambi):
        if self.method == 'projection':
            return np.dot(self.sph_mat, ambi)
        if self.method == 'pseudoinv':
            return np.dot(self.pinv.T, ambi)

def decode_ambix(ambi, pos, ambi_format=AmbiFormat(), method=DEFAULT_DECODING):
    return AmbiDecoder(speakers_pos=pos, ambi_format=ambi_format, method=method).decode(ambi)