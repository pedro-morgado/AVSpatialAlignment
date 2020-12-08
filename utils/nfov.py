# Adapted from https://github.com/NitishMutha/equirectangular-toolbox

# Copyright 2017 Nitish Mutha (nitishmutha.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import numpy as np

class NFOV():
    def __init__(self, height=400, width=800):
        self.FOV = [0.45, 0.45]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.fov_coord_rads = self._get_coord_rad(self._get_screen_img(), isScreenPts=True)

    def _get_coord_rad(self, points, isScreenPts=False):
        coord_rads = (points * 2 - 1) * np.array([self.PI, self.PI_2])
        if isScreenPts:
            coord_rads *= np.array(self.FOV)
        return coord_rads

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width, endpoint=False), np.linspace(0, 1, self.height, endpoint=False))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        rou[rou <= 1e-4] = 1e-4  # fix for zero rou
        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        # lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)
        lon = np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, frame, screen_coord):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        frame_channel = frame.shape[2]

        uf = np.mod(screen_coord.T[0], 1) * frame_width  # long - width
        vf = np.mod(screen_coord.T[1], 1) * frame_height  # lat - height

        disp_x = int((self.cp[0] / self.PI + 1) / 2 * frame_width - frame_width//2)
        frame = np.roll(frame, -disp_x, 1)

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        x2 = np.mod(x2, frame_width)
        y2 = np.minimum(y2, frame_height-1)

        base_y0 = np.multiply(y0, frame_width)
        base_y2 = np.multiply(y2, frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(frame, [-1, frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        return nfov

    def _nn_interpolation(self, frame, screen_coord):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        uf = np.mod(screen_coord.T[0], 1) * frame_width  # long - width
        vf = np.mod(screen_coord.T[1], 1) * frame_height  # lat - height

        disp_x = int((self.cp[0] / self.PI + 1) / 2 * frame_width - frame_width//2)
        frame = np.roll(frame, -disp_x, 1)

        x = np.round(uf).astype(int)
        y = np.round(vf).astype(int)

        x = np.mod(x, frame_width)
        y = np.minimum(y, frame_height-1)

        idx = x + y * frame_width
        flat_img = frame.reshape(-1)
        nfov = flat_img.take(idx, axis=0)
        nfov = nfov.reshape([self.height, self.width])
        return nfov

    def setFOV(self, FOV):
        self.FOV = FOV
        self.fov_coord_rads = self._get_coord_rad(self._get_screen_img(), isScreenPts=True)

    def setCenterPoit(self, center_point):
        self.cp = self._get_coord_rad(center_point, isScreenPts=False)
        self.spericalCoord = self._calcSphericaltoGnomonic(self.fov_coord_rads)

    def toNFOV(self, frame, interpolation='bilinear'):
        if interpolation == 'bilinear':
            return self._bilinear_interpolation(frame, self.spericalCoord)
        elif interpolation == 'nn':
            return self._nn_interpolation(frame, self.spericalCoord)



# test the class
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    img = plt.imread('data/360.jpg')
    nfov = NFOV(800, 800)
    nfov.setFOV([0.25, 0.5])

    f, ax = plt.subplots(3, 3)
    for idx, x in enumerate([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]):
        center_point = np.array([x, 0.5])  # camera center point (valid range [0,1])
        img2 = nfov.toNFOV(img, center_point)
        ax[0, 0].imshow(img)
        ax[0, 0].set_axis_off()
        coords = nfov.spericalCoord.reshape(nfov.height, nfov.width, 2) * np.array(
            [nfov.frame_width, nfov.frame_height])
        ax[0, 0].plot(coords[0, :, 0], coords[0, :, 1], 'r')
        ax[0, 0].plot(coords[-1, :, 0], coords[-1, :, 1], 'r')
        ax[0, 0].plot(coords[:, 0, 0], coords[:, 0, 1], 'r')
        ax[0, 0].plot(coords[:, -1, 0], coords[:, -1, 1], 'r')
        ax[(idx + 1) // 3, (idx + 1) % 3].imshow(img2)
        ax[(idx + 1) // 3, (idx + 1) % 3].set_axis_off()
    plt.show()
