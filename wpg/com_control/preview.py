#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of dynamic-walking
# <https://github.com/stephane-caron/dynamic-walking>.
#
# dynamic-walking is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# dynamic-walking is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# dynamic-walking. If not, see <http://www.gnu.org/licenses/>.

import os
import sys

from numpy import array, hstack, isreal, sqrt, zeros
from threading import Lock
from time import sleep

from pymanoid.draw import draw_line, draw_point
from pymanoid.sim import gravity

try:
    from fip_dynamics import integrate_fip
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/..')
    from fip_dynamics import integrate_fip


class ZMPPreviewBuffer(object):

    """
    Preview buffer for ZMP-based controllers.

    Parameters
    ----------
    contact_sequence : list of pymanoid.Contacts
        Contacts traversed during the preview.
    """

    def __init__(self, contact_sequence):
        self.P = None
        self.V = None
        self.Z = None
        self.cam = zeros(3)
        self.contact_sequence = contact_sequence
        self.cur_step = None
        self.dT = None
        self.lock = Lock()
        self.nb_steps = None
        self.omega2 = None
        self.rem_dT = None
        self.rem_duration = None

    @property
    def duration(self):
        return sum(self.dT)

    @property
    def is_empty(self):
        return self.P is None or self.cur_step >= self.nb_steps

    def update(self, P, V, Z, dT, omega2):
        """
        Update preview.

        Parameters
        ----------
        P : array, shape=(N, 3)
            Series of preview positions.
        V : array, shape=(N, 3)
            Series of preview velocities.
        Z : array, shape=(N, 3)
            Series of preview ZMPs.
        dT : array, shape=(N,)
            Series of timestep durations.
        omega2 : scalar or array, shape=(N,)
            FIP constant, or series of leg stiffness coefficients.
        """
        with self.lock:
            self.P = P
            self.V = V
            self.Z = Z
            self.cur_step = 0
            self.dT = dT
            self.nb_steps = len(dT)
            self.omega2 = omega2
            self.rem_dT = dT[0]
            self.rem_duration = sum(dT)

    def __get(self, attr, step=None):
        if step is None:
            step = self.cur_step
        if step >= self.nb_steps:
            raise Exception("read access after the end of the preview buffer")
        with self.lock:
            value = self.__dict__[attr][step]
        return value

    def get_contact(self, step=None):
        assert self.contact_sequence is not None
        nb_contacts = len(self.contact_sequence)
        if nb_contacts == 1:
            return self.contact_sequence[0]
        assert nb_contacts == 2, "invalid number of contacts in preview buffer"
        return self.contact_sequence[nb_contacts * step / self.nb_steps]

    def get_dT(self, step=None):
        return self.__get('dT', step)

    def get_omega2(self, step=None):
        if isreal(self.omega2):
            return self.omega2
        return self.__get('omega2', step)

    def get_zmp(self, step=None):
        return self.__get('Z', step)

    def forward(self, dt):
        if self.rem_dT is None:
            return
        self.rem_dT -= dt
        self.rem_duration -= dt
        if self.rem_dT > 1e-10:
            return
        self.cur_step += 1
        if self.cur_step >= self.nb_steps:
            print "Warning: reached end of receding horizon"
            return
        self.rem_dT = self.dT[self.cur_step]

    def discretize(self, nb_steps):
        """
        Discretize the remaining preview with a uniform timestep.

        Parameter
        ---------
        nb_steps : integer
            Number of discretization time steps.

        Returns
        -------
        X : array, shape=(N + 1, 9)
            Series of discretized states (com, comd, zmp).
        contacts : list of Contacts
            List of N + 1 contacts corresponding to each step k from 0 to N.
        """
        assert isreal(self.omega2), "Discretization only works on FIP previews"
        X = []
        contacts = []
        input_step = self.cur_step
        com, comd = self.P[0], self.V[0]
        output_timestep = self.rem_duration / nb_steps
        rem_dT = self.rem_dT
        omega2 = self.omega2
        for _ in xrange(nb_steps):
            X.append(hstack([com, comd, self.get_zmp(input_step)]))
            contacts.append(self.get_contact(input_step))
            time_to_fill = output_timestep
            while time_to_fill > 1e-10:
                if rem_dT < 1e-10:
                    input_step += 1
                    rem_dT = self.get_dT(input_step)
                zmp = self.get_zmp(input_step)
                dt = min(time_to_fill, rem_dT)
                com, comd = integrate_fip(com, comd, zmp, dt, omega2)
                time_to_fill -= dt
                rem_dT -= dt
        cp = com + comd / sqrt(omega2) + gravity / omega2
        X.append(hstack([com, comd, cp]))
        contacts.append(contacts[-1])
        return array(X), contacts

    def play(self, sim, slowdown=1., wrench_drawer=None,
             post_preview_duration=None, callback=None):
        handles = None
        com, comd = self.P[0], self.V[0]
        if wrench_drawer is not None:
            wrench_drawer.point_mass.set_pos(com)
            wrench_drawer.point_mass.set_vel(comd)
            wrench_drawer.point_mass.set_transparency(0.5)
        t, k, rem_time = 0., -1, -1.,
        duration = 2 * self.duration if post_preview_duration is None else \
            self.duration + post_preview_duration
        while t <= duration:
            if rem_time < 1e-10:
                if k < self.nb_steps - 1:
                    k += 1
                rem_time = self.get_dT(k)
            omega2 = self.get_omega2(k)
            cp = com + comd / sqrt(omega2) + sim.gravity / omega2
            zmp = self.get_zmp(k) if k < self.nb_steps else cp
            comdd = omega2 * (com - zmp) + sim.gravity
            handles = [
                draw_point(zmp, color='r'),
                draw_line(zmp, com, color='r', linewidth=4),
                draw_point(cp, color='b'),
                draw_line(cp, com, color='b', linewidth=2),
                draw_point(com, color='r', pointsize=0.05)]
            if wrench_drawer is not None:
                try:
                    wrench_drawer.recompute(self.get_contact(k), comdd, am=None)
                except ValueError:
                    print "Warning: wrench validation failed at t = %.2f s" % t
            dt = min(rem_time, sim.dt)
            com, comd = integrate_fip(com, comd, zmp, dt, omega2)
            if wrench_drawer is not None:
                wrench_drawer.point_mass.set_pos(com)
                wrench_drawer.point_mass.set_vel(comd)
            sleep(slowdown * dt)
            if callback is not None:
                callback()
            rem_time -= dt
            t += dt
        if wrench_drawer is not None:
            wrench_drawer.handles = []
        return handles

    def draw(self, color='b', pointsize=0.005):
        handles = []
        if self.is_empty:
            return handles
        com, comd = self.P[0], self.V[0]
        handles.append(draw_point(com, color, pointsize))
        for k in xrange(self.nb_steps):
            com_prev = com
            dT = self.get_dT(k)
            omega2 = self.get_omega2(k)
            zmp = self.get_zmp(k)
            com, comd = integrate_fip(com, comd, zmp, dT, omega2)
            handles.append(draw_point(com, color, pointsize))
            handles.append(draw_line(com_prev, com, color, linewidth=3))
        com_prev = com
        com = com + comd / sqrt(omega2)  # stationary value at CP
        handles.append(draw_point(com, color, pointsize))
        handles.append(draw_line(com_prev, com, color, linewidth=3))
        return handles
