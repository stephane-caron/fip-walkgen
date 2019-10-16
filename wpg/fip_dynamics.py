#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of fip-walking
# <https://github.com/stephane-caron/fip-walking>.
#
# fip-walking is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# fip-walking is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# fip-walking. If not, see <http://www.gnu.org/licenses/>.

from numpy import cosh, exp, sinh, sqrt, zeros
from numpy.random import normal

import pymanoid

from pymanoid import Point, PointMass
from pymanoid.gui import draw_line, draw_point
from pymanoid.sim import gravity


def integrate_fip(p, v, z, dt, omega2):
    """
    Integrate the equation of motion of the Floating-base Inverted Pendulum.

    Parameters
    ----------
    p : array, shape=(3,)
        Initial position.
    v : array, shape=(3,)
        Initial velocity.
    z : array, shape=(3,)
        ZMP location throughout the integration.
    dt : scalar
        Integration step.
    omega2 : scalar
        FIP constant.

    Returns
    -------
    p_next : array, shape=(3,)
        Position at the end of the integration step.
    v_next : array, shape=(3,)
        Velocity at the end of the integration step.

    Note
    ----
    The Linear Inverted Pendulum Mode (LIPM) is a special case of the FIP, so
    this function also applies to COP-based controllers.
    """
    omega = sqrt(omega2)
    a = omega2 * (p - z) + gravity
    p_next = p + v / omega * sinh(omega * dt) \
        + a / omega2 * (cosh(omega * dt) - 1.)
    v_next = v * cosh(omega * dt) + a / omega * sinh(omega * dt)
    return p_next, v_next


class FIP(pymanoid.Process):

    """
    Floating-base Inverted Pendulum dynamics.

    Parameters
    ----------
    com : array, shape=(3,)
        Initial COM position.
    comd : array, shape=(3,), optional
        Initial COM velocity (default is zero).
    zmp : array, shape=(3,), optional
        Initial ZMP position.
    """

    def __init__(self, mass, omega2, com, comd=None, zmp=None, zmp_delay=None,
                 zmp_noise=None):
        assert omega2 > 1e-10, "FIP constant must be positive"
        super(FIP, self).__init__()
        com_state = PointMass(com, mass, visible=False)
        comd = comd if comd is not None else zeros(3)
        zmp = zmp if zmp is not None else com + gravity / omega2
        zmp_delay = 0. if zmp_delay is None else zmp_delay
        zmp_noise = 0. if zmp_noise is None else zmp_noise
        zmp_state = Point(zmp, visible=False)
        zmp_state.hide()
        self.__time_ds = 0
        self.__time_nonqs = 0
        self.com_state = com_state
        self.mass = mass
        self.next_zmp_target = None
        self.omega = sqrt(omega2)
        self.omega2 = omega2
        self.zmp_delay = zmp_delay
        self.zmp_noise = zmp_noise
        self.zmp_state = zmp_state

    def on_tick(self, sim):
        if self.next_zmp_target is not None:
            self.__update_zmp(self.next_zmp_target, sim.dt)
            self.next_zmp_target = None
        self.__forward_com_dynamics(sim.dt)

    def __update_zmp(self, target, dt):
        dz = self.zmp_state.p - target
        delay = dz * exp(-dt / self.zmp_delay) if self.zmp_delay > 1e-4 else 0.
        if self.zmp_noise < 1e-4:
            self.zmp_state.set_pos(target + delay)
            return
        sigma = self.zmp_noise * dt
        noise = normal(0., sigma, size=target.shape)
        self.zmp_state.set_pos(target + delay + noise)

    def __forward_com_dynamics(self, dt):
        com, comd = self.com_state.p, self.com_state.pd
        zmp = self.zmp_state.p
        omega2 = self.omega2
        com_next, comd_next = integrate_fip(com, comd, zmp, dt, omega2)
        self.com_state.set_pos(com_next)
        self.com_state.set_vel(comd_next)

    def draw(self):
        com, comd = self.com_state.p, self.com_state.pd
        cp = com + comd / self.omega + gravity / self.omega2
        zmp = self.zmp_state.p
        handles = [
            draw_point(com, color='r', pointsize=0.025),
            draw_point(cp, color='b', pointsize=0.025),
            draw_point(zmp, color='r', pointsize=0.025),
            draw_line(com, cp, color='b', linewidth=1),
            draw_line(zmp, com, color='r', linewidth=4)]
        return handles
