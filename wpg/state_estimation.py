#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of fip-walkgen
# <https://github.com/stephane-caron/fip-walkgen>.
#
# fip-walkgen is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# fip-walkgen is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# fip-walkgen. If not, see <http://www.gnu.org/licenses/>.

from numpy import exp, random

import pymanoid


class StateEstimator(pymanoid.Process):

    def __init__(self, pendulum, delay, com_noise, comd_noise):
        super(StateEstimator, self).__init__()
        self.com = pendulum.com_state.p.copy()
        self.com_noise = com_noise
        self.comd = pendulum.com_state.pd.copy()
        self.comd_noise = comd_noise
        self.delay = delay
        self.pendulum = pendulum

    def estimate(self, dt, real, cur_est, noise_intensity):
        """
        Update an estimation under noise and delays.

        Parameters
        ----------
        dt : scalar
            Time since last estimation (usually one control cycle).
        real : array
            Ground-truth coordinates.
        cur_est : array
            Current estimation.
        noise_intensity : scalar
            Intensity of noise signal in [m] / [s].

        Returns
        -------
        estimate : array
            New estimate.
        """
        Delta = cur_est - real
        delay = Delta * exp(-dt / self.delay) if self.delay > 1e-4 else 0.
        if noise_intensity < 1e-4:
            return real + delay
        sigma = noise_intensity * dt
        noise = random.normal(0., sigma, size=real.shape)
        return real + delay + noise

    def on_tick(self, sim):
        """
        Update COM estimation after a tick of the control loop.

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
        """
        com_real = self.pendulum.com_state.p
        comd_real = self.pendulum.com_state.pd
        new_com_est = self.estimate(
            sim.dt, com_real, self.com, self.com_noise)
        new_comd_est = self.estimate(
            sim.dt, comd_real, self.comd, self.comd_noise)
        self.com = new_com_est
        self.comd = new_comd_est
