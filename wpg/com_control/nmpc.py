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

import casadi

import pymanoid

from pymanoid.optim import NonlinearProgram


"""
Note
----
By convention, timesteps throughout the code are denoted by `dt` when they are
fixed and `dT` when they are variable. You will hence see dT's in nonlinear
predictive controllers and dt's in linear ones.
"""


class NonlinearPredictiveController(pymanoid.Process):

    def __init__(self):
        super(NonlinearPredictiveController, self).__init__()
        self.nlp = NonlinearProgram()

    def add_com_boundary_constraint(self, contact, p, lb=0.5, ub=1.5):
        dist = casadi.dot(p - contact.p, contact.n)
        self.nlp.add_constraint(dist, lb=[lb], ub=[ub])

    def add_friction_constraint(self, contact, p, z):
        mu = contact.static_friction
        ZG = p - z
        ZG2 = casadi.dot(ZG, ZG)
        ZGn2 = casadi.dot(ZG, contact.n) ** 2
        slackness = ZG2 - (1 + mu ** 2) * ZGn2
        self.nlp.add_constraint(slackness, lb=[-self.nlp.infty], ub=[0])

    def add_linear_friction_constraints(self, contact, p, z):
        mu_inner = contact.static_friction / casadi.sqrt(2)
        ZG = p - z
        ZGt = casadi.dot(ZG, contact.t)
        ZGb = casadi.dot(ZG, contact.b)
        ZGn = casadi.dot(ZG, contact.n)
        c0 = ZGt - mu_inner * ZGn
        c1 = -ZGt - mu_inner * ZGn
        c2 = ZGb - mu_inner * ZGn
        c3 = -ZGb - mu_inner * ZGn
        self.nlp.add_constraint(c0, lb=[-self.nlp.infty], ub=[0])
        self.nlp.add_constraint(c1, lb=[-self.nlp.infty], ub=[0])
        self.nlp.add_constraint(c2, lb=[-self.nlp.infty], ub=[0])
        self.nlp.add_constraint(c3, lb=[-self.nlp.infty], ub=[0])

    def add_cop_constraint(self, contact, p, z, scaling=0.95):
        X = scaling * contact.X
        Y = scaling * contact.Y
        CZ, ZG = z - contact.p, p - z
        CZxZG = casadi.cross(CZ, ZG)
        Dx = casadi.dot(contact.b, CZxZG) / X
        Dy = casadi.dot(contact.t, CZxZG) / Y
        ZGn = casadi.dot(contact.n, ZG)
        slackness = Dx ** 2 + Dy ** 2 - ZGn ** 2
        self.nlp.add_constraint(slackness, lb=[-self.nlp.infty], ub=[-0.005])

    def add_linear_cop_constraints(self, contact, p, z, scaling=0.95):
        GZ = z - p
        for (i, v) in enumerate(contact.vertices):
            v_next = contact.vertices[(i + 1) % len(contact.vertices)]
            v = v + (1. - scaling) * (contact.p - v)
            v_next = v_next + (1. - scaling) * (contact.p - v_next)
            slackness = casadi.dot(v_next - v, casadi.cross(v - p, GZ))
            self.nlp.add_constraint(
                slackness, lb=[-self.nlp.infty], ub=[-0.005])

    @property
    def is_ready_to_switch(self):
        if self.nlp.optimal_found and self.cp_error < 2e-3:
            return True  # success
        return False

    def print_results(self):
        dT = self.preview.dT
        T_swing = sum(dT[:self.nb_steps / 2])
        T_total = sum(dT)
        print "\n"
        print "%14s:  " % "dT's", [round(x, 3) for x in dT]
        print "%14s:  " % "dT_min", "%.3f s" % self.dT_min
        if self.swing_duration is not None:
            print "%14s:  " % "T_swing", "%.3f s" % T_swing
            print "%14s:  " % "TTHS", "%.3f s" % self.swing_duration
            print "%14s:  " % "swing_dT_min", "%.3f s" % self.swing_dT_min
        print "%14s:  " % "T_total", "%.2f s" % T_total
        print "%14s:  " % "CP error", self.cp_error
        print "%14s:  " % "Comp. time", "%.1f ms" % (1000 * self.nlp.solve_time)
        print "%14s:  " % "Iter. count", self.nlp.iter_count
        print "%14s:  " % "Status", self.nlp.return_status
        print "\n"

    def update_dT_max(self, dT_max):
        """
        Update the maximum variable-timestep duration.

        Parameters
        ----------
        dT_max : scalar
            New maximum duration.
        """
        for k in xrange(self.nb_steps):
            dT_min = self.swing_dT_min if 2 * k < self.nb_steps else self.dT_min
            self.nlp.update_variable_bounds('dT_%d' % k, [dT_min], [dT_max])
        self.dT_max = dT_max

    def update_dT_min(self, dT_min):
        """
        Update the minimum variable-timestep duration.

        Parameters
        ----------
        dT_min : scalar
            New minimum duration.
        """
        dT_max = self.dT_max
        if dT_min < self.swing_dT_min:
            self.update_swing_dT_min(dT_min)
        for k in xrange(self.nb_steps / 2, self.nb_steps):
            self.nlp.update_variable_bounds('dT_%d' % k, [dT_min], [dT_max])
        self.dT_min = dT_min

    def update_swing_dT_min(self, dT_min):
        """
        Update the minimum variable-timestep duration for the swing phase.

        Parameters
        ----------
        dT_min : scalar
            New minimum duration. Only affects the swing phase.
        """
        dT_max = self.dT_max
        for k in xrange(self.nb_steps / 2):
            self.nlp.update_variable_bounds('dT_%d' % k, [dT_min], [dT_max])
        self.swing_dT_min = dT_min

    def update_swing_duration(self, T):
        """
        Update the duration of the swing phase.

        Parameters
        ----------
        T : scalar
            New duration.
        """
        if self.nlp.has_constraint('T_swing'):
            self.swing_duration = T
            self.nlp.update_constraint_bounds('T_swing', [T], [self.T_max])
