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

import casadi
import numpy

from numpy import array, sqrt
from time import time

from pymanoid.misc import norm
from pymanoid.sim import gravity

from nmpc import NonlinearPredictiveController
from preview import ZMPPreviewBuffer


class WrenchPredictiveController(NonlinearPredictiveController):

    weights = {
        'p':  1e-0,
        'v':  1e-2,
        't':  1e-2,
        'u':  1e-3,
        'l':  1e-6,
        'Ld': 1e-5,  # this one has a huge impact
    }

    T_max = 6.  # maximum duration of a step in [s]
    min_com_height = 0.5  # relative to contacts
    max_com_height = 1.5  # relative to contacts
    dT_max = 0.35  # as low as possible
    dT_min = 0.05   # as high as possible
    p_max = [+100, +100, +100]  # [m]
    p_min = [-100, -100, -100]  # [m]
    u_max = [+100, +100, +100]  # [m] / [s]^2
    u_min = [-100, -100, -100]  # [m] / [s]^2
    v_max = [+10, +10, +10]  # [m] / [s]
    v_min = [-10, -10, -10]  # [m] / [s]
    l_max = [+30] * 16

    dT_init = .5 * (dT_min + dT_max)
    omega2 = 9.81 / 0.8
    omega = sqrt(omega2)

    def __init__(self, nb_steps, state_estimator, com_target, contact_sequence,
                 omega2, swing_duration=None):
        super(WrenchPredictiveController, self).__init__()
        t_build_start = time()
        end_com = list(com_target.p)
        end_comd = list(com_target.pd)
        nb_contacts = len(contact_sequence)
        start_com = list(state_estimator.com)
        start_comd = list(state_estimator.comd)

        p_0 = self.nlp.new_constant('p_0', 3, start_com)
        v_0 = self.nlp.new_constant('v_0', 3, start_comd)
        p_k = p_0
        v_k = v_0
        T_swing = 0
        T_total = 0

        for k in range(nb_steps):
            contact = contact_sequence[nb_contacts * k / nb_steps]
            u_k = self.nlp.new_variable(
                'u_%d' % k, 3, init=[0., 0., 0.], lb=self.u_min, ub=self.u_max)
            dT_k = self.nlp.new_variable(
                'dT_%d' % k, 1, init=[self.dT_init], lb=[self.dT_min],
                ub=[self.dT_max])
            l_k = self.nlp.new_variable(
                'l_%d' % k, 16, init=[0.] * 16, lb=[0.] * 16, ub=self.l_max)

            c = contact.p
            w_k = casadi.mtimes(contact.wrench_span, l_k)
            f_k, tau_k = w_k[:3, :], w_k[3:, :]
            Ld_k = casadi.cross(c - p_k, f_k) + tau_k
            m = state_estimator.pendulum.com_state.mass
            self.nlp.add_equality_constraint(u_k, f_k / m + gravity)
            # hard angular-momentum constraint just don't work
            # nlp.add_constraint(Ld_k, lb=[0., 0., 0.], ub=[0., 0., 0.])

            self.nlp.extend_cost(
                self.weights['Ld'] * casadi.dot(Ld_k, Ld_k) * dT_k)
            self.nlp.extend_cost(
                self.weights['l'] * casadi.dot(l_k, l_k) * dT_k)
            self.nlp.extend_cost(
                self.weights['u'] * casadi.dot(u_k, u_k) * dT_k)

            p_next = p_k + v_k * dT_k + u_k * dT_k**2 / 2
            v_next = v_k + u_k * dT_k
            T_total = T_total + dT_k
            if 2 * k < nb_steps:
                T_swing = T_swing + dT_k

            p_k = self.nlp.new_variable(
                'p_%d' % (k + 1), 3, init=start_com, lb=self.p_min,
                ub=self.p_max)
            v_k = self.nlp.new_variable(
                'v_%d' % (k + 1), 3, init=start_comd, lb=self.v_min,
                ub=self.v_max)
            self.nlp.add_equality_constraint(p_next, p_k)
            self.nlp.add_equality_constraint(v_next, v_k)

        self.nlp.add_constraint(
            T_swing, lb=[swing_duration], ub=[100], name='T_swing')

        p_last, v_last = p_k, v_k
        p_diff = p_last - end_com
        v_diff = v_last - end_comd
        self.nlp.extend_cost(self.weights['p'] * casadi.dot(p_diff, p_diff))
        self.nlp.extend_cost(self.weights['v'] * casadi.dot(v_diff, v_diff))
        self.nlp.extend_cost(self.weights['t'] * T_total)
        self.nlp.create_solver()
        #
        self.build_time = time() - t_build_start
        self.com_target = com_target
        self.contact_sequence = contact_sequence
        self.end_com = array(end_com)
        self.end_comd = array(end_comd)
        self.nb_contacts = nb_contacts
        self.nb_steps = nb_steps
        self.omega2 = omega2
        self.preview = ZMPPreviewBuffer(contact_sequence)
        self.start_com = array(start_com)
        self.start_comd = array(start_comd)
        self.state_estimator = state_estimator
        self.swing_dT_min = self.dT_min

    def solve_nlp(self):
        t_solve_start = time()
        X = self.nlp.solve()
        self.solve_time = time() - t_solve_start
        # print "Symbols:", self.nlp.var_symbols
        N = self.nb_steps
        Y = X[:-6].reshape((N, 3 + 3 + 3 + 1 + 16))
        P = Y[:, 0:3]
        V = Y[:, 3:6]
        # U = Y[:, 6:9]  # not used
        dT = Y[:, 9]
        L = Y[:, 10:26]
        p_last = X[-6:-3]
        v_last = X[-3:]
        Z, Ld = self.compute_Z_and_Ld(P, L)
        cp_nog_last = p_last + v_last / self.omega
        cp_nog_last_des = self.end_com + self.end_comd / self.omega
        cp_error = norm(cp_nog_last - cp_nog_last_des)
        T_swing = sum(dT[:self.nb_steps / 2])
        if self.swing_duration is not None and \
                T_swing > 1.25 * self.swing_duration:
            self.update_swing_dT_min(self.swing_dT_min / 2)
        self.cp_error = cp_error
        if self.cp_error > 0.1:  # and not self.preview.is_empty:
            print("Warning: preview not updated as CP error was", cp_error)
        self.nlp.warm_start(X)
        self.preview.update(P, V, Z, dT, self.omega2)
        self.p_last = p_last
        self.v_last = v_last
        self.print_results()

    def warm_start(self, preview_step_controller):
        pass

    def compute_Z_and_Ld(self, P, L):
        Ld, Z = [], []
        contact_sequence = self.contact_sequence
        for k in range(self.nb_steps):
            contact = contact_sequence[2 * k / self.nb_steps]
            w_k = numpy.dot(contact.wrench_span, L[k])
            f_k, tau_k = w_k[:3], w_k[3:]
            # comdd_k = f_k + gravity
            # gravity - comdd = -f
            z_k = P[k] - f_k / self.omega2
            Ld_k = numpy.cross(contact.p - P[k], f_k) + tau_k
            Ld.append(Ld_k)
            Z.append(z_k)
        return array(Z), array(Ld)

    def on_tick(self, sim):
        self.preview.forward(sim.dt)
        start_com = list(self.state_estimator.com)
        start_comd = list(self.state_estimator.comd)
        self.nlp.update_constant('p_0', start_com)
        self.nlp.update_constant('v_0', start_comd)
        self.solve_nlp()
