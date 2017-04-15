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

from casadi import cosh, sinh
from numpy import array, sqrt
from time import time
from warnings import warn

from pymanoid.misc import norm
from pymanoid.sim import gravity

from nmpc import NonlinearPredictiveController
from preview import ZMPPreviewBuffer


class FIPPredictiveController(NonlinearPredictiveController):

    """
    Non-linear H-representation Predictive Controller.
    """

    weights = {         # EFFECT ON CONVERGENCE SPEED
        'cp': 1e-0,     # main task
        'time': 1e-2,   # sweet spot ~ 0.01 (> is faster but less stable)
        'acc': 1e-3,    # sweet spot ~ 1e-3
        'zmp': 1e-5,    # best range ~ [1e-5, 1e-4]
    }

    T_max = 6.  # maximum duration of a step in [s]
    dT_max = 0.35  # as low as possible
    dT_min = 0.03   # as high as possible, but caution when > sim.dt
    p_max = [+100, +100, +100]  # [m]
    p_min = [-100, -100, -100]  # [m]
    u_max = [+100, +100, +100]  # [m] / [s]^2
    u_min = [-100, -100, -100]  # [m] / [s]^2
    v_max = [+10, +10, +10]  # [m] / [s]
    v_min = [-10, -10, -10]  # [m] / [s]

    dT_init = .5 * (dT_min + dT_max)

    def __init__(self, nb_steps, state_estimator, com_target, contact_sequence,
                 omega2, swing_duration=None):
        super(FIPPredictiveController, self).__init__()
        t_build_start = time()
        omega = sqrt(omega2)
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

        for k in xrange(nb_steps):
            contact = contact_sequence[nb_contacts * k / nb_steps]
            z_min = list(contact.p - [1, 1, 1])  # TODO: smarter
            z_max = list(contact.p + [1, 1, 1])
            u_k = self.nlp.new_variable(
                'u_%d' % k, 3, init=[0, 0, 0], lb=self.u_min, ub=self.u_max)
            z_k = self.nlp.new_variable(
                'z_%d' % k, 3, init=list(contact.p), lb=z_min, ub=z_max)
            dT_k = self.nlp.new_variable(
                'dT_%d' % k, 1, init=[self.dT_init], lb=[self.dT_min],
                ub=[self.dT_max])

            CZ_k = z_k - contact.p
            self.nlp.add_equality_constraint(
                u_k, omega2 * (p_k - z_k) + gravity)
            self.nlp.extend_cost(
                self.weights['zmp'] * casadi.dot(CZ_k, CZ_k) * dT_k)
            self.nlp.extend_cost(
                self.weights['acc'] * casadi.dot(u_k, u_k) * dT_k)

            # Full precision (no Taylor expansion)
            p_next = p_k \
                + v_k / omega * casadi.sinh(omega * dT_k) \
                + u_k / omega2 * (cosh(omega * dT_k) - 1.)
            v_next = v_k * cosh(omega * dT_k) \
                + u_k / omega * sinh(omega * dT_k)
            T_total = T_total + dT_k
            if 2 * k < nb_steps:
                T_swing = T_swing + dT_k

            self.add_com_boundary_constraint(contact, p_k)
            self.add_friction_constraint(contact, p_k, z_k)
            # self.add_friction_constraint(contact, p_next, z_k)
            # self.add_linear_friction_constraints(contact, p_k, z_k)
            # self.add_linear_friction_constraints(contact, p_next, z_k)
            # self.add_cop_constraint(contact, p_k, z_k)
            # self.add_cop_constraint(contact, p_next, z_k)
            self.add_linear_cop_constraints(contact, p_k, z_k)
            # self.add_linear_cop_constraints(contact, p_next, z_k)

            p_k = self.nlp.new_variable(
                'p_%d' % (k + 1), 3, init=start_com, lb=self.p_min,
                ub=self.p_max)
            v_k = self.nlp.new_variable(
                'v_%d' % (k + 1), 3, init=start_comd, lb=self.v_min,
                ub=self.v_max)
            self.nlp.add_equality_constraint(p_next, p_k)
            self.nlp.add_equality_constraint(v_next, v_k)

        if swing_duration is not None:
            self.nlp.add_constraint(
                T_swing, lb=[swing_duration], ub=[self.T_max],
                name='T_swing')

        p_last, v_last, z_last = p_k, v_k, z_k
        cp_last = p_last + v_last / omega
        cp_last = end_com + end_comd / omega + gravity / omega2
        z_last = z_k
        # last_contact = contact_sequence[-1]
        # self.add_friction_constraint(last_contact, p_last, cp_last)
        # self.add_linear_cop_constraints(last_contact, p_last, cp_last)
        self.nlp.add_equality_constraint(z_last, cp_last)

        p_diff = p_last - end_com
        v_diff = v_last - end_comd
        cp_diff = p_diff + v_diff / omega
        # self.nlp.extend_cost(self.weights['pos'] * casadi.dot(p_diff, p_diff))
        # self.nlp.extend_cost(self.weights['vel'] * casadi.dot(v_diff, v_diff))
        self.nlp.extend_cost(self.weights['cp'] * casadi.dot(cp_diff, cp_diff))
        self.nlp.extend_cost(self.weights['time'] * T_total)
        self.nlp.create_solver()
        #
        self.build_time = time() - t_build_start
        self.contact_sequence = contact_sequence
        self.cp_error = 1e-6  # => self.is_ready_to_switch is initially True
        self.end_com = array(end_com)
        self.end_comd = array(end_comd)
        self.nb_contacts = nb_contacts
        self.nb_steps = nb_steps
        self.omega = omega
        self.omega2 = omega2
        self.p_last = None
        self.preview = ZMPPreviewBuffer(contact_sequence)
        self.state_estimator = state_estimator
        self.swing_dT_min = self.dT_min
        self.swing_duration = swing_duration
        self.v_last = None

    def solve_nlp(self):
        t_solve_start = time()
        X = self.nlp.solve()
        self.solve_time = time() - t_solve_start
        # print "Symbols:", self.nlp.var_symbols
        N = self.nb_steps
        Y = X[:-6].reshape((N, 3 + 3 + 3 + 3 + 1))
        P = Y[:, 0:3]
        V = Y[:, 3:6]
        # U = Y[:, 6:9]  # not used
        Z = Y[:, 9:12]
        dT = Y[:, 12]
        p_last = X[-6:-3]
        v_last = X[-3:]
        cp_nog_last = p_last + v_last / self.omega
        cp_nog_last_des = self.end_com + self.end_comd / self.omega
        cp_error = norm(cp_nog_last - cp_nog_last_des)
        T_swing = sum(dT[:self.nb_steps / 2])
        if self.swing_duration is not None and \
                T_swing > 1.25 * self.swing_duration:
            self.update_swing_dT_min(self.swing_dT_min / 2)
        self.cp_error = cp_error
        if self.cp_error > 0.1:  # and not self.preview.is_empty:
            print "Warning: preview not updated as CP error was", cp_error
            return
        self.nlp.warm_start(X)  # save as initial guess for next iteration
        self.preview.update(P, V, Z, dT, self.omega2)
        self.p_last = p_last
        self.v_last = v_last
        self.print_results()

    def warm_start(self, previous_step_controller):
        X = previous_step_controller.nlp.initvals
        N = previous_step_controller.nb_steps
        warm_start = X[(3 + 3 + 3 + 3 + 1) * (N / 2):]
        if len(warm_start) != len(self.nlp.initvals):
            warn("previous controller has no warm-start info for next phase")
            return
        self.update_dT_min(previous_step_controller.dT_min)
        self.nlp.warm_start(warm_start)

    def on_tick(self, sim):
        self.preview.forward(sim.dt)
        start_com = list(self.state_estimator.com)
        start_comd = list(self.state_estimator.comd)
        self.nlp.update_constant('p_0', start_com)
        self.nlp.update_constant('v_0', start_comd)
        self.solve_nlp()
