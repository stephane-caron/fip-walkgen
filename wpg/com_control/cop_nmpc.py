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
from numpy import array

from pymanoid.sim import gravity

from nmpc import NonlinearPredictiveController
from preview import ZMPPreviewBuffer


class COPPredictiveController(NonlinearPredictiveController):

    weights = {         # EFFECT ON CONVERGENCE SPEED
        'p': 1e-0,      # base task, sweet range ~ [1, 10]
        'v': 1e-2,      # sweet spot ~ 1/100
        't': 1e-2,      #
        'u': 1e-3,      # sweet spot ~ 1/1000
        'alpha': 1e-5,  # best range ~ [1e-5, 1e-4]
        'beta': 1e-5,   # idem
        'lambda': 0,    # dreadful effect: leave as 0!
    }

    alpha_lim = 0.5
    beta_lim = 0.5
    dT_max = 0.35  # as low as possible
    dT_min = 0.05   # as high as possible
    lambda_max = 10000
    lambda_min = 9.81 / 10
    p_max = [+100, +100, +100]  # [m]
    p_min = [-100, -100, -100]  # [m]
    u_max = [+100, +100, +100]  # [m] / [s]^2
    u_min = [-100, -100, -100]  # [m] / [s]^2
    v_max = [+10, +10, +10]  # [m] / [s]
    v_min = [-10, -10, -10]  # [m] / [s]

    dT_init = .5 * (dT_min + dT_max)
    lambda_init = 9.81 / 1.  # significant effect

    def __init__(self, nb_steps, com_state, com_target, contact_sequence,
                 omega2=None, swing_duration=None):
        super(COPPredictiveController, self).__init__()
        end_com = list(com_target.p)
        end_comd = list(com_target.pd)
        nb_contacts = len(contact_sequence)
        start_com = list(com_state.p)
        start_comd = list(com_state.pd)

        p_0 = self.nlp.new_constant('p_0', 3, start_com)
        v_0 = self.nlp.new_constant('v_0', 3, start_comd)
        p_k = p_0
        v_k = v_0
        T_swing = 0
        T_total = 0

        for k in xrange(nb_steps):
            contact = contact_sequence[nb_contacts * k / nb_steps]
            u_k = self.nlp.new_variable(
                'u_%d' % k, 3, init=[0, 0, 0], lb=self.u_min, ub=self.u_max)
            dT_k = self.nlp.new_variable(
                'dT_%d' % k, 1, init=[self.dT_init], lb=[self.dT_min],
                ub=[self.dT_max])
            alpha_k = self.nlp.new_variable(
                'alpha_%d' % k, 1, init=[0.], lb=[-self.alpha_lim],
                ub=[+self.alpha_lim])
            beta_k = self.nlp.new_variable(
                'beta_%d' % k, 1, init=[0.], lb=[-self.beta_lim],
                ub=[+self.beta_lim])
            lambda_k = self.nlp.new_variable(
                'lambda_%d' % k, 1, init=[self.lambda_init],
                lb=[self.lambda_min], ub=[self.lambda_max])

            z_k = contact.p + alpha_k * contact.X * contact.t + \
                beta_k * contact.Y * contact.b
            self.nlp.add_equality_constraint(
                u_k, lambda_k * (p_k - z_k) + gravity)
            self.nlp.extend_cost(
                self.weights['alpha'] * alpha_k * alpha_k * dT_k)
            self.nlp.extend_cost(
                self.weights['beta'] * beta_k * beta_k * dT_k)
            # self.nlp.extend_cost(
            #     self.weights['lambda'] * lambda_k ** 2 * dT_k)
            self.nlp.extend_cost(
                self.weights['u'] * casadi.dot(u_k, u_k) * dT_k)

            # Full precision (no Taylor expansion)
            omega_k = casadi.sqrt(lambda_k)
            p_next = p_k \
                + v_k / omega_k * casadi.sinh(omega_k * dT_k) \
                + u_k / lambda_k * (cosh(omega_k * dT_k) - 1.)
            v_next = v_k * cosh(omega_k * dT_k) \
                + u_k / omega_k * sinh(omega_k * dT_k)
            T_total = T_total + dT_k
            if 2 * k < nb_steps:
                T_swing = T_swing + dT_k

            self.add_friction_constraint(contact, p_k, z_k)
            self.add_friction_constraint(contact, p_next, z_k)
            # slower:
            # add_linear_friction_constraints(contact, p_k, z_k)
            # add_linear_friction_constraints(contact, p_next, z_k)

            p_k = self.nlp.new_variable(
                'p_%d' % (k + 1), 3, init=start_com, lb=self.p_min,
                ub=self.p_max)
            v_k = self.nlp.new_variable(
                'v_%d' % (k + 1), 3, init=start_comd, lb=self.v_min,
                ub=self.v_max)
            self.nlp.add_equality_constraint(p_next, p_k)
            self.nlp.add_equality_constraint(v_next, v_k)

        self.nlp.add_constraint(
            T_swing, lb=[swing_duration], ub=[100],
            name='T_swing')

        p_last, v_last = p_k, v_k
        p_diff = p_last - end_com
        v_diff = v_last - end_comd
        self.nlp.extend_cost(self.weights['p'] * casadi.dot(p_diff, p_diff))
        self.nlp.extend_cost(self.weights['v'] * casadi.dot(v_diff, v_diff))
        self.nlp.extend_cost(self.weights['t'] * T_total)
        self.nlp.create_solver()
        #
        self.com_target = com_target
        self.end_com = array(end_com)
        self.end_comd = array(end_comd)
        self.nb_steps = nb_steps
        self.preview = ZMPPreviewBuffer(contact_sequence)

    def solve_nlp(self):
        X = self.nlp.solve()
        self.nlp.warm_start(X)
        # print "Symbols:", self.nlp.var_symbols
        N = self.nb_steps
        Y = X[:-6].reshape((N, 3 + 3 + 3 + 1 + 1 + 1 + 1))
        P = Y[:, 0:3]
        V = Y[:, 3:6]
        # U = Y[:, 6:9]  # not used
        dT = Y[:, 9]
        Alpha = Y[:, 10]
        Beta = Y[:, 11]
        Lambda = Y[:, 12]
        p_last = X[-6:-3]
        v_last = X[-3:]
        Z = self.compute_Z(Alpha, Beta)
        self.preview.update(P, V, Z, dT, Lambda)
        self.p_last = p_last
        self.v_last = v_last
        self.print_results()

    def compute_Z(self, Alpha, Beta):
        Z = []
        contact_sequence = self.contact_sequence
        for k in xrange(self.nb_steps):
            contact = contact_sequence[2 * k / self.nb_steps]
            z_k = contact.p + Alpha[k] * contact.X * contact.t + \
                Beta[k] * contact.Y * contact.b
            Z.append(z_k)
        return array(Z)

    def on_tick(self, sim):
        self.preview.forward(sim.dt)
        start_com = list(self.robot.com)
        start_comd = list(self.robot.comd)
        self.nlp.update_constant('p_0', start_com)
        self.nlp.update_constant('v_0', start_comd)
        self.solve_nlp()
