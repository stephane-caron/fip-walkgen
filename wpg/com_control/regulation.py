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

from numpy import asarray, bmat, cosh, cross, dot, eye, hstack, sinh, sqrt
from numpy import vstack, zeros

from pymanoid.misc import norm
from pymanoid.mpc import LinearPredictiveControl

from preview import ZMPPreviewBuffer


class FIPRegulator(object):

    """
    Constrained linear quadratic regulator on the FIP.
    """

    def __init__(self, nb_steps, state_estimator, preview_ref):
        PVZ_ref, contacts = preview_ref.discretize(nb_steps)
        P_ref = PVZ_ref[:, 0:3]
        X_ref = PVZ_ref[:, 0:6]
        Z_ref = PVZ_ref[:, 6:9]
        GZ_ref = Z_ref - P_ref
        dt = preview_ref.duration / nb_steps
        I, fzeros = eye(3), zeros((4, 3))
        omega2 = preview_ref.omega2
        omega = sqrt(omega2)
        a = omega * dt
        A = asarray(bmat([
            [cosh(a) * I, sinh(a) / omega * I],
            [omega * sinh(a) * I,  cosh(a) * I]]))
        B = vstack([(1. - cosh(a)) * I, -omega * sinh(a) * I])
        x_init = hstack([state_estimator.com, state_estimator.comd])
        Delta_x_init = x_init - X_ref[0]
        Delta_x_goal = zeros((6,))
        C = [None] * nb_steps
        D = [None] * nb_steps
        e = [None] * nb_steps
        for k in xrange(nb_steps):
            contact = contacts[k]
            force_face = contact.force_face
            C_fric = hstack([force_face, fzeros])
            D_fric = -force_face
            e_fric = dot(force_face, GZ_ref[k])
            if not all(e_fric > -1e-10):
                print "Warning: reference violates friction cone constraints"
            C_cop = zeros((4, 6))
            D_cop = zeros((4, 3))
            e_cop = zeros(4)
            for (i, vertex) in enumerate(contact.vertices):
                successor = (i + 1) % len(contact.vertices)
                edge = contact.vertices[successor] - vertex
                normal = cross(P_ref[k] - vertex, edge)
                C_cop[i, 0:3] = cross(edge, Z_ref[k] - vertex)  # * Delta_p
                D_cop[i, :] = normal  # * Delta_z
                e_cop[i] = -dot(normal, GZ_ref[k])
            if not all(e_cop > -1e-10):
                print "Warning: reference violates friction cone constraints"
            C[k] = vstack([C_fric, C_cop])
            D[k] = vstack([D_fric, D_cop])
            e[k] = hstack([e_fric, e_cop])
        lmpc = LinearPredictiveControl(
            A, B, C, D, e, Delta_x_init, Delta_x_goal, nb_steps, wxt=1.,
            wxc=1e-3, wu=1e-3)
        lmpc.build()
        try:
            lmpc.solve()
            X = X_ref + lmpc.X
            Z = Z_ref[:-1] + lmpc.U
            P, V = X[:, 0:3], X[:, 3:6]
            assert len(X) == nb_steps + 1
            assert len(Z) == nb_steps
            preview = ZMPPreviewBuffer(preview_ref.contact_sequence)
            preview.update(P, V, Z, [dt] * nb_steps, omega2)
            if __debug__:
                self.Delta_X = lmpc.X
                self.Delta_Z = lmpc.U
                self.X_ref = X_ref
                self.P_ref = P_ref
                self.Z_ref = Z_ref
                self.contacts = contacts
        except ValueError as e:
            print "%s error:" % type(self).__name__, e
            preview = None
        self.lmpc = lmpc
        self.nb_steps = nb_steps
        self.preview = preview

    @property
    def optimal_found(self):
        return self.preview is not None

    @property
    def xprod_ratio(self):
        """Used to test the validity of our assumption on neglecting Dz x Dp."""
        t = 0.
        for k in xrange(self.nb_steps):
            contact = self.contacts[k]
            Delta_p = self.Delta_X[k, 0:3]
            Delta_z = self.Delta_Z[k]
            for (i, v) in enumerate(contact.vertices):
                e1 = cross(v - self.P_ref[k], Delta_z)
                e2 = cross(self.Z_ref[k] - v, Delta_p)
                min_norm = norm(e1 + e2)
                t += norm(cross(Delta_p, Delta_z)) / min_norm
        return t / self.nb_steps
