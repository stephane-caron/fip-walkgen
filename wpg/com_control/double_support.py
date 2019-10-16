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

from numpy import array, asarray, bmat, cross, dot, eye, hstack, vstack, zeros

from pymanoid.geometry import compute_polytope_hrep
from pymanoid.misc import normalize
from pymanoid.mpc import LinearPredictiveControl
from pymanoid.sim import gravity

from preview import ZMPPreviewBuffer


class COMTube(object):

    """
    Primal tube of COM locations computed along with its dual acceleration
    cone.

    Parameters
    ----------
    start_com : array
        Start position of the COM.
    target_com : array
        End position of the COM.
    stance : Stance
        Set of contacts used to compute the contact wrench cone.
    radius : scalar
        Side of the cross-section square (for ``shape`` > 2).
    margin : scalar
        Safety margin (in [m]) around boundary COM positions.
    """

    def __init__(self, start_com, end_com, stance, radius, margin):
        n = normalize(end_com - start_com)
        t = array([0., 0., 1.])
        t -= dot(t, n) * n
        t = normalize(t)
        b = cross(n, t)
        cross_section = [dx * t + dy * b for (dx, dy) in [
            (+radius, +radius),
            (+radius, -radius),
            (-radius, +radius),
            (-radius, -radius)]]
        tube_start = start_com - margin * n
        tube_end = end_com + margin * n
        primal_vrep = [tube_start + s for s in cross_section] + \
            [tube_end + s for s in cross_section]
        primal_hrep = compute_polytope_hrep(primal_vrep)
        dual_vrep = stance.compute_pendular_accel_cone(
            com_vertices=primal_vrep)
        dual_hrep = compute_polytope_hrep(dual_vrep)
        self.dual_hrep = dual_hrep
        self.dual_vrep = dual_vrep
        self.primal_hrep = primal_hrep
        self.primal_vrep = primal_vrep

    def draw(self, acc_scale=0.05):
        from pymanoid.gui import draw_cone, draw_polyhedron
        handles = []
        cyan, yellow = (0., 0.5, 0.5, 0.3), (0.5, 0.5, 0., 0.3)
        handles.extend(
            draw_polyhedron(self.primal_vrep, '*.-#', color=cyan))
        origin = self.primal_vrep[0]
        apex = origin + [0., 0., acc_scale * -9.81]
        vscale = [acc_scale * array(v) + origin for v in self.dual_vrep]
        handles.extend(draw_cone(
            apex=apex, axis=[0, 0, 1], section=vscale[1:], combined='r-#',
            color=yellow))
        return handles


class DoubleSupportController(object):

    """
    Feedback controller that continuously runs the preview controller and sends
    outputs to a COMAccelBuffer.
    """

    def __init__(self, nb_steps, duration, omega2, state_estimator, com_target,
                 stance, tube_radius=0.02, tube_margin=0.01):
        start_com = state_estimator.com
        start_comd = state_estimator.comd
        tube = COMTube(
            start_com, com_target.p, stance, tube_radius, tube_margin)
        dt = duration / nb_steps
        I = eye(3)
        A = asarray(bmat([[I, dt * I], [zeros((3, 3)), I]]))
        B = asarray(bmat([[.5 * dt ** 2 * I], [dt * I]]))
        x_init = hstack([start_com, start_comd])
        x_goal = hstack([com_target.p, com_target.pd])
        C1, e1 = tube.primal_hrep
        D2, e2 = tube.dual_hrep
        C1 = hstack([C1, zeros(C1.shape)])
        C2 = zeros((D2.shape[0], A.shape[1]))
        D1 = zeros((C1.shape[0], B.shape[1]))
        C = vstack([C1, C2])
        D = vstack([D1, D2])
        e = hstack([e1, e2])
        lmpc = LinearPredictiveControl(
            A, B, C, D, e, x_init, x_goal, nb_steps, wxt=1., wxc=1e-1, wu=1e-1)
        lmpc.build()
        try:
            lmpc.solve()
            U = lmpc.U
            P = lmpc.X[:-1, 0:3]
            V = lmpc.X[:-1, 3:6]
            Z = P + (gravity - U) / omega2
            preview = ZMPPreviewBuffer([stance])
            preview.update(P, V, Z, [dt] * nb_steps, omega2)
        except ValueError as e:
            print("%s error:" % type(self).__name__, e)
            preview = None
        self.lmpc = lmpc
        self.preview = preview
        self.tube = tube
