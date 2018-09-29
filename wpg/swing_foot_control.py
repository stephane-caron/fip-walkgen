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

import TOPP

from numpy import arange, dot, ones, zeros
from TOPP.Utilities import vect2str

import pymanoid

from pymanoid.body import Box
from pymanoid.gui import draw_line, draw_point
from pymanoid.misc import norm
from pymanoid.interp import interpolate_cubic_hermite, quat_slerp
from pymanoid.transformations import rotation_matrix_from_quat


def interpolate_uab_hermite(p0, u0, p1, u1):
    """
    Interpolate a Hermite path between :math:`p_0` and :math:`p_1` with tangents
    parallel to :math:`u_0` and :math:`u_1`, respectively. The output path
    `B(s)` minimizes a relaxation of the uniform acceleration bound:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & M \\\\
        \\mathrm{subject\\ to} & & \\forall s \\in [0, 1],\\
            \\|\\ddot{B}(s)\\|^2 \\leq M
        \\end{eqnarray}

    Parameters
    ----------
    p0 : (3,) array
        Start point.
    u0 : (3,) array
        Start tangent.
    p1 : (3,) array
        End point.
    u1 : (3,) array
        End tangent.

    Returns
    -------
    P : numpy.polynomial.Polynomial
        Polynomial function of the Hermite curve.

    Note
    ----
    We also impose that the output tangents share the sign of :math:`t_0` and
    :math:`t_1`, respectively.
    """
    Delta = p1 - p0
    _Delta_u0 = dot(Delta, u0)
    _Delta_u1 = dot(Delta, u1)
    _u0_u0 = dot(u0, u0)
    _u0_u1 = dot(u0, u1)
    _u1_u1 = dot(u1, u1)
    b0 = 6 * (3 * _Delta_u0 * _u1_u1 - 2 * _Delta_u1 * _u0_u1) / (
        9 * _u0_u0 * _u1_u1 - 4 * _u0_u1 * _u0_u1)
    if b0 < 0:
        b0 *= -1
    b1 = 6 * (-2 * _Delta_u0 * _u0_u1 + 3 * _Delta_u1 * _u0_u0) / (
        9 * _u0_u0 * _u1_u1 - 4 * _u0_u1 * _u0_u1)
    if b1 < 0:
        b1 *= -1
    return interpolate_cubic_hermite(p0, b0 * u0, p1, b1 * u1)


def interpolate_uab_hermite_topp(p0, v0, p1, v1):
    """
    Wrapper to :func:`pymanoid.interp.interpolate_uab_hermite`` for use with
    TOPP.

    Parameters
    ----------
    p0 : array, shape=(3,)
        Start point.
    v0 : array, shape=(3,)
        Start velocity tangent.
    p1 : array, shape=(3,)
        End point.
    v1 : array, shape=(3,)
        End velocity tangent.

    Returns
    -------
    path : TOPP.Trajectory.PiecewisePolynomialTrajectory
        Interpolated trajectory in TOPP format.
    """
    poly = interpolate_uab_hermite(p0, v0, p1, v1)
    C0, C1, C2, C3 = poly.coeffs
    path_str = "%f\n%d" % (1., 3)
    for k in xrange(3):
        path_str += "\n%f %f %f %f" % (C0[k], C1[k], C2[k], C3[k])
    return TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString(path_str)


class PathRetiming(object):

    def __init__(self):
        self.path = None
        self.s_traj = None

    def update(self, path, s_traj):
        self.path = path
        self.s_traj = s_traj

    @property
    def duration(self):
        return self.s_traj.duration

    def Eval(self, t):
        s = self.s_traj.Eval(t)
        return self.path.Eval(s)

    def Evald(self, t):
        s = self.s_traj.Eval(t)
        sd = self.s_traj.Evald(t)
        return self.path.Evald(s) * sd

    def Evaldd(self, t):
        s = self.s_traj.Eval(t)
        sd = self.s_traj.Evald(t)
        sdd = self.s_traj.Evaldd(t)
        return self.path.Evald(s) * sdd + self.path.Evaldd(s) * sd * sd


class SwingFootController(pymanoid.Process):

    def __init__(self, init_foot_link, max_foot_accel):
        super(SwingFootController, self).__init__()
        foot = Box(X=0.12, Y=0.06, Z=0.01, color='c', visible=False, dZ=-0.01)
        foot.set_transparency(0.5)
        self.discrtimestep = 0.1
        self.end_pose = None
        self.foot = foot
        self.foot_link = init_foot_link
        self.foot_vel = zeros(3)
        self.finished = False
        self.max_foot_accel = max_foot_accel
        self.path = None
        self.reparamstep = 0.01  # defaults to self.discrtimestep
        self.retimed_traj = PathRetiming()
        self.sd_beg = None
        self.start_pose = None
        self.time_to_heel_strike = None

    def reset(self, foot_link, swing_target):
        self.end_pose = swing_target.pose
        self.finished = False
        self.foot.set_pose(foot_link.pose)
        self.foot_link = foot_link
        self.start_pose = foot_link.pose
        #
        self.compute_trajectory()

    def interpolate_foot_path(self):
        p0 = self.foot.p
        p1 = self.end_pose[4:]
        R = rotation_matrix_from_quat(self.end_pose[:4])
        t, n = R[0:3, 0], R[0:3, 2]
        v1 = 0.5 * t - 0.5 * n
        if norm(self.foot_vel) > 1e-4:
            v0 = self.foot_vel
            self.path = interpolate_uab_hermite_topp(p0, v0, p1, v1)
            self.sd_beg = norm(v0) / norm(self.path.Evald(0.))
        else:  # choose initial direction
            v0 = 0.3 * self.foot.t + 0.7 * self.foot.n
            self.path = interpolate_uab_hermite_topp(p0, v0, p1, v1)
            self.sd_beg = 0.

    def create_topp_instance(self):
        assert self.path is not None, "interpolate a path first"
        amax = self.max_foot_accel * ones(self.path.dimension)
        id_traj = "1.0\n1\n0.0 1.0"
        discrtimestep = self.discrtimestep
        ndiscrsteps = int((self.path.duration + 1e-10) / discrtimestep) + 1
        constraints = str(discrtimestep)
        constraints += "\n" + str(0.)  # no velocity limit
        for i in range(ndiscrsteps):
            s = i * discrtimestep
            ps = self.path.Evald(s)
            pss = self.path.Evaldd(s)
            constraints += "\n" + vect2str(+ps) + " " + vect2str(-ps)
            constraints += "\n" + vect2str(+pss) + " " + vect2str(-pss)
            constraints += "\n" + vect2str(-amax) + " " + vect2str(-amax)
        self.topp = TOPP.TOPPbindings.TOPPInstance(
            None, "QuadraticConstraints", constraints, id_traj)
        self.topp.integrationtimestep = 1e-3

    def retime_path(self):
        sd_end = 0.0
        rc = self.topp.RunComputeProfiles(self.sd_beg, sd_end)
        if rc != TOPP.Errors.TOPP_OK:
            raise Exception("TOPP error: %s" % TOPP.Errors.MESSAGES[rc])
        self.topp.ReparameterizeTrajectory(self.reparamstep)
        self.topp.WriteResultTrajectory()
        s_str = self.topp.restrajectorystring
        s_traj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString(s_str)
        self.retimed_traj.update(self.path, s_traj)

    def update_target_pose(self, dt):
        assert self.retimed_traj is not None
        from_start = self.foot_link.p - self.start_pose[4:]
        to_goal = self.end_pose[4:] - self.foot_link.p
        dist_to_goal = norm(to_goal)
        dist_from_start = norm(from_start)
        x = dist_to_goal / (dist_from_start + dist_to_goal)
        quat = quat_slerp(self.foot.quat, self.end_pose[0:4], 1. - x)
        self.foot_vel = self.retimed_traj.Evald(dt)
        self.foot.set_pos(self.retimed_traj.Eval(dt))
        self.foot.set_quat(quat)

    def compute_trajectory(self):
        self.interpolate_foot_path()
        self.create_topp_instance()
        self.retime_path()
        self.time_to_heel_strike = self.retimed_traj.duration

    def on_tick(self, sim):
        if self.finished:
            return
        elif norm(self.end_pose[4:] - self.foot.p) < 5e-3:
            self.finished = True
            self.foot.set_pose(self.end_pose)
            self.foot_vel *= 0.
            return
        self.compute_trajectory()
        self.update_target_pose(sim.dt)

    def plot_profiles(self):
        """
        Plot TOPP profiles, e.g. for debugging.
        """
        import pylab
        pylab.ion()
        self.topp.WriteProfilesList()
        self.topp.WriteSwitchPointsList()
        profileslist = TOPP.TOPPpy.ProfilesFromString(
            self.topp.resprofilesliststring)
        switchpointslist = TOPP.TOPPpy.SwitchPointsFromString(
            self.topp.switchpointsliststring)
        TOPP.TOPPpy.PlotProfiles(profileslist, switchpointslist)
        TOPP.TOPPpy.PlotAlphaBeta(self.topp)
        pylab.title("%s phase profile" % type(self).__name__)
        pylab.axis([0, 1, 0, 10])

    def draw(self):
        """
        Draw the interpolated foot path.

        Returns
        -------
        handle : openravepy.GraphHandle
            OpenRAVE graphical handle. Must be stored in some variable,
            otherwise the drawn object will vanish instantly.
        """
        foot_path = self.retimed_traj.path
        if foot_path is None:
            return []
        ds = self.discrtimestep
        handles = [draw_point(foot_path.Eval(0), color='m', pointsize=0.007)]
        for s in arange(ds, foot_path.duration + ds, ds):
            handles.append(draw_point(
                foot_path.Eval(s), color='b', pointsize=0.01))
            handles.append(draw_line(
                foot_path.Eval(s - ds), foot_path.Eval(s), color='b',
                linewidth=3))
        return handles
