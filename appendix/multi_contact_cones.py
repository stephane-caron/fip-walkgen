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

from numpy import cross, dot, hstack

from pymanoid import Polyhedron
from pymanoid.draw import draw_polyhedron
from pymanoid.sim import gravity


def compute_cop_pendular_cone(contact_set, cop):
    """
    Cone of COM positions sustainable by a set of contacts under a fixed COP.

    Parameters
    ----------
    contact_set : pymanoid.ContactSet
        Set of contacts.
    cop : array, shape=(3,)
        COP position in the world frame.

    Returns
    -------
    cone : pymanoid.Polyhedron
        Cone of COM positions.
    """
    F = contact_set.compute_wrench_face([0, 0, 0])
    A_O, A = F[:, :3], F[:, 3:]
    A_C = A_O + cross(A, cop)
    c = dot(A_O, cop).reshape((A.shape[0], 1))
    return Polyhedron(hrep=hstack([c, -A_C]))


def draw_cop_pendular_cone(contact_set, cop):
    """
    Draw the cone of COM positions sustainable under a fixed COP.

    Parameters
    ----------
    contact_set : pymanoid.ContactSet
        Set of contacts.
    cop : array, shape=(3,)
        COP position in the world frame.

    Returns
    -------
    handles : list of openravepy.GraphHandle
        OpenRAVE graphical handles. Must be stored in some variable, otherwise
        the drawn object will vanish instantly.
    """
    polytope = contact_set.compute_cop_pendular_cone(cop)
    polytope.compute_vrep()
    return draw_polyhedron(
        polytope.vertices +
        [cop + 1 * r / dot(r, [0, 0, 1]) for r in polytope.rays])


def compute_vrp_com_cone(contact_set, vrp, gain):
    """
    Cone of COM positions sustainable with a fixed Virtual Repellent Point
    (VRP). The VRP was defined in [EOA15]_.

    Parameters
    ----------
    contact_set : pymanoid.ContactSet
        Set of contacts.
    vrp : array, shape=(3,)
        Virtual Repellent Point coordinates in the world frame.

    Returns
    -------
    cone : pymanoid.Polyhedron
        Cone of COM positions.

    References
    ----------
    .. [EOA15] J. Englsberger, C. Ott and A. Albu-Schäffer, "Three-dimensional
    bipedal walking control based on divergent component of motion," IEEE
    Transactions on Robotics, vol. 31, no. 2, pp. 355-368, April 2015.
    """
    from pymanoid.sim import gravity
    F = contact_set.compute_wrench_face([0, 0, 0])
    A_O, A = F[:, :3], F[:, 3:]
    B = gain * (A_O + cross(A, vrp)) + cross(A, gravity)
    c = dot(A_O, gain * vrp + gravity).reshape((B.shape[0], 1))
    return Polyhedron(hrep=hstack([c, -B]))


def draw_vrp_com_cone(contact_set, vrp, gain):
    """
    Draw the cone of COM positions sustainable under a fixed VRP.

    Parameters
    ----------
    contact_set : pymanoid.ContactSet
        Set of contacts.
    vrp : array, shape=(3,)
        Virtual Repellent Point coordinates in the world frame.

    Returns
    -------
    handles : list of openravepy.GraphHandle
        OpenRAVE graphical handles. Must be stored in some variable, otherwise
        the drawn object will vanish instantly.
    """
    polytope = contact_set.compute_vrp_com_cone(vrp, gain)
    polytope.compute_vrep()
    return draw_polyhedron(
        polytope.vertices +
        [vrp + 1 * r / dot(r, [0, 0, 1]) for r in polytope.rays])


def compute_attraction_polyhedron(contact_set, com_target, k):
    """
    Cone of COM positions sustainable when attracted by a desired position with
    stiffness k.

    Parameters
    ----------
    contact_set : pymanoid.ContactSet
        Set of contacts.
    com_target : array, shape=(3,)
        Target COM position (marginal attractor).
    k : scalar
        Attraction stiffness.

    Returns
    -------
    cone : pymanoid.Polyhedron
        Cone of COM positions.
    """
    F = contact_set.compute_wrench_face(p=[0, 0, 0])
    A_O, A = F[:, :3], F[:, 3:]
    B = -k * A_O + cross(A, gravity - k * com_target)
    c = dot(A_O, gravity - k * com_target).reshape((B.shape[0], 1))
    return Polyhedron(hrep=hstack([c, -B]))


def draw_attraction_polyhedron(contact_set, com_target, stiffness):
    """
    Draw the cone of COM positions sustainable for a given attractor.

    Parameters
    ----------
    contact_set : pymanoid.ContactSet
        Set of contacts.
    com_target : array, shape=(3,)
        Target COM position (marginal attractor).
    k : scalar
        Attraction stiffness.

    Returns
    -------
    handles : list of openravepy.GraphHandle
        OpenRAVE graphical handles. Must be stored in some variable, otherwise
        the drawn object will vanish instantly.
    """
    polytope = contact_set.compute_attraction_polyhedron(com_target, stiffness)
    polytope.compute_vrep()
    return draw_polyhedron(
        polytope.vertices +
        [com_target + r for r in polytope.rays])


def get_stiffness_range(contact_set, com, com_target):
    """
    Compute the minimum and maximum stiffness for a given COM attractor.

    Parameters
    ----------
    contact_set : pymanoid.ContactSet
        Set of contacts.
    com_target : array, shape=(3,)
        Target COM position (marginal attractor).

    Returns
    -------
    k_min : scalar
        Minimum stiffness.
    k_max : scalar
        Maximum stiffness.
    """
    F = contact_set.compute_wrench_face(p=[0, 0, 0])
    A_O, A = F[:, :3], F[:, 3:]
    p, pd = com, com_target
    b = dot(-A_O + cross(A, -pd), p) + dot(A_O, pd)
    c = dot(A_O, gravity) - dot(cross(A, gravity), p)
    # b * stiffness <= c
    k_min, k_max = -1e5, +1e5
    for i in xrange(b.shape[0]):
        m = c[i] / b[i]
        if b[i] > 0 and m < k_max:
            k_max = m
        elif b[i] < 0 and m > k_min:
            k_min = m
    return (k_min, k_max)
