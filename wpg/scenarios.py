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

from numpy import arange, array, cos, pi, sin
from numpy.random import random

from pymanoid import Contact


class ContactFeed(object):

    def __init__(self, cyclic=False):
        self.contacts = []
        self.cyclic = cyclic
        self.next_contact = 0

    def pop(self):
        i = self.next_contact
        self.next_contact += 1
        if self.next_contact >= len(self.contacts):
            if not self.cyclic:
                return None
            self.next_contact = 0
        return self.contacts[i]


class EllipticStaircase(ContactFeed):

    """
    Elliptic staircase with tilted steps.

    Parameters
    ----------
    robot : Robot
        Robot to which this contact feed is destined.
    radius : scalar
        Staircase radius in [m].
    angular_step : scalar
        Angular step between contacts in [rad].
    height : scalar
        Altitude variation in [m].
    roughness : scalar
        Amplitude of contact roll, pitch and yaw in [rad].
    contact_model : Contact
        Template contact giving shape and friction properties for all contacts.
    visible : bool, optional
        Show or hide stepping stones.
    """

    def __init__(self, robot, radius, angular_step, height, roughness,
                 contact_model, visible=True):
        super(EllipticStaircase, self).__init__(cyclic=True)
        for theta in arange(0., 2 * pi, angular_step):
            left_foot = Contact(
                shape=contact_model.shape,
                pos=[radius * cos(theta),
                     radius * sin(theta),
                     radius + .5 * height * sin(theta)],
                rpy=(roughness * (random(3) - 0.5) + [0, 0, theta + .5 * pi]),
                static_friction=contact_model.static_friction,
                kinetic_friction=contact_model.kinetic_friction,
                visible=visible,
                link=robot.left_foot)
            right_foot = Contact(
                shape=contact_model.shape,
                pos=[1.2 * radius * cos(theta + .5 * angular_step),
                     1.2 * radius * sin(theta + .5 * angular_step),
                     radius + .5 * height * sin(theta + .5 * angular_step)],
                rpy=(roughness * (random(3) - 0.5) + [0, 0, theta + .5 * pi]),
                static_friction=contact_model.static_friction,
                kinetic_friction=contact_model.kinetic_friction,
                visible=visible,
                link=robot.right_foot)
            self.contacts.append(left_foot)
            self.contacts.append(right_foot)
        self.contacts.insert(0, self.contacts.pop())


class RegularStaircase(ContactFeed):

    """
    Regular staircase with horizontal steps.

    Parameters
    ----------
    robot : Robot
        Robot to which this contact feed is destined.
    nb_steps : integer
        Number of steps.
    step_length : scalar
        Length of a step in [m].
    step_height : scalar
        Height of a step in [m].
    width : scalar
        Leg spread in [m].
    contact_model : Contact
        Template contact giving shape and friction properties for all contacts.
    visible : bool, optional
        Show or hide stepping stones.
    """

    def __init__(self, robot, nb_steps, step_length, step_height,
                 width, contact_model, visible=True):
        super(RegularStaircase, self).__init__(cyclic=False)
        init_dist = 0.2
        left_foot = Contact(
            shape=contact_model.shape,
            pos=[-step_length - init_dist, 0, -step_height],
            rpy=[0, 0, 0],
            static_friction=contact_model.static_friction,
            kinetic_friction=contact_model.kinetic_friction,
            visible=visible,
            name="StaircaseStep0",
            link=robot.left_foot)
        right_foot = Contact(
            shape=contact_model.shape,
            pos=[-step_length - init_dist, -width, -step_height],
            rpy=[0, 0, 0],
            static_friction=contact_model.static_friction,
            kinetic_friction=contact_model.kinetic_friction,
            visible=visible,
            name="StaircaseStep0",
            link=robot.right_foot)
        self.contacts.append(right_foot)
        self.contacts.append(left_foot)
        for i in xrange(nb_steps):
            left_foot = Contact(
                shape=contact_model.shape,
                pos=[(2*i + 1) * step_length, 0, (2*i + 1) * step_height],
                rpy=[0, 0, 0],
                static_friction=contact_model.static_friction,
                kinetic_friction=contact_model.kinetic_friction,
                visible=visible,
                name="StaircaseStep%d" % (len(self.contacts) + 1),
                link=robot.left_foot)
            right_foot = Contact(
                shape=contact_model.shape,
                pos=[2*i * step_length, -width, 2*i * step_height],
                rpy=[0, 0, 0],
                static_friction=contact_model.static_friction,
                kinetic_friction=contact_model.kinetic_friction,
                visible=visible,
                name="StaircaseStep%d" % (len(self.contacts) + 0),
                link=robot.right_foot)
            self.contacts.append(right_foot)
            self.contacts.append(left_foot)


class SteppingStones(ContactFeed):

    def __init__(self, robot, length, step, delta_pos, delta_rpy, roughness,
                 pvar, contact_model, visible=True):
        super(SteppingStones, self).__init__(cyclic=False)
        for x in arange(0., length, step):
            left_rand1 = 2 * random(3) - 1.
            left_rand2 = 2 * random(3) - 1.
            right_rand1 = 2 * random(3) - 1.
            right_rand2 = 2 * random(3) - 1.
            left_pos = array([x + 0.5 * step, 0, 0.])
            right_pos = array([x, 0, 0.]) + delta_pos
            left_foot = Contact(
                shape=contact_model.shape,
                pos=left_pos + pvar * left_rand1,
                rpy=roughness * left_rand2,
                static_friction=contact_model.static_friction,
                kinetic_friction=contact_model.kinetic_friction,
                visible=visible,
                name="SteppingStone%d" % (len(self.contacts) + 1),
                link=robot.left_foot)
            right_foot = Contact(
                shape=contact_model.shape,
                pos=right_pos + pvar * right_rand1,
                rpy=delta_rpy + roughness * right_rand2,
                static_friction=contact_model.static_friction,
                kinetic_friction=contact_model.kinetic_friction,
                visible=visible,
                name="SteppingStone%d" % (len(self.contacts) + 0),
                link=robot.right_foot)
            self.contacts.append(right_foot)
            self.contacts.append(left_foot)


class HorizontalFloor(SteppingStones):

    def __init__(self, robot, length, step, leg_spread, contact_model,
                 visible=True):
        super(HorizontalFloor, self).__init__(
            robot, length, step, [0., -leg_spread, 0.], 0., 0., 0.,
            contact_model, visible=visible)
