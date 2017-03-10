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

import IPython
import os
import sys

from numpy import hstack, random, zeros

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/pymanoid')
    import pymanoid

from pymanoid import Contact, ContactSet, PointMass
from pymanoid.drawers import TrajectoryDrawer
from pymanoid.sim import CameraRecorder, gravity
from pymanoid.tasks import DOFTask
from pymanoid.tasks import MinCAMTask

from wpg import WalkingPatternGenerator
from wpg.fip_dynamics import FIP
from wpg.scenarios import EllipticStaircase
# from wpg.scenarios import HorizontalFloor
from wpg.state_estimation import StateEstimator


#################
# USER SETTINGS #
#################

# State estimation

# Note: there is a typo in the first submitted version of the paper. In the
# tunings that generated the accompanying video, estimation delay is "fifteen"
# milliseconds, not "fifty". Sorry for this. (~SC)

COMD_NOISE = 0.1           # intensity of COM velocity noise in [m] / [s]^2
COM_NOISE = 0.1            # intensity of COM estimation noise in [m] / [s]
ESTIMATION_DELAY = 0.015   # delay in COM estimation, in [s]
ZMP_DELAY = 0.02           # delay in ZMP command, in [s]
ZMP_NOISE = 0.1            # intensity of ZMP noise in [mm] / [ms]

# Pattern generation

DISABLE_LQR = False           # set to True to try NMPC without LQ regulation
DRAW_DS_TUBE = False          # shows when the double-support controller is used
FORWARD_VELOCITY = 0.15       # reference forward walking velocity in [m] / [s]
MAX_SWING_FOOT_ACCEL = 4.     # maximum acceleration sent to the TOPP controller
MAX_SWING_IK_WEIGHT = 2e-2    # weight on swing-foot task by the end of the step
MIN_SWING_IK_WEIGHT = 1e-3    # weight on swing-foot task at beginning of step
NB_LQR_STEPS = 30             # number of fixed-duration timesteps in the LQR
NB_MPC_STEPS = 10             # number of variable-duration timesteps in the MPC
SIMULATE_INSTANT_MPC = False  # used to check an assertion we make in the paper


class StatsCollector(pymanoid.Process):

    def __init__(self, count_nonqs=False):
        super(StatsCollector, self).__init__()
        self.__time_ds = 0
        self.__time_nonqs = None

    def on_tick(self, sim):
        if wpg.is_in_double_support:
            self.__time_ds += 1
        elif self.__time_nonqs is not None:
            try:  # the robot is in single support
                cs = ContactSet([wpg.support_contact])
                p = pendulum.com_state.p
                cs.find_static_supporting_wrenches(p, robot.mass)
            except ValueError:
                self.__time_nonqs += 1

    def print_info(self):
        print "Fraction of time in DS: %.2f%%" % (
            100. * (1 + self.__time_ds) / (1 + sim.nb_steps))
        if self.__time_nonqs is not None:
            print "Fraction of time in non-QS: %.2f%%" % (
                100. * (1 + self.__time_nonqs) / (1 + sim.nb_steps))


class PreviewDrawer(pymanoid.Process):

    def __init__(self):
        super(PreviewDrawer, self).__init__()
        self.mpc_handle = None
        self.lqr_handle = None
        self.pendulum_handle = None
        self.swing_handle = None

    def on_tick(self, sim):
        if not wpg.com_mpc.preview.is_empty:
            self.mpc_handle = wpg.com_mpc.preview.draw('b')
        if not wpg.is_in_double_support:
            self.swing_handle = wpg.swing_controller.draw()
        if not DISABLE_LQR and wpg.com_lqr is not None and \
                wpg.com_lqr.preview is not None:
            self.lqr_handle = wpg.com_lqr.preview.draw('g')
        self.pendulum_handle = pendulum.draw()


class WrenchDrawer(pymanoid.drawers.PointMassWrenchDrawer):

    def __init__(self):
        point_mass = PointMass(robot.com, robot.mass, visible=False)
        contact_set = ContactSet([wpg.support_contact])
        super(WrenchDrawer, self).__init__(point_mass, contact_set)
        self.am = zeros(3)

    def find_supporting_wrenches(self, gravity):
        mass = self.point_mass.mass
        p = self.point_mass.p
        pdd = self.point_mass.pdd
        wrench = hstack([mass * (pdd - gravity), self.am])
        support = self.contact_set.find_supporting_wrenches(wrench, p)
        return support

    def hide(self):
        for l in self.handles:
            for h in l:
                h.SetShow(False)

    def on_tick(self, sim):
        com = pendulum.com_state.p
        comdd = pendulum.omega2 * (com - pendulum.zmp_state.p) + gravity
        self.point_mass.set_pos(com)
        self.point_mass.set_vel(pendulum.com_state.pd)
        self.contact_set = ContactSet([wpg.support_contact] + (
            [wpg.swing_start] if wpg.is_in_double_support else []))
        self.point_mass.pdd = comdd
        super(WrenchDrawer, self).on_tick(sim)
        if self.handles:
            self.handles[0][-1].SetShow(False)
        elif __debug__:
            print Exception("check contact wrenches")

    def recompute(self, contact, comdd, am):
        assert type(contact) in [Contact, ContactSet]
        if type(contact) is Contact:
            self.contact_set = ContactSet([contact])
        else:  # type(contact) is ContactSet
            self.contact_set = contact
        self.point_mass.pdd = comdd
        self.am = am if am is not None else zeros(3)
        super(WrenchDrawer, self).on_tick(sim)

    def show(self):
        for l in self.handles:
            for h in l:
                h.SetShow(True)


def generate_posture():
    """Initial posture generation."""
    init_stance = pymanoid.Stance(
        com=contact_feed.contacts[1].p + [0, 0, robot.leg_length],
        left_foot=contact_feed.contacts[1],
        right_foot=contact_feed.contacts[0])
    robot.init_ik(robot.whole_body)
    robot.set_ff_pos([0, 0, 2])  # start PG with the robot above contacts
    robot.generate_posture(init_stance, max_it=50)


def update_robot_ik():
    """Update IK targets once the WPG has been constructed."""
    robot.ik.tasks['COM'].update_target(pendulum.com_state)
    robot.ik.add_task(MinCAMTask(robot))
    # prevent robot from leaning backwards:
    robot.ik.add_task(DOFTask(robot, robot.ROT_P, 0., weight=1e-4))
    robot.ik.add_task(DOFTask(robot, robot.CHEST_P, 0.2, weight=1e-4))
    robot.ik.add_task(DOFTask(robot, robot.CHEST_Y, 0., weight=1e-4))
    # prevent lateral arm collisions with the chest:
    robot.ik.add_task(DOFTask(robot, robot.R_SHOULDER_R, -0.4))
    robot.ik.add_task(DOFTask(robot, robot.L_SHOULDER_R, +0.4))
    # prevent arms from leaning backward:
    robot.ik.add_task(DOFTask(robot, robot.L_SHOULDER_P, 0.))
    robot.ik.add_task(DOFTask(robot, robot.R_SHOULDER_P, 0.))
    robot.ik.tasks[robot.left_foot.name.upper()].weight = 1.
    robot.ik.tasks[robot.right_foot.name.upper()].weight = 1.
    robot.ik.tasks['COM'].weight = 1e-2
    robot.ik.tasks['MIN_CAM'].weight = 1e-4
    robot.ik.tasks['ROT_P'].weight = 1e-4
    robot.ik.tasks[robot.chest_p_name.upper()].weight = 1e-4
    robot.ik.tasks[robot.chest_y_name.upper()].weight = 1e-4
    if '--shoulders' in sys.argv:
        robot.ik.tasks['L_SHOULDER_P'].weight = 1e-4
        robot.ik.tasks['L_SHOULDER_R'].weight = 1e-4
        robot.ik.tasks['R_SHOULDER_P'].weight = 1e-4
        robot.ik.tasks['R_SHOULDER_R'].weight = 1e-4
    else:  # don't fix shoulders by default
        robot.ik.tasks['L_SHOULDER_P'].weight = 1e-5
        robot.ik.tasks['L_SHOULDER_R'].weight = 1e-5
        robot.ik.tasks['R_SHOULDER_P'].weight = 1e-5
        robot.ik.tasks['R_SHOULDER_R'].weight = 1e-5
    robot.ik.tasks['POSTURE'].weight = 1e-5


def set_camera_1():
    """Camera used to make the accompanying video."""
    sim.viewer.SetCamera([
        [0., -0.25, 1., -4],
        [-1., 0., 0., 0.],
        [-0., -1., -0.25, 2.7],
        [0, 0, 0, 1]])


def set_camera_2():
    """Camera used to make the accompanying video."""
    sim.viewer.SetCamera([
        [0.15, -0.2, 1., -3.5],
        [-1., 0., 0.25, 0.5],
        [0., -1., -0.2, 3.],
        [0, 0, 0, 1]])


def record_video():
    cam_recorder = CameraRecorder(sim, 'camera')
    sim.schedule_extra(cam_recorder)
    set_camera_1()

    def callback():
        cam_recorder.on_tick(sim)

    raw_input("Press [Enter] to start.\n")
    sim.step(425)
    # while raw_input("Is this the correct frame? [y/N] ").lower() != 'y':
    while not wpg.is_in_double_support:  # finish step
        sim.step()
    while wpg.is_in_double_support:  # now get right after the DS phase
        sim.step()
    cam_recorder.wait_for(2, sim)
    robot.hide()
    wrench_drawer.hide()
    cam_recorder.wait_for(5, sim)
    wpg.com_mpc.preview.play(sim, post_preview_duration=0.8, callback=callback)
    cam_recorder.wait_for(2, sim)
    wrench_drawer.show()
    cam_recorder.wait_for(5, sim)
    wpg.com_mpc.preview.play(
        sim, wrench_drawer=wrench_drawer, post_preview_duration=0.8,
        callback=callback)
    cam_recorder.wait_for(2, sim)
    for _ in xrange(15):
        sim.step()
        cam_recorder.on_tick(sim)
        cam_recorder.on_tick(sim)
    cam_recorder.wait_for(2, sim)
    robot.show()
    cam_recorder.wait_for(2, sim)
    wpg.com_mpc.preview.handles = None
    sim.step(600)


if __name__ == "__main__":
    random.seed(51)
    sim = pymanoid.Simulation(dt=0.03)
    try:
        robot = pymanoid.robots.HRP4()
    except:  # model not found
        robot = pymanoid.robots.JVRC1()
    sim.set_viewer()
    robot.set_transparency(0.3)
    contact_model = pymanoid.Contact(
        shape=(0.12, 0.06), static_friction=0.8, visible=False,
        name="ContactModel")
    contact_feed = EllipticStaircase(
        robot, radius=1.4, angular_step=0.5, height=1.2, roughness=0.5,
        contact_model=contact_model, visible=True)
    # contact_feed = HorizontalFloor(robot, 10, 0.6, 0.3, contact_model)
    generate_posture()
    pendulum = FIP(
        robot.mass, omega2=9.81 / robot.leg_length, com=robot.com,
        zmp_delay=ZMP_DELAY, zmp_noise=ZMP_NOISE)
    state_estimator = StateEstimator(
        pendulum, ESTIMATION_DELAY, COM_NOISE, COMD_NOISE)
    wpg = WalkingPatternGenerator(
        robot, state_estimator, pendulum, contact_feed, FORWARD_VELOCITY,
        NB_MPC_STEPS, NB_LQR_STEPS, MAX_SWING_FOOT_ACCEL)
    wpg.draw_support_tube(DRAW_DS_TUBE)
    wpg.set_swing_ik_weights(MIN_SWING_IK_WEIGHT, MAX_SWING_IK_WEIGHT)
    wpg.simulate_instant_mpc(SIMULATE_INSTANT_MPC)
    update_robot_ik()

    sim.schedule(state_estimator)  # should be called before wpg and pendulum
    sim.schedule(wpg)              # should be called before pendulum
    sim.schedule(pendulum)         # end of topological sorting ;)

    com_traj_drawer = TrajectoryDrawer(pendulum.com_state, 'b-')
    lf_traj_drawer = TrajectoryDrawer(robot.left_foot, 'g-')
    rf_traj_drawer = TrajectoryDrawer(robot.right_foot, 'r-')
    preview_drawer = PreviewDrawer()
    stats_collector = StatsCollector(count_nonqs=False)
    wrench_drawer = WrenchDrawer()

    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(lf_traj_drawer)
    sim.schedule_extra(preview_drawer)
    sim.schedule_extra(rf_traj_drawer)
    sim.schedule_extra(robot.ik)
    sim.schedule_extra(stats_collector)
    sim.schedule_extra(wrench_drawer)

    if '--record' in sys.argv:
        record_video()
    elif IPython.get_ipython() is None:
        IPython.embed()
