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

import pymanoid

from pymanoid import PointMass, Stance
from pymanoid.sim import gravity
from pymanoid.tasks import ContactTask, PoseTask

# from com_control import COPPredictiveController as PredictiveController
# from com_control import WrenchPredictiveController as PredictiveController
from com_control import DoubleSupportController
from com_control import FIPPredictiveController as PredictiveController
from com_control import FIPRegulator
from swing_foot_control import SwingFootController


class WalkingPatternGenerator(pymanoid.Process):

    """
    Main entry point for the WPG.

    Parameters
    ----------
    robot : Robot
        Controlled biped robot.
    state_estimator : StateEstimator
        Used by the predictive controller and regulator to read the COM state.
    pendulum : FIP
        Floating-base Inverted Pendulum.
    contact_feed : ContactFeed
        List of planned contacts.
    forward_velocity : scalar
        Forward walking velocity in [m] / [s].
    nb_mpc_steps : integer
        Number of variable-duration timesteps in the predictive controller.
    nb_lqr_steps : integer
        Number of fixed-duration timesteps in the regulator. Should be greater
        than the number of MPC steps.
    max_swing_foot_accel : scalar
        Maximum swing foot acceleration sent to the time-optimal controller.
    """

    def __init__(self, robot, state_estimator, pendulum, contact_feed,
                 forward_velocity, nb_mpc_steps, nb_lqr_steps,
                 max_swing_foot_accel):
        super(WalkingPatternGenerator, self).__init__()
        first_contact = contact_feed.pop()
        second_contact = contact_feed.pop()
        third_contact = contact_feed.pop()
        support_contact = second_contact
        swing_controller = SwingFootController(
            robot.left_foot, max_swing_foot_accel)
        swing_start = first_contact
        swing_target = third_contact
        com_target = PointMass(
            [0, 0, 0], 0.5 * robot.mass, color='g', visible=True)
        ds_com_target = PointMass(
            [0, 0, 0], 0.5 * robot.mass, color='b', visible=False)
        com_target.set_pos(swing_target.p + [0., 0., robot.leg_length])
        com_target.set_vel(forward_velocity * swing_target.t)
        time_to_heel_strike = 1.  # not computed yet
        com_mpc = PredictiveController(
            nb_mpc_steps, state_estimator, com_target,
            contact_sequence=[support_contact, swing_target],
            omega2=pendulum.omega2, swing_duration=time_to_heel_strike)
        self.__disable_lqr = False
        self.__draw_support_tube = False
        self.__ds_tube_handle = None
        self.__max_swing_weight = 2e-2
        self.__min_swing_weight = 1e-3
        self.__simulate_instant_mpc = False
        self.com_lqr = None
        self.com_mpc = None
        self.com_target = com_target
        self.contact_feed = contact_feed
        self.ds_com_target = ds_com_target
        self.forward_velocity = forward_velocity
        self.is_in_double_support = False
        self.last_mpc_success = None
        self.mpc_wait_time = 0.
        self.nb_ds_steps = nb_lqr_steps
        self.nb_lqr_steps = nb_lqr_steps
        self.nb_mpc_steps = nb_mpc_steps
        self.next_com_mpc = com_mpc
        self.pendulum = pendulum
        self.robot = robot
        self.startup_flag = True  # krooOon (ˆ(oo)ˆ)
        self.state_estimator = state_estimator
        self.strat_counts = {'cp': 0, 'ds': 0, 'lqr': 0, 'mpc': 0}
        self.support_contact = support_contact
        self.swing_controller = swing_controller
        self.swing_start = swing_start
        self.swing_target = swing_target
        #
        self.switch_controllers(None)
        self.switch_ik_tasks('SS')

    """
    Extra settings
    ==============
    """

    def disable_lqr(self):
        """Disable LQR correction in case NMPC fails."""
        self.__disable_lqr = True

    def draw_support_tube(self, activate):
        """
        Activate or deactive tube drawing in double support.

        Parameters
        ----------
        activate : bool
            Whether the setting should be activated or deactivated.
        """
        self.__draw_support_tube = activate

    def set_swing_ik_weights(self, w_min, w_max):
        self.__max_swing_weight = w_max
        self.__min_swing_weight = w_min

    def simulate_instant_mpc(self, activate):
        """
        Activate or deactive MPC with zero computation time.

        Parameters
        ----------
        activate : bool
            Whether the setting should be activated or deactivated.
        """
        self.__mpc_nowait = activate

    """
    Step Switching
    ==============
    """

    def switch_contacts(self):
        """Prepare contacts for next step."""
        self.swing_start = self.support_contact
        self.support_contact = self.swing_target
        self.swing_target = self.contact_feed.pop()

    def switch_controllers(self, sim):
        """Prepare swing and COM controllers for next step."""
        prev_foot = self.swing_controller.foot_link
        next_foot = self.robot.left_foot if 'right' in prev_foot.name.lower() \
            else self.robot.right_foot
        self.swing_controller.reset(next_foot, self.swing_target)
        self.com_target.set_pos(
            self.swing_target.p + [0., 0., self.robot.leg_length])
        self.com_target.set_vel(self.forward_velocity * self.swing_target.t)
        self.switch_com_mpc(sim)

    def switch_com_mpc(self, sim):
        self.com_mpc = self.next_com_mpc
        if self.com_mpc.nb_contacts == 2:
            self.next_com_mpc = PredictiveController(
                self.nb_mpc_steps / 2, self.state_estimator, self.com_target,
                contact_sequence=[self.swing_target],
                omega2=self.pendulum.omega2)
        else:  # self.com_mpc.nb_contacts == 1:
            self.next_com_mpc = PredictiveController(
                self.nb_mpc_steps, self.state_estimator, self.com_target,
                contact_sequence=[self.support_contact, self.swing_target],
                omega2=self.pendulum.omega2,
                swing_duration=self.swing_controller.time_to_heel_strike)
        if sim is not None:  # otherwise, being called by constructor
            sim.log_comp_time('mpc_build', self.next_com_mpc.build_time)

    def switch_ik_tasks(self, phase):
        """Prepare robot IK for next step."""
        assert phase in ['SS', 'DS']
        prev_lf_task = self.robot.ik.get_task(self.robot.left_foot.name)
        prev_rf_task = self.robot.ik.get_task(self.robot.right_foot.name)
        contact_weight = max(prev_lf_task.weight, prev_rf_task.weight)
        self.robot.ik.remove_task(self.robot.left_foot.name)
        self.robot.ik.remove_task(self.robot.right_foot.name)
        OtherTask = PoseTask if phase == 'SS' else ContactTask
        other_weight = self.__min_swing_weight if phase == 'SS' else \
            contact_weight
        if 'left' in self.support_contact.link.name.lower():
            left_foot_task = ContactTask(
                self.robot, self.robot.left_foot, self.support_contact,
                weight=contact_weight)
            right_foot_task = OtherTask(
                self.robot, self.robot.right_foot, self.swing_controller.foot,
                weight=other_weight)
            link_pose_task = right_foot_task
        else:  # support contact is on the right foot
            left_foot_task = OtherTask(
                self.robot, self.robot.left_foot, self.swing_controller.foot,
                weight=other_weight)
            right_foot_task = ContactTask(
                self.robot, self.robot.right_foot, self.support_contact,
                weight=contact_weight)
            link_pose_task = left_foot_task
        self.robot.ik.add_task(left_foot_task)
        self.robot.ik.add_task(right_foot_task)
        self.link_pose_task = link_pose_task

    """
    Control Loop
    ============
    """

    def on_tick(self, sim):
        """
        Update function called at each tick of the control loop.

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
        """
        just_switched = False
        if self.swing_controller.finished and not self.is_in_double_support:
            self.is_in_double_support = True
            self.ds_com_target.set_pos(self.com_target.p)
            self.ds_com_target.set_vel(0. * self.com_target.pd)
            self.switch_contacts()
            self.switch_controllers(sim)
            self.switch_ik_tasks('DS')
            just_switched = True
        self.update_time_to_heel_strike()
        self.update_com_mpc(sim)
        self.update_next_com_mpc(sim)
        if self.next_com_mpc.is_ready_to_switch and not just_switched:
            self.is_in_double_support = False
            self.__ds_tube_handle = None
            self.switch_com_mpc(sim)
            self.last_mpc_success = self.com_mpc.preview
            self.switch_ik_tasks('SS')
        if self.is_in_double_support:
            self.com_lqr = None
            self.update_com_ds(sim)
        else:  # SS phase
            self.com_ds = None
            self.update_com_lqr(sim)
        if self.last_mpc_success is None:
            print("\033[1;33mWarning: won't move before MPC success\033[0;0m")
            return
        self.update_swing_foot(sim)
        self.last_mpc_success.forward(sim.dt)
        self.pendulum.next_zmp_target = self.get_zmp()

    def update_time_to_heel_strike(self):
        tths = self.swing_controller.time_to_heel_strike
        if tths is None:
            return
        self.com_mpc.update_swing_duration(tths)
        if self.next_com_mpc is not None:
            self.next_com_mpc.update_swing_duration(tths)

    def update_com_mpc(self, sim):
        if self.mpc_wait_time < 1e-10:
            self.com_mpc.on_tick(sim)
            self.mpc_wait_time = self.com_mpc.solve_time
        self.mpc_wait_time -= sim.dt
        if self.__simulate_instant_mpc:
            self.mpc_wait_time = 0.
        if self.startup_flag:
            # good behavior is to make swing controller wait for first preview
            # before starting next step, but I'm going for this dirty fix now
            self.startup_flag = False
        elif self.mpc_wait_time > 1e-10:
            return
        if self.com_mpc.nlp.optimal_found:
            # only update when optimal found => more stable
            self.last_mpc_success = self.com_mpc.preview
            sim.log_comp_time('mpc_solve', self.com_mpc.solve_time)

    def update_next_com_mpc(self, sim):
        if self.next_com_mpc is None:
            return
        if self.com_mpc.nb_contacts == 2:
            self.next_com_mpc.warm_start(self.com_mpc)
            return
        self.next_com_mpc.on_tick(sim)

    def update_com_lqr(self, sim):
        if self.last_mpc_success is None or self.last_mpc_success.is_empty \
                or self.__disable_lqr:
            return
        self.com_lqr = FIPRegulator(
            self.nb_lqr_steps, self.state_estimator, self.last_mpc_success)
        sim.log_comp_time('lqr_build', self.com_lqr.lmpc.build_time)
        sim.log_comp_time('lqr_solve', self.com_lqr.lmpc.solve_time)

    def update_com_ds(self, sim):
        try:
            stance = Stance(
                com=self.ds_com_target, left_foot=self.swing_start,
                right_foot=self.support_contact)
            self.com_ds = DoubleSupportController(
                self.nb_ds_steps, 0.5, self.pendulum.omega2,
                self.state_estimator, self.ds_com_target, stance)
            sim.log_comp_time('dsqp_build', self.com_ds.lmpc.build_time)
            if self.com_ds.lmpc.solve_time is not None:
                sim.log_comp_time('dsqp_solve', self.com_ds.lmpc.solve_time)
            if self.__draw_support_tube:
                self.__ds_tube_handle = self.com_ds.tube.draw()
        except RuntimeError as e:  # this one is cdd...
            print("%s error:" % type(self).__name__, e)
            self.preview = None
            self.tube = None

    def update_swing_foot(self, sim):
        if self.is_in_double_support:
            return
        if self.link_pose_task.weight < self.__max_swing_weight:
            self.link_pose_task.weight += 2e-3
        self.swing_controller.on_tick(sim)

    def get_zmp(self):
        zmp = None
        if self.is_in_double_support:
            if self.com_ds is not None and self.com_ds.preview is not None:
                zmp = self.com_ds.preview.get_zmp()
                self.strat_counts['ds'] += 1
            else:  # go for the last-resort strategy
                print("\033[1;33mWarning: sending emergency break!\033[0;0m")
                com = self.state_estimator.com
                comd = self.state_estimator.comd
                omega, omega2 = self.pendulum.omega, self.pendulum.omega2
                zmp = com + comd / omega + gravity / omega2
                self.strat_counts['cp'] += 1
        elif self.com_lqr is not None and self.com_lqr.optimal_found:
            zmp = self.com_lqr.preview.get_zmp()
            self.strat_counts['lqr'] += 1
        else:
            zmp = self.last_mpc_success.get_zmp()
            self.strat_counts['mpc'] += 1
        return zmp
