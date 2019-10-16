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

import pylab

from pymanoid.misc import Statistics


"""
Tests
=====

The functions below are meant to be imported at runtime from an IPython shell.
"""


def test_discretization(nmpc, nb_steps):
    dT = nmpc.preview.dT
    pylab.ion()
    pylab.clf()
    ax = pylab.subplot(311)
    ax.set_color_cycle(['r', 'g', 'b'])
    pylab.plot(
        [sum(dT[:i]) for i in range(len(dT))],
        nmpc.preview.P, marker='o')
    pylab.plot(
        pylab.linspace(0., sum(dT), nb_steps + 1),
        [x[0:3] for x in nmpc.preview.discretize(nb_steps)],
        marker='s', linestyle='--')
    ax = pylab.subplot(312)
    ax.set_color_cycle(['r', 'g', 'b'])
    pylab.plot(
        [sum(dT[:i]) for i in range(len(dT))],
        nmpc.preview.V, marker='o')
    pylab.plot(
        pylab.linspace(0., sum(dT), nb_steps + 1),
        [x[3:6] for x in nmpc.preview.discretize(nb_steps)],
        marker='s', linestyle='--')
    ax = pylab.subplot(313)
    ax.set_color_cycle(['r', 'g', 'b'])
    pylab.plot(
        [sum(dT[:i]) for i in range(len(dT))],
        nmpc.preview.Z, marker='o')
    pylab.plot(
        pylab.linspace(0., sum(dT), nb_steps + 1),
        [x[6:9] for x in nmpc.preview.discretize(nb_steps)],
        marker='s', linestyle='--')


def test_dT_impact(xvals, f, nmpc, sim, start=0.1, end=0.8, step=0.02,
                   ymax=200, sample_size=100, label=None):
    """Used to generate Figure XX of the paper."""
    # c = raw_input("Did you remove iter/time caps in IPOPT settings? [y/N] ")
    # if c.lower() not in ['y', 'yes']:
    #     print("Then go ahead and do it.")
    #     return
    stats = [Statistics() for _ in range(len(xvals))]
    fails = [0. for _ in range(len(xvals))]
    pylab.ion()
    pylab.clf()
    for (i, dT) in enumerate(xvals):
        f(dT)
        for _ in range(sample_size):
            nmpc.on_tick(sim)
            if 'Solve' in nmpc.nlp.return_status:
                stats[i].add(nmpc.nlp.solve_time)
            else:  # max CPU time exceeded, infeasible problem detected, ...
                fails[i] += 1.
    yvals = [1000 * ts.avg if ts.avg is not None else 0. for ts in stats]
    yerr = [1000 * ts.std if ts.std is not None else 0. for ts in stats]
    pylab.bar(
        xvals, yvals, width=step, yerr=yerr, color='y', capsize=5,
        align='center', error_kw={'capsize': 5, 'elinewidth': 5})
    pylab.xlim(start - step / 2, end + step / 2)
    pylab.ylim(0, ymax)
    pylab.grid(True)
    if label is not None:
        pylab.xlabel(label, fontsize=24)
    pylab.ylabel('Comp. time (ms)', fontsize=20)
    pylab.tick_params(labelsize=16)
    pylab.twinx()
    yfails = [100. * fails[i] / sample_size for i in range(len(xvals))]
    pylab.plot(xvals, yfails, 'ro', markersize=12)
    pylab.plot(xvals, yfails, 'r--', linewidth=3)
    pylab.xlim(start - step / 2, end + step / 2)
    pylab.ylabel("Failure rate [%]", fontsize=20)
    pylab.tight_layout()


def test_dT_max_impact(nmpc, sim, start=0.1, end=0.8, step=0.02, ymax=200,
                       sample_size=200):
    xvals = pylab.arange(start, end + step, step)
    f = nmpc.update_dT_max
    label = '$\\Delta t_\\mathsf{max}$ (s)'
    return test_dT_impact(
        xvals, f, nmpc, sim, start, end, step, ymax, sample_size, label)


def test_dT_min_impact(nmpc, sim, start=0.002, end=0.2, step=0.002, ymin=200,
                       sample_size=20):
    xvals = pylab.arange(start, end + step, step)
    f = nmpc.update_dT_min
    label = '$\\Delta t_\\mathsf{min}$ (s)'
    return test_dT_impact(
        xvals, f, nmpc, sim, start, end, step, ymin, sample_size, label)


def test_nmpc_comp_times(nmpc, sim, nb_steps):
    stats = Statistics()
    for _ in range(nb_steps):
        nmpc.on_tick(sim)
        stats.add(nmpc.nlp.solve_time)
    print(stats.as_comp_times('ms'))
    return stats


def test_xprod(fsm, sim, nb_steps):
    stats = Statistics()
    for _ in range(nb_steps):
        sim.step()
        if fsm.com_lqr is not None:
            stats.add(fsm.com_lqr.xprod_ratio)
    print(stats)
