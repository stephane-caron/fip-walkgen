# FIP Dynamic Walking

Source code for https://hal.archives-ouvertes.fr/hal-01481052

## Abstract

We present a real-time rough-terrain dynamic walking pattern generator. Our
method automatically finds step durations, which is a critical issue over rough
terrains where they depend on terrain topology. To achieve this level of
generality, we introduce the Floating-base Inverted Pendulum (FIP) model where
the center of mass can translate freely and the zero-tilting moment point is
allowed to leave the contact surface. We show that this model is equivalent to
the linear-inverted pendulum mode with variable center of mass height, aside
from the fact that its equations of motion remain linear. Our design then
follows three steps: (i) we characterize the FIP contact-stability condition;
(ii) we compute feedforward controls by solving a nonlinear optimization over
receding-horizon FIP trajectories. Despite running at 30 Hz in a
model-predictive fashion, simulations show that the latter is too slow to
stabilize dynamic motions. To remedy this, we (iii) linearize FIP feedback
control computations into a quadratic program, resulting in a constrained
linear-quadratic regulator that runs at 300 Hz. We finally demonstrate our
solution in simulations with a model of the HRP-4 humanoid robot, including
noise and delays over both state estimation and foot force control. 

<img src="https://scaron.info/figures/dynamic-walking.png" height="400" />

Authors:
[Stéphane Caron](https://scaron.info) and
[Abderrahmane Kheddar](http://www.lirmm.fr/lirmm_eng/users/utilisateurs-lirmm/equipes/idh/abderrahmane-kheddar)

## Installation

On Ubuntu 14.04, once you have [installed
OpenRAVE](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html),
do:

```bash
sudo apt-get install cython python python-dev python-pip python-scipy python-shapely
sudo pip install pycddlib quadprog pyclipper
```

Then, clone the repository and its submodule via:

```bash
git clone --recursive https://github.com/stephane-caron/dynamic-walking.git
```

If you already have [pymanoid](https://github.com/stephane-caron/pymanoid)
installed on your system, be sure that its version matches that of the
submodule.

## Questions?

Feel free to post your questions or comments in the issue tracker.