# 3D walking by NMPC of the floating-base pendulum

Source code for https://hal.archives-ouvertes.fr/hal-01481052/document

## Installation

The following instructions were tested on Ubuntu 14.04 and 16.04.

- Install OpenRAVE: here are [instructions for Ubuntu 14.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html) as well as [for Ubuntu 16.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html)
- Install Python and related dependencies: ``sudo apt-get install cython python python-dev python-pip python-scipy python-shapely``
- Install Python packages: ``sudo pip install pycddlib quadprog pyclipper``
- Install [CasADi](http://casadi.org). Pre-compiled binaries are available, but
  I recommend you [build it from
  source](https://github.com/casadi/casadi/wiki/InstallationLinux). When
  installing IPOPT, make sure to install the MA27 linear solver
  (``ThirdParty/HSL`` folder).

- Install [TOPP](https://github.com/quangounet/TOPP.git):
```bash
git clone https://github.com/quangounet/TOPP.git
cd TOPP && mkdir build && cd build
cmake ..
sudo make install
```

Finally, clone this repository and its submodule via:

```bash
git clone --recursive https://github.com/stephane-caron/fip-walkgen.git
```

If you already have [pymanoid](https://github.com/stephane-caron/pymanoid)
installed on your system, make sure to clone submodules and run the ``walk.py``
script from this repository (so that it uses the submodule rather than
system-wide pymanoid version).

## Usage

Run the main script via:
- ``./walk.py -e`` for the elliptic staircase scenario
- ``./walk.py -r`` for a regular staircase

## Questions?

Feel free to post your questions or comments in the issue tracker.
