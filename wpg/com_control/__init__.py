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

from cop_nmpc import COPPredictiveController
from double_support import DoubleSupportController
from fip_nmpc import FIPPredictiveController
from regulation import FIPRegulator
from wrench_nmpc import WrenchPredictiveController

__all__ = [
    'COPPredictiveController',
    'DoubleSupportController',
    'FIPPredictiveController',
    'FIPRegulator',
    'WrenchPredictiveController',
]
