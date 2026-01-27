r"""
Created on March 17, 2019 by Andrew Abi-Mansour
Updated to Martini 3 force field on January 31, 2025 by Magdalena Szczuka

This is the::
    _   _   _ _____ ___     __  __    _    ____ _____ ___ _   _ ___   __  __ _____
   / \ | | | |_   _/ _ \   |  \/  |  / \  |  _ \_   _|_ _| \ | |_ _|  |  \/  |___ /  
  / _ \| | | | | || | | |  | |\/| | / _ \ | |_) || |  | ||  \| || |   | |\/| | |_ \  
 / ___ \ |_| | | || |_| |  | |  | |/ ___ \|  _ < | |  | || |\  || |   | |  | |___) | 
/_/  _\_\___/  |_| \___/   |_|  |_/_/   \_\_| \_\|_| |___|_| \_|___|  |_|  |_|____/    
                                                

A tool for automatic MARTINI 3 force field mapping and parametrization of small organic molecules

Developers::
        Magdalena Szczuka (magdalena.szczuka at univ-tlse3.fr)
        Tristan BEREAU (bereau at mpip-mainz.mpg.de)
        Kiran Kanekal (kanekal at mpip-mainz.mpg.de)
        Andrew Abi-Mansour (andrew.gaam at gmail.com)

AUTO_MARTINI M3 is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENSE files.
"""

import os

from . import solver, topology
from ._version import __version__


def _select_optimization_module():
                """Select optimization backend.

                Controlled via env var:
                        AUTO_MARTINI_OPTIMIZATION=legacy  -> optimization_legacy
                        (anything else / unset)           -> optimization

                This is intentionally evaluated at import time so downstream modules can do:
                        from auto_martiniM3 import optimization
                """

                mode = os.getenv("AUTO_MARTINI_OPTIMIZATION", "")
                if mode is None:
                                mode = ""
                mode = str(mode).strip().lower()

                if mode in {"legacy", "old", }:
                                from . import optimization_legacy as optimization_module

                                return optimization_module

                from . import optimization as optimization_module

                return optimization_module


# Public alias
optimization = _select_optimization_module()

__all__ = ["solver", "topology", "optimization", "__version__"]
