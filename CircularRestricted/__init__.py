""" Zack Funke made this to explore the feasibility
    of controlling a constellation of refractive
    sailcraft at L1 to mitigate catastrophic warming,
    possibly as an augmentation to MIT's 'Space Bubbles'
    idea (https://senseable.mit.edu/space-bubbles/).
"""
import numpy as np
from poliastro.bodies import Sun, Moon, Earth
from poliastro.constants import GM_sun, GM_earth, GM_moon
from poliastro.threebody import restricted
from astropy import units as u
from skyfield.api import load
from skyfield.planetarylib import PlanetaryConstants
import matplotlib.pyplot as plt
import matplotlib as mpl
from poliastro.bodies import Sun, Moon, Earth
from poliastro.constants import GM_sun, GM_earth, GM_moon
from skyfield.api import load
from skyfield.planetarylib import PlanetaryConstants
from .core import *
# from utils import *

__version__ = '0.0.1'
