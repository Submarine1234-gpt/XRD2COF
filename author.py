import numpy as np
import os
from multiprocessing import Pool
from pymatgen.core import Structure
from pymatgen.analysis.diffraction import xrd
import random
import math

def get_xrd(new_structure, name):
    
        the_stru = new_structure
        calculator = xrd.XRDCalculator()
        pattern = calculator.get_pattern(the_stru, two_theta_range=(5.0, 110))

        angles, intensities = pattern.x, pattern.y
        #steps = np.linspace(5.0, 110, 5250)
        steps = np.linspace(2,30,554)
        signals = np.zeros(steps.shape[0])
        possible_domains = np.linspace(1, 100)
        tau = random.choice(possible_domains)

        for i, ang in enumerate(angles):
            idx = np.argmin(np.abs(ang - steps))
            signals[idx] = intensities[i]

        conv = []
        for (ang, int) in zip(steps, signals):
            if int != 0:
                K = 0.9
                wavelength = 0.15406
                theta = math.radians(ang / 2.)
                beta = (K / wavelength) * (math.cos(theta) / tau)

                std_dev = beta / 2.35482

                gauss = [int * np.exp((-(val - ang) ** 2) / std_dev) for val in steps]
                conv.append(gauss)

        mixed_data = zip(*conv)
        all_I = []
        for values in mixed_data:
            noise = random.choice(np.linspace(-0.75, 0.75, 1000))
            all_I.append(sum(values) + noise)

        shifted_vals = np.array(all_I) - min(all_I)
        scaled_vals = 100 * np.array(shifted_vals) / max(shifted_vals)
        all_I = [val for val in scaled_vals]
        return steps, np.array(all_I)

import matplotlib.pyplot

a="all_cifs/5.cif"
b= Structure.from_file(a)
c=get_xrd(b,"5")
matplotlib.pyplot.plot(c[0],c[1])
matplotlib.pyplot.show()