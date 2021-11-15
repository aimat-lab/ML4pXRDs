"""
This code originates from https://github.com/njszym/XRD-AutoAnalyzer

Probabilistic Deep Learning Approach to Automate the Interpretation of Multi-phase Diffraction Spectra
Nathan J. Szymanski, Christopher J. Bartel, Yan Zeng, Qingsong Tu, and Gerbrand Ceder
Chemistry of Materials 2021 33 (11), 4204-4215
DOI: 10.1021/acs.chemmater.1c01071

"""


from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import random
import numpy as np
from pymatgen.io.cif import CifParser
import matplotlib.pyplot as plt


class BroadGen(object):
    """
    Class used to simulate xrd spectra with broad peaks
        that are associated with small domain size
    """

    def __init__(
        self,
        struc,
        wavelength=1.207930,
        min_domain_size=1,
        max_domain_size=100,
        min_angle=10.0,
        max_angle=90.0,
    ):
        """
        Args:
            struc: structure to simulate augmented xrd spectra from
            min_domain_size: smallest domain size (in nm) to be sampled,
                leading to the broadest peaks
            max_domain_size: largest domain size (in nm) to be sampled,
                leading to the most narrow peaks
        """
        self.calculator = xrd.XRDCalculator(wavelength=wavelength)
        self.struc = struc

        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.pattern = self.calculator.get_pattern(
            struc, two_theta_range=(self.min_angle, self.max_angle)
        )

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensities(self):
        return self.pattern.y

    @property
    def hkl_list(self):
        return [v[0]["hkl"] for v in self.pattern.hkls]

    def calc_std_dev(self, two_theta, tau):
        """
        calculate standard deviation based on angle (two theta) and domain size (tau)
        Args:
            two_theta: angle in two theta space
            tau: domain size in nm
        Returns:
            standard deviation for gaussian kernel
        """
        ## Calculate FWHM based on the Scherrer equation
        K = 0.9  ## shape factor
        wavelength = self.calculator.wavelength * 0.1  ## angstrom to nm
        theta = np.radians(two_theta / 2.0)  ## Bragg angle in radians
        beta = (K * wavelength) / (np.cos(theta) * tau)  # in radians

        ## Convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
        return sigma ** 2

    def broadened_spectrum(self, domain_size=None, N=4008):

        angles = self.angles
        intensities = self.intensities

        steps = np.linspace(self.min_angle, self.max_angle, N)

        signals = np.zeros([len(angles), steps.shape[0]])

        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang - steps))
            signals[i, idx] = intensities[i]

        if domain_size is None:
            # Convolute every row with unique kernel
            # Iterate over rows; not vectorizable, changing kernel for every row
            domain_size = random.uniform(self.min_domain_size, self.max_domain_size)

        step_size = (self.max_angle - self.min_angle) / N
        for i in range(signals.shape[0]):
            row = signals[i, :]
            ang = steps[np.argmax(row)]
            std_dev = self.calc_std_dev(ang, domain_size)
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i, :] = gaussian_filter1d(
                row, np.sqrt(std_dev) * 1 / step_size, mode="constant"
            )

        # Combine signals
        signal = np.sum(signals, axis=0)

        # Normalize signal
        norm_signal = signal / max(signal)

        return norm_signal, domain_size


if __name__ == "__main__":

    parser = CifParser("/home/henrik/Dokumente/Big_Files/ICSD/cif/100.cif")
    crystals = parser.get_structures()
    crystal = crystals[0]

    broadener = BroadGen(
        crystal,
        wavelength=1.207930,
        min_domain_size=30,
        max_domain_size=90,
        min_angle=0,
        max_angle=90,
    )

    diffractogram, domain_size = broadener.broadened_spectrum(N=9001)

    peak_positions = broadener.angles
    peak_sizes = np.array(broadener.intensities) / np.max(broadener.intensities)

    plt.plot(np.linspace(0, 90, 9001), diffractogram)
    plt.show()
