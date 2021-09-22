import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as plt


crystal = xu.materials.Crystal.fromCIF("test.cif")

powder = xu.simpack.Powder(
    crystal,
    1,
    crystallite_size_lor=2e-07,  # default
    crystallite_size_gauss=2e-07,  # default
    strain_lor=0,
    strain_gauss=0,
    preferred_orientation=(0, 0, 0),
    preferred_orientation_factor=1,
)

# Further information on the settings: https://nvlpubs.nist.gov/nistpubs/jres/120/jres.120.014.c.py
powder_model = xu.simpack.PowderModel(
    powder,
    I0=100,
    fpsettings={
        "classoptions": {
            "anglemode": "twotheta",
            "oversampling": 4,
            "gaussian_smoother_bins_sigma": 1.0,
            "window_width": 20.0,
        },
        "global": {
            "geometry": "symmetric",
            "geometry_incidence_angle": None,
            "diffractometer_radius": 0.3,
            "equatorial_divergence_deg": 0.5,
            # "dominant_wavelength": 1.207930e-10, # this is a read-only setting!
        },
        "emission": {
            "emiss_wavelengths": (1.207930e-10),
            "emiss_intensities": (1.0),
            "emiss_gauss_widths": (3e-14),
            "emiss_lor_widths": (3e-14),
            # "crystallite_size_lor": 2e-07,  # this needs to be set for the powder
            # "crystallite_size_gauss": 2e-07,  # this needs to be set for the powder
            # "strain_lor": 0,  # this needs to be set for the powder
            # "strain_gauss": 0,  # this needs to be set for the powder
            # "preferred_orientation": (0, 0, 0),  # this needs to be set for the powder
            # "preferred_orientation_factor": 1,  # this needs to be set for the powder
        },
        "axial": {
            "axDiv": "full",
            "slit_length_source": 0.008001,
            "slit_length_target": 0.008,
            "length_sample": 0.01,
            "angI_deg": 2.5,
            "angD_deg": 2.5,
            "n_integral_points": 10,
        },
        "absorption": {"absorption_coefficient": 100000.0},
        "si_psd": {"si_psd_window_bounds": None},
        "receiver_slit": {"slit_width": 5.5e-05},
        "tube_tails": {
            "main_width": 0.0002,
            "tail_left": -0.001,
            "tail_right": 0.001,
            "tail_intens": 0.001,
        },
    },
)

xs = np.arange(0, 90, 0.01)
diffractogram = powder_model.simulate(xs)

# Default settings:
# print(powder_model.pdiff[0].settings)

powder_model.plot(xs)

# plt.savefig("test.pdf")

plt.show()
