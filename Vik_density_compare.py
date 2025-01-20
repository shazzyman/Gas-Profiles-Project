#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script analyzes gas density profiles from galaxy clusters using TNG100 and TNG300 simulation data
# It computes normalized density profiles, performs mass integrations, and compares them to observational data.

# Import necessary libraries for computations, file handling, and visualization
import numpy as np
import h5py
import matplotlib.pyplot as plt
from colossus.utils import constants
from colossus.cosmology import cosmology
from scipy.interpolate import interp1d
from scipy.integrate import quad
from colossus.halo import mass_so

# SIMULATION DEFINITIONS
# ====================================================================================
# Define filenames for gas density profile data and load them using h5py
# These files contain information about gas density profiles from TNG100 and TNG300 simulations
gasprofs100_v5_filename = 'groups_gasprofs_v5_tng100_099.hdf5'
gasprofs100_v5_vals = h5py.File(gasprofs100_v5_filename, 'r', libver='earliest', swmr=True)

gasprofs300_v5_filename = 'groups_gasprofs_v5_tng300_099.hdf5'
gasprofs300_v5_vals = h5py.File(gasprofs300_v5_filename, 'r', libver='earliest', swmr=True)

# Define the critical density of the universe in kpc^-3, a reference density used for normalization
rho_crit = constants.RHO_CRIT_0_KPC3

# Extract radii data for TNG100 and normalize it by the R500 radius for each cluster
# R500 is the radius within which the average density is 500 times the critical density
radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
R_Crit500_100 = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])
radii_all100_norm = radii_all100 / R_Crit500_100[:, None]

# Extract gas density values for TNG100 and normalize them by the critical density
rho_vals100 = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'])
rho_vals100_norm = rho_vals100 / rho_crit

# Repeat the same process for TNG300 simulation data
radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
R_Crit500_300 = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
radii_all300_norm = radii_all300 / R_Crit500_300[:, None]

rho_vals300 = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'])
rho_vals300_norm = rho_vals300 / rho_crit

# Set cosmological parameters using the Colossus library
# These parameters are used to calculate properties such as the Hubble constant and critical density
Vik_Cosmo = cosmology.setCosmology('Vik_Cosmo', params=cosmology.cosmologies['planck18'],
                                   Om0=0.3, Ode0=0.7, H0=72, sigma8=0.9)
print(Vik_Cosmo)

# Define constants used in the calculations
gamma = 3  # Parameter used in density profile calculations
mp = constants.M_PROTON  # Mass of a proton in kg
G = constants.G  # Gravitational constant in SI units (m^3 kg^-1 s^-2)
G_CGS = constants.G_CGS  # Gravitational constant in CGS units (cm^3 g^-1 s^-2)

# Function to calculate normalized gas density and total density for a given radial distance
def get_rho_norm(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    # Calculate the electron density profile using a double beta model
    npne = ((n0 * (10**-3))**2) * (((r / rc)**(-a)) / ((1 + (r**2) / (rc**2))**(3 * B + (-a / 2)))) \
        * (1 / ((1 + ((r**gamma) / (rs**gamma)))**(epsilon / gamma))) \
        + (((n02 * (10**-1))**2) / ((1 + (r**2) / (rc2**2))**(3 * B2)))

    # Calculate the gas density from the electron density
    rho_g = 1.624 * mp * (npne)**(1/2)

    # Compute the critical density of the universe at a given redshift
    h_col = Vik_Cosmo.Ez(z)  # The Hubble parameter at redshift z
    rho_c_col = Vik_Cosmo.rho_c(z) * (constants.MSUN) * ((h_col)**2) * (constants.KPC**-3)

    # Normalize the gas density by the critical density
    rho_norm_col = rho_g / rho_c_col

    return rho_g, rho_norm_col

# CLUSTER DATA AND CALCULATIONS
# ====================================================================================
# Load observational parameters for galaxy clusters from a CSV file
n0, rc, rs, a, B, epsilon, n02, rc2, B2 = np.loadtxt('Vikhlinin_tab2.csv', skiprows=0, unpack=True, delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))

# Define a dictionary containing properties of galaxy clusters
# Each entry includes the cluster name, redshift (z), R500, and parameters for density profile calculations
cluster_dict = {
    'A133': ('A133', 0.0569, 1007, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02[0], rc2[0], B2[0], 3.17, 0.083, 40),
    # Additional clusters can be added here...
}

# Generate radial bins for density calculations and comparisons
# Radial range is based on observational data and simulation limits
Vik_bins = np.linspace(78, 23400, 500)

# Example: Calculate gas density and normalized density for the cluster A133
z = cluster_dict['A133'][1]  # Redshift of the cluster
rho_g_A133, rho_norm_Vik_A133 = get_rho_norm(Vik_bins, cluster_dict['A133'][3], cluster_dict['A133'][4],
                                             cluster_dict['A133'][5], cluster_dict['A133'][6], cluster_dict['A133'][7],
                                             cluster_dict['A133'][8], cluster_dict['A133'][9], cluster_dict['A133'][10],
                                             cluster_dict['A133'][11])

# Repeat calculations for other clusters (e.g., A262, A383, etc.)


# DENSITY INTERPOLATION
# ====================================================================================
# Define consistent radial bins for interpolating density profiles from TNG simulations
my_bins100 = np.logspace(np.log10(np.min(radii_all100_norm)), np.log10(np.max(radii_all100_norm)), 50)
my_bins100_centers = (my_bins100[:-1] + my_bins100[1:]) / 2.0

# Interpolate normalized density values for TNG100 simulation data
rho_vals_interp100 = np.empty((len(radii_all100_norm), len(my_bins100_centers)))
rho_vals_interp100[:] = np.nan  # Initialize with NaN for missing data handling

for cluster in range(len(radii_all100_norm)):
    # Create an interpolation function for each cluster
    rho_vals_interp_func = interp1d(radii_all100_norm[cluster], rho_vals100_norm[cluster], bounds_error=False, fill_value=np.nan)
    rho_vals_interp100[cluster] = rho_vals_interp_func(my_bins100_centers)

# Compute the median density profile for TNG100 by taking the median across all clusters
median_rho_100 = np.nanmedian(rho_vals_interp100, axis=0)

# MASS CALCULATIONS
# ====================================================================================
# Function to compute the total mass enclosed within a radius by integrating the density profile
def mass_integral(cluster_name):
    r = cluster_dict[cluster_name][2]  # R500 for the cluster
    n0, rc, rs, a, B, epsilon, n02, rc2, B2 = cluster_dict[cluster_name][3:12]  # Density profile parameters
    # Perform numerical integration over the radial range
    res, _ = quad(lambda r: 4 * np.pi * (r**2) * get_rho_norm(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2)[0], 0, r)
    return res / constants.MSUN  # Convert mass to solar masses

# Example: Compute mass for the cluster A133
mass_values = {'A133': mass_integral('A133')}
# Additional clusters can be added similarly...

# PLOTTING
# ====================================================================================
# Create a plot comparing TNG simulation density profiles and observational data
plt.figure(figsize=(10, 8))
plt.semilogy(my_bins100_centers, median_rho_100, label='TNG100 Median Density')  # Plot TNG100 median density
plt.semilogy(Vik_bins / cluster_dict['A133'][2], rho_norm_Vik_A133, label='A133 Observed Density')  # Plot observational data for A133
plt.xlabel('r/R500c')  # Label for x-axis
plt.ylabel(r'$\rho/\rho_c$')  # Label for y-axis
plt.title('Density Profiles: TNG Simulations vs Observations')  # Title of the plot
plt.legend()  # Add legend to the plot
plt.show()  # Display the plot
