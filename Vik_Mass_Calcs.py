#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Import necessary libraries for scientific computation, file handling, and visualization
import numpy as np
import h5py
import matplotlib.pyplot as plt
from colossus.utils import constants
from colossus.cosmology import cosmology
from scipy.optimize import curve_fit
from tabulate import tabulate
from scipy.integrate import quad
from scipy.interpolate import interp1d

# SIMULATION DEFINITIONS
# ====================================================================================
# Define filenames for gas profile data and load them using h5py for TNG100 and TNG300 simulations
gasprofs100_v5_filename = 'groups_gasprofs_v5_tng100_099.hdf5'
gasprofs100_v5_vals = h5py.File(gasprofs100_v5_filename, 'r', libver='earliest', swmr=True)

gasprofs300_v5_filename = 'groups_gasprofs_v5_tng300_099.hdf5'
gasprofs300_v5_vals = h5py.File(gasprofs300_v5_filename, 'r', libver='earliest', swmr=True)

# Critical density of the universe in units of kpc^-3
rho_crit = constants.RHO_CRIT_0_KPC3

# Define a dictionary containing information about galaxy clusters:
# Cluster name, redshift (z), and R500 radius (kpc)
cluster_dict = {
    'A133': ('A133', 0.0569, 1007),
    'A262': ('A262', 0.0162, 650),
    'A383': ('A383', 0.1883, 944),
    # ... other clusters ...
}

# Example list of clusters to analyze
clusters = ['A2390', 'A133']

# Set cosmological parameters using Colossus library for calculations
Vik_Cosmo = cosmology.setCosmology('Vik_Cosmo', params=cosmology.cosmologies['planck18'],
                                   Om0=0.3, Ode0=0.7, H0=72, sigma8=0.9)
print(Vik_Cosmo)

# Constants used in density calculations
gamma = 3
mp = constants.M_PROTON  # Mass of proton
G = constants.G  # Gravitational constant in SI units
G_CGS = constants.G_CGS  # Gravitational constant in CGS units

# Function to calculate normalized gas density and normalized total density
def get_rho_norm(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    # Compute electron density using a double beta model
    npne = ((n0 * (10**-3))**2) * (((r / rc)**(-a)) / ((1 + (r**2) / (rc**2))**(3 * B + (-a / 2)))) \
        * (1 / ((1 + ((r**gamma) / (rs**gamma)))**(epsilon / gamma))) \
        + (((n02 * (10**-1))**2) / ((1 + (r**2) / (rc2**2))**(3 * B2)))

    # Calculate gas density from electron density
    rho_g = 1.624 * mp * (npne)**(1/2)

    # Compute critical density of the universe at the cluster's redshift
    H_col = Vik_Cosmo.Hz(z)
    h_col = H_col * (10**-2)
    rho_c_col = Vik_Cosmo.rho_c(z) * (constants.MSUN) * ((h_col)**2) / (constants.KPC**3)

    # Normalize the gas density by the critical density
    rho_norm_col = rho_g / rho_c_col

    return rho_g, rho_norm_col

# Load simulation data, normalize radii and densities for TNG100 and TNG300
radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
R_Crit500_100 = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])
radii_all100_norm = radii_all100 / R_Crit500_100[:, None]
rho_vals100 = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'])
rho_vals100_norm = rho_vals100 / rho_crit

radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
R_Crit500_300 = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
radii_all300_norm = radii_all300 / R_Crit500_300[:, None]
rho_vals300 = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'])
rho_vals300_norm = rho_vals300 / rho_crit

# MASS CALCULATIONS
# ====================================================================================
# Integrate the gas density profile to calculate the gas mass within R500 for each cluster
def integrand(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    rho_g, rho_norm = get_rho_norm(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2)
    return rho_g * 4 * np.pi * (r**2)

# Median gas mass calculations for TNG100 and TNG300
volume_bins100 = np.array(gasprofs100_v5_vals['profile_bins_volume'])
volume_bins300 = np.array(gasprofs300_v5_vals['profile_bins_volume'])

gas_mass_array100 = np.zeros(100)
for j in range(0, 100):
    i = np.argmax(radii_all100_norm[j, :] >= 1)  # Find index for R500
    gas_mass_array100[j] = np.sum(volume_bins100[j, :i] * rho_vals100[j, :i])
median_gas_mass100 = np.median(gas_mass_array100)

# Similarly, calculate for TNG300
gas_mass_array300 = np.zeros(100)
for j in range(0, 100):
    i = np.argmax(radii_all300_norm[j, :] >= 1)
    gas_mass_array300[j] = np.sum(volume_bins300[j, :i] * rho_vals300[j, :i])
median_gas_mass300 = np.median(gas_mass_array300)

# PLOTTING
# ====================================================================================
# Create a scatter plot comparing gas mass and total mass for different clusters
plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10, 8), dpi=200)
ax = plt.subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")

# Plot TNG100 and TNG300 data points
for i in range(len(gas_mass_array100)):
    ax.scatter(TNG100_cluster_mass_tot[i], gas_mass_array100[i], color="#1F77B4")
    ax.scatter(TNG300_cluster_mass_tot[i], gas_mass_array300[i], color="#FF7F0E")

# Add labels and legend
plt.xlabel('M$_{total,500c}$', fontsize=20)
plt.ylabel('M$_{gas,500c}$', fontsize=18)
plt.title("Gas Mass vs. Total Mass", fontsize=30)
plt.legend(loc='best', fontsize=14)
plt.show()

