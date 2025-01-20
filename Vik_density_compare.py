#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# Load data from TNG100 and TNG300 simulation gas profiles (HDF5 files)
gasprofs100_v5_filename = 'groups_gasprofs_v5_tng100_099.hdf5'
gasprofs100_v5_vals = h5py.File(gasprofs100_v5_filename, 'r', libver='earliest', swmr=True)

gasprofs300_v5_filename = 'groups_gasprofs_v5_tng300_099.hdf5'
gasprofs300_v5_vals = h5py.File(gasprofs300_v5_filename, 'r', libver='earliest', swmr=True)

# Define the critical density of the universe (kpc^-3)
rho_crit = constants.RHO_CRIT_0_KPC3

# Extract and normalize radii and density data from TNG100 simulation
radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
R_Crit500_100 = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])
radii_all100_norm = radii_all100 / R_Crit500_100[:, None]
rho_vals100 = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'])
rho_vals100_norm = rho_vals100 / rho_crit

# Extract and normalize radii and density data from TNG300 simulation
radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
R_Crit500_300 = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
radii_all300_norm = radii_all300 / R_Crit500_300[:, None]
rho_vals300 = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'])
rho_vals300_norm = rho_vals300 / rho_crit

# Define cosmology with parameters based on Planck 2018
Vik_Cosmo = cosmology.setCosmology(
    'Vik_Cosmo',
    params=cosmology.cosmologies['planck18'],
    Om0=0.3,
    Ode0=0.7,
    H0=72,
    sigma8=0.9
)
print(Vik_Cosmo)

# Define constants
gamma = 3  # Parameter for density profile
mp = constants.M_PROTON  # Proton mass
G = constants.G  # Gravitational constant
G_CGS = constants.G_CGS  # Gravitational constant in CGS units

# Define a function to calculate gas density and normalized density profiles
def get_rho_norm(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    npne = (
        ((n0 * (10**-3))**2)
        * (((r / rc)**(-a)) / ((1 + (r**2) / (rc**2))**(3 * B + (-a / 2))))
        * (1 / ((1 + ((r**gamma) / (rs**gamma)))**(epsilon / gamma)))
        + (((n02 * (10**-1))**2) / ((1 + (r**2) / (rc2**2))**(3 * B2)))
    )
    rho_g = 1.624 * mp * (npne)**(1 / 2)
    h_col = Vik_Cosmo.Ez(z)  # Cosmological scaling factor
    rho_c_col = Vik_Cosmo.rho_c(z) * (constants.MSUN) * ((h_col)**2) * (constants.KPC**-3)
    rho_norm_col = rho_g / rho_c_col
    return rho_g, rho_norm_col

#EXTRACT CLUSTER OBSERVATION DATA + PLUG AND CHUG
#====================================================================================
#CLUSTER DICTIONARY (CLUSTER NAME, Z, R500, n0, rc, rs, a, B, epsilon, n02, rc2, B2, M500, fg500, rmin)

n0, rc, rs, a, B, epsilon, n02, rc2, B2 = np.loadtxt('Vikhlinin_tab2.csv', skiprows = 0, unpack=True, delimiter = ',', usecols=(0,1,2,3,4,5,6,7,8))
                          
cluster_dict = {'A133': ('A133', 0.0569, 1007, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02[0], rc2[0], B2[0], 3.17, 0.083, 40),
                'A262': ('A262', 0.0162 , 650, n0[1], rc[1], rs[1], a[1], B[1], epsilon[1], n02[1], rc2[1], B2[1], None, None, 10),
                'A383': ('A383', 0.1883, 944, n0[2], rc[2], rs[2], a[2], B[2], epsilon[2], n02[2], rc2[2], B2[2], 3.06, 0.124, 25),
                'A478': ('A478', 0.0881, 1337, n0[3], rc[3], rs[3], a[3], B[3], epsilon[3], n02[3], rc2[3], B2[3], 7.68, 0.120, 30),
                'A907': ('A907', 0.1603, 1096, n0[4], rc[4], rs[4], a[4], B[4], epsilon[4], n02[4], rc2[4], B2[4], 4.56, 0.124, 40),
                'A1413':('A1413', 0.1429, 1299, n0[5], rc[5], rs[5], a[5], B[5], epsilon[5], n02[5], rc2[5], B2[5], 7.57, 0.107, 20),
                'A1795':('A1795', 0.0622, 1235, n0[6], rc[6], rs[6], a[6], B[6], epsilon[6], n02[6], rc2[6], B2[6], 6.03, 0.104, 40),
                'A1991':('A1991', 0.0592, 732, n0[7], rc[7], rs[7], a[7], B[7], epsilon[7], n02[7], rc2[7], B2[7], 1.23, 0.102, 10),
                'A2029':('A2029', 0.0779, 1362, n0[8], rc[8], rs[8], a[8], B[8], epsilon[8], n02[8], rc2[8], B2[8], 8.01, 0.123, 20),
                'A2390':('A2390', 0.2302, 1416, n0[9], rc[9], rs[9], a[9], B[9], epsilon[9], n02[9], rc2[9], B2[9], 10.74, 0.141, 80),
                'RX':('RX J1159+5531', 0.0810, 700, n0[10], rc[10], rs[10], a[10], B[10], epsilon[10], n02[10], rc2[10], B2[10], None, None, 10),
                'MKW 4':('MKW 4', 0.0199, 634, n0[11], rc[11], rs[11], a[11], B[11], epsilon[11], n02[11], rc2[11], B2[11], 0.77, 0.062, 5),
                'USGC S152':('USGC S152', 0.0153, None, n0[12], rc[12], rs[12], a[12], B[12], epsilon[12], n02[12], rc2[12], B2[12], None, None, 20)
        }

Vik_bins = np.linspace(78, 23400, 500) #Minimum and maximum radial bins in both  TNG 100 and 300


# Normalize radii for clusters

z=cluster_dict['A133'][1]
rho_g_A133, rho_norm_Vik_A133 = get_rho_norm(Vik_bins, cluster_dict['A133'][3], cluster_dict['A133'][4], cluster_dict['A133'][5], 
                                           cluster_dict['A133'][6], cluster_dict['A133'][7], cluster_dict['A133'][8], 
                                           cluster_dict['A133'][9], cluster_dict['A133'][10], cluster_dict['A133'][11])
z=cluster_dict['A262'][1]
rho_g_A262, rho_norm_Vik_A262 = get_rho_norm(Vik_bins, n0[1], rc[1], rs[1], a[1], B[1], epsilon[1], n02[1], rc2[1], B2[1])

z=cluster_dict['A383'][1]
rho_g_A383, rho_norm_Vik_A383 = get_rho_norm(Vik_bins, n0[2], rc[2], rs[2], a[2], B[2], epsilon[2], n02[2], rc2[2], B2[2])

z=cluster_dict['A478'][1]
rho_g_A478, rho_norm_Vik_A478 = get_rho_norm(Vik_bins, n0[3], rc[3], rs[3], a[3], B[3], epsilon[3], n02[3], rc2[3], B2[3])

z=cluster_dict['A907'][1]
rho_g_A907, rho_norm_Vik_A907 = get_rho_norm(Vik_bins, n0[4], rc[4], rs[4], a[4], B[4], epsilon[4], n02[4], rc2[4], B2[4])

z=cluster_dict['A1413'][1]
rho_g_A1413, rho_norm_Vik_A1413 = get_rho_norm(Vik_bins, n0[5], rc[5], rs[5], a[5], B[5], epsilon[5], n02[5], rc2[5], B2[5])

z=cluster_dict['A1795'][1]
rho_g_A1795, rho_norm_Vik_A1795 = get_rho_norm(Vik_bins, n0[6], rc[6], rs[6], a[6], B[6], epsilon[6], n02[6], rc2[6], B2[6])

z=cluster_dict['A1991'][1]
rho_g_A1991, rho_norm_Vik_A1991 = get_rho_norm(Vik_bins, n0[7], rc[7], rs[7], a[7], B[7], epsilon[7], n02[7], rc2[7], B2[7])

z=cluster_dict['A2029'][1]
rho_g_A2029, rho_norm_Vik_A2029 = get_rho_norm(Vik_bins, n0[8], rc[8], rs[8], a[8], B[8], epsilon[8], n02[8], rc2[8], B2[8])

z=cluster_dict['A2390'][1]
rho_g_A2390, rho_norm_Vik_A2390 = get_rho_norm(Vik_bins, n0[9], rc[9], rs[9], a[9], B[9], epsilon[9], n02[9], rc2[9], B2[9])

z=cluster_dict['RX'][1]
rho_g_RX, rho_norm_Vik_RX = get_rho_norm(Vik_bins, n0[10], rc[10], rs[10], a[10], B[10], epsilon[10], n02[10], rc2[10], B2[10])

z=cluster_dict['MKW 4'][1]
rho_g_MKW, rho_norm_Vik_MKW = get_rho_norm(Vik_bins, n0[11], rc[11], rs[11], a[11], B[11], epsilon[11], n02[11], rc2[11], B2[11])

# Uncomment the following line to calculate the emission profile for the USGC S152 cluster
# emis_profile_USGC = get_rho_norm(Vik_bins, n0[12], rc[12], rs[12], a[12], B[12], epsilon[12], n02[12], rc2[12], B2[12])


m_radii_norm_A133 = Vik_bins/cluster_dict['A133'][2]
m_radii_norm_A262 = Vik_bins/cluster_dict['A262'][2]
m_radii_norm_A383 = Vik_bins/cluster_dict['A383'][2]
m_radii_norm_A478 = Vik_bins/cluster_dict['A478'][2]
m_radii_norm_A907 = Vik_bins/cluster_dict['A907'][2]
m_radii_norm_A1413 = Vik_bins/cluster_dict['A1413'][2]
m_radii_norm_A1795 = Vik_bins/cluster_dict['A1795'][2]
m_radii_norm_A1991 = Vik_bins/cluster_dict['A1991'][2]
m_radii_norm_A2029 = Vik_bins/cluster_dict['A2029'][2]
m_radii_norm_A2390 = Vik_bins/cluster_dict['A2390'][2]
m_radii_norm_RX = Vik_bins/cluster_dict['RX'][2]
m_radii_norm_MKW = Vik_bins/cluster_dict['MKW 4'][2]

# Median Density Calculations
# ====================================================================================
# This section calculates the median density profiles for TNG100 and TNG300 clusters.

# TNG100: Define radial bins based on normalized radii
my_bins100 = [np.min(radii_all100_norm[:, 0]), np.max(radii_all100_norm[:, -1])]
my_bins100 = np.logspace(np.log10(my_bins100[0]), np.log10(my_bins100[1]), 50)
my_bins100_centers = (my_bins100[:-1] + my_bins100[1:]) / 2.0

# Prepare lists to store bins for each cluster and their midpoints
bins_4_clus100 = [[] for _ in range(100)]
bins_4_clus100_centers = [[] for _ in range(100)]

# Assign bins to each cluster and calculate midpoints
for cluster in range(len(radii_all100_norm)):
    mask = (my_bins100 >= radii_all100_norm[cluster][0]) & (my_bins100 <= radii_all100_norm[cluster][-1])
    bins_4_clus100[cluster].extend(my_bins100[mask])

    for i in range(len(bins_4_clus100[cluster]) - 1):
        midpoints = (bins_4_clus100[cluster][i] + bins_4_clus100[cluster][i + 1]) / 2
        bins_4_clus100_centers[cluster].append(midpoints)

# Interpolate density values for each cluster
rho_vals_interp100 = np.empty((len(radii_all100_norm), len(my_bins100_centers)))
rho_vals_interp100[:] = np.nan  # Initialize with NaN

for cluster in range(len(radii_all100_norm)):
    rho_vals_interp_func = interp1d(radii_all100_norm[cluster], rho_vals100_norm[cluster], bounds_error=False, fill_value=np.nan)
    for j in range(len(my_bins100) - 1):
        if bins_4_clus100[cluster][0] == my_bins100[j]:
            interp_values = rho_vals_interp_func(bins_4_clus100_centers[cluster])
            rho_vals_interp100[cluster, j:j + len(interp_values)] = interp_values

# Remove bins with too many NaN values
for j in range(len(my_bins100_centers)):
    if np.sum(np.isnan(rho_vals_interp100[:, j])) > 95:
        rho_vals_interp100[:, j] = np.nan

# Calculate the median density for TNG100 clusters
median_rho_100 = np.nanmedian(rho_vals_interp100, axis=0)

# TNG300: Repeat the process for TNG300 clusters
# Extract and normalize radii for TNG300
radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
R_Crit500_300 = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
radii_all300_norm = radii_all300 / R_Crit500_300[:, None]

# Define radial bins and their centers
my_bins300 = [np.min(radii_all300_norm[:, 0]), np.max(radii_all300_norm[:, -1])]
my_bins300 = np.logspace(np.log10(my_bins300[0]), np.log10(my_bins300[1]), 50)
my_bins300_centers = (my_bins300[:-1] + my_bins300[1:]) / 2.0

# Extract normalized gas density values
rho_vals300 = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'])
rho_vals300_norm = rho_vals300 / rho_crit

# Prepare bins for TNG300 clusters
bins_4_clus300 = [[] for _ in range(100)]
bins_4_clus300_centers = [[] for _ in range(100)]

for cluster in range(len(radii_all300_norm)):
    mask = (my_bins300 >= radii_all300_norm[cluster][0]) & (my_bins300 <= radii_all300_norm[cluster][-1])
    bins_4_clus300[cluster].extend(my_bins300[mask])

    for i in range(len(bins_4_clus300[cluster]) - 1):
        midpoints = (bins_4_clus300[cluster][i] + bins_4_clus300[cluster][i + 1]) / 2
        bins_4_clus300_centers[cluster].append(midpoints)

# Interpolate density values for TNG300 clusters
rho_vals_interp300 = np.empty((len(radii_all300_norm), len(my_bins300_centers)))
rho_vals_interp300[:] = np.nan  # Initialize with NaN

for cluster in range(len(radii_all300_norm)):
    rho_vals_interp_func = interp1d(radii_all300_norm[cluster], rho_vals300_norm[cluster], bounds_error=False, fill_value=np.nan)
    for j in range(len(my_bins300) - 1):
        if bins_4_clus300[cluster][0] == my_bins300[j]:
            interp_values = rho_vals_interp_func(bins_4_clus300_centers[cluster])
            rho_vals_interp300[cluster, j:j + len(interp_values)] = interp_values

# Remove bins with too many NaN values
for j in range(len(my_bins300_centers)):
    if np.sum(np.isnan(rho_vals_interp300[:, j])) > 95:
        rho_vals_interp300[:, j] = np.nan

# Calculate the median density for TNG300 clusters
median_rho_300 = np.nanmedian(rho_vals_interp300, axis=0)


# MASS DERIVATION
# ====================================================================================
# This section calculates the mass of galaxy clusters by integrating gas density profiles and compares
# the results with alternative methods, including the Colossus library and Vikhlinin model.

# Define a function to compute the surface area for a sphere at radius `r`
def SA_func(r):
    return 4 * np.pi * (r ** 2)

# Define a function to integrate the mass profile for a given cluster
def mass_integral(cluster_name):
    # Extract cluster parameters
    r = cluster_dict[cluster_name][2]  # R500 radius
    n0, rc, rs, a, B, epsilon, n02, rc2, B2 = (
        cluster_dict[cluster_name][3], cluster_dict[cluster_name][4], cluster_dict[cluster_name][5], 
        cluster_dict[cluster_name][6], cluster_dict[cluster_name][7], cluster_dict[cluster_name][8], 
        cluster_dict[cluster_name][9], cluster_dict[cluster_name][10], cluster_dict[cluster_name][11]
    )
    # Convert normalization parameters to kpc^3
    n0_kpc = n0 * (constants.KPC ** 3)
    n02_kpc = n02 * (constants.KPC ** 3)
    # Integrate the density profile
    res, _ = quad(lambda r: SA_func(r) * get_rho_norm(r, n0_kpc, rc, rs, a, B, epsilon, n02_kpc, rc2, B2)[0], 0, r)
    return res / constants.MSUN  # Return mass in solar masses

# Calculate mass values for specific clusters using the integral method
mass_values = {
    'A133': mass_integral('A133'),
    'A383': mass_integral('A383'),
    'A478': mass_integral('A478'),
    'A907': mass_integral('A907'),
    'A1413': mass_integral('A1413'),
    'A1795': mass_integral('A1795'),
    'A1991': mass_integral('A1991'),
    'A2029': mass_integral('A2029'),
    'A2390': mass_integral('A2390'),
    'MKW 4': mass_integral('MKW 4')
}


# Calculate mass values using the Colossus library
colosus_values = {
    'A133': mass_so.R_to_M(cluster_dict['A133'][2]*Vik_Cosmo.Ez(cluster_dict['A133'][1]), cluster_dict['A133'][1],  '500c'),
    'A383': mass_so.R_to_M(cluster_dict['A383'][2]*Vik_Cosmo.Ez(cluster_dict['A383'][1]), cluster_dict['A383'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A383'][1]) * cluster_dict['A383'][13],
    'A478': mass_so.R_to_M(cluster_dict['A478'][2]*Vik_Cosmo.Ez(cluster_dict['A478'][1]), cluster_dict['A478'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A478'][1]) * cluster_dict['A478'][13],
    'A907': mass_so.R_to_M(cluster_dict['A907'][2]*Vik_Cosmo.Ez(cluster_dict['A907'][1]), cluster_dict['A907'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A907'][1]) * cluster_dict['A907'][13],
    'A1413': mass_so.R_to_M(cluster_dict['A1413'][2]*Vik_Cosmo.Ez(cluster_dict['A1413'][1]), cluster_dict['A1413'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A1413'][1])  * cluster_dict['A1413'][13],
    'A1795': mass_so.R_to_M(cluster_dict['A1795'][2]*Vik_Cosmo.Ez(cluster_dict['A1795'][1]), cluster_dict['A1795'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A1795'][1]) * cluster_dict['A1795'][13],
    'A1991': mass_so.R_to_M(cluster_dict['A1991'][2]*Vik_Cosmo.Ez(cluster_dict['A1991'][1]), cluster_dict['A1991'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A1991'][1])  * cluster_dict['A1991'][13],
    'A2029': mass_so.R_to_M(cluster_dict['A2029'][2]*Vik_Cosmo.Ez(cluster_dict['A2029'][1]), cluster_dict['A2029'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A2029'][1]) * cluster_dict['A2029'][13],
    'A2390': mass_so.R_to_M(cluster_dict['A2390'][2]*Vik_Cosmo.Ez(cluster_dict['A2390'][1]), cluster_dict['A2390'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['A2390'][1])  * cluster_dict['A2390'][13],
    'MKW 4': mass_so.R_to_M(cluster_dict['MKW 4'][2]*Vik_Cosmo.Ez(cluster_dict['MKW 4'][1]), cluster_dict['MKW 4'][1],  '500c')/Vik_Cosmo.Ez(cluster_dict['MKW 4'][1])  * cluster_dict['MKW 4'][13],
}

# Calculate mass values using the Vikhlinin model (V06)
V06_values =  {
    'A133': ((cluster_dict['A133'][12]*1e14) * cluster_dict['A133'][13], 0.38 * 1e14),
    'A383': ((cluster_dict['A383'][12]*1e14) * cluster_dict['A383'][13], 0.31 * 1e14),
    'A478': ((cluster_dict['A478'][12]*1e14) * cluster_dict['A478'][13], 1.01 * 1e14),
    'A907': ((cluster_dict['A907'][12]*1e14) * cluster_dict['A907'][13], 0.37 * 1e14),
    'A1413': ((cluster_dict['A1413'][12]*1e14) * cluster_dict['A1413'][13], 0.76 * 1e14),
    'A1795': ((cluster_dict['A1795'][12]*1e14) * cluster_dict['A1795'][13], 0.52 * 1e14),
    'A1991': ((cluster_dict['A1991'][12]*1e14) * cluster_dict['A1991'][13], 0.17 * 1e14),
    'A2029': ((cluster_dict['A2029'][12]*1e14) * cluster_dict['A2029'][13], 0.74* 1e14),
    'A2390': ((cluster_dict['A2390'][12]*1e14) * cluster_dict['A2390'][13], 1.01* 1e14),
    'MKW 4': ((cluster_dict['MKW 4'][12]*1e14) * cluster_dict['MKW 4'][13], 0.10* 1e14)
}

# Format Vikhlinin model values with scientific notation
V06_values_SN =  {
    'A133': (("{:.2e}".format(cluster_dict['A133'][12]*1e14)), "{:.2e}".format((cluster_dict['A133'][12]* 1e14 * cluster_dict['A133'][13]))),
    'A383': (("{:.2e}".format(cluster_dict['A383'][12]*1e14)), "{:.2e}".format((cluster_dict['A383'][12]* 1e14) * cluster_dict['A383'][13])),
    'A478': (("{:.2e}".format(cluster_dict['A478'][12]*1e14)), "{:.2e}".format((cluster_dict['A478'][12]* 1e14) * cluster_dict['A478'][13])),
    'A907': (("{:.2e}".format(cluster_dict['A907'][12]*1e14)), "{:.2e}".format((cluster_dict['A907'][12]* 1e14) * cluster_dict['A907'][13])),
    'A1413': (("{:.2e}".format(cluster_dict['A1413'][12]*1e14)), "{:.2e}".format((cluster_dict['A1413'][12]*  1e14) * cluster_dict['A1413'][13])),
    'A1795': (("{:.2e}".format(cluster_dict['A1795'][12]*1e14)), "{:.2e}".format((cluster_dict['A1795'][12]* 1e14) * cluster_dict['A1795'][13])),
    'A1991': (("{:.2e}".format(cluster_dict['A1991'][12]*1e14)), "{:.2e}".format((cluster_dict['A1991'][12]* 1e14) * cluster_dict['A1991'][13])),
    'A2029': (("{:.2e}".format(cluster_dict['A2029'][12]*1e14)), "{:.2e}".format((cluster_dict['A2029'][12]* 1e14) * cluster_dict['A2029'][13])),
    'A2390': (("{:.2e}".format(cluster_dict['A2390'][12]*1e14)), "{:.2e}".format((cluster_dict['A2390'][12]* 1e14) * cluster_dict['A2390'][13])),
    'MKW 4': (("{:.2e}".format(cluster_dict['MKW 4'][12]*1e14)), "{:.2e}".format((cluster_dict['MKW 4'][12]* 1e14) * cluster_dict['MKW 4'][13]))
}

# Extract total mass and error values for plotting or further calculations
values = [value[0] for value in V06_values.values()]
errors = [value[1] for value in V06_values.values()]


r500 = []
for c in range(len(my_bins100_centers)):
    if my_bins100_centers[c] > 1:
        r500 = np.append(r500, c)
        break


# Using pysr_TNG, define a symbolic regression model to compare with observed and simulated density profiles
# ====================================================================================

def get_rho_sr_comp1(r):
    rho1 = (((4.9257 / (r - 0.034715)) - (np.log10(r ** 4.9257) / 0.11442)) + (4.9257 - -0.085583)) 
    rho2 = (((4.9257 / (r - 0.034715)) + (r ** 3.5259)) - ((np.log10(r ** 4.9257) + -0.53084) / 0.11336))
    rho3 = (((12.353 / (r - -0.15531)) + (4.9257 / (r - 0.034715))) - (r / ((r / r) + np.log10(r))))  
    
  
    return rho1, rho2, rho3

# def get_rho_sr_comp2(r):
#     rho1 = (((16.808 / ((r ** (0.64312 + r)) -0.096964)) + (0.083056 / (2.0957 + np.log(r)))) / 1.6225)
#     rho2 = ((((63.838 / np.exp(r / 1.214)) + (r ** 0.0026082)) / np.exp(r)) + 0.16116)
#     rho3 = (((1.6977 / ((r** 2) - ((2.1559 + r) - np.log(np.log(r))))) + (0.58536 - 0.16809)) -0.39173) 
#     return rho1, rho2, rho3

def get_rho_sr_comp2(r):
    rho1 = ((0.16973 ** (r - 2.3344)) - (((np.log10(r) **r) ** r) + -0.16626))
    rho2 =(((((11.111 / r) - 2.4562) + (r + r)) / (r ** r)) + 0.20065) 
    rho3 =  ((((((11.111 + (r * r)) / r) - 2.4562) + r) / (r ** r)) + 0.20065) 

    return rho1, rho2, rho3

# def get_rho_sr_comp3(r):
#     rho1 = ((((16.808 / ((r ** (0.64312 + r)) -0.096964)) + (0.083056 / (2.0957 + np.log(r)))) - r) / 1.6225) 
#     rho2 = (((((r + 63.884) + 0.51159) / (np.exp(r) * 0.82631)) / np.exp(r ** 0.83057)) + 0.13583)  
#     rho3 = ((1.6296 / (((r**2) - (r +1.9503)) - (np.log10(np.exp(r ** 0.31851)) ** r))) + 0.026312)    
#     return rho1, rho2, rho3

# def get_rho_sr_comp3(r):
#     rho1 = ((np.exp((-29.458 + np.exp(np.log(r - 0.30922))) * 0.86424) - 1.7659) + (4.1964 / r))  
#     rho2 = ((np.exp(((-29.458 + (r - 0.30922)) + (0.5553 / r)) * 0.86424) - 1.7659) + (4.1964 / r))  
#     rho3 = ((np.exp(((-29.458 + (r - 0.30922)) + (np.log(1.7367) / r)) * 0.86424) - 1.7659) + (4.1964 / r))  
#     rho1 = 10 ** rho1
#     rho2 = 10 ** rho2
#     rho3 = 10 ** rho3
#     return rho1, rho2, rho3

rho1_comp1, rho2_comp1, rho3_comp1 = get_rho_sr_comp1(my_bins100_centers)
rho1_comp2, rho2_comp2, rho3_comp2 = get_rho_sr_comp2(my_bins100_centers)

comp1_ratio =[median_rho_100[1:23]/rho1_comp1[1:23], median_rho_100[22:33]/rho2_comp1[22:33], median_rho_100[32:-1]/rho3_comp1[32:-1]]
comp2_ratio = [median_rho_100[1:23]/rho1_comp2[1:23], median_rho_100[22:33]/rho2_comp2[22:33], median_rho_100[32:-1]/rho3_comp2[32:-1]]

    
#PLOTTING

#=============================================================================
fig, (ax1,ax2) = plt.subplots(1, 2, sharex='col', dpi=100)
fig.set_size_inches(20,9)
fig.tight_layout()
fig.patch.set_facecolor('white')
fig.suptitle('SR and TNG vs Radius', y=1.1, fontsize=30)

ax1.semilogy(my_bins100_centers, median_rho_100, color="#1F77B4", lw=5, alpha=1, label='TNG100')

ax1.semilogy(my_bins100_centers[1:int(r500[0])+1], rho1_comp1[1:int(r500[0])+1], color="#ff0e8e", lw=5, alpha=1, label='Complexity 7')
ax1.semilogy(my_bins100_centers[1:int(r500[0])+1], rho1_comp2[1:int(r500[0])+1], color="#eb0eff", lw=5, alpha=1, label='Complexity 8')
#ax1.semilogy(my_bins100_centers[1:int(r500[0])+1], rho1_comp3[1:int(r500[0])+1], color="#620eff", lw=5, alpha=1, label='Complexity 9')


ax1.semilogy(my_bins100_centers[int(r500[0]):32+1], rho2_comp1[int(r500[0]):32+1], color="#ff0e8e", lw=5, alpha=1, label='Complexity 7')
ax1.semilogy(my_bins100_centers[int(r500[0]):32+1], rho2_comp2[int(r500[0]):32+1], color="#eb0eff", lw=5, alpha=1, label='Complexity 8')
#ax1.semilogy(my_bins100_centers[int(r500[0]):32+1], rho2_comp3[int(r500[0]):32+1], color="#620eff", lw=5, alpha=1, label='Complexity 9')

ax1.semilogy(my_bins100_centers[32:-1], rho2_comp1[32:-1], color="#ff0e8e", lw=5, alpha=1, label='Complexity 7')
ax1.semilogy(my_bins100_centers[32:-1], rho2_comp2[32:-1], color="#eb0eff", lw=5, alpha=1, label='Complexity 8')
#ax1.semilogy(my_bins100_centers[32:-1], rho2_comp3[32:-1], color="#620eff", lw=5, alpha=1, label='Complexity 9')

sixteenth_percentile= np.nanpercentile(rho_vals_interp100, 16, axis=0)
ax1.semilogy(my_bins100_centers, sixteenth_percentile, color="#1F77B4", lw=3, alpha=0.2)

eightfour_percentile= np.nanpercentile(rho_vals_interp100, 84, axis=0)
ax1.semilogy(my_bins100_centers, eightfour_percentile, color="#1F77B4", lw=3, alpha=0.2)

ax1.fill_between(my_bins100_centers, sixteenth_percentile, eightfour_percentile, color="#1F77B4", alpha=0.2)

ax1.set_xlabel('r/R500c', fontsize=20)
ax1.axvline(my_bins100_centers[-1], color="black", linestyle='--', lw=3)
ax1.axvline(my_bins100_centers[1], color="black", linestyle='--', lw=3)
ax1.axvline(my_bins100_centers[22], color="black", linestyle='--', lw=3)
ax1.axvline(my_bins100_centers[32], color="black", linestyle='--', lw=3)
ax1.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
ax1.set_title("TNG Simulations Fit vs. radius", fontsize=20)
ax1.set_xscale("log")

ax2.plot(my_bins100_centers[1:23], comp1_ratio[0], color="#ff0e8e", lw=8, alpha=1, label='TNG100')
ax2.plot(my_bins100_centers[1:23], comp2_ratio[0], color="#eb0eff", lw=8, alpha=1, label='TNG100')
#ax2.plot(my_bins100_centers[1:23], comp3_ratio[0], color="#620eff", lw=8, alpha=1, label='TNG100')

ax2.plot(my_bins100_centers[22:33], comp1_ratio[1], color="#ff0e8e", lw=8, alpha=1, label='TNG100')
ax2.plot(my_bins100_centers[22:33], comp2_ratio[1], color="#eb0eff", lw=8, alpha=1, label='TNG100')
#ax2.plot(my_bins100_centers[22:33], comp3_ratio[1], color="#620eff", lw=8, alpha=1, label='TNG100')

ax2.plot(my_bins100_centers[32:-1], comp1_ratio[2], color="#ff0e8e", lw=8, alpha=1, label='TNG100')
ax2.plot(my_bins100_centers[32:-1], comp2_ratio[2], color="#eb0eff", lw=8, alpha=1, label='TNG100')
#ax2.plot(my_bins100_centers[32:-1], comp3_ratio[2], color="#620eff", lw=8, alpha=1, label='TNG100')
ax2.axvline(my_bins100_centers[22], color="black", linestyle='--', lw=3)
ax2.axvline(my_bins100_centers[32], color="black", linestyle='--', lw=3)

ax2.set_yscale("log")
ax2.set_xscale("log")

ax2.set_ylim(.1, 15)
ax2.axhline(1.0, color="black", linestyle='--', lw=3)
ax2.set_xlabel('r/R500c', fontsize=20)
ax2.set_ylabel('\u03C1$_{fit}$/\u03C1$_{c}$', fontsize=18)
ax2.set_title(" Fit/Simulations vs radius", fontsize=20)
plt.show()

#=============================================================================
plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10,8), dpi=200)
ax = plt.subplot(111)
plt.xscale("log")



# for i in range(0, 99):
#     radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
#     bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
#     first_bin_center100 = (radii_all100[:, 0]/2)
#     bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
#     R_500c = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])
#     r_normalized_per = bin_centers100[i]/R_500c[i]
#     rho_vals_per = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'][i])/rho_crit
#     plt.xscale("log")
#     plt.semilogy(r_normalized_per, rho_vals_per, color="yellow", lw=.5, alpha=0.25)            

# for i in range(0, 99):
#     radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
#     bin_centers300 = (radii_all300[:, :-1] + radii_all300[:, 1:]) / 2
#     first_bin_center300 = (radii_all300[:, 0]/2)
#     bin_centers300 = np.insert(bin_centers300, 0, first_bin_center300, axis=1)
#     R_500c = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
#     r_normalized_per = bin_centers300[i]/R_500c[i]
#     rho_vals_per = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'][i])/rho_crit
#     plt.xscale("log")
#     plt.semilogy(r_normalized_per, rho_vals_per, color="purple", lw=.5, alpha=0.25)            

plt.semilogy(m_radii_norm_A133, rho_norm_Vik_A133, color="red", lw=1, alpha=1, label='A133')
plt.semilogy(m_radii_norm_A262, rho_norm_Vik_A262, color="indianred", lw=1, alpha=1, label='A262')
plt.semilogy(m_radii_norm_A383, rho_norm_Vik_A383, color="tomato", lw=1, alpha=1, label='A383')
plt.semilogy(m_radii_norm_A478, rho_norm_Vik_A478, color="mistyrose", lw=1, alpha=1, label='A478')
plt.semilogy(m_radii_norm_A907, rho_norm_Vik_A907, color="sienna", lw=1, alpha=1, label='A907')
plt.semilogy(m_radii_norm_A1413, rho_norm_Vik_A1413, color="peru", lw=1, alpha=1, label='A1413')
plt.semilogy(m_radii_norm_A1795, rho_norm_Vik_A1795, color="darkgoldenrod", lw=1, alpha=1, label='A1795')
plt.semilogy(m_radii_norm_A1991, rho_norm_Vik_A1991, color="khaki", lw=1, alpha=1, label='A1991')
plt.semilogy(m_radii_norm_A2029, rho_norm_Vik_A2029, color="olive", lw=1, alpha=1, label='A2029')
plt.semilogy(m_radii_norm_A2390, rho_norm_Vik_A2390, color="limegreen", lw=1, alpha=1, label='A2390')
plt.semilogy(m_radii_norm_RX, rho_norm_Vik_RX, color="blue", lw=1, alpha=1, label='RX J1159+5531')
plt.semilogy(m_radii_norm_MKW, rho_norm_Vik_MKW, color="navy", lw=1, alpha=1, label='MKW 4')


plt.semilogy(my_bins100_centers, median_rho_100, color="#1F77B4", lw=5, alpha=1, label='TNG100')
plt.semilogy(my_bins300_centers, median_rho_300, color="#FF7F0E", lw=5, alpha=1, label='TNG300')


sixteenth_percentile= np.nanpercentile(rho_vals_interp100, 16, axis=0)
plt.semilogy(my_bins100_centers, sixteenth_percentile, color="#1F77B4", lw=3, alpha=0.2)

eightfour_percentile= np.nanpercentile(rho_vals_interp100, 84, axis=0)
plt.semilogy(my_bins100_centers, eightfour_percentile, color="#1F77B4", lw=3, alpha=0.2)

plt.fill_between(my_bins100_centers, sixteenth_percentile, eightfour_percentile, color="#1F77B4", alpha=0.2)



sixteenth_percentile= np.nanpercentile(rho_vals_interp300, 16, axis=0)
plt.semilogy(my_bins300_centers, sixteenth_percentile, color="#FF7F0E", lw=3, alpha=0.2)

eightfour_percentile= np.nanpercentile(rho_vals_interp300, 84, axis=0)
plt.semilogy(my_bins300_centers, eightfour_percentile, color="#FF7F0E", lw=3, alpha=0.2)

plt.fill_between(my_bins300_centers, sixteenth_percentile, eightfour_percentile, color="#FF7F0E", alpha=0.2)

plt.xlabel('r/R500c', fontsize=20)
plt.ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
plt.title("Density of TNG simulations and cluster observations vs. radius", fontsize=30)


box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()



