#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
import statistics

#TNG100 Results
## =============================================================================
gasprofs_filename = 'groups_gasprofs_tng100_099.hdf5'
gasprofs100_vals = h5py.File(gasprofs_filename, 'r', libver='earliest', swmr=True)

# radii = np.array(gasprofs_vals['profile_bins'][0][:])
R_Mean200 = np.array(gasprofs100_vals['catgrp_Group_R_Mean200'])
# r_normalized = radii/R_Mean200[0]
# rho_vals_cluster_1 = np.array(gasprofs_vals['profile_gas_rho_3d'][0])

# plt.rcParams.update({'font.size': 16})
# fig2 = plt.figure(figsize=(10,8), dpi=200)
# ax = plt.subplot(111)
# ax.set_yscale('log')
# plt.semilogy(r_normalized, rho_vals_cluster_1, color="#1F77B4", lw=2, alpha=0.8)
# plt.xlabel('r/R200m', fontsize=20)
# plt.ylabel('density (Msun/kpc^3)', fontsize=18)
# plt.title("Density vs. radius for first cluster in T100 simulation", fontsize=30)
# plt.show()

rho_vals = np.array(gasprofs100_vals['profile_gas_rho_3d'])
median_rho = np.median(rho_vals, axis=0)
radii_all100 = np.array(gasprofs100_vals['profile_bins'])
bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
first_bin_center100 = (radii_all100[:, 0]/2)
bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
median_radii100 = np.median(bin_centers100, axis=0)
median_R_Mean200 = np.median(R_Mean200, axis=0)
m_radii_norm = median_radii100/median_R_Mean200

plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10,8), dpi=200)
ax = plt.subplot(111)
plt.xscale("log")
plt.semilogy(m_radii_norm, median_rho, color="#1F77B4", lw=5, alpha=1, label='median TNG100 density')
plt.semilogy(m_radii_norm, rho_vals[0], color="black", lw=.5, alpha=0.25, label='individual clusters')
sixteenth_percentile= np.percentile(rho_vals, 16, axis=0)
plt.semilogy(m_radii_norm, sixteenth_percentile, color="gray", lw=3, alpha=0.5, label='scatter')
eightfour_percentile= np.percentile(rho_vals, 84, axis=0)
plt.semilogy(m_radii_norm, eightfour_percentile, color="gray", lw=3, alpha=0.5)
plt.fill_between(m_radii_norm, sixteenth_percentile, eightfour_percentile, color="gray", alpha=0.5)
for i in range(0, len(rho_vals)-1):
    radii_all100 = np.array(gasprofs100_vals['profile_bins'])
    bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
    first_bin_center100 = (radii_all100[:, 0]/2)
    bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
    R_Mean200 = np.array(gasprofs100_vals['catgrp_Group_R_Mean200'])
    r_normalized_per = bin_centers100[i]/R_Mean200[i]
    rho_vals_per = np.array(gasprofs100_vals['profile_gas_rho_3d'][i])
    plt.xscale("log")
    plt.semilogy(r_normalized_per, rho_vals_per, color="black", lw=.5, alpha=0.25)
    
    
    
    
plt.xlabel('r/R200m', fontsize=20)
plt.ylabel('median density(Msun/kpc^3)', fontsize=18)
plt.title("Median and cluster densities vs. radius for all clusters in T100 simulation", fontsize=30)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
## =============================================================================

#TNG300 Results
## =============================================================================
gasprofs_filename = 'groups_gasprofs_tng300_099.hdf5'
gasprofs300_vals = h5py.File(gasprofs_filename, 'r', libver='earliest', swmr=True)

radii = np.array(gasprofs300_vals['profile_bins'][0][:])
R_Mean200 = np.array(gasprofs300_vals['catgrp_Group_R_Mean200'])
r_normalized = radii/R_Mean200[0]
# rho_vals_cluster_1 = np.array(gasprofs_vals['profile_gas_rho_3d'][0])
# plt.rcParams.update({'font.size': 16})
# fig2 = plt.figure(figsize=(10,8), dpi=200)
# ax = plt.subplot(111)
# ax.set_yscale('log')
# plt.semilogy(r_normalized, rho_vals_cluster_1, color="#FF7F0E", lw=2, alpha=0.8)
# plt.xlabel('r/R200m', fontsize=20)
# plt.ylabel('density (Msun/kpc^3)', fontsize=18)
# plt.title("Plot of density vs. radius for first cluster in T300 simulation", fontsize=30)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
# plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
# plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# plt.show()

rho_vals = np.array(gasprofs300_vals['profile_gas_rho_3d'])
median_rho = np.median(rho_vals, axis=0)
radii_all300 = np.array(gasprofs300_vals['profile_bins'])
bin_centers300 = (radii_all300[:, :-1] + radii_all300[:, 1:]) / 2
first_bin_center300 = (radii_all300[:, 0]/2)
bin_centers300 = np.insert(bin_centers300, 0, first_bin_center300, axis=1)
median_radii300 = np.median(bin_centers300, axis=0)
m_radii_norm = median_radii300/median_R_Mean200

plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10,8), dpi=200)
ax = plt.subplot(111)
plt.xscale("log")
plt.semilogy(m_radii_norm, median_rho, color="#FF7F0E", lw=5, alpha=1, label='median density')
plt.semilogy(m_radii_norm, rho_vals[0], color="black", lw=.5, alpha=0.25, label='individual clusters')
sixteenth_percentile= np.percentile(rho_vals, 16, axis=0)
plt.semilogy(m_radii_norm, sixteenth_percentile, color="gray", lw=3, alpha=0.5, label='scatter')
eightfour_percentile= np.percentile(rho_vals, 84, axis=0)
plt.semilogy(m_radii_norm, eightfour_percentile, color="gray", lw=3, alpha=0.5)
plt.fill_between(m_radii_norm, sixteenth_percentile, eightfour_percentile, color="gray", alpha=0.5)
for i in range(0, len(rho_vals)-1):
    radii_all300 = np.array(gasprofs300_vals['profile_bins'])
    bin_centers300 = (radii_all300[:, :-1] + radii_all300[:, 1:]) / 2
    first_bin_center300 = (radii_all300[:, 0]/2)
    bin_centers300 = np.insert(bin_centers300, 0, first_bin_center300, axis=1)
    R_Mean200 = np.array(gasprofs100_vals['catgrp_Group_R_Mean200'])
    r_normalized_per = bin_centers300[i]/R_Mean200[i]
    rho_vals_per = np.array(gasprofs300_vals['profile_gas_rho_3d'][i])
    plt.xscale("log")
    plt.semilogy(r_normalized_per, rho_vals_per, color="black", lw=.5, alpha=0.25)
    
    
    
    
plt.xlabel('r/R200m', fontsize=20)
plt.ylabel('median density(Msun/kpc^3)', fontsize=18)
plt.title("Median and cluster densities vs. radius for all clusters in T300 simulation", fontsize=30)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
## =============================================================================




gasprofs100_filename = 'groups_gasprofs_tng100_099.hdf5'
gasprofs100_vals = h5py.File(gasprofs100_filename, 'r', libver='earliest', swmr=True)

gasprofs300_filename = 'groups_gasprofs_tng300_099.hdf5'
gasprofs300_vals = h5py.File(gasprofs300_filename, 'r', libver='earliest', swmr=True)

rho_vals100 = np.array(gasprofs100_vals['profile_gas_rho_3d'])
median_rho100 = np.median(rho_vals100, axis=0)
radii_all100 = np.array(gasprofs100_vals['profile_bins'])
bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
first_bin_center100 = (radii_all100[:, 0]/2)
bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
median_radii100 = np.median(bin_centers100, axis=0)

R_Mean200_100 = np.array(gasprofs100_vals['catgrp_Group_R_Mean200'])
median_R_Mean200_100 = np.median(R_Mean200_100, axis=0)
m_radii_norm_100 = median_radii100/median_R_Mean200_100

rho_vals300 = np.array(gasprofs300_vals['profile_gas_rho_3d'])
median_rho300 = np.median(rho_vals300, axis=0)
radii_all300 = np.array(gasprofs300_vals['profile_bins'])
bin_centers300 = (radii_all300[:, :-1] + radii_all300[:, 1:]) / 2
first_bin_center300 = (radii_all300[:, 0]/2)
bin_centers300 = np.insert(bin_centers300, 0, first_bin_center300, axis=1)
median_radii300 = np.median(bin_centers300, axis=0)

R_Mean200_300 = np.array(gasprofs300_vals['catgrp_Group_R_Mean200'])
median_R_Mean200_300 = np.median(R_Mean200_300, axis=0)
m_radii_norm_300 = median_radii300/median_R_Mean200_300

plt.rcParams.update({'font.size': 16})
fig2 = plt.figure(figsize=(10,8), dpi=200)
ax = plt.subplot(111)
plt.xscale("log")
plt.semilogy(m_radii_norm_100, median_rho100, color="#1F77B4", lw=5, alpha=1, label='TNG100')
plt.semilogy(m_radii_norm_300, median_rho300, color="#FF7F0E", lw=5, alpha=1, label='TNG300')
sixteenth_percentile= np.percentile(rho_vals100, 16, axis=0)
plt.semilogy(m_radii_norm_100, sixteenth_percentile, color="#1F77B4", lw=3, alpha=0.2)
eightfour_percentile= np.percentile(rho_vals100, 84, axis=0)
plt.semilogy(m_radii_norm_100, eightfour_percentile, color="#1F77B4", lw=3, alpha=0.2)
plt.fill_between(m_radii_norm_100, sixteenth_percentile, eightfour_percentile, color="#1F77B4", alpha=0.2)
sixteenth_percentile= np.percentile(rho_vals300, 16, axis=0)
plt.semilogy(m_radii_norm_300, sixteenth_percentile, color="#FF7F0E", lw=3, alpha=0.2)
eightfour_percentile= np.percentile(rho_vals300, 84, axis=0)
plt.semilogy(m_radii_norm_300, eightfour_percentile, color="#FF7F0E", lw=3, alpha=0.2)
plt.fill_between(m_radii_norm_300, sixteenth_percentile, eightfour_percentile, color="#FF7F0E", alpha=0.2)
plt.xlabel('r/R200m', fontsize=20)
plt.ylabel('median density(Msun/kpc^3)', fontsize=18)
plt.title("Medians and scatter of TNG100 and TNG300 densities vs. radius", fontsize=30)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0, box.width, box.height * 0.9])
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

## =============================================================================



