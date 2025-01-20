#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
from colossus.utils import constants
from colossus.cosmology import cosmology
from scipy.optimize import curve_fit
from tabulate import tabulate
import Vik_density_compare as Vik_rho
import pickle
import os

#SIMULATION DEFINITIONS
#====================================================================================

gasprofs100_v5_filename = 'groups_gasprofs_v5_tng100_099.hdf5'
gasprofs100_v5_vals = h5py.File(gasprofs100_v5_filename, 'r', libver='earliest', swmr=True)

gasprofs300_v5_filename = 'groups_gasprofs_v5_tng300_099.hdf5'
gasprofs300_v5_vals = h5py.File(gasprofs300_v5_filename, 'r', libver='earliest', swmr=True)


rho_crit = constants.RHO_CRIT_0_KPC3

#CLUSTER DICTIONARY (CLUSTER NAME, Z, R500)


cluster_dict = {'A133': ('A133', 0.0569, 1007),
                'A262': ('A262', 0.0162 , 650),
                'A383': ('A383', 0.1883, 944),
                'A478': ('A478', 0.0881, 1337),
                'A907': ('A907', 0.1603, 1096),
                'A1413':('A1413', 0.1429, 1299),
                'A1795':('A1795', 0.0622, 1235),
                'A1991':('A1991', 0.0592, 732),
                'A2029':('A2029', 0.0779, 1362),
                'A2390':('A2390', 0.2302, 1416),
                'RX':('RX J1159+5531', 0.0810, 700),
                'MKW 4':('MKW 4', 0.0199, 634),
                'USGC S152':('USGC S152', 0.0153, None)
        }

clusters = ['A2390', 'A133']

Vik_Cosmo = cosmology.setCosmology('Vik_Cosmo', params = cosmology.cosmologies['planck18'], Om0 = 0.3, Ode0=0.7, H0 = 72, sigma8 = 0.9)
print(Vik_Cosmo)

gamma=3

mp = constants.M_PROTON

G = constants.G

G_CGS = constants.G_CGS

def get_rho_norm(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    npne = ((n0*(10**-3))**2)*(((r/rc)**(-a))/((1+(r**2)/(rc**2))**(3*B + (-a/2))))* \
        (1/((1+((r**gamma)/(rs**gamma)))**(epsilon/gamma))) \
            + (((n02*(10**-1))**2)/ ((1 + (r**2)/(rc2**2))**(3*B2)))

            
    rho_g = 1.624 * mp * (npne)**(1/2)
    
    #rho_c_ill = mycosmo.rho_c(0.2302) * (constants.MSUN)*(.677**2)/(constants.KPC**3)
    
    H_col = Vik_Cosmo.Hz(z)
    
    h_col = H_col * (10 ** (-2))
    
    rho_c_col = Vik_Cosmo.rho_c(z) * (constants.MSUN)*((h_col)**2)/(constants.KPC**3)
    
    rho_norm_col = rho_g/rho_c_col #* 100
    
    
    return rho_g, rho_norm_col

def get_rho_norm_fit(r, n0, rc, rs, a, B, epsilon, n02, rc2, B2):
    npne = ((n0*(10**-3))**2)*(((r/rc)**(-a))/((1+(r**2)/(rc**2))**(3*B + (-a/2))))* \
        (1/((1+((r**gamma)/(rs**gamma)))**(epsilon/gamma))) \
            + (((n02*(10**-1))**2)/ ((1 + (r**2)/(rc2**2))**(3*B2)))

            
    rho_g = 1.624 * mp * (npne)**(1/2)
    
    #rho_c_ill = mycosmo.rho_c(0.2302) * (constants.MSUN)*(.677**2)/(constants.KPC**3)
    
    H_col = Vik_Cosmo.Hz(z)
    
    h_col = H_col * (10 ** (-2))
    
    rho_c_col = Vik_Cosmo.rho_c(z) * (constants.MSUN)*((h_col)**2)/(constants.KPC**3)
    
    rho_norm_col = rho_g/rho_c_col #* 100
    
    
    return rho_norm_col

radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
R_Crit500_100 = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])

radii_all100_norm = radii_all100/R_Crit500_100[:,None]

radii_all100_norm_mid = (radii_all100_norm[:-1] + radii_all100_norm[1:]) / 2.0
first_bin_midpoint = radii_all100_norm[0] / 2.0
radii_all100_norm_mid = np.insert(radii_all100_norm_mid, 0, first_bin_midpoint)

              
rho_vals100 = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'])
rho_vals100_norm = rho_vals100/rho_crit

radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
R_Crit500_300 = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])

radii_all300_norm = radii_all300/R_Crit500_300[:,None]

radii_all300_norm_mid = (radii_all300_norm[:-1] + radii_all300_norm[1:]) / 2.0
first_bin_midpoint = radii_all300_norm[0] / 2.0
radii_all300_norm_mid = np.insert(radii_all300_norm_mid, 0, first_bin_midpoint)

rho_vals300 = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'])
rho_vals300_norm = rho_vals300/rho_crit

rho_vals_interp100 = Vik_rho.rho_vals_interp100
rho_vals_interp300 = Vik_rho.rho_vals_interp300

#EXTRACT CLUSTER OBSERVATION DATA + PLUG AND CHUG
#====================================================================================

Vik_bins = np.linspace(78, 23400, 500) #Minimum and maximum radial bins in both  TNG 100 and 300

n0, rc, rs, a, B, epsilon, n02, rc2, B2 = np.loadtxt('Vikhlinin_tab2.csv', skiprows = 0, unpack=True, delimiter = ',', usecols=(0,1,2,3,4,5,6,7,8))
                                                                 
z=cluster_dict['A133'][1]
rho_g_A133, rho_norm_Vik_A133 = get_rho_norm(Vik_bins, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02[0], rc2[0], B2[0])

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
#emis_profile_USGC = get_rho_norm(Vik_bins, n0[12], rc[12], rs[12], a[12], B[12], epsilon[12], n02[12], rc2[12], B2[12])


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


#FITTING PARAMETERS ( For Median and Individual clusters)
#====================================================================================
bounds = [[0.1, 0, 200, 0.05, 0.1, 0.1, 0, 0, 0], [35, 1000, 3000, 3, 2, 5, 6, 100, 4]]


median_rho_100 = Vik_rho.median_rho_100
my_bins100_centers = Vik_rho.my_bins100_centers
median_rho_300 = Vik_rho.median_rho_300
my_bins300_centers = Vik_rho.my_bins300_centers

nan_mask = ~np.isnan(median_rho_100)
median_rho_100_red = median_rho_100[nan_mask]
my_bins100_centers_red = my_bins100_centers[nan_mask]

nan_mask = ~np.isnan(median_rho_300)
median_rho_300_red = median_rho_300[nan_mask]
my_bins300_centers_red = my_bins300_centers[nan_mask]



my_bins100_centers_red_2x = my_bins100_centers_red[:28]
my_bins300_centers_red_2x = my_bins300_centers_red[:28]

bin_1_100 = my_bins100_centers_red_2x[0]/8
bin_1_300 = my_bins300_centers_red_2x[0]/8
bin_2_100 = my_bins100_centers_red_2x[0]/4
bin_2_300 = my_bins300_centers_red_2x[0]/4
bin_3_100 = my_bins100_centers_red_2x[0]/2
bin_3_300 = my_bins300_centers_red_2x[0]/2

my_bins100_centers_red_2x_ex = np.insert(my_bins100_centers_red_2x, 0,[bin_1_100, bin_2_100, bin_3_100])
my_bins300_centers_red_2x_ex = np.insert(my_bins300_centers_red_2x, 0,[bin_1_300, bin_2_300, bin_3_300])

median_rho_100_red_2x = median_rho_100_red[:28]
median_rho_300_red_2x = median_rho_300_red[:28]

popt100_p1, pcov100 = curve_fit(get_rho_norm_fit, my_bins100_centers_red_2x, median_rho_100_red_2x, bounds=bounds)
popt300_p1, pcov300 = curve_fit(get_rho_norm_fit, my_bins300_centers_red_2x, median_rho_300_red_2x, bounds=bounds)


col_names = ["n0, rc, rs, a, B, epsilon, n02, rc2, B2"]

param_table_1 = [[popt100_p1],
          [popt300_p1]]



print(tabulate(param_table_1, headers=col_names[:], tablefmt='fancy_grid'))


curve_fit_result100_p1 = get_rho_norm_fit(my_bins100_centers_red_2x, popt100_p1[0], popt100_p1[1], popt100_p1[2], popt100_p1[3], popt100_p1[4], popt100_p1[5], popt100_p1[6], popt100_p1[7], popt100_p1[8])
curve_fit_result300_p1 = get_rho_norm_fit(my_bins300_centers_red_2x, popt300_p1[0], popt300_p1[1], popt300_p1[2], popt300_p1[3], popt300_p1[4], popt300_p1[5], popt300_p1[6], popt300_p1[7], popt300_p1[8])

curve_fit_result100_p1_ex = get_rho_norm_fit(my_bins100_centers_red_2x_ex, popt100_p1[0], popt100_p1[1], popt100_p1[2], popt100_p1[3], popt100_p1[4], popt100_p1[5], popt100_p1[6], popt100_p1[7], popt100_p1[8])
curve_fit_result300_p1_ex = get_rho_norm_fit(my_bins300_centers_red_2x_ex, popt300_p1[0], popt300_p1[1], popt300_p1[2], popt300_p1[3], popt300_p1[4], popt300_p1[5], popt300_p1[6], popt300_p1[7], popt300_p1[8])

TNG100_sim_ratio = curve_fit_result100_p1 / median_rho_100_red_2x
TNG300_sim_ratio = curve_fit_result300_p1 / median_rho_300_red_2x


    
twor500_index_array100 = np.zeros(100, dtype=int)
for j in range(0,100):
    for i in range(0, 50):
        if radii_all100_norm[j,i] >= 0.5:
            twor500_index_array100 = np.append(twor500_index_array100, i)
            break
twor500_index_array100 = twor500_index_array100[100:]



twor500_index_array300 = np.zeros(100, dtype=int)
for j in range(0,100):
    for i in range(0, 50):
        if radii_all300_norm[j,i] >= 0.5:
            twor500_index_array300 = np.append(twor500_index_array300, i)
            break
twor500_index_array300 = twor500_index_array300[100:]



#CREATE PICKLE FIKE LOOP 
#====================================================================================
folder_path = '/Users/shazaliaudu/Downloads/Gas_Profiles_Project'
pickle_file_name100 = 'TNG100_fit_1.7.pkl'
pickle_file_name300 = 'TNG300_fit_1.7.pkl'


pickle_file_path100 = folder_path + pickle_file_name100

pickle_file_path300 = folder_path + pickle_file_name300

if os.path.exists(pickle_file_path100):
    with open(pickle_file_path100, 'rb') as file:
        ac_fits100 = pickle.load(file)
    print("Loaded data from TNG100 pickle file succesfully")
    
else:
    ac_fits100 = [[] for _ in range(100)]
    for j in range(0, 100):
        i = twor500_index_array100[j]
        popt100, cov100 = curve_fit(get_rho_norm_fit, radii_all100_norm[j, :i], rho_vals100_norm[j, :i], bounds=bounds)
        ac_fits100[j] = np.append(ac_fits100[j], popt100)

    with open(pickle_file_path100, 'wb') as file:
        pickle.dump(ac_fits100, file)
    print("Saved data to TNG100 pickle file.")
    
if os.path.exists(pickle_file_path300):
    with open(pickle_file_path300, 'rb') as file:
        ac_fits300 = pickle.load(file)
    print("Loaded data from TNG300 pickle file ssuccesfully")
    
else:   
    ac_fits300 = [[] for _ in range(100)]
    for j in range(0, 100):
        i = twor500_index_array300[j]
        popt300, cov300 = curve_fit(get_rho_norm_fit, radii_all300_norm[j, :i], rho_vals300_norm[j, :i], bounds=bounds)
        ac_fits300[j] = np.append(ac_fits300[j], popt300)

    with open(pickle_file_path300, 'wb') as file:
        pickle.dump(ac_fits300, file)
    print("Saved data to TNG300 pickle file.")

ac_curve100 = [[] for _ in range(len(ac_fits100))]
for j in range(len(ac_fits100)):
        i = twor500_index_array100[j]
        curve100 = get_rho_norm_fit(radii_all100_norm[j,:i], ac_fits100[j][0], ac_fits100[j][1], ac_fits100[j][2], ac_fits100[j][3], ac_fits100[j][4], ac_fits100[j][5], ac_fits100[j][6], ac_fits100[j][7], ac_fits100[j][8])
        ac_curve100[j] = np.append(ac_curve100[j], curve100)

ac_curve300 = [[] for _ in range(len(ac_fits300))]
for j in range(len(ac_fits300)):
    i = twor500_index_array300[j]
    curve300 = get_rho_norm_fit(radii_all300_norm[j,:i], ac_fits300[j][0], ac_fits300[j][1], ac_fits300[j][2], ac_fits300[j][3], ac_fits300[j][4], ac_fits300[j][5], ac_fits300[j][6], ac_fits300[j][7], ac_fits300[j][8])
    ac_curve300[j] = np.append(ac_curve300[j], curve300)
    

TNG_100_ratio_arr = [[] for _ in range(100)]
for j in range(len(ac_fits100)):
    i = twor500_index_array100[j]
    TNG_100_ratio = ac_curve100[j] / rho_vals100_norm[j, :i]
    TNG_100_ratio_arr[j] = np.append(TNG_100_ratio_arr[j], TNG_100_ratio)
    
TNG_300_ratio_arr = [[] for _ in range(100)]
for j in range(len(ac_fits300)):
    i = twor500_index_array300[j]
    TNG_300_ratio = ac_curve300[j] / rho_vals300_norm[j, :i]
    TNG_300_ratio_arr[j] = np.append(TNG_300_ratio_arr[j], TNG_300_ratio)

    

#CHI SQUARED
#===================================================================================
def chi_squared(rho_fit, rho_sim, num_o_bins, num_o_param):
    deg_o_free = num_o_bins - num_o_param
    
    num = abs(rho_fit - rho_sim)
    chi_square_res = (np.log10(num))/(deg_o_free)
    
    return chi_square_res

        
chi_square_res_arr100 = [[] for _ in range(100)]
sus_clus_arr100 = [[] for _ in range(100)]

for j in range(len(rho_vals100_norm)):
    i = twor500_index_array100[j]
    chi_square_res = chi_squared(rho_vals100_norm[j, :i], ac_curve100[j], len(rho_vals100_norm[j, :i]), 9)
    chi_square_res_arr100[j] = np.append(chi_square_res_arr100[j], chi_square_res)
    
    if np.any(chi_square_res_arr100[j] >= 0.05):
        sus_clus_arr100[j] = np.append(sus_clus_arr100[j], j)
sus_clus_arr100 = [element for subarray in sus_clus_arr100 for element in subarray if subarray]


chi_square_res_arr300 = [[] for _ in range(100)]
sus_clus_arr300 = [[] for _ in range(100)]

for j in range(len(rho_vals300_norm)):
    i = twor500_index_array300[j]
    chi_square_res = chi_squared(rho_vals300_norm[j, :i], ac_curve300[j], len(rho_vals300_norm[j, :i]), 9)
    chi_square_res_arr300[j] = np.append(chi_square_res_arr300[j], chi_square_res)
    
    if np.any(chi_square_res_arr300[j] >= 0.05):
        sus_clus_arr300[j] = np.append(sus_clus_arr300[j], j)

sus_clus_arr300 = [element for subarray in sus_clus_arr300 for element in subarray if subarray]

n0_ac100 = [ac_fits100[i][0] for i in range(100) if i not in sus_clus_arr100]
rc_ac100 = [ac_fits100[i][1] for i in range(100) if i not in sus_clus_arr100]
rs_ac100 = [ac_fits100[i][2] for i in range(100) if i not in sus_clus_arr100]
a_ac100 = [ac_fits100[i][3] for i in range(100) if i not in sus_clus_arr100]
B_ac100 = [ac_fits100[i][4] for i in range(100) if i not in sus_clus_arr100]
epsilon_ac100 = [ac_fits100[i][5] for i in range(100) if i not in sus_clus_arr100]
n02_ac100 = [ac_fits100[i][6] for i in range(100) if i not in sus_clus_arr100]
rc2_ac100 = [ac_fits100[i][7] for i in range(100) if i not in sus_clus_arr100]
B2_ac100 = [ac_fits100[i][8] for i in range(100) if i not in sus_clus_arr100]

n0_ac300 = [ac_fits300[i][0] for i in range(100) if i not in sus_clus_arr300]
rc_ac300 = [ac_fits300[i][1] for i in range(100) if i not in sus_clus_arr300]
rs_ac300 = [ac_fits300[i][2] for i in range(100) if i not in sus_clus_arr300]
a_ac300 = [ac_fits300[i][3] for i in range(100) if i not in sus_clus_arr300]
B_ac300 =[ac_fits300[i][4] for i in range(100) if i not in sus_clus_arr300]
epsilon_ac300 = [ac_fits300[i][5] for i in range(100) if i not in sus_clus_arr300]
n02_ac300 = [ac_fits300[i][6] for i in range(100) if i not in sus_clus_arr300]
rc2_ac300 = [ac_fits300[i][7] for i in range(100) if i not in sus_clus_arr300]
B2_ac300 = [ac_fits300[i][8] for i in range(100) if i not in sus_clus_arr300]

#PARAMTER COMPARISON
#====================================================================================

z=cluster_dict['A133'][1]

n0_vary = np.linspace(np.min(n0), np.max(n0), 10)

Vik_bins_extended = np.linspace(10, 50000, 1000)


m_radii_norm_A133_ex = Vik_bins_extended/cluster_dict['A133'][2]

rho_g_A133_n0_vary =[[] for _ in range(len(n0_vary-1))]
for i in range(len(n0_vary)-1):
    rho_g_A133_n0, rho_norm_Vik_A133_n0 = get_rho_norm(Vik_bins_extended, n0_vary[i], rc[0], rs[0], a[0], B[0], epsilon[0], n02[0], rc2[0], B2[0])
    rho_g_A133_n0_vary[i] = np.append(rho_g_A133_n0_vary[i], rho_norm_Vik_A133_n0)

rc_vary = np.linspace(np.min(rc), np.max(rc), 10)

rho_g_A133_rc_vary =[[] for _ in range(len(rc_vary-1))]
for i in range(len(rc_vary)-1):
    rho_g_A133_rc, rho_norm_Vik_A133_rc = get_rho_norm(Vik_bins_extended, n0[0], rc_vary[i], rs[0], a[0], B[0], epsilon[0], n02[0], rc2[0], B2[0])
    rho_g_A133_rc_vary[i] = np.append(rho_g_A133_rc_vary[i], rho_norm_Vik_A133_rc)

rs_vary = np.linspace(np.min(rs), np.max(rs), 10)

rho_g_A133_rs_vary =[[] for _ in range(len(rs_vary-1))]
for i in range(len(rs_vary)-1):
    rho_g_A133_rs, rho_norm_Vik_A133_rs = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs_vary[i], a[0], B[0], epsilon[0], n02[0], rc2[0], B2[0])
    rho_g_A133_rs_vary[i] = np.append(rho_g_A133_rs_vary[i], rho_norm_Vik_A133_rs)

a_vary = np.linspace(np.min(a)-2, np.max(a)+2, 10)

rho_g_A133_a_vary =[[] for _ in range(len(a_vary-1))]
for i in range(len(a_vary)-1):
    rho_g_A133_a, rho_norm_Vik_A133_a = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs[0], a_vary[i], B[0], epsilon[0], n02[0], rc2[0], B2[0])
    rho_g_A133_a_vary[i] = np.append(rho_g_A133_a_vary[i], rho_norm_Vik_A133_a)

B_vary = np.linspace(np.min(B), np.max(B), 10)

rho_g_A133_B_vary =[[] for _ in range(len(B_vary-1))]
for i in range(len(B_vary)-1):
    rho_g_A133_B, rho_norm_Vik_A133_B = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs[0], a[0], B_vary[i], epsilon[0], n02[0], rc2[0], B2[0])
    rho_g_A133_B_vary[i] = np.append(rho_g_A133_B_vary[i], rho_norm_Vik_A133_B)

epsilon_vary = np.linspace(np.min(epsilon), np.max(epsilon), 10)

rho_g_A133_epsilon_vary =[[] for _ in range(len(epsilon_vary-1))]
for i in range(len(epsilon_vary)-1):
    rho_g_A133_epsilon, rho_norm_Vik_A133_epsilon = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs[0], a[0], B[0], epsilon_vary[i], n02[0], rc2[0], B2[0])
    rho_g_A133_epsilon_vary[i] = np.append(rho_g_A133_epsilon_vary[i], rho_norm_Vik_A133_epsilon)

n02_vary = np.logspace(-0, 1, 10)
n02_vary = np.linspace(np.min(n02), np.max(n02), 10)

rho_g_A133_n02_vary =[[] for _ in range(len(n02_vary-1))]
for i in range(len(n02_vary)-1):
    rho_g_A133_n02, rho_norm_Vik_A133_n02 = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02_vary[i], rc2[0], B2[0])
    rho_g_A133_n02_vary[i] = np.append(rho_g_A133_n02_vary[i], rho_norm_Vik_A133_n02)

# rho_g_A133_n02_vary =[[] for _ in range(len(n02-1))]
# for i in range(len(n02)-1):
#     rho_g_A133_n02, rho_norm_Vik_A133_n02 = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02[i], rc2[0], B2[0])
#     rho_g_A133_n02_vary[i] = np.append(rho_g_A133_n02_vary[i], rho_norm_Vik_A133_n02)

rc2_vary = np.logspace(-1, 3, 10)

rho_g_A133_rc2_vary =[[] for _ in range(len(rc2_vary-1))]
for i in range(len(rc2_vary)-1):
    rho_g_A133_rc2, rho_norm_Vik_A133_rc2 = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02[0], rc2_vary[i], B2[0])
    rho_g_A133_rc2_vary[i] = np.append(rho_g_A133_rc2_vary[i], rho_norm_Vik_A133_rc2)

B2_vary = np.logspace(-1, 1, 10)

rho_g_A133_B2_vary =[[] for _ in range(len(B2_vary-1))]
for i in range(len(B2_vary)-1):
    rho_g_A133_B2, rho_norm_Vik_A133_B2 = get_rho_norm(Vik_bins_extended, n0[0], rc[0], rs[0], a[0], B[0], epsilon[0], n02[0], rc2[0], B2_vary[i])
    rho_g_A133_B2_vary[i] = np.append(rho_g_A133_B2_vary[i], rho_norm_Vik_A133_B2)



#PLOTTING
#===================================================================================
fig, (ax1,ax2) = plt.subplots(1, 2, sharex='col', dpi=100)
fig.set_size_inches(20,9)
fig.tight_layout()
fig.patch.set_facecolor('white')
fig.suptitle('Fit of VIK density formula to TNG results (chi^2<=0.05)', y=1.1, fontsize=30)

ax1.semilogy(my_bins100_centers, median_rho_100, color="#1F77B4", lw=8, alpha=1, label='TNG100')
ax1.semilogy(my_bins300_centers, median_rho_300, color="#FF7F0E", lw=8, alpha=1, label='TNG300')

ax1.semilogy(my_bins100_centers_red_2x_ex, curve_fit_result100_p1_ex, color="red", linestyle = 'dashed', lw=3, alpha=1)
ax1.semilogy(my_bins300_centers_red_2x_ex, curve_fit_result300_p1_ex, color="black", linestyle = 'dashed', lw=3, alpha=1)


for i in range(0, 99):
    radii_all100 = np.array(gasprofs100_v5_vals['profile_bins'])
    bin_centers100 = (radii_all100[:, :-1] + radii_all100[:, 1:]) / 2
    first_bin_center100 = (radii_all100[:, 0]/2)
    bin_centers100 = np.insert(bin_centers100, 0, first_bin_center100, axis=1)
    R_500c = np.array(gasprofs100_v5_vals['catgrp_Group_R_Crit500'])
    r_normalized_per = bin_centers100[i]/R_500c[i]
    rho_vals_per = np.array(gasprofs100_v5_vals['profile_gas_rho_3d'][i])/rho_crit
    plt.xscale("log")
    ax1.semilogy(r_normalized_per[2:], rho_vals_per[2:], color="yellow", lw=.5, alpha=0.25)            

for i in range(0, 99):
    radii_all300 = np.array(gasprofs300_v5_vals['profile_bins'])
    bin_centers300 = (radii_all300[:, :-1] + radii_all300[:, 1:]) / 2
    first_bin_center300 = (radii_all300[:, 0]/2)
    bin_centers300 = np.insert(bin_centers300, 0, first_bin_center300, axis=1)
    R_500c = np.array(gasprofs300_v5_vals['catgrp_Group_R_Crit500'])
    r_normalized_per = bin_centers300[i]/R_500c[i]
    rho_vals_per = np.array(gasprofs300_v5_vals['profile_gas_rho_3d'][i])/rho_crit
    plt.xscale("log")
    ax1.semilogy(r_normalized_per[2:], rho_vals_per[2:], color="purple", lw=.5, alpha=0.25)            

sixteenth_percentile= np.nanpercentile(rho_vals_interp100, 16, axis=0)
ax1.semilogy(my_bins100_centers, sixteenth_percentile, color="#1F77B4", lw=3, alpha=0.2)

eightfour_percentile= np.nanpercentile(rho_vals_interp100, 84, axis=0)
ax1.semilogy(my_bins100_centers, eightfour_percentile, color="#1F77B4", lw=3, alpha=0.2)

ax1.fill_between(my_bins100_centers, sixteenth_percentile, eightfour_percentile, color="#1F77B4", alpha=0.2)



sixteenth_percentile= np.nanpercentile(rho_vals_interp300, 16, axis=0)
ax1.semilogy(my_bins300_centers, sixteenth_percentile, color="#FF7F0E", lw=3, alpha=0.2)

eightfour_percentile= np.nanpercentile(rho_vals_interp300, 84, axis=0)
ax1.semilogy(my_bins300_centers, eightfour_percentile, color="#FF7F0E", lw=3, alpha=0.2)

ax1.fill_between(my_bins300_centers, sixteenth_percentile, eightfour_percentile, color="#FF7F0E", alpha=0.2)


ax1.set_xlabel('r/R500c', fontsize=20)
ax1.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
ax1.set_title("TNG Simulations Fit vs. radius", fontsize=20)


ax1.set_xscale("log")

ax1.set_xlabel('r/R500c', fontsize=20)
ax1.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
ax1.set_title("TNG Simulations Fit vs. radius", fontsize=20)


ax1.set_xscale("log")


ax2.plot(my_bins100_centers_red_2x, TNG100_sim_ratio, color="#1F77B4", lw=8, alpha=1, label='TNG100')
ax2.plot(my_bins300_centers_red_2x, TNG300_sim_ratio, color="#FF7F0E", lw=8, alpha=1, label='TNG300')

for j in range(len(ac_fits100)):
    if j in sus_clus_arr100:
        continue
    
    i = twor500_index_array100[j]
    ax2.plot(radii_all100_norm[j, :i], np.array(TNG_100_ratio_arr[j]).reshape(len(radii_all100_norm[j, :i])), alpha = 0.3)
    
for j in range(len(ac_fits300)):
    if j in sus_clus_arr300:
        continue
    i = twor500_index_array300[j]
    ax2.plot(radii_all300_norm[j, :i], np.array(TNG_300_ratio_arr[j]).reshape(len(radii_all300_norm[j, :i])), alpha = 0.3)
    


ax2.axhline(1.0, color="black", linestyle='--', lw=3)

ax2.set_ylim(.1, 15)
ax2.set_yscale("log")
ax2.set_xscale("log")


ax2.set_xlabel('r/R500c', fontsize=20)
ax2.set_ylabel('\u03C1$_{fit}$/\u03C1$_{c}$', fontsize=18)
ax2.set_title(" Fit/Simulations vs radius", fontsize=20)

textstr = '\n'.join((
    r'TNG100',
    r'$n_{0}=%.2f$' % (popt100_p1[0], ),
    r'$r_{c}=%.2f$' % (popt100_p1[1], ),
    r'$r_{s}=%.2f$' % (popt100_p1[2], ),
    r'$\alpha=%.2f$' % (popt100_p1[3], ),
    r'$\beta=%.2f$' % (popt100_p1[4], ),
    r'$\epsilon=%.2f$' % (popt100_p1[5], ),
    r'$n_{02}=%.2f$' % (popt100_p1[6], ),
    r'$r_{c2}=%.2f$' % (popt100_p1[7], ),
    r'$\beta_{2}=%.2f$' % (popt100_p1[8], )))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.35, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


textstr = '\n'.join((
    r'TNG300',
    r'$n_{0}=%.2f$' % (popt300_p1[0], ),
    r'$r_{c}=%.2f$' % (popt300_p1[1], ),
    r'$r_{s}=%.2f$' % (popt300_p1[2], ),
    r'$\alpha=%.2f$' % (popt300_p1[3], ),
    r'$\beta=%.2f$' % (popt300_p1[4], ),
    r'$\epsilon=%.2f$' % (popt300_p1[5], ),
    r'$n_{02}=%.2f$' % (popt300_p1[6], ),
    r'$r_{c2}=%.2f$' % (popt300_p1[7], ),
    r'$\beta_{2}=%.2f$' % (popt300_p1[8], )))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.75, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


plt.show()

#===================================================================================
# fig = plt.figure(figsize=(20, 9))
# fig.patch.set_facecolor('white')
# fig.suptitle('Fit of individual TNG clusters', y=1.1, fontsize=30)

# gs = GridSpec(2, 3, figure=fig)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[1, 0])
# ax5 = fig.add_subplot(gs[1, 1])
# ax6 = fig.add_subplot(gs[1, 2])

# ax1.set_xscale("log")
# ax2.set_xscale("log")
# # ax3.set_xscale("log")
# # ax4.set_xscale("log")
# # ax5.set_xscale("log")
# # ax6.set_xscale("log")

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', dpi=100)
# fig.set_size_inches(12, 8)
# fig.tight_layout()
# fig.patch.set_facecolor('white')
# fig.suptitle('Fit of 1st, 9th, 19th, and 49th TNG clusters', y=1.1, fontsize=30)

# ax1.set_xscale("log")
# ax2.set_xscale("log")
# ax3.set_xscale("log")
# ax4.set_xscale("log")

# ax1.set_xlabel('r/R500c', fontsize=12)
# ax1.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
# ax2.set_xlabel('r/R500c', fontsize=12)
# ax2.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
# ax3.set_xlabel('r/R500c', fontsize=12)
# ax3.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
# ax4.set_xlabel('r/R500c', fontsize=12)
# ax4.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)



# ax1.semilogy(radii_all100_norm[0, :twor500_index_array100[0]], rho_vals100_norm[0, :twor500_index_array100[0]], color="#1F77B4", lw=8, alpha=1, label='TNG100')
# ax1.semilogy(radii_all300_norm[0, :twor500_index_array300[0]], rho_vals300_norm[0, :twor500_index_array300[0]], color="#FF7F0E", lw=8, alpha=1, label='TNG300')
# ax1.semilogy(radii_all100_norm[0, :twor500_index_array100[0]], ac_curve100[0][0], color="red", lw=3, alpha=1)
# ax1.semilogy(radii_all300_norm[0, :twor500_index_array300[0]], ac_curve300[0][0], color="black", lw=3, alpha=1)

# ax2.semilogy(radii_all100_norm[10, :twor500_index_array100[10]], rho_vals100_norm[10, :twor500_index_array100[10]], color="#1F77B4", lw=8, alpha=1, label='TNG100')
# ax2.semilogy(radii_all300_norm[10, :twor500_index_array300[10]], rho_vals300_norm[10, :twor500_index_array300[10]], color="#FF7F0E", lw=8, alpha=1, label='TNG300')
# ax2.semilogy(radii_all100_norm[10, :twor500_index_array100[10]], ac_curve100[9][0], color="red", lw=3, alpha=1)
# ax2.semilogy(radii_all300_norm[10, :twor500_index_array300[10]], ac_curve300[10][0], color="black", lw=3, alpha=1)
# ax2_secondary = ax2.twinx()
# ax2_secondary.set_yscale(ax1.get_yaxis().get_scale())

# ax3.semilogy(radii_all100_norm[20, :twor500_index_array100[20]], rho_vals100_norm[20, :twor500_index_array100[20]], color="#1F77B4", lw=8, alpha=1, label='TNG100')
# ax3.semilogy(radii_all300_norm[20, :twor500_index_array300[20]], rho_vals300_norm[20, :twor500_index_array300[20]], color="#FF7F0E", lw=8, alpha=1, label='TNG300')
# ax3.semilogy(radii_all100_norm[20, 1:twor500_index_array100[20]], ac_curve100[18][0], color="red", lw=3, alpha=1)
# ax3.semilogy(radii_all300_norm[20, :twor500_index_array300[20]], ac_curve300[19][0][1:], color="black", lw=3, alpha=1)

# ax4.semilogy(radii_all100_norm[50, :twor500_index_array100[50]], rho_vals100_norm[50, :twor500_index_array100[50]], color="#1F77B4", lw=8, alpha=1, label='TNG100')
# ax4.semilogy(radii_all300_norm[50, :twor500_index_array300[50]], rho_vals300_norm[50, :twor500_index_array300[50]], color="#FF7F0E", lw=8, alpha=1, label='TNG300')
# ax4.semilogy(radii_all100_norm[50, :twor500_index_array100[50]], ac_curve100[45][0], color="red", lw=3, alpha=1)
# ax4.semilogy(radii_all300_norm[50, :twor500_index_array300[50]], ac_curve300[47][0], color="black", lw=3, alpha=1)
# ax4_secondary = ax4.twinx()
# ax4_secondary.set_yscale(ax1.get_yaxis().get_scale())

# # ax5.semilogy(radii_all100_norm[70, 2:twor500_index_array100[70]], rho_vals100_norm[70, 2:twor500_index_array100[70]], color="#1F77B4", lw=8, alpha=1, label='TNG100')
# # ax5.semilogy(radii_all300_norm[70, :twor500_index_array300[70]], rho_vals300_norm[70, :twor500_index_array300[70]], color="#FF7F0E", lw=8, alpha=1, label='TNG300')
# # ax5.semilogy(radii_all100_norm[70, 4:twor500_index_array100[70]], ac_curve100[65][0][2:], color="red", lw=3, alpha=1)
# # ax5.semilogy(radii_all300_norm[70, 1:twor500_index_array300[70]], ac_curve300[65][0], color="black", lw=3, alpha=1)

# # ax6.semilogy(radii_all100_norm[91, :twor500_index_array100[91]], rho_vals100_norm[91, :twor500_index_array100[91]], color="#1F77B4", lw=8, alpha=1, label='TNG100')
# # ax6.semilogy(radii_all300_norm[91, :twor500_index_array300[91]], rho_vals300_norm[91, :twor500_index_array300[91]], color="#FF7F0E", lw=8, alpha=1, label='TNG300')
# # ax6.semilogy(radii_all100_norm[91, :twor500_index_array100[91]], ac_curve100[86][0], color="red", lw=3, alpha=1)
# # ax6.semilogy(radii_all300_norm[91, 1:twor500_index_array300[91]], ac_curve300[86][0], color="black", lw=3, alpha=1)

# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0, box.width, box.height * 0.9])
# ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=14)

# box = ax3.get_position()
# ax3.set_position([box.x0, box.y0 + box.height * 0, box.width, box.height * 0.9])
# ax3.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=14)

# plt.tight_layout()
# plt.show()



fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row', dpi=100)
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.patch.set_facecolor('white')
fig.suptitle('Histograms of Parameter Distributions', y=1.1, fontsize=30)

ax1.set_title("n$_{0}$", fontsize=18)
ax2.set_title("r$_{c}$", fontsize=18)
ax3.set_title("r$_{s}$", fontsize=18)
ax4.set_title("\u03B1", fontsize=18)
ax5.set_title("\u03B2", fontsize=18) 
ax6.set_title("\u03B5", fontsize=18)
ax7.set_title("n$_{02}$", fontsize=18)
ax8.set_title("r$_{c2}$", fontsize=18)
ax9.set_title("\u03B2$_{2}$", fontsize=18)

bins_n0 = np.logspace(np.log10(bounds[0][0]), np.log10(bounds[1][0]) ,20)

ax1.set_xscale("log")

ax1.hist(n0_ac100, bins=bins_n0, label='TNG100',color="#1F77B4")
ax1.hist(n0_ac300, bins=bins_n0, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(n0)):
    ax1.axvline(n0[i], color = "green")
    
ax1.axvline(n0[0], label = "Vikhlinin Values", color = "green")

bins_rc = np.logspace(np.log10(bounds[0][1]), np.log10(bounds[1][1]) ,50)

ax1.set_xscale("log")

ax2.hist(rc_ac100, bins=bins_rc, label='TNG100',color="#1F77B4")
ax2.hist(rc_ac300, bins=bins_rc, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(n0)):
    ax2.axvline(rc[i], color = "green")
    
ax2.axvline(rc[0], label = "Vikhlinin Values", color = "green")

bins_rs = np.logspace(np.log10(bounds[0][2]), np.log10(bounds[1][2]) ,20)

ax3.set_xscale("log")

ax3.hist(rs_ac100, bins=bins_rs, label='TNG100',color="#1F77B4")
ax3.hist(rs_ac300, bins=bins_rs, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(n0)):
    ax3.axvline(rs[i], color = "green")
    
bins_a = np.logspace(np.log10(bounds[0][3]), np.log10(bounds[1][3]) ,40)

ax4.set_xscale("log")


ax4.hist(a_ac100, bins=bins_a, label='TNG100',color="#1F77B4")
ax4.hist(a_ac300, bins=bins_a, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax4.axvline(a[i], color = "green")
    
bins_B = np.logspace(np.log10(bounds[0][4]), np.log10(bounds[1][4]) ,10)

ax5.set_xscale("log")

ax5.hist(B_ac100, bins=bins_B, label='TNG100',color="#1F77B4")
ax5.hist(B_ac300, bins=bins_B,label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax5.axvline(B[i], color = "green")
    
bins_ep = np.logspace(np.log10(bounds[0][5]), np.log10(bounds[1][5]) ,10)


ax6.set_xscale("log")

ax6.hist(epsilon_ac100, bins=bins_ep, label='TNG100',color="#1F77B4")
ax6.hist(epsilon_ac300, bins=bins_ep, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax6.axvline(epsilon[i], color = "green")
    
bins_n02 = np.logspace(np.log10(1), np.log10(10) ,10)

ax7.set_xscale("log")

ax7.hist(n02_ac100, bins=bins_n02, label='TNG100',color="#1F77B4")
ax7.hist(n02_ac300, bins=bins_n02, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax7.axvline(n02[i], color = "green")
    
bins_rc2 = np.logspace(np.log10(1), np.log10(10) ,10)

ax8.set_xscale("log")

ax8.hist(rc2_ac100, bins = bins_rc2, label='TNG100',color="#1F77B4")
ax8.hist(rc2_ac300, bins = bins_rc2, label ='TNG300', color="#FF7F0E", alpha=0.75)


for i in range(len(a)):
    ax8.axvline(rc2[i], color = "green")
    
ax8.axvline(rc2[0], label = "Vikhlinin Values", color = "green")

bins_B2 = np.logspace(np.log10(1), np.log10(10) ,10)

ax9.set_xscale("log")

ax9.hist(B2_ac100, bins=bins_B2,  label='TNG100',color="#1F77B4")
ax9.hist(B2_ac300, bins=bins_B2, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax9.axvline(B2[i], color = "green")
    
ax9.axvline(B2[0], label = "Vikhlinin Values", color = "green")



plt.show()
#===================================================================================

# #===================================================================================

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row', dpi=100)
# fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', dpi=100)
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.patch.set_facecolor('white')
fig.suptitle('Comparsion of V06 Paramters (A133)', y=1.1, fontsize=30)


ax1.set_xscale("log")
ax2.set_xscale("log")
ax3.set_xscale("log")
ax4.set_xscale("log")
ax5.set_xscale("log")
ax6.set_xscale("log")
ax7.set_xscale("log")
ax8.set_xscale("log")
ax9.set_xscale("log")

plt.ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)

ax1.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
ax4.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)
ax7.set_ylabel('\u03C1/\u03C1$_{c}$', fontsize=18)


ax7.set_xlabel('r/R500c', fontsize=18)
ax8.set_xlabel('r/R500c', fontsize=18)
ax9.set_xlabel('r/R500c', fontsize=18)


ax1.set_title("n$_{0}$", fontsize=18)
ax2.set_title("r$_{c}$", fontsize=18)
ax3.set_title("r$_{s}$", fontsize=18)
ax4.set_title("\u03B1", fontsize=18)
ax5.set_title("\u03B2", fontsize=18) 
ax6.set_title("\u03B5", fontsize=18)
ax7.set_title("n$_{02}$", fontsize=18)
ax8.set_title("r$_{c2}$", fontsize=18)
ax9.set_title("\u03B2$_{2}$", fontsize=18)


ax1.yaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)
ax3.yaxis.set_tick_params(labelsize=12)
ax4.yaxis.set_tick_params(labelsize=12)
ax5.yaxis.set_tick_params(labelsize=12)
ax6.yaxis.set_tick_params(labelsize=12)
ax7.yaxis.set_tick_params(labelsize=12)
ax8.yaxis.set_tick_params(labelsize=12)
ax9.yaxis.set_tick_params(labelsize=12)

ax1.set_yticks(ax1.get_yticks())
ax2.set_yticks(ax1.get_yticks())
ax3.set_yticks(ax1.get_yticks())
ax4.set_yticks(ax1.get_yticks())
ax5.set_yticks(ax1.get_yticks())
ax6.set_yticks(ax1.get_yticks())
ax7.set_yticks(ax1.get_yticks())
ax8.set_yticks(ax1.get_yticks())
ax9.set_yticks(ax1.get_yticks())

ax1.tick_params(axis='y', which='both', left=True, labelleft=True)
ax2.tick_params(axis='y', which='both', left=True, labelleft=True)
ax3.tick_params(axis='y', which='both', left=True, labelleft=True)
ax4.tick_params(axis='y', which='both', left=True, labelleft=True)
ax5.tick_params(axis='y', which='both', left=True, labelleft=True)
ax6.tick_params(axis='y', which='both', left=True, labelleft=True)
ax7.tick_params(axis='y', which='both', left=True, labelleft=True)
ax8.tick_params(axis='y', which='both', left=True, labelleft=True)
ax9.tick_params(axis='y', which='both', left=True, labelleft=True)
 
ax1.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax2.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax3.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax4.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax5.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax6.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax7.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax8.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')
ax9.semilogy(m_radii_norm_A133,  rho_norm_Vik_A133, color="black", lw=3, alpha=1, label='Median Curve')

# label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[0])
# ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[0], color="red", lw=1, alpha=1, label = label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[1])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[1], color="blue", lw=1, alpha=1, label = label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[2])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[2], color="green", lw=1, alpha=1, label = label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[3])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[3], color="yellow", lw=1, alpha=1, label = label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[4])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[4], color="purple", lw=1, alpha=1, label =  label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[5])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[5], color="indianred", lw=1, alpha=1, label = label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[6])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[6], color="mistyrose", lw=1, alpha=1, label = label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[7])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[7], color="olive", lw=1, alpha=1, label = label)
label = 'n$_{}$ = {:.2f}'.format(0, n0_vary[8])
ax1.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n0_vary[8], color="limegreen", lw=1, alpha=1, label = label)

# label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[0])
# ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[0], color="red", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[1])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[1], color="blue", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[2])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[2], color="green", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[3])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[3], color="yellow", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[4])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[4], color="purple", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[5])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[5], color="indianred", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[6])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[6], color="mistyrose", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[7])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[7], color="olive", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c', rc_vary[8])
ax2.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc_vary[8], color="limegreen", lw=1, alpha=1, label = label)

label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[0])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[0], color="red", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[1])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[1], color="blue", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[2])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[2], color="green", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[3])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[3], color="yellow", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[4])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[4], color="purple", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[5])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[5], color="indianred", lw=1, alpha=1, label =label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[6])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[6], color="mistyrose", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[7])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[7], color="olive", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('s', rs_vary[8])
ax3.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rs_vary[8], color="limegreen", lw=1, alpha=1, label =  label)

label = '\u03B1 = {:.2f}'.format(a_vary[0])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[0], color="red", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[1])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[1], color="blue", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[2])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[2], color="green", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[3])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[3], color="yellow", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[4])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[4], color="purple", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[5])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[5], color="indianred", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[6])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[6], color="mistyrose", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[7])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[7], color="olive", lw=1, alpha=1, label=label)
label = '\u03B1 = {:.2f}'.format(a_vary[8])
ax4.semilogy(m_radii_norm_A133_ex,  rho_g_A133_a_vary[8], color="limegreen", lw=1, alpha=1, label=label)

label = '\u03B2 = {:.2f}'.format(B_vary[0])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[0], color="red", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[1])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[1], color="blue", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[2])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[2], color="green", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[3])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[3], color="yellow", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[4])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[4], color="purple", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[5])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[5], color="indianred", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[6])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[6], color="mistyrose", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[7])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[7], color="olive", lw=1, alpha=1, label=label)
label = '\u03B2 = {:.2f}'.format(B_vary[8])
ax5.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B_vary[8], color="limegreen", lw=1, alpha=1, label=label)

label = '\u03B5 = {:.2f}'.format(epsilon_vary[0])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[0], color="red", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[1])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[1], color="blue", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[2])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[2], color="green", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[3])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[3], color="yellow", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[4])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[4], color="purple", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[5])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[5], color="indianred", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[6])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[6], color="mistyrose", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[7])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[7], color="olive", lw=1, alpha=1, label=label)
label = '\u03B5 = {:.2f}'.format(epsilon_vary[8])
ax6.semilogy(m_radii_norm_A133_ex,  rho_g_A133_epsilon_vary[8], color="limegreen", lw=1, alpha=1, label=label)

label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[0])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[0], color="red", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[1])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[1], color="blue", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[2])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[2], color="green", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[3])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[3], color="yellow", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[4])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[4], color="purple", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[5])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[5], color="indianred", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[6])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[6], color="mistyrose", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[7])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[7], color="olive", lw=1, alpha=1, label=label)
label = 'n$_{}$ = {:.5f}'.format(2, n02_vary[8])
ax7.semilogy(m_radii_norm_A133_ex,  rho_g_A133_n02_vary[8], color="limegreen", lw=1, alpha=1, label=label)

label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[0])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[0], color="red", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[1])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[1], color="blue", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[2])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[2], color="green", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[3])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[3], color="yellow", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[4])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[4], color="purple", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[5])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[5], color="indianred", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[6])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[6], color="mistyrose", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[7])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[7], color="olive", lw=1, alpha=1, label = label)
label = 'r$_{}$ = {:.2f}'.format('c2', rc2_vary[8])
ax8.semilogy(m_radii_norm_A133_ex,  rho_g_A133_rc2_vary[8], color="limegreen", lw=1, alpha=1, label = label)

label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[0])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[0], color="red", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[1])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[1], color="blue", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[2])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[2], color="green", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[3])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[3], color="yellow", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[4])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[4], color="purple", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[5])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[5], color="indianred", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[6])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[6], color="mistyrose", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[7])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[7], color="olive", lw=1, alpha=1, label = label)
label = 'B$_{}$ = {:.2f}'.format('2', B2_vary[8])
ax9.semilogy(m_radii_norm_A133_ex,  rho_g_A133_B2_vary[8], color="limegreen", lw=1, alpha=1, label = label)



axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
for i, ax in enumerate(axs):
    ax.legend(fontsize = 4)

plt.tight_layout()
plt.show()

#THINGS TO DO
#5) GO THRU LECTURES 14 AND 15
#6) READ OVER PAPER FOR NEW FITTING FUNC




