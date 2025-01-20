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

def get_rho_norm_fit(r, n0, rc, rs, a, B, epsilon):
    npne = ((n0*(10**-3))**2)*(((r/rc)**(-a))/((1+(r**2)/(rc**2))**(3*B + (-a/2))))* \
        (1/((1+((r**gamma)/(rs**gamma)))**(epsilon/gamma))) 

            
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
bounds = [[0.1, 1, 200, 0.05, 0.1, 0.1], [35, 600, 3000, 3, 2, 5]]


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


col_names = ["n0, rc, rs, a, B, epsilon"]

param_table_1 = [[popt100_p1],
          [popt300_p1]]
# param_table_2 = [[popt100_p2],
#           [popt300_p2]]


print(tabulate(param_table_1, headers=col_names[:], tablefmt='fancy_grid'))
# print(tabulate(param_table_2, headers=col_names[:], tablefmt='fancy_grid'))


curve_fit_result100_p1 = get_rho_norm_fit(my_bins100_centers_red_2x, popt100_p1[0], popt100_p1[1], popt100_p1[2], popt100_p1[3], popt100_p1[4], popt100_p1[5])
curve_fit_result300_p1 = get_rho_norm_fit(my_bins300_centers_red_2x, popt300_p1[0], popt300_p1[1], popt300_p1[2], popt300_p1[3], popt300_p1[4], popt300_p1[5])

curve_fit_result100_p1_ex = get_rho_norm_fit(my_bins100_centers_red_2x_ex, popt100_p1[0], popt100_p1[1], popt100_p1[2], popt100_p1[3], popt100_p1[4], popt100_p1[5])
curve_fit_result300_p1_ex = get_rho_norm_fit(my_bins300_centers_red_2x_ex, popt300_p1[0], popt300_p1[1], popt300_p1[2], popt300_p1[3], popt300_p1[4], popt300_p1[5])

TNG100_sim_ratio = curve_fit_result100_p1 / median_rho_100_red_2x
TNG300_sim_ratio = curve_fit_result300_p1 / median_rho_300_red_2x


# bin_1_100 = radii_all100_norm[:,0]/8
# bin_1_300 = radii_all300_norm[:,0]/8
# bin_2_100 = radii_all100_norm[:,0]/4
# bin_2_300 = radii_all300_norm[:,0]/4
# bin_3_100 = radii_all100_norm[:,0]/2
# bin_3_300 = radii_all300_norm[:,0]/2

# radii_all100_norm_ex = np.zeros((radii_all100_norm.shape[0], radii_all100_norm.shape[1] + 3))
# radii_all300_norm_ex = np.zeros((radii_all300_norm.shape[0], radii_all300_norm.shape[1] + 3))

# for i in range(len(radii_all100_norm)-1):
#     radii_all100_norm_ex[i] = np.insert(radii_all100_norm[i], 0, [bin_1_100[i], bin_2_100[i], bin_3_100[i]])
# for i in range(len(radii_all300_norm)-1):
#     radii_all300_norm_ex[i] = np.insert(radii_all300_norm[i], 0, [bin_1_300[i], bin_2_300[i], bin_3_300[i]])
   
    
twor500_index_array100 = np.zeros(100, dtype=int)
for j in range(0,100):
    for i in range(0, 50):
        if radii_all100_norm[j,i] >= 0.5:
            twor500_index_array100 = np.append(twor500_index_array100, i)
            break
twor500_index_array100 = twor500_index_array100[100:]

# twor500_index_array100_ex = np.zeros(100, dtype=int)
# for j in range(0,100):
#     for i in range(0, 50):
#         if radii_all100_norm_ex[j,i] >= 2:
#             twor500_index_array100_ex = np.append(twor500_index_array100_ex, i)
#             break
# twor500_index_array100_ex = twor500_index_array100_ex[100:]

twor500_index_array300 = np.zeros(100, dtype=int)
for j in range(0,100):
    for i in range(0, 50):
        if radii_all300_norm[j,i] >= 0.5:
            twor500_index_array300 = np.append(twor500_index_array300, i)
            break
twor500_index_array300 = twor500_index_array300[100:]


# twor500_index_array300_ex = np.zeros(100, dtype=int)
# for j in range(0,100):
#     for i in range(0, 50):
#         if radii_all300_norm_ex[j,i] >= 2:
#             twor500_index_array300_ex = np.append(twor500_index_array300_ex, i)
#             break
# twor500_index_array300_ex = twor500_index_array300_ex[100:]


#CREATE PICKLE FIKE LOOP 
#====================================================================================
folder_path = '/Users/shazaliaudu/Downloads/Gas_Profiles_Project'
pickle_file_name100 = 'TNG100_fit_6par_0.7_3.pkl'
pickle_file_name300 = 'TNG300_fit_6par_0.7_3.pkl'


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
#====================================================================================
#Clusters where optimal paramters couldn't be found, 
#TNG100: 6th, 12th, 30th, 37th, 40th clusters; TNG300: 13th, 41st, 44th, 57th, 63rd

ac_curve100 = [[] for _ in range(len(ac_fits100))]
for j in range(len(ac_fits100)):
        i = twor500_index_array100[j]
        curve100 = get_rho_norm_fit(radii_all100_norm[j,:i], ac_fits100[j][0], ac_fits100[j][1], ac_fits100[j][2], ac_fits100[j][3], ac_fits100[j][4], ac_fits100[j][5])
        ac_curve100[j] = np.append(ac_curve100[j], curve100)
    
# ac_curve100_ex = [[] for _ in range(len(ac_fits100))]
# for j in range(len(ac_fits100)):
#     if j == 7:
#         continue
#     if j == 13:
#         continue
#     if j == 31:
#         continue
#     if j == 38:
#         continue
#     if j == 41:
#         continue
#     i = twor500_index_array100_ex[j]
#     curve100_ex = get_rho_norm_fit(radii_all100_norm_ex[j,:i], ac_fits100[j][0][0], ac_fits100[j][0][1], ac_fits100[j][0][2], ac_fits100[j][0][3], ac_fits100[j][0][4], ac_fits100[j][0][5], ac_fits100[j][0][6], ac_fits100[j][0][7], ac_fits100[j][0][8])
#     ac_curve100_ex[j].append(curve100_ex)

ac_curve300 = [[] for _ in range(len(ac_fits300))]
for j in range(len(ac_fits300)):
    i = twor500_index_array300[j]
    curve300 = get_rho_norm_fit(radii_all300_norm[j,:i], ac_fits300[j][0], ac_fits300[j][1], ac_fits300[j][2], ac_fits300[j][3], ac_fits300[j][4], ac_fits300[j][5])
    ac_curve300[j] = np.append(ac_curve300[j], curve300)
    
# ac_curve300_ex = [[] for _ in range(len(ac_fits300))]
# for j in range(len(ac_fits300)):
#     if j == 14:
#         continue
#     if j == 42:
#         continue
#     if j == 45:
#         continue
#     if j == 58:
#         continue
#     if j == 64:
#         continue
#     i = twor500_index_array300_ex[j]
#     curve300_ex = get_rho_norm_fit(radii_all300_norm_ex[j,:i], ac_fits300[j][0][0], ac_fits300[j][0][1], ac_fits300[j][0][2], ac_fits300[j][0][3], ac_fits300[j][0][4], ac_fits300[j][0][5], ac_fits300[j][0][6], ac_fits300[j][0][7], ac_fits300[j][0][8])
#     ac_curve300_ex[j].append(curve300_ex)
 
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
    
    if np.any(chi_square_res_arr100[j] >= 0.08):
        sus_clus_arr100[j] = np.append(sus_clus_arr100[j], j)
sus_clus_arr100 = [element for subarray in sus_clus_arr100 for element in subarray if subarray]


chi_square_res_arr300 = [[] for _ in range(100)]
sus_clus_arr300 = [[] for _ in range(100)]

for j in range(len(rho_vals300_norm)):
    i = twor500_index_array300[j]
    chi_square_res = chi_squared(rho_vals300_norm[j, :i], ac_curve300[j], len(rho_vals300_norm[j, :i]), 9)
    chi_square_res_arr300[j] = np.append(chi_square_res_arr300[j], chi_square_res)
    
    if np.any(chi_square_res_arr300[j] >= 0.08):
        sus_clus_arr300[j] = np.append(sus_clus_arr300[j], j)

sus_clus_arr300 = [element for subarray in sus_clus_arr300 for element in subarray if subarray]

n0_ac100 = [ac_fits100[i][0] for i in range(100) if i not in sus_clus_arr100]
rc_ac100 = [ac_fits100[i][1] for i in range(100) if i not in sus_clus_arr100]
rs_ac100 = [ac_fits100[i][2] for i in range(100) if i not in sus_clus_arr100]
a_ac100 = [ac_fits100[i][3] for i in range(100) if i not in sus_clus_arr100]
B_ac100 = [ac_fits100[i][4] for i in range(100) if i not in sus_clus_arr100]
epsilon_ac100 = [ac_fits100[i][5] for i in range(100) if i not in sus_clus_arr100]

n0_ac300 = [ac_fits300[i][0] for i in range(100) if i not in sus_clus_arr300]
rc_ac300 = [ac_fits300[i][1] for i in range(100) if i not in sus_clus_arr300]
rs_ac300 = [ac_fits300[i][2] for i in range(100) if i not in sus_clus_arr300]
a_ac300 = [ac_fits300[i][3] for i in range(100) if i not in sus_clus_arr300]
B_ac300 =[ac_fits300[i][4] for i in range(100) if i not in sus_clus_arr300]
epsilon_ac300 = [ac_fits300[i][5] for i in range(100) if i not in sus_clus_arr300]




#PLOTTING
#===================================================================================
fig, (ax1,ax2) = plt.subplots(1, 2, sharex='col', dpi=100)
fig.set_size_inches(20,9)
fig.tight_layout()
fig.patch.set_facecolor('white')
fig.suptitle('Fit of VIK density formula to TNG results (chi^2<=0.07)', y=1.1, fontsize=30)

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
    r'$\beta=%.2f$' % (popt100_p1[4], )))

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
    r'$\epsilon=%.2f$' % (popt300_p1[5])))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.75, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


plt.show()

#===================================================================================

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row', dpi=100)
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

bins_n0 = np.logspace(np.log10(bounds[0][0]), np.log10(bounds[1][0]) ,20)

ax1.set_xscale("log")

ax1.hist(n0_ac100, bins=bins_n0, label='TNG100',color="#1F77B4")
ax1.hist(n0_ac300, bins=bins_n0, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(n0)):
    ax1.axvline(n0[i], color = "green", linewidth = 0.5)
    
ax1.axvline(n0[0], label = "Vikhlinin Values", color = "green")

bins_rc = np.logspace(np.log10(bounds[0][1]), np.log10(bounds[1][1]) ,50)

ax1.set_xscale("log")

ax2.hist(rc_ac100, bins=bins_rc, label='TNG100',color="#1F77B4")
ax2.hist(rc_ac300, bins=bins_rc, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(n0)):
    ax2.axvline(rc[i], color = "green" , linewidth = 0.5)
    
ax2.axvline(rc[0], label = "Vikhlinin Values", color = "green")

bins_rs = np.logspace(np.log10(bounds[0][2]), np.log10(bounds[1][2]) ,20)

ax3.set_xscale("log")

ax3.hist(rs_ac100, bins=bins_rs, label='TNG100',color="#1F77B4")
ax3.hist(rs_ac300, bins=bins_rs, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(n0)):
    ax3.axvline(rs[i], color = "green", linewidth = 0.5)
    
bins_a = np.logspace(np.log10(bounds[0][3]), np.log10(bounds[1][3]) ,40)

ax4.set_xscale("log")


ax4.hist(a_ac100, bins=bins_a, label='TNG100',color="#1F77B4")
ax4.hist(a_ac300, bins=bins_a, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax4.axvline(a[i], color = "green", linewidth = 0.5)
    
bins_B = np.logspace(np.log10(bounds[0][4]), np.log10(bounds[1][4]) ,10)

ax5.set_xscale("log")

ax5.hist(B_ac100, bins=bins_B, label='TNG100',color="#1F77B4")
ax5.hist(B_ac300, bins=bins_B,label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax5.axvline(B[i], color = "green", linewidth = 0.5)
    
bins_ep = np.logspace(np.log10(bounds[0][5]), np.log10(bounds[1][5]) ,10)


ax6.set_xscale("log")

ax6.hist(epsilon_ac100, bins=bins_ep, label='TNG100',color="#1F77B4")
ax6.hist(epsilon_ac300, bins=bins_ep, label ='TNG300', color="#FF7F0E", alpha=0.75)

for i in range(len(a)):
    ax6.axvline(epsilon[i], color = "green", linewidth = 0.5)
    


plt.show()






