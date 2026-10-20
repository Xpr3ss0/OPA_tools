from tools import PFT_change_interface, PFT_prism_sym, PFT_prism_sym_test, PFT_prism_sym_test2, prism_pair_angles, fresnel_reflectance
import numpy as np
from refractiveindex import RefractiveIndexMaterial, NoExtinctionCoefficient
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pandas import DataFrame
from os.path import join
from pathlib  import Path

######################
## SETUP PARAMETERS ##
######################

theta_1_range_deg = (4, 75)                                 # apex angle of first prism in degrees
absorption_limit = 0.05                                     # for filtering datapoints before plotting
minimum_pft = 2.0                                           # for filtering datapoints before plotting
lmd = 400                                                   # wavelength in nm
main_plot_savedir = r'D:\VSCode Projects\OPA_tools\pulse_front_tilt\plots\material_comparison'
materials_data_savedir = r'D:\VSCode Projects\OPA_tools\pulse_front_tilt\material_data'
save_plots = False
save_data = True


###############
## MATERIALS ##
###############

db_base = Path.home() / ".refractiveindex.info-database" / "data"
db_specs = db_base / "specs"
db_main = db_base / "main"

# SCHOTT catalogue
db_schott = db_specs / "schott" / "optical"

# OHARA catalogue
db_ohara = db_specs / "ohara" / "optical"

# CDGM catalogue
db_cdgm = db_specs / "cdgm" / "optical"

use_catalogue = db_cdgm
book_name = f"{use_catalogue.parts[-2].upper()}-{use_catalogue.parts[-1]}"




materials = [
    ["glass", "fused_silica", "Malitson", "FS"],
]


# add common materials
materials.append(["popular_glass", "BK7", "SCHOTT", "N-BK7"])
materials.append(["popular_glass", "BAF10", "SCHOTT", "N-BaF10"])
materials.append(["popular_glass", "FK51A", "SCHOTT", "N-FK51A"])
materials.append(["popular_glass", "LASF9", "SCHOTT", "N-LASF9"])
materials.append(["main", "Al2O3", "Malitson-o", "Al2O3-o"])
materials.append(["main", "Al2O3", "Malitson-o", "Al2O3-e"])
materials.append(["main", "Y3Al5O12", "Hrabovsky", "YAG"])

# Fluorides
materials.append(["main", "CaF2", "Zheng-21C", "CaF2"])
materials.append(["main", "BaF2", "Zheng-21C", "BaF2"])
materials.append(["main", "LiF", "Li", "LiF"])
materials.append(["main", "MgF2", "Zheng-o-21C", "MgF2-o"])
materials.append(["main", "MgF2", "Zheng-e-21C", "MgF2-e"])

# Selenides

materials.append(["main", "ZnSe", "Amotchkina", "ZnSe"])

# add specified catalogue
catalogue_materials = sorted(p.stem for p in use_catalogue.glob("*.yml"))

for mat in catalogue_materials:
    print(f"Adding material {mat}")
    materials.append(["specs", book_name, str(mat), str(mat)])

"""
# add SCHOTT flint
materials.extend([["specs", "SCHOTT-optical", f"F{sufix}", f"F{sufix}"] for sufix in [2, "2G12", "2HT", 5]])
materials.append(["specs", "SCHOTT-optical", f"N-F2", f"N-F2"])

# add SCHOTT dense flint
materials.extend([["specs", "SCHOTT-optical", f"SF{i}", f"SF{i}"] for i in [1,2,4,5,6,10,11,14,15]])
materials.extend([[f"specs", "SCHOTT-optical", f"N-SF{i}", f"N-SF{i}"] for i in [1,2,4,5,6,8,10,11,14,15,19,56,57,64,66]])
materials.extend([[f"specs", "SCHOTT-optical", f"P-SF{i}", f"P-SF{i}"] for i in [8,67,68,69]])

# add SCHOTT light flint
materials.extend([["specs", "SCHOTT-optical", f"LF{sufix}", f"LF{sufix}"] for sufix in [5, "5G15", "5G19", "5HTi"]])

# add SCHOTT very light flint
materials.extend([["specs", "SCHOTT-optical", f"LLF-{sufix}", f"LLF-{sufix}"] for sufix in [1,"1HTi"]])

# add SCHOTT special short flint
materials.extend([["specs", "SCHOTT-optical", f"KZFS{sufix}", f"KZFS{sufix}"] for sufix in [12,"N4",]])
materials.extend([["specs", "SCHOTT-optical", f"N-KZFS{sufix}", f"N-KZFS{sufix}"] for sufix in [2,4,"4HT",5,8,11]])

# add SCHOTT Barium flint
materials.extend([["specs", "SCHOTT-optical", f"N-BAF{i}", f"N-BAF{i}"] for i in [3,4,10,51,52]])
materials.append(["specs", "SCHOTT-optical", f"BAFN6", f"BAFN6"])

# add SCHOTT Barium light flint
materials.extend([["specs", "SCHOTT-optical", f"N-BALF{i}", f"N-BALF{i}"] for i in [4,5]])

# add SCHOTT dense flint
materials.extend([["specs", "SCHOTT-optical", f"N-BASF{i}", f"N-BASF{i}"] for i in [2,64]])

# add SCHOTT Lanthanum flint
materials.extend([["specs", "SCHOTT-optical", f"N-LAF{i}", f"N-LAF{i}"] for i in [2,3,7,21,32,33,34,35,36]])
materials.append(["specs", "SCHOTT-optical", f"LAFN7", f"LAFN7"])
materials.append(["specs", "SCHOTT-optical", f"P-LAF37", f"P-LAF37"])

# add SCHOTT Lanthanum dense flint
materials.extend([["specs", "SCHOTT-optical", f"N-LASF{i}", f"N-LASF{i}"] for i in ["9HT",9,31,"31A",40,41,43,44,45,"45HT",46,"46A","46B"]])
materials.extend([["specs", "SCHOTT-optical", f"P-LASF{i}", f"P-LASF{i}"] for i in [47, 50, 51]])
materials.extend([["specs", "SCHOTT-optical", f"LASF{i}", f"LASF{i}"] for i in ["N9",35]])


# add SCHOTT crown
materials.extend([["specs", "SCHOTT-optical", f"K{sufix}", f"K{sufix}"] for sufix in [7,10,"5G20"]])
materials.extend([["specs", "SCHOTT-optical", f"N-ZK{sufix}", f"N-ZK{sufix}"] for sufix in [7,"7A"]])
materials.append(["specs", "SCHOTT-optical", f"N-K5", f"N-K5"])

# add SCHOTT dense crown
materials.extend([["specs", "SCHOTT-optical", f"N-SK{sufix}", f"N-SK{sufix}"] for sufix in [2, "2HT", 4,5,10,11,14,15,16,]])
materials.extend([["specs", "SCHOTT-optical", f"P-SK{sufix}", f"P-SK{sufix}"] for sufix in [57,"57Q1","58A",60]])

# add SCHOTT very dense crown
materials.extend([["specs", "SCHOTT-optical", f"N-SSK{sufix}", f"N-SSK{sufix}"] for sufix in [2,5,8]])

# add SCHOTT Borosilicate crown
materials.extend([["specs", "SCHOTT-optical", f"N-BK{sufix}", f"N-BK{sufix}"] for sufix in [7, "7HT", "7HTi", 10]])
materials.append(["specs", "SCHOTT-optical", f"P-BK7", f"P-BK7"])
materials.append(["specs", "SCHOTT-optical", f"BK7G18", f"BK7G18"])

# add SCHOTT Barium crown
materials.extend([["specs", "SCHOTT-optical", f"N-BAK{sufix}", f"N-BAK{sufix}"] for sufix in [1,2,4,"4HT"]])

# add SCHOTT Fluor crown
materials.extend([["specs", "SCHOTT-optical", f"N-FK{sufix}", f"N-FK{sufix}"] for sufix in [5,51,"51A",58]])
materials.extend([["specs", "SCHOTT-optical", f"FK{sufix}", f"FK{sufix}"] for sufix in [3,"5HTi"]])

# add SCHOTT Lanthanum crown
materials.extend([["specs", "SCHOTT-optical", f"N-LAK{sufix}", f"N-LAK{sufix}"] for sufix in [7,8,9,10,12,14,21,22,"33A","33B",34,35]])
materials.extend([["specs", "SCHOTT-optical", f"LAK{sufix}", f"LAK{sufix}"] for sufix in ["9G15","L12"]])
materials.append(["specs", "SCHOTT-optical", f"P-LAK35", f"P-LAK35"])

# add SCHOTT Phosphate crown
materials.extend([["specs", "SCHOTT-optical", f"N-PK{sufix}", f"N-PK{sufix}"] for sufix in [51,"51A"]])
materials.append(["specs", "SCHOTT-optical", f"P-PK35", f"P-PK35"])

# add SCHOTT Dense Phosphate crown
materials.extend([["specs", "SCHOTT-optical", f"N-PSK{sufix}", f"N-PSK{sufix}"] for sufix in [3,53,"53A"]])

# add SCHOTT crown flint
materials.append(["specs", "SCHOTT-optical", f"N-KF9", f"N-KF9"])

"""


##########
# SCRIPT #
##########


def get_angles(theta_1_range_deg, material_1: RefractiveIndexMaterial, material_2: RefractiveIndexMaterial, lmd=400):

    # refractive indices
    n1_func = material_1.get_refractive_index
    n2_func = material_2.get_refractive_index
    n1 = material_1.get_refractive_index(lmd)                           # refractive index of first prism
    n2 = material_2.get_refractive_index(lmd)                           # refractive index of second prism

    # setup parameter grid
    theta_1_array = np.linspace(np.radians(theta_1_range_deg[0]), np.radians(theta_1_range_deg[1]), 100)
    theta_2_array = np.zeros_like(theta_1_array)
    PFT_1_array = np.zeros_like(theta_1_array)
    PFT_2_array = np.zeros_like(theta_1_array)
    PFT_total_array = np.zeros_like(theta_1_array)
    PFT_total_simple_array = np.zeros_like(theta_1_array)


    def get_objective_1(theta_1):

        def objective(delta_theta):
            phi_i_2 = np.arcsin(n2 * np.sin(theta_1 /2 + delta_theta /2))
            phi_e_1 = np.arcsin(n1 * np.sin(theta_1 /2))
            diff = np.abs(phi_e_1 - (phi_i_2 - delta_theta/2))
            return diff
        
        return objective

    def get_objective_2(theta_1):
        
        phi_e_1 = np.arcsin(n1 * np.sin(theta_1 /2))

        def objective(delta_theta):
            theta_2 = theta_1 + delta_theta
            phi_i_2 = np.arcsin(n2 * np.sin(theta_2 /2))
            diff = np.abs(2*phi_e_1 - theta_1 - (2*phi_i_2 - theta_2))
            return diff

        return objective

    get_objective_used = get_objective_1

        
    # compute theta_2 for all theta_1 values
    for i, theta_1 in enumerate(theta_1_array):

        # objective for finding theta_2
        objective = get_objective_used(theta_1)

        delta_theta_min = - theta_1
        delta_theta_max = np.pi/2 - theta_1
        res = minimize_scalar(objective, bounds=(delta_theta_min, delta_theta_max), method='bounded')
        theta_2_array[i] = theta_1 + res.x

        # compute pulse front tilt angles
        PFT_1 = PFT_prism_sym(theta_1, n1_func, lmd=lmd)
        PFT_2 = PFT_prism_sym(theta_2_array[i], n2_func, lmd=lmd)
        PFT_1_array[i] = PFT_1
        PFT_2_array[i] = PFT_2
        PFT_total_array[i] = np.arctan(np.tan(PFT_1) - np.tan(PFT_2))
        PFT_total_simple_array[i] = PFT_1 - PFT_2

    return theta_1_array, theta_2_array, PFT_1_array, PFT_2_array, PFT_total_array, PFT_total_simple_array


def get_transmission(theta_1_array, theta_2_array, material_1: RefractiveIndexMaterial, material_2: RefractiveIndexMaterial, lmd=400):

    n1 = material_1.get_refractive_index(lmd)                           # refractive index of first prism
    n2 = material_2.get_refractive_index(lmd)                           # refractive index of second prism
    phi_i1_array = np.arcsin(n1 * np.sin(theta_1_array /2)) # incident angle for symmetric configuration
    result_arrays = prism_pair_angles(phi_i1_array, theta_1_array, theta_2_array, n1, n2)
    R_1_s, R_1_p = fresnel_reflectance(1, n1, result_arrays['phi_i1'])
    R_2_s, R_2_p = fresnel_reflectance(n1, n2, theta_1_array - result_arrays['phi_r1'])
    R_3_s, R_3_p = fresnel_reflectance(n2, 1, result_arrays['phi_i2'])
    R_4_s, R_4_p = fresnel_reflectance(n2, 1, theta_2_array - result_arrays['phi_r2'])
    T_total_s = (1 - R_1_s) * (1 - R_2_s) * (1 - R_3_s) * (1 - R_4_s)
    T_total_p = (1 - R_1_p) * (1 - R_2_p) * (1 - R_3_p) * (1 - R_4_p)
    return T_total_s, T_total_p

def get_absorption(theta_1_val, theta_2_val, material_1: RefractiveIndexMaterial, material_2: RefractiveIndexMaterial, lmd=400):
    
    # obtain approximate extinction at max. PFT, assuming beam traversal through center prisms, with 3cm side length
    travel_distance_1 = np.sin(theta_1_valid_max / 2) * 3e-2 # per m
    travel_distance_2 = np.sin(theta_2_valid_max / 2) * 3e-2 # per m
    try:
        k1 = mat1.get_extinction_coefficient(lmd)                    # extinction coefficient of first prism
    except NoExtinctionCoefficient:
        k1 = 0.0

    try:
        k2 = mat2.get_extinction_coefficient(lmd)                    # extinction coefficient of second prism
    except NoExtinctionCoefficient:
        k2 = 0.0

    alpha1 = 4 * np.pi * k1 / (lmd*1e-9)
    alpha2 = 4 * np.pi * k2 / (lmd*1e-9)
    
    abs1 = (1 - np.exp(-alpha1 * travel_distance_1))
    abs2 = (1 - np.exp(-alpha2 * travel_distance_2))
    return alpha1, alpha2, abs1, abs2


if __name__=="__main__":

    # loop through all material combinations
    combination_names = []
    combination_details = []
    max_pft_vals = []
    max_apex_vals = []
    absorption_vals = []
    fresnel_trans_vals = []


    for i, mat1_keys in tqdm(enumerate(materials[:-1]), total=len(materials)):
        try:
                mat1 = RefractiveIndexMaterial(shelf=mat1_keys[0], book=mat1_keys[1], page=mat1_keys[2])
        except KeyError:
            print(f"Cannot find material {mat1_keys[3]} in Database. Skipping")
            continue
        
        for mat2_keys in materials[i+1:]:
            try:
                mat2 = RefractiveIndexMaterial(shelf=mat2_keys[0], book=mat2_keys[1], page=mat2_keys[2])
            except KeyError:
                print(f"Cannot find material {mat2_keys[3]} in Database. Skipping")
                continue

            # obtain angles
            theta_1_array, theta_2_array, PFT_1_array, PFT_2_array, PFT_total_array, PFT_total_simple_array = get_angles(
                theta_1_range_deg=theta_1_range_deg,
                material_1=mat1,
                material_2=mat2,
                lmd=lmd
            )

            # obtain transmissions
            T_s, T_p = get_transmission(theta_1_array=theta_1_array, theta_2_array=theta_2_array, 
                                        material_1=mat1, material_2=mat2, lmd=lmd)
            
            # obtain PFT value within 80% transmission threshold (p-pol)
            transmission_filter = T_p > 0.8
            if not np.any(transmission_filter):
                continue
            PFT_filtered = PFT_total_array[transmission_filter]
            PFT_valid_max_index = np.argmax(np.abs(PFT_filtered))
            PFT_valid_max = PFT_filtered[PFT_valid_max_index]
            theta_1_valid_max = theta_1_array[transmission_filter][PFT_valid_max_index]
            theta_2_valid_max = theta_2_array[transmission_filter][PFT_valid_max_index]
            transmission_valid_max = T_p[transmission_filter][PFT_valid_max_index]

            alpha1, alpha2, abs1, abs2 = get_absorption(
                theta_1_val=theta_1_valid_max, theta_2_val=theta_2_valid_max,
                material_1=mat1, material_2=mat2, lmd=lmd
            )
            total_abs = 1 - (1 - abs1) * (1 - abs2)


            combination_names.append((mat1_keys[3], mat2_keys[3]))
            max_pft_vals.append(np.abs(np.degrees(PFT_valid_max)))
            max_apex_vals.append((np.degrees(theta_1_valid_max), np.degrees(theta_2_valid_max)))
            absorption_vals.append(total_abs)
            fresnel_trans_vals.append(transmission_valid_max)
            combination_details.append((mat1_keys, mat2_keys))


    fig, ax = plt.subplots()

    # save to csv
    if save_data:
        save_path = join(materials_data_savedir, f"pft_materials_{lmd}nm_{book_name}.csv")
        comb_names = [f"{comb[0]}+{comb[1]}" for i, comb in enumerate(combination_names)]
        apex_1_vals = [angles[0] for angles in max_apex_vals]
        apex_2_vals = [angles[1] for angles in max_apex_vals]
        data = {
            "combination": comb_names,
            "max_viable_pft": max_pft_vals,
            "apex_angle_1": apex_1_vals,
            "apex_angle_2": apex_2_vals,
            "total_fresnel_transmission": fresnel_trans_vals,
            "total_bulk_absorption": absorption_vals,
            "material_1_book": [combination_details[i][0][0] for i in range(len(combination_details))],
            "material_2_book": [combination_details[i][1][0] for i in range(len(combination_details))],
            "material_1_shelf": [combination_details[i][0][1] for i in range(len(combination_details))],
            "material_2_shelf": [combination_details[i][1][1] for i in range(len(combination_details))],
            "material_1_page": [combination_details[i][0][2] for i in range(len(combination_details))],
            "material_2_page": [combination_details[i][1][2] for i in range(len(combination_details))],
        }

        df = DataFrame(data=data)
        df.sort_values(by="max_viable_pft", ascending=False)
        df.to_csv(save_path, sep=",")

    # filter
    absorption_vals = np.array(absorption_vals)
    max_pft_vals = np.array(max_pft_vals)
    filter = (absorption_vals < absorption_limit) & (max_pft_vals > minimum_pft)


    ax.scatter(absorption_vals[filter], max_pft_vals[filter])
    labels = [f"{comb[0]}+{comb[1]}" for i, comb in enumerate(combination_names) if filter[i]]
    for i in range(len(labels)):
        ax.text(absorption_vals[filter][i], max_pft_vals[filter][i], labels[i])

    ax.grid()
    plt.xlabel("absorption (bulk)")
    plt.ylabel("max. pft angle (degree)")

    plt.tight_layout()
    plt.show()


            

