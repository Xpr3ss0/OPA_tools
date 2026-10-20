import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = r'D:\VSCode Projects\OPA_tools\pulse_front_tilt\material_data\pft_materials_400nm_SCHOTT-optical.csv'
max_absorption = 0.05
min_pft_angle = 1.5

must_include = ["MgF2-e"]   #["PBL35Y",]
must_exclude = ["TIH6", "TIH11", "TIH14", "TIH23", "FTM16", "PBM11", "LAM66", "LAH75", 
                "LAH60", "LAM55", "LAH63", "BAH54", "LAH58", "LAH53", "LAH52", "LAH55",
                "LAH51","LAM2","PBL7","LAH54","LAH67","LAH65","LAH64","LAH59","LAM3","LAM54","LAL10",
                "LAM60","LAL56","LAH66","YGH52","YGH51","LAL13","LAL61","LAL60","BSM71","LAL59"]


if __name__=="__main__":

    df = pd.read_csv(filepath, delimiter=",")
    print(df)

    df_filtered = df.loc[df["total_bulk_absorption"] <= max_absorption]
    df_filtered = df_filtered.loc[df_filtered["max_viable_pft"] >= min_pft_angle]

    for s in must_include:
        parts = df_filtered["combination"].str.split("+", expand=True)
        df_filtered = df_filtered[(parts[0] == s) | (parts[1] == s)]
    
    for s in must_exclude:
        parts = df_filtered["combination"].str.split("+", expand=True)
        df_filtered = df_filtered[(parts[0] != s) & (parts[1] != s)]

    df_sorted = df_filtered.sort_values(by="max_viable_pft", ascending=False)

    fig, ax = plt.subplots()

    xdata = df_sorted["total_bulk_absorption"]
    ydata = df_sorted["max_viable_pft"]
    combs = df_sorted["combination"]
    ax.scatter(xdata, ydata)
    for i in range(len(df_sorted)):
        ax.text(xdata.to_list()[i], ydata.to_list()[i], combs.to_list()[i])


    plt.xlabel("total bulk absorption")
    plt.ylabel("max. viable PFT angle")
    plt.grid()
    plt.tight_layout()
    plt.show()