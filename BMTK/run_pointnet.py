import os, sys
import math
import numpy as np
from bmtk.simulator import pointnet
from bmtk.simulator.pointnet.pyfunction_cache import synaptic_weight
from bmtk.simulator.pointnet.io_tools import io

import pandas as pd
import json
import csv
import nest


try:
    nest.Install('glifmodule')
except Exception as e:
    pass


@synaptic_weight
def DirectionRule_others(edges, src_nodes, trg_nodes):
    src_tuning = src_nodes['tuning_angle'].values
    tar_tuning = trg_nodes['tuning_angle'].values
    sigma = edges['weight_sigma'].values
    nsyn = edges['nsyns'].values
    syn_weight = edges['syn_weight'].values

    delta_tuning_180 = np.abs(np.abs(np.mod(np.abs(tar_tuning - src_tuning), 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180 / sigma) ** 2)
    
    return syn_weight * w_multiplier_180 * nsyn


@synaptic_weight
def DirectionRule_EE(edges, src_nodes, trg_nodes):
    src_tuning = src_nodes['tuning_angle'].values
    tar_tuning = trg_nodes['tuning_angle'].values
    x_tar = trg_nodes['x'].values
    x_src = src_nodes['x'].values
    z_tar = trg_nodes['z'].values
    z_src = src_nodes['z'].values
    sigma = edges['weight_sigma'].values
    nsyn = edges['nsyns'].values
    syn_weight = edges['syn_weight'].values
    
    delta_tuning_180 = np.abs(np.abs(np.mod(np.abs(tar_tuning - src_tuning), 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180 / sigma) ** 2)

    delta_x = (x_tar - x_src) * 0.07
    delta_z = (z_tar - z_src) * 0.04

    theta_pref = tar_tuning * (np.pi / 180.)
    xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
    sigma_phase = 1.0
    phase_scale_ratio = np.exp(- (xz ** 2 / (2 * sigma_phase ** 2)))

    # To account for the 0.07 vs 0.04 dimensions. This ensures the horizontal neurons are scaled by 5.5/4 (from the
    # midpoint of 4 & 7). Also, ensures the vertical is scaled by 5.5/7. This was a basic linear estimate to get the
    # numbers (y = ax + b).
    theta_tar_scale = abs(abs(abs(180.0 - np.mod(np.abs(tar_tuning), 360.0)) - 90.0) - 90.0)
    phase_scale_ratio = phase_scale_ratio * (5.5 / 4.0 - 11.0 / 1680.0 * theta_tar_scale)

    return syn_weight * w_multiplier_180 * phase_scale_ratio * nsyn

# The next part corresponds to the resulting data format, which is modified to facilitate the visualization of the results, especially in the analysis with the two impact metric.

parameters_txt = []
n_exec = 10
def type_attack():
    global typ_attack 
    with open("/home/type_attack.txt", 'r') as file:
        line = file.readline()
        attack = line.split(":")
        attack_ = attack[1]

        if "FLO" in attack_ :
            typ_attack = "FLO"

            with open("/home/FLO_attributes.txt", 'r') as file_attributes:
                for line in file_attributes:
                    parameter = line.split(":")
                    parameter = parameter[1].replace("\n", "")
                    parameters_txt.append(parameter)
        elif "JAM" in attack_:
            typ_attack = "JAM"
            with open("/home/JAM_attributes.txt", 'r') as file_attributes:
                for line in file_attributes:
                    parameter = line.split(":")
                    parameter = parameter[1].replace("\n", "")
                    parameters_txt.append(parameter)
        else:
            typ_attack ="Spontaneous"




def main(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)    
    sim = pointnet.PointSimulator.from_config(configure, graph)
    sim.run()


if __name__ == '__main__':
    type_attack()
    dfs=[]
    for exec in range(n_exec):

        if __file__ != sys.argv[-1]:
            main(sys.argv[-1])
        else:
            main('config.json')
            
        # Once the simulation ends and the files of resulting data are created and saved in the directory output,
        # the code changes its format, adding the attack executed, the number of execution, the stimulus and 
        # background used, the attack instants, and the number of neurons attacked.

        with open('config.json','r') as file:
            data = json.load(file)

            output_dir = data["manifest"]["$OUTPUT_DIR"].replace("$BASE_DIR", ".")
            spikes_file_csv = data["output"]["spikes_file_csv"]
            resulting_path = output_dir + "/" + spikes_file_csv

            df_spikes = pd.read_csv(resulting_path, delimiter=" ")
    
            if "LGN_spikes" in data["inputs"]:

                inputs_dir_LGN = data["inputs"]["LGN_spikes"]["input_file"]
                inputs_dir_LGN_trial = inputs_dir_LGN.split(".")

            else:
                inputs_dir_LGN = ""

            inputs_dir_BKG = data["inputs"]["BKG_spikes"]["input_file"]

            inputs_dir_BKG = inputs_dir_BKG.split(".")
            
        df_spikes['neuron_ids'] = df_spikes["node_ids"]
        df_spikes = df_spikes.drop(['population'], axis = 1)

	# The attack executed.
        merged_df['attack'] = typ_attack
        
        # The number of executions.
        merged_df['n_exec'] = exec
 	
 	# The stimulus and BKG used.
        if "fullField" in inputs_dir_LGN:
            merged_df['stimulus'] = "Flash_" + inputs_dir_LGN_trial[1]
        elif "movie" in inputs_dir_LGN:
            merged_df['stimulus'] = "Movie_" + inputs_dir_LGN_trial[1]
        elif "full3_production" in inputs_dir_LGN:
            inputs_dir_LGN_spikes_trial = inputs_dir_LGN.split("/")
            inputs_dir_LGN_trial = inputs_dir_LGN_spikes_trial[2].split(".")
            merged_df['stimulus'] = "Gratings_" + inputs_dir_LGN_trial[1]

        merged_df["BKG"] = "BKG_" + inputs_dir_BKG[1]
	
	# The attack instants, the number of neurons attacked, and the voltage used.
        if typ_attack == "FLO":
            merged_df["instant_attack"] = parameters_txt[0]
            merged_df["v_increment"] = parameters_txt[1]
            merged_df["n_neurons"] = parameters_txt[2]

            merged_df = merged_df.reindex(columns=['attack','n_exec','stimulus','BKG','instant_attack','v_increment','n_neurons','timestamps','neuron_ids']) 

        elif typ_attack == "JAM":
            merged_df["init_attack"] = parameters_txt[0]
            merged_df["end_attack"] = parameters_txt[1]
            merged_df["n_neurons"] = parameters_txt[2]

            merged_df = merged_df.reindex(columns=['attack','n_exec','stimulus','BKG','init_attack','end_attack','n_neurons','timestamps','neuron_ids']) 
 
        else:
            merged_df = merged_df.reindex(columns=['attack','n_exec','stimulus','BKG','timestamps','neuron_ids']) 

        dfs.append(merged_df)

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df.to_csv(resulting_path, index=False, sep=';')









    
