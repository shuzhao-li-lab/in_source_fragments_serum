import io
import sys
import json
import contextlib
from matchms.importing import load_from_msp
from khipu.extended import peaklist_to_khipu_list, export_empCpd_khipu_list
from mass2chem.search import build_centurion_tree, find_all_matches_centurion_indexed_list

import tqdm
import time

sys.path.insert(0, '..')
from mining import * 
from isf_helper import (extract_ms2_spectrum, 
                        get_comprehensive_stats_per_dataset, 
                        explain_a_dataset_by_mz_deltas, 
                        explain_a_dataset_byMS2)

def load_from_mona(path):
    '''read from mona ms2 file from a given path
    
    path: the path to mona .msp file
    
    return: the inchikey-spectra pair.
    '''
    # reused from JM
    spectral_registry = {}
    total = 0
    for x in tqdm.tqdm(load_from_msp(path)):
        try:
            inchikey = x.metadata_dict()['inchikey']
            if inchikey:
                if inchikey not in spectral_registry:
                    spectral_registry[inchikey] = []
                spectral_registry[inchikey].append(x)
                total += 1
        except:
            pass

    print("MS2 #: ", str(len(spectral_registry)), " Compounds with ", str(total), " MS2 Spectra")
    return spectral_registry

def build_centurion_tree_from_msp(ion_mode):
    import logging
    logging.getLogger("matchms").setLevel(logging.ERROR)
    spectral_registry = {}
    if ion_mode == 'pos':
        spectral_registry = load_from_mona('../MoNA_MS2/filtered_MoNA-export-LC-MS-MS_Positive_Mode.msp')
    else:
        spectral_registry = load_from_mona('../MoNA_MS2/filtered_MoNA-export-LC-MS-MS_Negative_Mode.msp')

    # ms2List is usable MoNA MS/MS compounds
    ms2_list, no_precursor = [], []
    for sp in spectral_registry.values(): 
        try:
            ms2_list.append(extract_ms2_spectrum(sp[0])) 
        except KeyError:
            no_precursor.append(sp[0])
            
    print(f'{len(ms2_list)} spectra are found with precursors.')

    ms2_tree = build_centurion_tree([{'mz': x[0], 'rtime': 0, 'name': x[1], 'peaks': x[2]} for x in ms2_list])
    return ms2_tree



dict_rtwindow = {}
for line in open('elution_parameters_16studies_tof.tsv').readlines()[1:]:
    a = line.rstrip().split('\t')
    dict_rtwindow[a[0]] = float(a[5])
    

pos_candidate_fragments = '''18.01	113	18.010565	water	{'H': 2, 'O': 1}
14.015	83	14.015649	addition of acetic acid and loss of CO2. Reaction: (+C2H2O2) and (-CO2)	{'C': 1, 'H': 2}
2.015	80	2.014552	2H	{'H': 2}
28.0305	59	28.0313	± C2H4, natural alkane chains such as fatty acids	{'C': 2, 'H': 4}
46.0055	54	46.00548	± CO+H2O (carboxylic acid)	{'C': 1, 'O': 2, 'H': 2}
17.0265	53	17.0265	addition of ammonia. Reaction: (+NH3)	{'N': 1, 'H': 3}
11.9995	45	12.0	methylation and reduction	{'C': 1}
44.0255	41	44.0262	hydroxyethylation	{'C': 2, 'H': 4, 'O': 1}
26.015	41	26.01565	acetylation and loss of oxygen. Reaction: (+C2H2O) and (-O)	{'C': 2, 'H': 2}
15.995	40	15.99492	± O, e.g. oxidation/reduction	{'O': 1}
16.031	39	16.0313	Methylation + reduction	{'C': 1, 'H': 4}
32.026	37	32.026215	MeOH	{'C': 1, 'H': 4, 'O': 1}
39.993	37	39.9925	extra OH sodium adduct	{'H': 1, 'O': 1}
27.9945	37	27.9949	addition of CO. Reaction: (+CO)	{'C': 1, 'O': 1}
23.999	36	24.0	acetylation and loss of water. Reaction: (+C2H2O) and (-H2O)	{'C': 2}
42.0465	35	42.04695	± C3H6, propylation	{'C': 3, 'H': 6}
9.984	32	9.98435	addition of CO and loss of water. Reaction: (+CO) and (-H2O)	{'C': 1, 'H': -2}
30.0105	31	30.010564	addition of acetic acid and loss of CO. Reaction: (+C2H2O2) and (-CO)	{'C': 1, 'H': 2, 'O': 1}
56.0625	30	56.0626	± C4H8, butylation	{'C': 4, 'H': 8}
13.979	30	13.979264	nitrification and loss of oxygen. Reaction: (NH2 -> NO2) and (-O)	{'H': -2, 'O': 1}
'''
pos_candidate_fragments = [
    (float(x.split()[0]), x) for x in pos_candidate_fragments.splitlines()
]
pos_isf_candidate_fragments = [x[0] for x in pos_candidate_fragments]

neg_candidate_fragments = '''67.9875	140	67.987424	NaCOOH	"{'C': 1, 'O': 2, 'Na': 1, 'H': 1}"
2.015	70	2.014552	2H	{'H': 2}
135.974	49	135.974848	2X NaCOOH	"{'C': 2, 'O': 4, 'H': 2, 'Na': 2}"
82.002	48	82.003035	methylation and addition of trifluoromethyl. Reaction: (+CH2) and (+CF3-H)	"{'C': 2, 'H': 1, 'F': 3}"
1.011	41	1.007276467	1H	{'H':1 }
0.996	39	0.996585	addition of Guanine and loss of D-ribose. Reaction: (+C5H3N5) and (-C5H8O4)	"{'H': -5, 'N': 5, 'O': -4}"
43.989	35	43.9898	addition of CO2. Reaction: (+CO2)	"{'C': 1, 'O': 2}"
14.015	32	14.015649	addition of acetic acid and loss of CO2. Reaction: (+C2H2O2) and (-CO2)	"{'C': 1, 'H': 2}"
46.005	31	46.005305	addition of Phosphate and dechlorination. Reaction: (+HPO3) and (-Cl+H)	"{'H': 2, 'O': 3, 'P': 1, 'Cl': -1}"
26.015	29	26.01565	acetylation and loss of oxygen. Reaction: (+C2H2O) and (-O)	"{'C': 2, 'H': 2}"
61.97	29	61.975755	addition of Phosphate and defluorination. Reaction: (+HPO3) and (-F+H)	"{'H': 2, 'O': 3, 'P': 1, 'F': -1}"
129.957	29	129.958482	addition of di-phosphate and denitrification. Reaction: (+H2P2O6) and (NO2 -> NH2)	"{'H': 4, 'O': 4, 'P': 2}"
60.021	29	-60.0211	desmolysis	"{'C': -2, 'H': -4, 'O': -2}"
74.0365	28	74.03678	propionylation	"{'C': 3, 'H': 6, 'O': 2}"
23.9995	27	24	acetylation and loss of water. Reaction: (+C2H2O) and (-H2O)	{'C': 2}
10.029	27	10.028802	addition of CO2 and dechlorination. Reaction: (+CO2) and (-Cl+H)	"{'C': 1, 'H': 1, 'O': 2, 'Cl': -1}"
44.026	26	44.0262	hydroxyethylation	"{'C': 2, 'H': 4, 'O': 1}"
18.0105	26	18.010565	water	"{'H': 2, 'O': 1}"
6.0165	24	6.010565	addition of tiglyl and loss of phenyl. Reaction: (+C5H6O) and (-C6H5+H)	"{'C': -1, 'H': 2, 'O': 1}"
15.994	23	15.9949	oxidation	{'O': 1}
'''
neg_candidate_fragments = [
    (float(x.split()[0]), x) for x in neg_candidate_fragments.splitlines()
]

neg_isf_candidate_fragments = [x[0] for x in neg_candidate_fragments]

# pos
isp_pos = [ (1.003355, '13C/12C', (0, 0.8)),
                            (2.00671, '13C/12C*2', (0, 0.8)),
                            # (3.9948, '44Ca/40Ca', (0, 0.1)), # 2%
                            (1.9970, '37Cl/35Cl', (0.1, 0.8)), # 24.24%
                            ]

asp_pos = [  # initial patterns are relative to M+H+
                            (21.98194, 'Na/H'),
                            (17.0265, 'NH3'),
                            (41.026549, 'ACN'),     # Acetonitrile
                            (67.987424, 'NaCOOH'),
                            (32.026215, 'MeOH')
                            ]

# neg 
isp_neg = [ (1.003355, '13C/12C', (0, 0.8)),
                            (2.00671, '13C/12C*2', (0, 0.8)),
                            (1.9970, '37Cl/35Cl', (0.1, 0.8)), # 24.24%
                            (1.9958, '32S/34S', (0, 0.1)), # 4%
                            ]

asp_neg = [  # initial patterns are relative to M+H+
                            (21.98194, 'Na/H'), 
                            (67.987424, 'NaCOOH'),
                            (135.974848, 'NaCOOH*2'),
                            (82.0030, 'NaCH2COOH'),
                            # (1.99566, 'F <-> OH'), 
                            ]


from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

# Function to process each dataset
def process_dataset(f):
    
    ion_mode = 'pos' if 'pos' in f else 'neg'
    ms2_tree = build_centurion_tree_from_msp(ion_mode)

    start_time = time.time()
    print(f"Starting dataset: {f}")
    results = {}
    try:
        # Step 1: Get comprehensive stats
        list_khipus, all_assigned_fids, list_features = get_comprehensive_stats_per_dataset(
            f'../input_data_tof/{f}/full_feature_table.tsv', 
            dict_rtwindow[f], 
            isp_pos if ion_mode == 'pos' else isp_neg,
            asp_pos if ion_mode == 'pos' else asp_neg,
            ion_mode
        )
        print(f"Dataset {f} step 1 finished")
        # Step 2: Sort and process remaining features
        remaining_features = [feature for feature in list_features if feature['id'] not in all_assigned_fids]
        # all_representative_features = [] + remaining_features
        # for khipu in list_khipus:
        #     all_representative_features.append(get_M0(khipu['MS1_pseudo_Spectra']))
        
        list_khipus = sorted(list_khipus, key=lambda x: x['neutral_formula_mass'], reverse=True)
        print(f"Dataset {f} step 2 finished")
        # Step 3: Explain by mz deltas
        explained_khipu_ids, explained_feature_ids, delta_values_used = explain_a_dataset_by_mz_deltas(
            list_khipus, 
            remaining_features, 
            pos_isf_candidate_fragments if ion_mode == 'pos' else neg_isf_candidate_fragments, 
            rt_stdev=dict_rtwindow[f]
        )
        print(f"Dataset {f} step 3 finished")
        # Step 4: Explain by MoNA MS2
        have_precursors, matched2 = explain_a_dataset_byMS2(
            list_features, 
            ms2_tree, 
            rt_stdev=dict_rtwindow[f]
        )
        delta_values_ms2 = []
        for x in matched2:
            delta_values_ms2.append(x)
        print(f"Dataset {f} step 4 finished")
        # Step 5: Compile results
        results = {
            'study': f,
            'num_khipus': len(list_khipus),
            'num_features': len(list_features),
            'mzdelta_explained_khipus': len(set(explained_khipu_ids)), 
            'mzdelta_explained_features': len(set(explained_feature_ids)),
            'freq_delta_values_used': delta_values_used,
            'have_precursors': len(have_precursors),
            'ms2_explained_features': len(matched2),
            'delta_values_ms2': delta_values_ms2,
        }
        os.makedirs(f'output/tof/01292025/{ion_mode}', exist_ok=True)
        # Save results to JSON file
        with open(f'output/tof/01292025/{ion_mode}/{f}.json', 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2) 
        
        print(f"Dataset {f} processed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing dataset {f}: {e}")
    return results


# Use ProcessPoolExecutor for multiprocessing
if __name__ == "__main__":
    tof_datasets = [x.rstrip() for x in open('selected_16_tof_datasets.txt').readlines()]
    # pos_orbi_datasets = [x for x in orbi_datasets if 'pos' in x]
    # neg_orbi_datasets = [x for x in tof_datasets if 'neg' in x]
    # Set number of workers to number of available CPUs
    max_workers = 9  # Adjust based on available cores or system capacity

    # List to track completed results
    tally_pos = []

    # Submit tasks to the process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_dataset, f): f for f in tof_datasets}
        
        # Process results as they complete
        for future in as_completed(futures):
            active_processes = sum(1 for f in futures if not f.done())
            print(f"Currently active processes: {active_processes}")
            
            dataset_name = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"Dataset {dataset_name} finished.")
                    tally_pos.append(result)
            except Exception as e:
                print(f"Error in future for dataset {dataset_name}: {e}")