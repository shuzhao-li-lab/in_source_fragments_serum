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


def get_features_in_rtwindow(list_features, rt_ref, rt_stdev):
    '''returns features in list_features that are within rt_stdev
    
    list_features: full list of features
    rt_ref: the reference retention time as the center of window
    rt_stdev: tolerance of retention time selecting
    
    return: list of features inside the given window
    '''
    return [f for f in list_features if abs(f['rtime']-rt_ref) <= rt_stdev]   

def get_khipus_in_rtwindow(list_khipus, rt_ref, rt_stdev):
    '''returns khipus in list_khipus that are within rt_stdev
    
    list_khipus: given list of khipus
    rt_ref: the reference retention time as the center of window
    rt_stdev: tolerance of retention time selecting
    
    return: list of khipus inside the given window
    '''
    return [f for f in list_khipus if abs(f['MS1_pseudo_Spectra'][0]['rtime']-rt_ref) <= rt_stdev]  

def get_comprehensive_stats_per_dataset(full_table_path, rt_tolerance, isotope_search_patterns, adduct_search_patterns, ion_mode):
    '''construct khipus from given features and related information
    
    full_table_path: the path to the full feature table of current dataset
    rt_tolerance: the retention time tolerance of current dataset. The tolerance is generated in step2 in elution_parameters_45studies.tsv
    ion_mode: the ionization mode of current dataset
    
    returns: 
    list_khipus: khipu list from given feature list
    all_assigned_fids: feature id list of the features being used in khipus
    list_features: all feature list
    '''
    with contextlib.redirect_stdout(io.StringIO()):
        _n, list_features = read_features_from_asari_table(open(full_table_path).read())
        
        for f in list_features:
            f['representative_intensity'] = f['peak_area']
        list_khipus, all_assigned_fids = peaklist_to_khipu_list(
                                list_features, 
                                isotope_search_patterns=isotope_search_patterns, 
                                adduct_search_patterns=adduct_search_patterns,
                                extended_adducts=[],    # not to confuse later analysis of ISF
                                mz_tolerance_ppm=5,
                                rt_tolerance=rt_tolerance,
                                mode=ion_mode,
                                charges=[1, 2, 3],
                                )
        # remaining_features = [f for f in list_features if f['id'] not in all_assigned_fids]
    return export_empCpd_khipu_list(list_khipus), all_assigned_fids, list_features

def explain_a_dataset_by_mz_deltas(list_khipus, remaining_features, isf_candidate_fragments, rt_stdev=0.613):
    '''map the pairwise mass distance of the khipus and 'free' features to isf_candidate_fragments
    
    list_khipus: list of khipus
    remaining_features: list of orphan features
    isf_candidate_fragments: list of the most frequent delta mass values 
    rt_stdev: half window of rt tolerance
    
    return:
    explained_khipu_ids: list of explained khipu ids
    explained_feature_ids: list of explained feature ids
    delta_values_used: list of delta mz values used for explanation
    '''
    def mz_delta_in_list(mz, mlist, max_diff=0.0005, ppm=5):
        '''check if the given mz value is in given mz list(meet the requirement of certain number or ppm)
        
        mz: a float number of m/z delta value
        mlist: a list of m/z value
        max_diff: fixed number tolerance
        ppm: ppm tolerance
        
        return: a boolean value indicating if the given mz sits in the window of any mz value in the given mlist.
        '''
        r = False
        if mz > max_diff:
            deltas = sorted([abs(x-mz) for x in mlist])
            if deltas[0] <= max_diff or deltas[0]/mz < ppm*1e-6:
                r = True
        return r 
    
    explained_khipu_ids, explained_feature_ids, delta_values_used = [], [], []

    # iterate through given khipu list
    for ii in range(len(list_khipus)-1):
        # get rtime of current khipu
        rt_ref = list_khipus[ii]['MS1_pseudo_Spectra'][0]['rtime']
        # get mz of M0 feature in current khipu
        base_mz = get_M0(list_khipus[ii]['MS1_pseudo_Spectra'])['mz']
        # get list of khipus whose mass value is bigger than current one
        khipus_in_rtwindow = get_khipus_in_rtwindow(
            list_khipus[ii+1:], 
            rt_ref, 
            rt_stdev)
        # iterate through the khipus in rtime window to get whose delta to given one matching isf_candidate_fragments
        for k in khipus_in_rtwindow:
            _d = list_khipus[ii]['neutral_formula_mass']-k['neutral_formula_mass']
            if mz_delta_in_list(_d, isf_candidate_fragments):
                explained_khipu_ids.append(k['interim_id'])
                delta_values_used.append((_d, k['interim_id'], rt_ref-get_M0(k['MS1_pseudo_Spectra'])['rtime']))
        
        # iterate through the features in rtime window to get whose delta to given one matching isf_candidate_fragments
        features_in_rtwindow = get_features_in_rtwindow(
            remaining_features, 
            rt_ref, 
            rt_stdev)
        for f in features_in_rtwindow:
            _d = base_mz - f['mz']
            if mz_delta_in_list(_d, isf_candidate_fragments):
                explained_feature_ids.append(f['id'])
                delta_values_used.append((_d, f['id'], rt_ref-f['rtime']))
                
    return explained_khipu_ids, explained_feature_ids, delta_values_used

def find_match_ms2_from_mzs_in_rtbin(mzs_in_rtbin, ms2_fragments, limit_ppm=5):
    '''returns ms2 fragments that are matched in mzs_in_rtbin
    
    mzs_in_rtbin: list of mz values in a certain rt bin. Ex, [81.0178, 83.0863, 75.0996, 71.05, 72.0715]
    ms2_fragments: list of tuple (intenisty, mz). Ex. [(100.0, 104.05261), (73.74, 56.04967), (63.76, 133.03153), (26.88, 61.01076), (25.4, 102.05479)]
    
    note: ms2_precursor and khipu M0 should already match before this.
    
    return: list of matched mz values
    '''
    found = []
    if mzs_in_rtbin and ms2_fragments:
        for x in mzs_in_rtbin:
            for mz in ms2_fragments:
                if abs(mz-x['mz']) < 0.000001*limit_ppm*mz:
                    found.append((mz, x['id']))
    return found

def explain_a_dataset_byMS2(all_features, ms2_tree, rt_stdev, limit_ppm=5):
    '''given a dataset, map the precursor and related ms2 peaks to ms2 tree
    
    all_features: list of features in metDataModel format. Ex. 
        {'id_number': 'F1',
        'id': 'F1',
        'mz': 70.0066,
        'rtime': 540.23,
        'apex': 540.23,
        'left_base': 539.97,
        'right_base': 541.3,
        'parent_masstrack_id': '0',
        'peak_area': '132716',
        'cSelectivity': '0.27',
        'goodness_fitting': '0.56',
        'snr': '75',
        'detection_counts': '4',
        'representative_intensity': '132716'}
    
    ms2_tree: the dict built by function: `build_centurion_tree` from ms2 spectra list
    example ms2_tree: {20908: [{'mz': 209.081,
        'rtime': 0,
        'name': 'Pyrenocin A - NCGC00169582-02_C11H12O4_1H-2-Benzopyran-1-one, 3,4-dihydro-6-hydroxy-8-methoxy-3-methyl-',
        'peaks': [(100.0, 163.075516),
        (96.239927, 191.070282),
        (26.604602, 103.053802),
        (19.662868, 135.080414),
        (18.420266, 148.05191)]}]}
        
    note: example result from test_match_ms2: [191.070282]
    
    return:
    _have_precursors: the list of feature id who has precursor matched
    matched: the tuple containing matched feature id, precursor name and the delta: (precursor's mz - ms2 peak's mz)
    '''   
    matched, _have_precursors = [], []
    for ff in all_features:
        in_rtwindow = get_features_in_rtwindow(all_features, ff['rtime'], rt_stdev)
        in_rtwindow = [x for x in in_rtwindow if x['mz'] < ff['mz']*(1+limit_ppm*1e-6)]
        # find matched precursor
        precursors = find_all_matches_centurion_indexed_list(ff['mz'], ms2_tree, limit_ppm=limit_ppm)
        if precursors:
            _have_precursors.append(ff['id_number'])
        for sp in precursors:
            matched_ms2_mzs = find_match_ms2_from_mzs_in_rtbin(in_rtwindow, [x[1] for x in sp['peaks']], limit_ppm)
            
            if matched_ms2_mzs:
                # _deltas = [sp['mz']-x for x in matched_ms2_mzs] # delta between ms2 peaks and precursors
                matched.append((ff['id_number'], ff['mz'], sp['name'], matched_ms2_mzs))
                
    return _have_precursors, matched



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

def extract_ms2_spectrum(sp, N=5):
    '''get precursor, name and top N (intensity, mz) from given ms2 spectrum

    sp: given spectrum to extract
    N: top N intensity:mz pair to extract
    
    return: (precursor mz, compound name, (intensity, mz) list). Ex.
            (248.0585,
            'Forchlorfenuron',
            [(100.0, 111.0553),
            (95.390586, 129.0214),
            (17.257158, 93.0448),
            (11.163815, 155.0007),
            (10.442742, 137.0346)])
    '''
    _d = sp.metadata_dict()
    _precursor, _name = _d['precursor_mz'], _d['compound_name']
    imz = zip(sp.peaks.intensities, sp.peaks.mz)
    imz = [x for x in imz if x[1] < _precursor - 0.01 and x[0] > 0.1] # excluding _precursor and small peaks
    return _precursor, _name, sorted(imz, reverse=True)[:N]

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
22.9845	54	22.989218	 Na	  { 'Na':1 }
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
                            (82.0030, 'C2HF3'),
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
        
        # Save results to JSON file
        with open(f'output/tof/12272024/{ion_mode}/{f}.json', 'w', encoding='utf-8') as json_file:
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