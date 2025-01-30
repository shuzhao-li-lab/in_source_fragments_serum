import io
import contextlib
from mass2chem.search import find_all_matches_centurion_indexed_list
from khipu.extended import peaklist_to_khipu_list, export_empCpd_khipu_list
from mining import * 

def calculate_bin_deltas_fulldataset(good_khipus, stdev_shift, feature_list):
    '''calculate all delta mz values in every rtime window
    
    good_khipus: a list of khipus(snr>=5, shape>=0.9)
    stdev_shift: M1-M0 retention time shift
    feature_list: full list of features in this dataset
    
    return: list of tuples containing khipu id and related list of delta mz values. 
        Ex, [('kp4_69.0587', [0.0,
                5.033599999999993,
                22.167199999999994,
                11.455299999999994,
                1.0032999999999959]),
            ('kp5_69.0587', [0.0,
                10.871099999999998,
                1.0032999999999959])]
    '''
    mz_delta_list = []
    for kp in good_khipus:
        M0 = get_M0(kp['MS1_pseudo_Spectra']) # get the M0 khipu
        features_in_rt_window = [f for f in feature_list if abs(f['rtime'] - M0['rtime']) <= stdev_shift] # get the list of features sitting in the stdev from M0 feature
        
        base = min([x['mz'] for x in kp['MS1_pseudo_Spectra']]) # the smallest mz value in this khipu
        mz_delta_list.append((kp['interim_id'], [x['mz']-base for x in features_in_rt_window]))
    return mz_delta_list


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
                
    return explained_khipu_ids, list(set(explained_feature_ids)), delta_values_used


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

def find_match_ms2_from_feature(all_representative_features, feature, rt_half_width, ms2_tree, limit_ppm=5):
    ''' Given a feature, find the ms2 values matched with features in the same rtime window.
    
    all_representative_features: list of all features, from which we are looking for featues in rtwindow
    feature: the given feature we want to find matched ms2 mzs
    rt_half_width: the radius we used for selecting features in rtwindow
    ms2_tree: the centurion tree built from MoNA positive MS2 spectra
    
    return list of tuple containing mapping information. Ex,
    [('F49362',
        {'mz': 655.457,
        'rtime': 0,
        'name': 'phorbol-12,13-didecanoate',
        'peaks': [(100.0, 311.164459),
            (55.231621, 199.111877),
            (49.650773, 293.153961),
            (39.600037, 107.086075),
            (31.224962, 265.158875)]},
        [265.158875])]
    '''
    in_rtwindow = get_features_in_rtwindow(all_representative_features, feature['rtime'], rt_half_width)
    in_rtwindow = [f for f in in_rtwindow if f['mz'] < feature['mz']*(1+limit_ppm*1e-6)] # all mzs in the rt window with M0 feature
    
    matched = []
    # find ms2-precursor-matched khipu
    matched_precursors = find_all_matches_centurion_indexed_list(feature['mz'], ms2_tree, limit_ppm=5)
    for p in matched_precursors:
        matched_ms2_mzs = find_match_ms2_from_mzs_in_rtbin(in_rtwindow, [x[1] for x in p['peaks']], 5) # [x[1] for x in p['peaks']] is the mz list under one precursor
        if matched_ms2_mzs:
            matched.append((feature['id_number'], p, matched_ms2_mzs))
    
    return matched

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