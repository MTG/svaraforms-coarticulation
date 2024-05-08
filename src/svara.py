import os
import tqdm

import numpy as np
from scipy.signal import savgol_filter

from src.dtw import dtw
from src.utils import cpath, write_pkl, myround
from src.tools import get_derivative

SVARAS = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']

def chop_time_series(ts, s, e, timestep):
    s = round(s/timestep)
    e = round(e/timestep)
    return ts[s:e]


def get_unique_svaras(annotations):
    svaras = list(set([s.strip().lower() for s in annotations['label']]))
    return [s for s in SVARAS if s in  svaras]


def get_gamaka(p, timestep):
    # gamaka
    t = [x*timestep for x in range(len(p))]
    dp, dt = get_derivative(p, t)
    asign = np.sign(dp)
    sumchange = sum([1 for i in range(len(asign))[1:] if np.sign(asign[i]) != np.sign(asign[i-1])])
    if sumchange == 0:
        gamaka = 'jaaru'
    elif sumchange == 1:
        gamaka = 'none'
    else:
        gamaka = 'kampita'

    return gamaka

def get_svara_dict(annotations, pitch_cents, timestep, track, tonic, min_length=0.145, smooth_window=0.145, prec_suc=0.3, path=None, plot_dir=None, verbose=True):
    
    if min_length < smooth_window:
        raise Exception(f'<min_length> cannot be smaller than <smooth_window>')
    if prec_suc < min_length:
        raise Exception('prec_suc must be larger than minimum lenght of svara')

    svara_dict = {}
    for i,row in annotations.iterrows():
        start = row['start']
        end = row['end']
        label = row['label'].strip().lower()
        duration = end-start
        if i != 0:
            prev = annotations.iloc[i-1]
            if start - prev['end'] < 0.5:
                prev_svara = prev['label'].strip().lower()
                prev_start = prev['start']
                prev_end = prev['end']
            else:
                prev_svara = 'silence'
                prev_start = prev['start']
                prev_end = prev['end']
        else:
            prev_svara = None
            prev_start = None
            prev_end = None

        if i != len(annotations)-1:
            nex = annotations.iloc[i+1]
            if nex['start'] - end < 0.5:
                next_svara = nex['label'].strip().lower()
                next_start = nex['start']
                next_end = nex['end']
            else:
                next_svara = 'silence'
                next_start = nex['start']
                next_end = nex['end']
        else:
            next_svara = None
            next_start = None
            next_end = None

        if label not in SVARAS:
            continue

        pitch_curve = chop_time_series(pitch_cents, start, end, timestep)
        prec_pitch_curve = chop_time_series(pitch_cents, start-prec_suc, start, timestep)
        succ_pitch_curve = chop_time_series(pitch_cents, end, end+prec_suc, timestep)

        #pitch_curve = pitch_curve[pitch_curve!=None]

        L = len(pitch_curve)*timestep
        if L < min_length:
            print(f'Index {i} discarded, below minimum length, {L}s')
            continue
        
        if None in pitch_curve:
            continue
        
        if max(pitch_curve) - min(pitch_curve) > 1000:
            print(f'Index {i} discarded, time series octave error')
            continue

        wl = round(smooth_window/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        pitch_curve = savgol_filter(pitch_curve, polyorder=2, window_length=wl, mode='interp')
        pitch_curve = savgol_filter(pitch_curve, polyorder=2, window_length=wl, mode='interp')

        prec_pitch_curve = savgol_filter(prec_pitch_curve, polyorder=2, window_length=wl, mode='interp')
        prec_pitch_curve = savgol_filter(prec_pitch_curve, polyorder=2, window_length=wl, mode='interp')
        
        succ_pitch_curve = savgol_filter(succ_pitch_curve, polyorder=2, window_length=wl, mode='interp')
        succ_pitch_curve = savgol_filter(succ_pitch_curve, polyorder=2, window_length=wl, mode='interp')

        gamaka = get_gamaka(pitch_curve, timestep)

        d = {
                'pitch': pitch_curve,
                'track': track,
                'start': start,
                'end': end,
                'duration': duration,
                'annotation_index': i,
                'preceeding_svara': prev_svara,
                'preceeding_start': prev_start,
                'preceeding_end': prev_end,
                'succeeding_svara': next_svara,
                'succeeding_start': next_start,
                'succeeding_end': next_end,
                'prec_pitch':prec_pitch_curve,
                'succ_pitch':succ_pitch_curve,
                'gamaka': gamaka,
                'timestep':timestep,
                'tonic':tonic
            }

        if label in svara_dict:
            svara_dict[label].append(d)
        else:
            svara_dict[label] = [d]


    if verbose:
        for k,v in svara_dict.items():
            print(f'{len(v)} occurrences of {k}')

    if path:
        svara_dict_path = cpath(path)
        write_pkl(svara_dict, svara_dict_path)

    if plot_dir:
        for svara in svara_dict.keys():
            all_svaras = svara_dict[svara]
            for i in range(len(all_svaras)):
                pt = all_svaras[i]['pitch']
                times = [x*timestep for x in range(len(pt))]
                path = cpath(plot_dir, svara, f'{i}.png')
                plt.plot(times, pt)
                plt.savefig(path)
                plt.close('all')

    return svara_dict


def create_bad_svaras(annotations, pitch_cents, timestep, track, tonic, min_length, thresh=0.1, prec_suc=0.3, n=500, smooth_window=0.1):

    if min_length < smooth_window:
        raise Exception(f'<min_length> cannot be smaller than <smooth_window>')


    annotations_svara = annotations[annotations['label'].isin(SVARAS)]
    annotations_svara['l'] = annotations_svara['end'] - annotations_svara['start']
    min_start = min(annotations_svara['start'])
    max_end = max(annotations_svara['end'])
    shortened_curve = pitch_cents[int(round(min_start)):int(round(max_end))]
    ps_timestep = int(round(prec_suc/timestep))

    bad_svaras = []
    total = 0
    while total < n:
        start = np.random.randint(0, len(shortened_curve))
        l = np.random.choice(annotations_svara['l'])

        end = start + int(round(l/timestep))
        pitch_curve = shortened_curve[start:end]

        prec_pitch_curve = pitch_cents[start+int(round(min_start))-ps_timestep:start+int(round(min_start))]
        succ_pitch_curve = pitch_cents[end+int(round(min_start)):end+int(round(min_start))+ps_timestep]

        L = len(pitch_curve)*timestep

        start_match = [myround(start,thresh)==myround(x,thresh) for x in annotations['start'].values]
        end_match = [myround(start,thresh)==myround(x,thresh) for x in annotations['end'].values]
        start_end = list(zip(start_match, end_match))
        coinc = any([i and j for i,j in start_end])

        if coinc:
            print('coincidence')

        if (None in pitch_curve) or (L < min_length) or coinc:
            continue

        wl = round(smooth_window/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        pitch_curve = savgol_filter(pitch_curve, polyorder=2, window_length=wl, mode='interp')
        pitch_curve = savgol_filter(pitch_curve, polyorder=2, window_length=wl, mode='interp')

        prec_pitch_curve = savgol_filter(prec_pitch_curve, polyorder=2, window_length=wl, mode='interp')
        prec_pitch_curve = savgol_filter(prec_pitch_curve, polyorder=2, window_length=wl, mode='interp')

        succ_pitch_curve = savgol_filter(succ_pitch_curve, polyorder=2, window_length=wl, mode='interp')
        succ_pitch_curve = savgol_filter(succ_pitch_curve, polyorder=2, window_length=wl, mode='interp')

        gamaka = get_gamaka(pitch_curve, timestep)

        d = {
                'pitch': pitch_curve,
                'track': track,
                'start': start*timestep,
                'end': end+timestep,
                'duration': L,
                'annotation_index': None,
                'preceeding_svara': None,
                'preceeding_start': None,
                'preceeding_end': None,
                'succeeding_svara': None,
                'succeeding_start': None,
                'succeeding_end': None,
                'prec_pitch':prec_pitch_curve,
                'succ_pitch':succ_pitch_curve,
                'gamaka': gamaka,
                'timestep':timestep,
                'tonic':tonic
            }

        bad_svaras.append(d)
        total += 1

    return bad_svaras


def pairwise_distances_to_file(ix, all_svaras, path, r=0.05, mean_norm=False):
    try:
        print('Removing previous distances file')
        os.remove(path)
    except OSError:
        pass

    header = 'index1,index2,distance'
    with open(path,'a') as file:
        file.write(header)
        file.write('\n')
        for i in tqdm.tqdm(ix):
            for j in ix:
                if i <= j:
                    continue
                pat1 = all_svaras[i]['pitch']
                pat2 = all_svaras[j]['pitch']

                pi = len(pat1)
                pj = len(pat2)
                l_longest = max([pi, pj])

                path, dtw_val = dtw(pat1, pat2, radius=round(l_longest*r), mean_norm=mean_norm)
                l = len(path)
                dtw_norm = dtw_val/l

                row = f'{i},{j},{dtw_norm}'
                
                file.write(row)
                file.write('\n')


def get_centered_svaras(svara):
    MASSVARAS = SVARAS + SVARAS + SVARAS
    occs = [i for i,x in enumerate(MASSVARAS) if x==svara]
    secocc = occs[1]
    svaras = MASSVARAS[secocc-3: secocc+4]
    return svaras



def asc_desc(n0, n, n2):
    cent_svara = get_centered_svaras(n)
    ni = cent_svara.index(n)
        
    # ADD REPEATED
    if n0 not in cent_svara: # i.e. silence or unknown
        n2i = cent_svara.index(n2)
        if ni < n2i:
            return 'asc'
        elif ni > n2i:
            return 'desc'
        else:
            return 'cp'  

    elif n2 not in cent_svara:
        n0i = cent_svara.index(n0)
        if n0i < ni:
            return 'asc'
        elif n0i > ni:
            return 'desc'
        else:
            return 'cp'

    elif n0 not in cent_svara and n2 not in cent_svara:
        return np.nan
    
    n0i = cent_svara.index(n0)
    n2i = cent_svara.index(n2)

    if n0i < ni < n2i:
        return 'asc'
    elif n0i > ni > n2i:
        return 'desc'
    else:
        return 'cp'