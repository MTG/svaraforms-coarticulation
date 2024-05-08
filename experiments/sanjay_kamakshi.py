%load_ext autoreload
%autoreload 2

import math
from itertools import groupby
from collections import Counter
import compiam
import os 
import librosa
import mirdata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tqdm

from matplotlib.patches import Rectangle
from numpy.linalg import norm
from numpy import dot
from scipy.interpolate import interp1d 

from src.dtw import dtw
from src.tools import compute_novelty_spectrum, get_loudness, interpolate_below_length, get_derivative, chaincode
from src.utils import load_audacity_annotations, load_elan_annotations, cpath,  write_pkl, write_pitch_track, load_pitch_track, read_txt, load_pkl, myround
from src.visualisation import get_plot_kwargs, plot_subsequence, get_arohana_avarohana
from src.pitch import pitch_seq_to_cents
from src.svara import get_svara_dict, get_unique_svaras, pairwise_distances_to_file, chop_time_series
from scipy.signal import savgol_filter
from src.clustering import duration_clustering, cadence_clustering, hier_clustering

import numpy as np

def is_stable(seq, max_var):
    mu = np.nanmean(seq)
    maximum = np.nanmax(seq)
    minimum = np.nanmin(seq)
    if (maximum < mu + max_var) and (minimum > mu - max_var):
        return 1
    else:
        return 0


def reduce_stability_mask(stable_mask, min_stability_length_secs, timestep):
    min_stability_length = int(min_stability_length_secs/timestep)
    num_one = 0
    indices = []
    for i,s in enumerate(stable_mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            if num_one < min_stability_length:
                for ix in indices:
                    stable_mask[ix] = 0
            num_one = 0
            indices = []
    return stable_mask


def get_stability_mask(raw_pitch, min_stability_length_secs, stability_hop_secs, var_thresh, timestep):
    stab_hop = int(stability_hop_secs/timestep)
    reverse_raw_pitch = np.flip(raw_pitch)

    # apply in both directions to array to account for hop_size errors
    stable_mask_1 = [is_stable(raw_pitch[s:s+stab_hop], var_thresh) for s in range(len(raw_pitch))]
    stable_mask_2 = [is_stable(reverse_raw_pitch[s:s+stab_hop], var_thresh) for s in range(len(reverse_raw_pitch))]
    
    silence_mask = raw_pitch == 0

    zipped = zip(stable_mask_1, np.flip(stable_mask_2), silence_mask)
    
    stable_mask = np.array([int((any([s1,s2]) and not sil)) for s1,s2,sil in zipped])

    stable_mask = reduce_stability_mask(stable_mask, min_stability_length_secs, timestep)

    return stable_mask


out_dir = cpath('data', 'short_test')

track = 'kamakshi'
raga = 'bhairavi'

###########
# Get paths
###########
annotations_path = os.path.join('data', 'annotation', f'{track}.txt')

vocal_path = 'data/audio/Kamakshi.mp3'

###########
# Load Data
###########
sr = 44100
vocal, _ = librosa.load(vocal_path, sr=sr)

##########
# Cut Data
##########
#import soundfile as sf
#startends = [
#    (197, 'ni', (3*60+53.399), (3*60+53.677)),
#    (198, 'ga', (3*60+53.677), (3*60+54.291)),
#    (199, 'ri', (3*60+54.291), (3*60+54.699)),
#    (200, 'sa', (3*60+54.699), (3*60+55.159))
#]
#for i,svara,s,e in startends:
#    this_vocal = vocal[round(sr*s):round(sr*e)]
#    sf_path = cpath('data', 'kamakshi_cuts', f'{i}_{svara}.wav')
#    sf.write(sf_path, this_vocal, sr)


tonic = 146.84
plot_kwargs = get_plot_kwargs(raga, tonic)
yticks_dict = plot_kwargs['yticks_dict']
aro, avaro = get_arohana_avarohana(raga)

# Load annotations
#annotations = load_audacity_annotations(annotations_path)

annotations = load_elan_annotations(annotations_path)

max_t = len(vocal)/sr
annotations = annotations[annotations['end']<max_t]
##################
# Extract Features
##################
# Pitch track -> change points
ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
pitch_track_path = cpath('data', 'pitch_tracks', f'{track}.tsv')
#ftanet_pitch_track = ftanet_carnatic.predict(vocal_path,hop_size=30)
#write_pitch_track(ftanet_pitch_track, pitch_track_path, sep='\t')
ftanet_pitch_track = load_pitch_track(pitch_track_path)


pitch = ftanet_pitch_track[:,1]
time = ftanet_pitch_track[:,0]
timestep = time[3]-time[2]

pitch = interpolate_below_length(pitch, 0, (250*0.001/timestep))
null_ind = pitch==0

pitch[pitch<50]=0
pitch[null_ind]=0

pitch_cents = pitch_seq_to_cents(pitch, tonic)


#####################
# Get Svaras Features
#####################
svara_dict_path = cpath(out_dir, 'data', 'svara_dict', f'{track}.csv')
unique_svaras = get_unique_svaras(annotations)

svara_dict = get_svara_dict(annotations, pitch_cents, timestep, track, min_length=0.1, smooth_window=0.1, path=None)



##############################
# Create Time Series Dataset #
##############################
train_size = 0.7
max_len = 600
np_data_out_dir = '/Users/thomasnuttall/code/dtc-tensorflow/data/kamakshi/'
label_lookup = {i:s for i,s in enumerate(unique_svaras)}

def reject_outliers(data, m = 3):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]


def resample(arr, n):
    Told = np.linspace(0,len(arr),len(arr))
    F = interp1d(Told, arr, fill_value='extrapolate')
    Tnew = np.linspace(Told.min(), Told.max(), n)
    return F(Tnew)

np_data = []
labels = []
for svara, sv_sd in svara_dict.items():
    for i in range(len(sv_sd)):
        s1 = sv_sd[i]['start']
        s2 = sv_sd[i]['end']
        p = chop_time_series(pitch_cents, s1, s2, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = np.trim_zeros(p)

        if sum(np.isnan(p))>0:
            continue
        
        if len(p)>max_len:
            continue

        #p2 = resample(p, resample_size)

        l = unique_svaras.index(svara)
        
        np_data.append(p)
        labels.append(l)

labels = np.array(labels)

# ensure same length
lens = np.array([x.shape[0] for x in np_data])
max_len = min([max_len, max(lens)])

np_data_pad = []

for d in np_data:
    n_p = max_len-len(d)
    p1 = int(n_p/2)
    p = np.pad(d, pad_width=p1)
    if n_p%2:
        p = np.concatenate([p,np.array([0])])
    np_data_pad.append(np.array(p))


np_data_pad = np.array(np_data_pad)

np_data_pad = np.c_[labels, np_data_pad]

# Shuffle
random.seed(42)
random.shuffle(np_data_pad)


train_data = np.array(np_data_pad[:round(train_size*len(np_data_pad))])
test_data = np.array(np_data_pad[round(train_size*len(np_data_pad)):])

# Write
train_path = cpath(np_data_out_dir, 'TRAIN.pkl')
test_path = cpath(np_data_out_dir, 'TEST.pkl')
label_path = cpath(np_data_out_dir, 'LABELS.pkl')
write_pkl(train_data, train_path)
write_pkl(test_data, test_path)
write_pkl(label_lookup, label_path)


#####################
## N Gram Chain Codes
#####################
all_chain_codes = []
all_chain_code_p = []
all_chain_code_s = []
for svara in unique_svaras:
    print(svara)
    sv_sd = svara_dict[svara]

    for i in range(len(sv_sd)):
        
        s1 = sv_sd[i]['start']
        s2 = sv_sd[i]['end']
        p = chop_time_series(pitch_cents, s1, s2, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')

        if sum(np.isnan(p)) > 0:
            continue

        cci = chaincode(p, min_length=5, reduce_length=1)

        all_chain_codes.append(cci)
        all_chain_code_p.append(p)
        all_chain_code_s.append(svara)

from nltk import ngrams

def extract_ngram(seq, n=3):
    return [''.join([str(y) for y in x]) for x in list(ngrams(seq, n))]

def extract_all_ngrams(seqs, n=3, thresh=None):
    ng = [extract_ngram(s, n=n) for s in seqs]
    ng = [y for x in ng for y in x]

    n_poss = len(set([y for x in seqs for y in x]))
    n_combos = math.factorial(n_poss)
    n_exp = len(ng)/n_combos

    counts = Counter(ng)
    probs = [(n,c/n_exp) for n,c in counts.items()]
    sorts = sorted(probs, key=lambda y: -y[1])
    if thresh:
        sorts = [(n,c) for n,c in sorts if c > thresh]
    return sorts



all_sig_ng = []
for n in [3]:#[3,4,5,6,7,8]:
    ang = extract_all_ngrams(all_chain_codes, n=n, thresh=1.5)
    all_sig_ng += [x[0] for x in ang]
print(all_sig_ng)

##############################
# Create Time Series Dataset #
##############################
train_size = 0.7
resample_size = 250
np_data_out_dir = '/Users/thomasnuttall/code/DTCR/DTCR_code/Data/Bhairavi_FEATURES/'
label_lookup = {i:s for i,s in enumerate(unique_svaras)}

def reject_outliers(data, m = 3):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]


def resample(arr, n):
    Told = np.linspace(0,len(arr),len(arr))
    F = interp1d(Told, arr, fill_value='extrapolate')
    Tnew = np.linspace(Told.min(), Told.max(), n)
    return F(Tnew)

# Features
    # - chaincode tf-idf, n-gram class
    # - chaincode how many of each coordinate
    # - pitch range (remove outliers)
    # - highest pitch (octave normalised) (removing outliers)
    # - lowest pitch (octave normalised) (removing outliers)
    # - average pitch of first 5% of sequence
    # - average pitch of last 5% of sequence
    # - number of change points
    # - duration
    # - Timbral features (MFCC across 5 partitions)
    # - Maximum Loudness
    # - Loudness range
    # - Loudness change points
    # - Stability 
    # -   - number of stable points
    # -   - total duration of stable points
    # - Whether it is part of ascending/descending/change point or repetition
    # - mean loudness (energy)
    # - number of intervals traversed
    # - how many of each svara
    # -   - how many sa
    # -   - how many pa
    # -   - how many ....
    # - duration of each svara
    #   - duration of ...
    #   - duration of ...
    # - length of chaincode


cols = ['label'] + [f'fcc{f}' for f in all_sig_ng] + [f'fcc_count{i}' for i in [0,1,2,3,4]] + ['pitch_range', 'min_pitch', 'highest_pitch']+\
       ['av_first_pitch', 'av_end_pitch', 'num_change_points', 'duration', 'max_loudness', 'loudness_range', 'n_loudness_change_points']+\
       ['n_stable_regions', 'total_duration_stable', 'mean_loudness'] + ['chaincode_length']# + [f'n_{s}' for s in unique_svaras]+\
       #[f'duration_{s}' for s in unique_svaras] 

data = pd.DataFrame(columns=cols)
np_data = []
for svara, sv_sd in svara_dict.items():
    for i in range(len(sv_sd)):
        s1 = sv_sd[i]['start']
        s2 = sv_sd[i]['end']
        p = chop_time_series(pitch_cents, s1, s2, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = np.trim_zeros(p)
        l = unique_svaras.index(svara)
        t = [i*timestep for i in range(len(p))]

        stab_mask =  get_stability_mask(p, 0.05, 0.02, 3, timestep)
        
        n_stable_regions = sum([key for key, _group in groupby(stab_mask)])
            

        dp, dt = get_derivative(p,t)
        asign = np.sign(dp)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        cp_p = sum(signchange)

        y = vocal[round(s1*sr):round(s2*sr)]

        loudness = get_loudness(y)
        asign = np.sign(loudness)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        cp_l = sum(signchange)

        if sum(np.isnan(p)) > 0:
            continue

        cci = chaincode(p, min_length=5, reduce_length=1)
        ccistr = "".join([str(c) for c in cci])
        cci_count = Counter(cci)

        vals  = [l]
        vals += [int(a in ccistr) for a in all_sig_ng]
        vals += [cci_count[i] for i in [0,1,2,3,4]]
        vals += [max(p)-min(p), min(p), max(p)]
        vals += [np.mean(p[:int(len(p)*0.05)]), np.mean(p[-int(len(p)*0.05):])]
        vals += [cp_p]
        vals += [len(p)*timestep]
        vals += [max(loudness), max(loudness)-min(loudness), cp_l]
        vals += [n_stable_regions, sum(stab_mask)*timestep, np.mean(loudness), len(cci)]

        row = dict(zip(cols, vals))

        data = data.append(row, ignore_index=True)

np_data = data.values

# Shuffle
random.seed(42) 
random.shuffle(np_data)

train_data = np.array(np_data[:round(train_size*len(np_data))])
test_data = np.array(np_data[round(train_size*len(np_data)):])

# Write
train_path = cpath(np_data_out_dir, 'BHAIRAVI_TRAIN.pkl')
test_path = cpath(np_data_out_dir, 'BHAIRAVI_TEST.pkl')
label_path = cpath(np_data_out_dir, 'BHAIRAVI_LABELS.pkl')
write_pkl(train_data, train_path)
write_pkl(test_data, test_path)
write_pkl(label_lookup, label_path)




#####
# DTW
#####
distance_paths = {}
for svara in unique_svaras:
    print(f'Computing distances for {svara}')
    all_svaras = svara_dict[svara]
    all_ix = list(range(len(all_svaras)))
    dtw_distances_path = cpath(out_dir, 'data', 'dtw_distances', f'{track}', f'{svara}.csv')
    pairwise_distances_to_file(all_ix, all_svaras, dtw_distances_path, mean_norm=True)
    distance_paths[svara] = dtw_distances_path


############
# Plot Svara
############
if plot:
    for svara, sds in svara_dict.items():
        print(f'Plotting all {svara}...')
        for i,sd in tqdm.tqdm(list(enumerate(sds))):
            s1o = sd['preceeding_start']
            s2o = sd['succeeding_end']
            s1 = sd['start']
            s2 = sd['end']

            if s1o is None or s2o is None:
                continue
            
            p = chop_time_series(pitch, s1o, s2o, timestep)
            p[p==0] = None
            none_mask = p==None

            wl = round(0.1/timestep)
            wl = wl if not wl%2 == 0 else wl+1
            p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
            p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
            
            pitch_masked = np.ma.masked_where(none_mask, p)

            gamaka = sd['gamaka']
            preceeding_svara = sd['preceeding_svara']
            succeeding_svara = sd['succeeding_svara']
            t = [s1o+x*timestep for x in range(len(p))]
            
            
            fig = plt.figure(figsize=(10,5))
            ax = plt.gca()

            plt.plot(t,pitch_masked)
            plt.grid()

            tick_names = list(yticks_dict.keys())
            tick_loc = [p for p in yticks_dict.values()]
            ax.set_yticks(tick_loc)
            ax.set_yticklabels(tick_names)

            xmin = myround(min(t), 1)
            xmax = myround(max(t), 1)

            try:
                ymin = min([x for x in p if not np.isnan(x)])-20
                ymax = max([x for x in p if not np.isnan(x)])+20
            except:
                continue

            ax.set_ylim((ymin, ymax))
            ax.set_xlim((xmin, xmax))
            ax.set_facecolor('#f2f2f2')
    
            max_y = ax.get_ylim()[1]
            min_y = ax.get_ylim()[0]
            rect = Rectangle((s1, min_y), s2-s1, max_y-min_y, facecolor='lightgrey')
            ax.add_patch(rect)
            
            ax.axvline(x=s1, linestyle="dashed", color='black', linewidth=0.8)
            ax.axvline(x=s2, linestyle="dashed", color='black', linewidth=0.8)

            plt.title(f'[{preceeding_svara}] {svara} ({gamaka}) [{succeeding_svara}]')
            plt.xlabel('Time (s)')
            plt.ylabel('Pitch (cents)')
            path = cpath(out_dir, 'plots', f'{track}', 'raw_svaras', f'{svara}', f'{i}_{preceeding_svara}-{succeeding_svara}__{gamaka}.png')
            plt.savefig(path)
            plt.close('all')

            path = cpath(out_dir, 'plots', f'{track}', 'raw_svaras', f'{svara}', f'{i}_{preceeding_svara}-{succeeding_svara}__{gamaka}_full.wav')
            this_vocal = vocal[round(sr*s1o):round(sr*s2o)]
            sf.write(path, this_vocal, sr)

            path = cpath(out_dir, 'plots', f'{track}', 'raw_svaras', f'{svara}', f'{i}_{preceeding_svara}-{succeeding_svara}__{gamaka}_svara.wav')
            this_vocal = vocal[round(sr*s1):round(sr*s2)]
            sf.write(path, this_vocal, sr)
            # Replace






######################
## Freeman Chain Codes
######################
import Levenshtein
for svara in ['ri']:
    print(svara)
    sv_sd = svara_dict[svara]

    distances = []
    for i in range(len(sv_sd)):
        
        s1 = sv_sd[i]['start']
        s2 = sv_sd[i]['end']
        p = chop_time_series(pitch_cents, s1, s2, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')

        if sum(np.isnan(p)) > 0:
            distances.append((i, j, ['nan'], ['nan'], 1))
            continue

        cci = chaincode(p, min_length=10, reduce_length=1)

        for j in range(len(sv_sd)):
            if i <= j:
                continue

            s1 = sv_sd[j]['start']
            s2 = sv_sd[j]['end']
            p = chop_time_series(pitch_cents, s1, s2, timestep)
            p = np.trim_zeros(p)
            p = interpolate_below_length(p, 0, (250*0.001/timestep))
            wl = round(0.1/timestep)
            wl = wl if not wl%2 == 0 else wl+1
            p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
            p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')

            if sum(np.isnan(p)) > 0:
                distances.append((i, j, ['nan'], ['nan'], 1))
                continue

            ccj = chaincode(p, min_length=2, reduce_length=1)

            edit_distance = Levenshtein.distance(cci, ccj)
            ed = edit_distance/(max([len(cci), len(ccj)]))
            distances.append((i, j, cci, ccj, ed))

    distances = sorted(distances, key=lambda y: y[4])

    for I in range(len(distances)):
        # plot_pair
        i,j,cc1,cc2,d = distances[I]
        d = round(d, 2)
        s1i = sv_sd[i]['start']
        s2i = sv_sd[i]['end']
        s1j = sv_sd[j]['start']
        s2j = sv_sd[j]['end']

        p = chop_time_series(pitch_cents, s1i, s2i, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        pi = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')

        p = chop_time_series(pitch_cents, s1j, s2j, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        pj = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')


        ti = [i*timestep for i in range(len(pi))]
        tj = [i*timestep for i in range(len(pj))]

        fig, axs = plt.subplots(2)
        fig.tight_layout()
        fig.suptitle(f'edit distance={d} rank={I+1}')
        axs[0].plot(ti, pi)
        axs[0].set_title(cc1, fontsize=6)
        axs[1].plot(tj, pj)
        axs[1].set_title(cc2, fontsize=6)
        lv_path = cpath('plots','Levenshtein_reduced_2',track,svara,f'{I}_{i}_{j}.png')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch (cents)')
        plt.savefig(lv_path)
        plt.close('all')



all_chain_codes = []
all_chain_code_p = []
all_chain_code_s = []
for svara in unique_svaras:
    print(svara)
    sv_sd = svara_dict[svara]

    for i in range(len(sv_sd)):
        
        s1 = sv_sd[i]['start']
        s2 = sv_sd[i]['end']
        p = chop_time_series(pitch_cents, s1, s2, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')

        if sum(np.isnan(p)) > 0:
            continue

        cci = chaincode(p, min_length=5, reduce_length=1)

        all_chain_codes.append(cci)
        all_chain_code_p.append(p)
        all_chain_code_s.append(svara)

from nltk import ngrams

def extract_ngram(seq, n=3):
    return [''.join([str(y) for y in x]) for x in list(ngrams(seq, n))]

def extract_all_ngrams(seqs, n=3, thresh=None):
    ng = [extract_ngram(s, n=n) for s in seqs]
    ng = [y for x in ng for y in x]

    n_poss = len(set([y for x in seqs for y in x]))
    n_combos = math.factorial(n_poss)
    n_exp = len(ng)/n_combos

    counts = Counter(ng)
    probs = [(n,c/n_exp) for n,c in counts.items()]
    sorts = sorted(probs, key=lambda y: -y[1])
    if thresh:
        sorts = [(n,c) for n,c in sorts if c > thresh]
    return sorts



all_sig_ng = []
for n in [3]:#[3,4,5,6,7,8]:
    ang = extract_all_ngrams(all_chain_codes, n=n, thresh=1.5)
    all_sig_ng += [x[0] for x in ang]
print(all_sig_ng)



for i in range(len(all_chain_codes)):
    p = all_chain_code_p[i]
    s = all_chain_code_s[i]
    cc = all_chain_codes[i]
    cc_str = ''.join([str(y) for y in cc])

    matches = [i for i,x in enumerate(all_sig_ng) if x in cc_str]
    if not any(matches):
        continue

    title = '_'.join([all_sig_ng[m] for m in matches])

    t = [j*timestep for j in range(len(p))]
         
    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()

    plt.plot(t,p)
    plt.title(f'{cc} [{s}]')

    path = cpath('plots','chain_code_clustering',f'{i}__{title}.png')
    plt.savefig(path)
    plt.close('all')








#####################
## Plot all svaras ##
#####################
for svara in unique_svaras:
    print(svara)
    sv_sd = svara_dict[svara]

    for i in range(len(sv_sd)):
        
        s1 = sv_sd[i]['start']
        s2 = sv_sd[i]['end']
        p = chop_time_series(pitch_cents, s1, s2, timestep)
        p = np.trim_zeros(p)
        p = interpolate_below_length(p, 0, (250*0.001/timestep))
        wl = round(0.1/timestep)
        wl = wl if not wl%2 == 0 else wl+1
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')
        p = savgol_filter(p, polyorder=2, window_length=wl, mode='interp')

        if sum(np.isnan(p)) > 0:
            continue

        t = [s1+i*timestep for i in range(len(p))]

        plt.plot(t,p)
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch (cents)')
        path = cpath(out_dir, 'plots', f'{track}', f'{svara}', f'{round(s1,2)}_{round(s2,2)}.png')
        plt.savefig(path)
        plt.close('all')
        
        path = cpath(out_dir, 'plots', f'{track}', f'{svara}', f'{round(s1,2)}_{round(s2,2)}.wav')
        this_vocal = vocal[round(sr*s1):round(sr*s2)]
        sf.write(path, this_vocal, sr)

#############
## Clustering
#############
wl = round(145*0.001/timestep)
wl = wl if not wl%2 == 0 else wl+1

min_samples = 1 # duration min samp
eps = 0.05 # duration epsilon

t = 1 # hierarchical clustering t
min_in_group = 1 # min in group for hierarchical

plot = False # plot final clusters?

cluster_dict = {}
for svara, sd in svara_dict.items():
    print(f'Duration clustering, {svara}')
    cluster_dict[svara] = duration_clustering(sd, eps=0.05)
    print(f'    {len(cluster_dict[svara])} clusters')

# Force Sa and Pa to be uniform
for svara,c_val in [('sa',0), ('pa',700)]:
    print(f'Removing gamaka from {svara}')
    sd = cluster_dict[svara]
    new_sd = []
    for s1 in sd:
        new_s1 = []
        for n,s2 in s1:
            this_dict = {
                'pitch': np.array([c_val for s in s2['pitch']]),
                'track': s2['track'],
                'start': s2['start'],
                'end': s2['end'],
                'duration': s2['duration'],
                'annotation_index': s2['annotation_index'],
                'preceeding_svara': s2['preceeding_svara'],
                'succeeding_svara': s2['succeeding_svara'],
                'gamaka': 'none'
            }
            new_s1.append((n, this_dict))
        new_sd.append(new_s1)
    cluster_dict[svara] = new_sd


for svara, clusters in cluster_dict.items():
    if svara in ['sa','pa']:
        continue
    print(f'Cadence clustering, {svara}')
    new_clusters = []
    for cluster in clusters:
        cadclust = cadence_clustering(cluster, svara)
        new_clusters += cadclust
    cluster_dict[svara] = new_clusters
    print(f'    {len(cluster_dict[svara])} clusters')


for svara, clusters in cluster_dict.items():
    if svara in ['sa','pa']:
        continue
    print(f'Hierarchical clustering, {svara}')
    distance_path = distance_paths[svara]
    distances = pd.read_csv(distance_path)
    distances_flip = distances.copy()
    distances_flip.columns = ['index2', 'index1', 'distance']
    distances = pd.concat([distances, distances_flip])
    new_clusters = []
    for cluster in clusters:
        hier = hier_clustering(cluster, distances, t=t, min_in_group=min_in_group)
        if hier: # hier can return empty array if min_in_group > 1
            new_clusters += hier
    cluster_dict[svara] = new_clusters
    print(f'    {len(cluster_dict[svara])} clusters')


if plot:
    for svara, clusters in cluster_dict.items():
        print(f'Plotting {svara} cluster...')
        for i,cluster in tqdm.tqdm(list(enumerate(clusters))):
            for j,sd in cluster:
                p = sd['pitch']
                gamaka = sd['gamaka']
                t = [x*timestep for x in range(len(p))]
                
                plt.plot(t,p)
                plt.xlabel('Time (s)')
                plt.ylabel('Pitch (cents)')
                path = cpath(out_dir, 'plots', f'{track}', 'clustering', f'{svara}', f'cluster_{i}', f'{gamaka}_{j}.png')
                plt.savefig(path)
                plt.close('all')


#########################
## Pick cluster candidate
#########################
final_clusters = {}
for svara, clusters in cluster_dict.items():
    print(f'Reducing {svara} clusters to 1 candidate')
    clusts = []
    for c in clusters:
        clusts.append(random.choice(c)[1])
    final_clusters[svara] = clusts


########################
## Get Distance profiles
########################
sample = 30

if sample:
    sample_pitch_cents = pitch_cents[:round(sample/timestep)]
else:
    sample_pitch_cents = pitch_cents

distance_profiles = {}
for svara, clusters in final_clusters.items():
    print(f'Computing {svara} distance profile')
    distance_profiles[svara] = {i:[] for i in range(len(clusters))}
    for c, s in enumerate(clusters):
        print(f'Cluster {c}')
        unit = s['pitch']
        for i in tqdm.tqdm(list(range(len(sample_pitch_cents)))):
            target = sample_pitch_cents[i:i+len(unit)]
            if len(target) < wl:
                break
            target = savgol_filter(target, polyorder=2, window_length=wl, mode='interp')

            if np.isnan(target).any():
                distance_profiles[svara][c].append(np.Inf)
                continue

            pi = len(target)
            pj = len(unit)
            l_longest = max([pi, pj])

            path, dtw_val = dtw(target, unit, radius=round(l_longest*0.05))
            l = len(path)
            dtw_norm = dtw_val/l
            distance_profiles[svara][c].append(dtw_norm)

sample_str = f"_{sample}" if sample else ""

dp_path = cpath(out_dir, 'data', 'distance_profiles', f'{track}{sample_str}.pkl')
write_pkl(distance_profiles, dp_path)

dp_path = cpath(out_dir, 'data', 'distance_profiles', f'{track}{sample_str}.pkl')
distance_profiles = load_pkl(dp_path)

###########
## Annotate
###########
annot_dist_thresh = 5

occurences = []
distances = []
lengths = []
labels = []
gamakas = []
for s,ap in distance_profiles.items():
    print(f'Annotating svara, {s}')
    for ix,i in enumerate(ap):
        gamaka = svara_dict[s][ix]['gamaka']
        dp = np.array(ap[i]).copy()

        max_dist = 0
        while max_dist < annot_dist_thresh:
            
            l = len(final_clusters[s][i]['pitch'])
            
            ix = dp.argmin()
            dist = dp[ix]
            
            dp[max(0,int(ix-l)):min(len(dp),int(ix+l))] = np.Inf

            distances.append(dist)
            lengths.append(l)
            occurences.append(ix)
            labels.append(s)
            gamakas.append(gamaka)

            max_dist = dist

occurences = np.array(occurences)
distances = np.array(distances)
lengths = np.array(lengths)
labels = np.array(labels)
gamakas = np.array(gamakas)

###########
## Clean Up
###########
def reduce_labels(occurences, labels, lengths, distances, gamakas, timestep, ovl=0.5, chunk_size_seconds=0.1):
    chunk_size = round(chunk_size_seconds/timestep)
    chunkovl = chunk_size*ovl
    starts = occurences
    ends = starts + lengths
    l_track = ends.max()

    occs = []
    lens = []
    gams = []
    dists = []
    labs = []
    for i1 in np.arange(0, l_track, chunk_size):
        i2 = i1 + chunk_size

        # start before end during
        sbed = ((starts <= i1) & (ends >= i1) & (ends <= i2))
        # occupy at least <ovl> of the chunk?
        oalo1 = sbed & (ends - i1 > chunkovl)
        
        # start during end after
        sdea = ((starts >= i1) & (starts <= i2) & (ends >= i2))
        # occupy at least <ovl> of the chunk?
        oalo2 = sdea & (abs(starts - i2) > chunkovl)

        # start during end during
        sded = ((starts >= i1) & (starts <= i2) & (ends >= i1) & (ends <= i2))
        # occupy at least <ovl> of the chunk?
        oalo3 = sded & (starts - ends > chunkovl)

        # start before end after
        sbea = ((starts <= i1) & (ends >= i2))

        # occupy at least <ovl> of the chunk?
        oalo = oalo1 | oalo2 | oalo2

        all_options = np.where(sbed | sdea | sded | sbea | oalo)[0]

        if len(all_options) > 0:
            # For chunk select svara with lowest distance
            winner = np.argmin(distances[all_options])
            winner_ix = all_options[winner]

            occs.append(i1)
            lens.append(chunk_size)
            gams.append(gamakas[winner_ix])
            dists.append(distances[winner_ix])
            labs.append(labels[winner_ix])

    return np.array(occs), np.array(labs), np.array(lens), np.array(dists), np.array(gams)


def join_neighbouring_svaras(occurences, labels, lengths, distances, gamakas, include_gamaka=True):
    # make sure in chronological order
    ix = sorted(range(len(occurences)), key=lambda y: occurences[y])
    occs = occurences[ix]
    lens = lengths[ix]
    dist = distances[ix]
    gams = gamakas[ix]
    labs = labels[ix]

    batches = [[0]]
    for i,_ in enumerate(occurences[1:],1):
        
        o1 = occurences[i]
        o2 = o1 + lens[i]
        ol = labs[i]
        og = gams[i]

        p1 = occurences[i-1]
        p2 = p1 + lens[i-1]
        pl = labs[i-1]
        pg = gams[i-1]

        gcheck = (pg == og) if include_gamaka else True

        if (p2 == o1) and (pl == ol) and (pg == og) and gcheck:
            # append to existing batch
            batches[-1].append(i)
        else:
            # create new batch
            batches.append([i])
    

    j_occs = []
    j_lens = []
    j_dist = []
    j_gams = []
    j_labs = []
    for batch in batches:
        j_occs.append(occs[batch].min())
        j_lens.append(lens[batch].sum())
        j_dist.append(dist[batch].mean())
        j_gams.append(gams[batch][0])
        j_labs.append(labs[batch][0])

    return np.array(j_occs), np.array(j_labs), np.array(j_lens), np.array(j_dist), np.array(j_gams)


def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found

        
occs, labs, lens, dists, gams = reduce_labels(occurences, labels, lengths, distances, gamakas, timestep, chunk_size_seconds=0.05)
occs, labs, lens, dists, gams = join_neighbouring_svaras(occs, labs, lens, dists, gams)

gams = [gams[i] if l not in ['sa','pa'] else 'none' for i,l in enumerate(labs)]
#occs, labs, lens, dists, gams = occurences, labels, lengths, distances, gamakas


# Check arohana and avorahana errors 
def check_trio_error(trio, aro, avaro, n=1):

    succ = trio[1:]
    prec = trio[:2]

    # successive svara does not obey either arohana or avarahana
    c1 = search_sequence_numpy(aro, succ)
    c2 = search_sequence_numpy(avaro, succ)
    c3 = succ[0] == succ[1] # same svara
    if not any([len(c1), len(c2), c3]):
        return True
    
    # preceeding svara does not obey either arohana or avorahana
    c4 = search_sequence_numpy(aro, prec)
    c5 = search_sequence_numpy(avaro, prec)
    c6 = prec[0] == prec[1] # same svara
    if not any([len(c4), len(c5), c6]):
        return True

    return False


def replace_svara_txt(s):
    if 'S' in s:
        return 'sa'
    if 'R' in s:
        return 'ri'
    if 'G' in s:
        return 'ga'
    if 'M' in s:
        return 'ma'
    if 'P' in s:
        return 'pa'
    if 'D' in s:
        return 'dha'
    if 'N' in s:
        return 'ni'

aro_txt = [replace_svara_txt(x) for x in aro]
avaro_txt = [replace_svara_txt(x) for x in avaro]

# Check errors
errors = []
for i,_ in enumerate(occs[1:-1],1):
    trio_ix = [i-1, i, i-2]
    
    ls = labs[trio_ix]
    gs = gams[trio_ix]

    e = check_trio_error(ls, aro_txt, avaro_txt)

    if e:
        print(ls)
        errors.append(i)



# Check ornamented Sa and Pa
errors = []
for i,_ in enumerate(occs):
    l = labs[i]
    g = gams[i]
    
    if l in ['sa', 'pa']:
        



#########
## Export
#########
starts = [o*timestep for o in occs]
ends   = [starts[i]+(lens[i]*timestep) for i in range(len(starts))]
transcription = pd.DataFrame({
        'start':starts,
        'end':ends,
        'label':[f"{l} ({g})" for l,g in zip(labs, gams)],
    }).sort_values(by='start')

trans_path = cpath(out_dir, 'data', 'transcription', f'{track}.txt')
transcription.to_csv(trans_path, index=False, header=False, sep='\t')

#  - Clean up
#      - dtw distances between them all to trim
#      - remove silences
#  - Characterize svaras
#      - cluster into profiles based on features
#      - what are the distinguishing characteristics
#      - Markov chains
#  - Query track with MP style approach to identify other occurrences
#  - Clean up annotations with raga grammar
#  - Clean annotations with onsets from spectral flux?
#  - Which characteristics are due to the pronunciation and which due to the melody?
#  - Evaluate somehow?
