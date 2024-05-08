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

svara_dict = get_svara_dict(annotations, pitch_cents, timestep, track, min_length=0.1, smooth_window=0.1, path=None, tonic=tonic)

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

del data['label']

static_features = data.values

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

