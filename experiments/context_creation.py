### list of annontations files

### Load each, tonic normalise pitch extraction

### 
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
import soundfile as sf

from src.dtw import dtw
from src.tools import compute_novelty_spectrum, get_loudness, interpolate_below_length, get_derivative, chaincode
from src.utils import load_audacity_annotations, load_elan_annotations, load_processed_annotations, cpath,  write_pkl, write_pitch_track, load_pitch_track, read_txt, load_pkl, myround
from src.visualisation import get_plot_kwargs, plot_subsequence, get_arohana_avarohana
from src.pitch import pitch_seq_to_cents
from src.svara import get_svara_dict, get_unique_svaras, pairwise_distances_to_file, chop_time_series, create_bad_svaras
from scipy.signal import savgol_filter
from src.clustering import duration_clustering, cadence_clustering, hier_clustering

import numpy as np

analysis_name = 'bhairaviContext'

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

include_bad = False

tracks = ['kamakshi']
tonics = [146.84, 130.81]
loader = ['elan','proc']

raga = 'bhairavi'

###########
# Get paths
###########
all_unique_svaras = []
svara_dicts = {}
vocals = {}
sr = 44100
for i,track in enumerate(tracks):
    print(f'Extracting for {track}...')
    if track == 'chintayama_kanda':
        print('Skipping...')
        continue
    annotations_path = os.path.join('data', 'annotation', f'{track}.txt')
    vocal_path = os.path.join('data','audio', f'{track}.mp3')

    ###########
    # Load Data
    ###########
    vocal, _ = librosa.load(vocal_path, sr=sr)

    vocals[track] = vocal

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


    tonic = tonics[i]
    lder = loader[i]
    plot_kwargs = get_plot_kwargs(raga, tonic)
    yticks_dict = plot_kwargs['yticks_dict']
    aro, avaro = get_arohana_avarohana(raga)

    # Load annotations
    if lder == 'aud':
        annotations = load_audacity_annotations(annotations_path)
    elif lder == 'elan':
        annotations = load_elan_annotations(annotations_path)
    elif lder == 'proc':
        annotations = load_processed_annotations(annotations_path)

    annotations['label'] = annotations['label'].apply(lambda y: y.strip().replace('da','dha'))

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

    svara_dict = get_svara_dict(annotations, pitch_cents, timestep, track, tonic, min_length=0.1, smooth_window=0.1, path=None)
    n_svaras = max([len(v) for v in svara_dict.values()])
    n = n_svaras

    if include_bad:
        print(f'Creating {n} bad svaras')
        bad_svaras = create_bad_svaras(annotations, pitch_cents, timestep, track, tonic, min_length=0.1, thresh=0.1, n=n, smooth_window=0.1)

        svara_dict['none'] = bad_svaras

    svara_dicts[track] = svara_dict
    all_unique_svaras = all_unique_svaras + unique_svaras

all_unique_svaras = list(set(unique_svaras)) + ['none'] if include_bad else list(set(unique_svaras)) # for bad svaras

##############
## Combine all
##############
svara_dict = {}
for s in all_unique_svaras:
    this = []
    for sd in svara_dicts.values():
        this += sd[s]
    svara_dict[s] = this

for k,v in svara_dict.items():
    print(f'{len(v)} total occurences of {k}')


###########################
## Transpose to same octave
###########################
def get_prop_octave(pitch, o=1):
    madyhama = len(np.where(np.logical_and(pitch>=0, pitch<=1200))[0]) # middle octave
    tara = len(np.where(pitch>1200)[0]) # higher octave
    mandhira = len(np.where(pitch<0)[0]) # lower octave

    octs = [mandhira, madyhama, tara]
    return octs[o]/len(pitch)

def transpose_pitch(pitch):
    ## WARNING: Assumes no pitch values in middle octave+2 or middle octave-2
    ## and that no svara traverses two octaves (likely due to pitch errors)
    r_prop = get_prop_octave(pitch, 0)
    p_prop = get_prop_octave(pitch, 1)
    i_prop = get_prop_octave(pitch, 2)

    if r_prop == 0 and i_prop == 0:
        # no transposition
        return pitch, False

    if r_prop == 0 and p_prop == 0:
        # transpose down
        return pitch-1200, True
    
    if i_prop == 0 and p_prop == 0:
        # transpose up
        return pitch+1200, True

    if i_prop > 0.8:
        # transpose down
        return pitch-1200, True

    return pitch, False

t_count = 0
for svara, sds in svara_dict.items():
    print(f'Transposing {svara}...')
    for i,sd in tqdm.tqdm(list(enumerate(sds))):
        pitch = sd['pitch']
        transposed, t = transpose_pitch(pitch)
        sd['pitch'] = transposed
        sd['transposed'] = t
        if t:
            t_count +=1
print(f'{t_count} svaras transposed')

write_pkl(svara_dict, cpath('data', 'analysis', f'{analysis_name}', 'svara_dict.pkl'))

##############################
# Create Time Series Dataset #
##############################
train_size = 0.7
ts_data_out_dir = f'/Users/thomasnuttall/code/DeepGRU/data/{analysis_name}/'
label_lookup = {i:s for i,s in enumerate(all_unique_svaras)}
write_pkl(label_lookup, cpath('data', 'analysis', f'{analysis_name}', 'label_lookup.pkl'))

# static features
cols = ['label', 'pitch_range', 'av_pitch', 'min_pitch', 'max_pitch', 'pitch75', 'pitch25', 'av_first_pitch', 'av_end_pitch', 'num_change_points_pitch', 
        'num_change_points_loudness', 'max_loudness', 'min_loudness', 'loudness75', 'loudness25', 
        'prec_pitch_range', 'av_prec_pitch', 'min_prec_pitch', 'max_prec_pitch', 'prec_pitch75', 'prec_pitch25', 
        'succ_pitch_range', 'av_succ_pitch', 'min_succ_pitch', 'max_succ_pitch', 'succ_pitch75', 'succ_pitch25', 
        'direction_asc', 'direction_desc',
        'duration']

data = pd.DataFrame(columns=cols)
ts_data = []
labels = []
for svara, sv_sd in svara_dict.items():
    for ni,sd in enumerate(sv_sd):
        
        l = all_unique_svaras.index(svara)
        
        track = sd['track']
        pitch = sd['pitch']
        s1 = sd['start']
        s2 = sd['end']
        
        prec_pitch = sd['prec_pitch']
        succ_pitch = sd['succ_pitch']
        
        context_pitch = np.concatenate([prec_pitch[-round(0.3/timestep):], pitch, succ_pitch[:round(0.3/timestep)]])
        
        if None in context_pitch or sum(np.isnan(context_pitch))!=0:
            print('excluding')
            continue

        time = [i*timestep for i in range(len(pitch))]


        dp, dt = get_derivative(pitch, time)
        asign = np.sign(dp)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        cp_p = sum(signchange)
        
        y = vocals[track][round(s1*sr):round(s2*sr)]

        loudness = get_loudness(y)
        asign = np.sign(loudness)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        cp_l = sum(signchange)
        
        ran = max(pitch)-min(pitch)
        
        # exclude incorrectly extracted time series
        if ran > 1000:
            continue

        row = {
            'label': l,
            'track': track,
            'svara': svara,
            'index': int(ni),
            'timestep': timestep,
            'pitch_range': max(pitch)-min(pitch),
            'av_pitch': np.nanmean(pitch),
            'min_pitch': min(pitch),
            'max_pitch': max(pitch),
            'pitch75': np.quantile(pitch, 0.75),
            'pitch25': np.quantile(pitch, 0.25),
            'max_pitch': max(pitch),
            'max_pitch': max(pitch),
            'av_first_pitch': np.mean(pitch[:int(len(pitch)*0.1)]),
            'av_end_pitch':  np.mean(pitch[-int(len(pitch)*0.1):]),
            'num_change_points_pitch': cp_p,
            'num_change_points_loudness': cp_l,
            'max_loudness': max(loudness),
            'min_loudness':min(loudness),
            'loudness75': np.quantile(loudness, 0.75),
            'loudness25': np.quantile(loudness, 0.25),
            'prec_pitch_range': max(prec_pitch)-min(prec_pitch), 
            'av_prec_pitch': np.nanmean(prec_pitch),
            'min_prec_pitch': min(prec_pitch),
            'max_prec_pitch': max(prec_pitch),
            'prec_pitch75': np.quantile(prec_pitch, 0.75),
            'prec_pitch25': np.quantile(prec_pitch, 0.25),
            'succ_pitch_range': max(succ_pitch)-min(succ_pitch), 
            'av_succ_pitch': np.nanmean(succ_pitch),
            'min_succ_pitch': min(succ_pitch), 
            'max_succ_pitch': max(succ_pitch),
            'succ_pitch75': np.quantile(succ_pitch, 0.75),
            'succ_pitch25': np.quantile(succ_pitch, 0.25),
            'direction_asc': int(np.mean(pitch[:int(len(pitch)*0.1)]) < np.mean(pitch[-int(len(pitch)*0.1):])),
            'direction_desc': int(np.mean(pitch[:int(len(pitch)*0.1)]) > np.mean(pitch[-int(len(pitch)*0.1):])),
            'duration': len(pitch)*timestep,
        }

        data = data.append(row, ignore_index=True)
        
        ts_data.append(context_pitch)
        labels.append(l)

data = data.fillna(data.mean())

data.to_csv(cpath('data', 'analysis', f'{analysis_name}', 'data.csv'), index=False)

for c in cols:
    print(f'{c}')
    for s in all_unique_svaras:
        sv = all_unique_svaras.index(s)
        v = data[data['label']==sv][c].values
        print(f' {s}: mean={np.mean(v)}, min={np.min(v)}, max={np.max(v)}')

del data['label']
del data['track']
del data['svara']
#del data['index']
del data['timestep']

static_features = data.values

all_data = list(zip(labels, ts_data, static_features))

# Shuffle
random.seed(42) 
random.shuffle(all_data)

feat_data = np.array([x[2] for x in all_data])
ts_data = [x[:2] for x in all_data]

feat_train_data = np.array(feat_data[:round(train_size*len(feat_data))])
feat_test_data = np.array(feat_data[round(train_size*len(feat_data)):])

train_data = np.array(ts_data[:round(train_size*len(ts_data))])
test_data = np.array(ts_data[round(train_size*len(ts_data)):])

# Write
train_path = cpath(ts_data_out_dir, 'TRAIN.pkl')
test_path = cpath(ts_data_out_dir, 'TEST.pkl')
write_pkl(train_data, train_path)
write_pkl(test_data, test_path)

feat_train_path = cpath(ts_data_out_dir, 'TRAIN_FEAT.pkl')
feat_test_path = cpath(ts_data_out_dir, 'TEST_FEAT.pkl')
write_pkl(feat_train_data, feat_train_path)
write_pkl(feat_test_data, feat_test_path)

label_path = cpath(ts_data_out_dir, 'LABELS.pkl')
write_pkl(label_lookup, label_path)



