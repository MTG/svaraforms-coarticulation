%load_ext autoreload
%autoreload 2

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

from numpy.linalg import norm
from numpy import dot

from src.dtw import dtw
from src.tools import compute_novelty_spectrum, get_loudness, interpolate_below_length, get_derivative
from src.utils import load_audacity_annotations, load_elan_annotations, cpath,  write_pkl, write_pitch_track, load_pitch_track, read_txt, load_pkl
from src.visualisation import get_plot_kwargs, plot_subsequence, get_arohana_avarohana
from src.pitch import pitch_seq_to_cents
from src.svara import get_svara_dict, get_unique_svaras, pairwise_distances_to_file
from scipy.signal import savgol_filter
from src.clustering import duration_clustering, cadence_clustering, hier_clustering

track = 'omkar_bhoopali'
raga = 'bhoopali'


###########
# Get paths
###########
vocal_path = f'data/audio/{track}.mp3'

tonic = 311.13
sr = 44100
vocal, _ = librosa.load(vocal_path, sr=sr)

pitch_track_path = cpath('data', 'pitch_tracks', f'{track}.tsv')
tidy_pitch_track_path = cpath('data', 'pitch_tracks', f'{track}_tidy.csv')

pt = load_pitch_track(pitch_track_path)

pitch = pt[:,1]
time = pt[:,0]
timestep = time[3]-time[2]


###################
## Tidy Pitch Track
###################
min_gap = 250*0.001 # in seconds
min_track = 100*0.001 # in seconds
min_f = 50
max_f = 500

# Remove above frequency
null_ind = pitch==0

pitch[pitch<min_f]=0
pitch[pitch>max_f]=0

# Interpolate small gaps
pitch = interpolate_below_length(pitch, 0, (min_gap/timestep))

# remove below length
pitch = remove_below_length(pitch, (min_track/timestep))

# cents
pitch_cents = pitch_seq_to_cents(pitch, tonic)

pt[:,1] = pitch
pt[:,0] = time

tidy_pitch_track_path = cpath('data', 'pitch_tracks', f'{track}_tidy.csv')
write_pitch_track(pt, tidy_pitch_track_path, sep=',')
