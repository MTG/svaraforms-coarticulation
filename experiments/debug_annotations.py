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


path1 = 'data/annotation/kamakshi.txt'
path2 = 'data/annotation/kamakshi_new.txt'

annotations_old = load_elan_annotations(path1)
annotations_new = load_elan_annotations(path2)