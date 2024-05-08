# transpose is current
# with_static is transpose with static features (88% compared to 86.7%)
# context_ablation_norm_ts

# Batch size changed to 80 then 32

%load_ext autoreload
%autoreload 2
import sys
sys.path.append('../DeepGRU/')
import os
import pickle
from dataset.datafactory import DataFactory
from model import DeepGRU
import torch
from utils.logger import log
import numpy as np

import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd 
from sklearn.manifold import TSNE
from scipy import stats

from src.utils import cpath, load_pkl, myround
from src.visualisation import get_plot_kwargs

from PIL import Image
import umap

import os

folder = 'data/analysis/bhairaviTransposedIndex/current_best/kmeans/'

files = [os.path.join(d, x).replace(folder,'').replace('_TRANSPOSED','')
            for d, dirs, files in os.walk(folder)
            for x in files if x.endswith(".png")]
allsvaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']
all_data =[]

for f in files:
	splitted = f.split('/')
	svara = splitted[0]
	clus = splitted[1]
	namesplit = splitted[2].replace('.png','').split('_')
	prec = namesplit[0]
	suc = namesplit[2]
	precsuc = f'{prec}_{suc}'
	all_data.append((svara, f'{svara}_{clus}', prec, suc, precsuc, f))


df = pd.DataFrame(all_data, columns=['svara', 'cluster', 'prec', 'suc', 'precsuc', 'filepath'])
df = df[~df['filepath'].str.contains('-')]
df['oi'] = df['filepath'].apply(lambda y: int(y.split('.')[0].replace('_TRANSPOSED','').split('_')[-2]))

### Feature Data
################
analysis_name = 'bhairaviTransposedIndex'
model_path = '/Users/thomasnuttall/code/DeepGRU/models/best/transpose'

direc = f'/Users/thomasnuttall/code/carnatic_segmentation/data/analysis/{analysis_name}'
pldirec = f'/Users/thomasnuttall/code/carnatic_segmentation/data/analysis/{analysis_name}'
raga = 'bhairavi'

dataset_name = 'bhairavi'
seed = 1570254494

plots_dir = os.path.join(pldirec, 'plots', '')

features_path = os.path.join(direc, 'data.csv')
svara_dict_path = os.path.join(direc, 'svara_dict.pkl')
label_lookup_path = os.path.join(direc, 'label_lookup.pkl')
plots_path_path =  os.path.join(plots_dir, 'plot_paths.csv')

data = pd.read_csv(features_path)
plots_paths = pd.read_csv(plots_path_path)
svara_data = data[data['svara']!='none']
svara_dict = load_pkl(svara_dict_path)
label_lookup = load_pkl(label_lookup_path)
lookup_label = {v:k for k,v in label_lookup.items()}
svara_data['av_start_pitch'] = svara_data['av_first_pitch']



### Plots
#########
def transpose_pitch(pitch, prop=0.6):
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

    if i_prop > prop:
        # transpose down
        return pitch-1200, True

    if r_prop > prop:
        # transpose up
        return pitch+1200, True

    return pitch, False

def get_prop_octave(pitch, o=1):
    madyhama = len(np.where(np.logical_and(pitch>=0, pitch<=1200))[0]) # middle octave
    tara = len(np.where(pitch>1200)[0]) # higher octave
    mandhira = len(np.where(pitch<0)[0]) # lower octave

    octs = [mandhira, madyhama, tara]
    return octs[o]/len(pitch)

def get_timestamp(secs, divider='-'):
    """
    Convert seconds into timestamp

    :param secs: seconds
    :type secs: int
    :param divider: divider between minute and second, default "-"
    :type divider: str

    :return: timestamp
    :rtype: str
    """
    minutes = int(secs/60)
    seconds = round(secs%60)
    return f'{minutes}min{divider}{seconds}sec'



### Cluster plots
cluster = 'dha_cluster_4'
this_data = df[df['cluster']==cluster]
incorrect_ix = [41, 77, 62, 5] # for dha
this_data = this_data[~this_data['oi'].isin(incorrect_ix)]

starts = []
pitches = []
times = []
max_pitch = -10000
min_pitch = 10000
for i,row in this_data.iterrows():
	oi = row['oi']
	r = svara_dict[cluster.split('_')[0]][oi]
	pitch,_ = transpose_pitch(r['pitch'])
	start = r['start']
	timestep = r['timestep']
	starts.append(start)
	pitches.append(pitch)
	times.append([i*timestep for i in range(len(pitch))])
	tonic = r['tonic']
	max_pitch = max([max_pitch, max(pitch)])
	min_pitch = min([min_pitch, min(pitch)])

plot_kwargs = get_plot_kwargs(raga, tonic, cents=True)
yticks_dict = plot_kwargs['yticks_dict']

yticks_dict = {k:v for k,v in yticks_dict.items() if v<=myround(max_pitch,100)+100 and v>=myround(min_pitch)-100}
tick_names = list(yticks_dict.keys())
tick_loc = [p for p in yticks_dict.values()]

zipped = sorted(zip(starts, pitches, times), key=lambda y: y[0])
starts = [x[0] for x in zipped]
pitches = [x[1] for x in zipped]
times = [x[2] for x in zipped]

plt.rcParams["font.family"] = "DejaVu Sans"

plt.close('all')
plt.figure(figsize=(10,5))

for i in range(len(starts)):
	s = round(starts[i],1)
	ts = get_timestamp(s)
	plt.plot(times[i], pitches[i], label=ts, linewidth=0.75)

plt.legend(title='Start time')
plt.grid()
plt.title(cluster.replace('_', ' ').capitalize())
plt.xlabel('Time (s)', fontname='')
plt.ylabel(f'Pitch (cents) as svara positions')
ax = plt.gca()
ax.set_yticks(tick_loc)
ax.set_yticklabels(tick_names)

ax.set_facecolor('#f2f2f2')
plt.savefig('test.png', bbox_inches='tight')
plt.close('all')



### Random svara plots
svara = 'ni'
n=5
this_data = df[df['svara']==svara].sample(n)
ixs = [0, 69, 66, 25, 23]
#this_data = this_data[this_data['oi'].isin(ixs)]
starts = []
pitches = []
times = []
indices = []
max_pitch = -10000
min_pitch = 10000
for i,row in this_data.iterrows():
	oi = row['oi']
	r = svara_dict[svara][oi]
	pitch,_ = transpose_pitch(r['pitch'])
	start = r['start']
	timestep = r['timestep']
	indices.append(oi)
	starts.append(start)
	pitches.append(pitch)
	times.append([i*timestep for i in range(len(pitch))])
	tonic = r['tonic']
	max_pitch = max([max_pitch, max(pitch)])
	min_pitch = min([min_pitch, min(pitch)])

plot_kwargs = get_plot_kwargs(raga, tonic, cents=True)
yticks_dict = plot_kwargs['yticks_dict']

yticks_dict = {k:v for k,v in yticks_dict.items() if v<=myround(max_pitch,100)+100 and v>=myround(min_pitch)-100}
tick_names = list(yticks_dict.keys())
tick_loc = [p for p in yticks_dict.values()]

zipped = sorted(zip(starts, pitches, times), key=lambda y: y[0])
starts = [x[0] for x in zipped]
pitches = [x[1] for x in zipped]
times = [x[2] for x in zipped]

plt.rcParams["font.family"] = "DejaVu Sans"

size=4
plt.close('all')
fig=plt.figure()
fig, axs = plt.subplots(1, len(this_data), figsize=(size*len(this_data),size), sharey=True)

for i in range(len(starts)):
	s = round(starts[i],1)
	ts = get_timestamp(s)
	axs[i].plot(times[i], pitches[i], linewidth=0.75, color='darkgreen')
	axs[i].set_title(f'i={indices[i]}, start: {ts}', fontsize=10)
	axs[i].set_yticks(tick_loc)
	axs[i].set_yticklabels(tick_names)
	axs[i].grid()
	axs[i].set_facecolor('#f2f2f2')

for ax in axs.flat:
    ax.set(xlabel='Time (s)', ylabel='Pitch (cents) as svara positions')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

#fig.suptitle(f'{n} {svara.capitalize()}s')

plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('test.png', bbox_inches='tight')
plt.close('all')


