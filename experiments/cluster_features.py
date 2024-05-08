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

from src.utils import cpath, load_pkl
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

best = -100
### Get Features
for clus in df['cluster'].unique():
	clus_svara = clus.split('_')[0]
	this_svara_data = svara_data[(svara_data['svara']==clus_svara)]
	clus_ix = df[df['cluster']==clus]['oi'].values
	clus_data = svara_data[(svara_data['svara']==clus_svara) & (svara_data['index'].isin(clus_ix))]
	assert len(clus_data)==len(clus_ix)

	plt.close('all')

	# feature plots 
	cluschar_path = cpath('data', 'analysis', analysis_name, 'features')
	feature_plots_path = cpath(cluschar_path, clus_svara, f'{clus}.png')
	featcols = ['duration', 'av_pitch', 'min_pitch', 'max_pitch', 'pitch_range', 'av_start_pitch', 'av_end_pitch', 'num_change_points_pitch']

	featcols.reverse()	
	featmeans = {f:np.mean(this_svara_data[f]) for f in featcols}
	featstds = {f:np.std(this_svara_data[f]) for f in featcols}
	
	features = {f:clus_data[f].values for f in featcols}

	d = {k:(np.mean(v)-featmeans[k])/featstds[k] for k,v in features.items()}
	
	best = max([max(d.values()),best])
	val = {k:np.mean(v) for k,v in features.items() if k in featcols}
	
	plt.close('all')
	plt.figure(figsize=(12,6), tight_layout=True)
	plt.grid(zorder=0)
	bar = plt.barh(list(d.keys()), list(d.values()), zorder=2, color='purple')
	plt.title(f'Features, {clus} (n={len(clus_ix)})')
	plt.xlabel('Z-score')
	plt.xlim((-4.5,4.5))
	ax = plt.gca()
	for bar, v in zip(ax.patches, val.values()):
	    t = ax.text(-4.45, bar.get_y()+bar.get_height()/2, round(v,2), color = 'white', ha = 'left', va = 'center')
	    t.set_bbox(dict(facecolor='black', alpha=0.7))

	plt.savefig(feature_plots_path)
	plt.close('all')

