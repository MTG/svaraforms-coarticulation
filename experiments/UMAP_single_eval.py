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

np.random.seed(42)

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

feature_names = [c for c in data.columns if c not in ['label', 'track', 'svara', 'index', 'timestep']]

contextual_features =[]# ['num_change_points_loudness', 'max_loudness', 'min_loudness', 'loudness75', 'loudness25']
                      #['prec_pitch_range', 'av_prec_pitch', 'min_prec_pitch', 'max_prec_pitch', 'prec_pitch75', 'prec_pitch25', 
                      # 'succ_pitch_range', 'av_succ_pitch', 'min_succ_pitch', 'max_succ_pitch', 'succ_pitch75', 'succ_pitch25']
# remove contextual features
feature_names = [x for x in feature_names if x not in contextual_features]

svara_data['index']=svara_data['index'].astype(int)

#############################
### GET MODEL ANDD EMBEDDINGS
#############################
# Load model
log.set_dataset_name(dataset_name)
dataset = DataFactory.instantiate(dataset_name)
hyperparameters = dataset.get_hyperparameter_set()

# Instantiate the model, loss measure and optimizer
model = DeepGRU(dataset.num_features, dataset.num_classes, dataset.static_features.shape[1], with_static=False)

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
model.load_state_dict(state_dict)


# Predict and evaluate
all_labels = []
all_embds = []
all_oi = []
all_ts = []
all_lengths = []
model.eval()
for fold_idx in range(len(dataset.train_indices)):
    train_loader, test_loader = dataset.get_data_loaders(fold_idx,
                                                         shuffle=True,
                                                         random_seed=seed+fold_idx,
                                                         normalize_ts=True,
                                                         normalize_static=False)
    batch_size = hyperparameters.batch_size

    samples = [x for x in train_loader]

    for i in range(len(samples)):
        batch = samples[i]
        examples, length, labels, static_features, oi = batch

        # Predict
        outputs = model.embed(examples, length)
        all_embds += list(outputs.detach().numpy())
        all_labels += list([lookup_label[dataset.idx_to_class[l]] for l in labels.numpy()])
        all_oi += oi
        all_ts += examples
        all_lengths += list(length.numpy())

for fold_idx in range(len(dataset.test_indices)):
    train_loader, test_loader = dataset.get_data_loaders(fold_idx,
                                                         shuffle=True,
                                                         random_seed=seed+fold_idx,
                                                         normalize_ts=True,
                                                         normalize_static=False)
    batch_size = hyperparameters.batch_size

    samples = [x for x in test_loader]

    for i in range(len(samples)):
        batch = samples[i]
        examples, length, labels, static_features, oi = batch

        # Predict
        outputs = model.embed(examples, length)
        all_embds += list(outputs.detach().numpy())
        all_labels += list([lookup_label[dataset.idx_to_class[l]] for l in labels.numpy()])
        all_oi += oi
        all_ts += examples
        all_lengths += list(length.numpy())

#joined = zip(all_labels, all_embds, all_oi, all_ts, all_lengths)
#joined = [x for x in joined if not ((x[0]==5) and (x[2] in [52, 29, 12]))]

#all_labels = [x[0] for x in joined]
#all_embds = [x[1] for x in joined]
#all_oi = [x[2] for x in joined]
#all_ts = [x[3] for x in joined]
#all_lengths = [x[4] for x in joined]

####################
## Cluster functions
####################
from sklearn.cluster import KMeans
import umap
from sklearn.metrics import silhouette_score
import hdbscan
def cluster(X, n=3, min_samples=2, min_cluster_size=2, algo='hdbscan'):
    
    if algo == 'kmeans':
        
        # kmeans
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(X)
        l = kmeans.labels_

    elif algo == 'hdbscan':
        
        l = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        ).fit_predict(X)

    return l


def output_clusters(clusdir, clus_labels, this_label):
    this_ix = [i for i,l in enumerate(all_labels) if l == this_label]
    this_pp = [pp for i,pp in enumerate(all_plot_paths) if i in this_ix]
    for i,l in enumerate(clus_labels):
        cluschardir = cpath(clusdir, f'cluster_{l}/')
        pp = this_pp[i]
        ap = pp.replace('.png', '.wav').replace('_TRANSPOSED','')
        shutil.copy(pp, cluschardir)
        shutil.copy(ap, cluschardir)

def context(clus_labels, this_label, path):
    this_ix = [i for i,l in enumerate(all_labels) if l == this_label]
    this_pp = [pp for i,pp in enumerate(all_plot_paths) if i in this_ix]
    for clus in set(clus_labels):
        if not clus >= 0:
            continue
        
        clindices = [i for i,l in enumerate(clus_labels) if l == clus]

        cluschardir = cpath(clusdir, f'cluster_{l}/')
        pp = this_pp[i]

from collections import Counter
def evaluate_clusters(clus_labels, path):
    eval_str = []
    winners_prec = []
    winners_suc = []
    winners_precsuc = []
    this_ix = [i for i,l in enumerate(all_labels) if l == this_label]
    this_pp = np.array([pp for i,pp in enumerate(all_plot_paths) if i in this_ix])
    for clus in set(clus_labels):
        clindices = [i for i,l in enumerate(clus_labels) if l == clus]
        clus_pps = this_pp[clindices]
        eval_str.append('\n')
        eval_str.append(f'Cluster {clus}, population: {len(clus_pps)}...\n')
        
        context = [pp.replace('_TRANSPOSED','').split('.')[0].split('/')[-1].split('_') for pp in clus_pps]
        prec_suc = [(x[0],x[2]) for x in context]
        
        prec_suc_comb = [f'{x[0]}-{x[1]}' for x in prec_suc]
        ps_c = Counter(prec_suc_comb)
        prec = [x[0] for x in prec_suc]
        p_c = Counter(prec)
        suc = [x[1] for x in prec_suc]
        s_c = Counter(suc)
        eval_str.append('   Full Context:\n')
        eval_str.append('   -------------\n')
        for l,v in ps_c.items():
            eval_str.append(f'      {l}: {v} ({round(v*100/len(clus_pps), 2)}%)\n')
        eval_str.append('   Preceeding Svara:\n')
        eval_str.append('   -----------------\n')
        for l,v in p_c.items():
            eval_str.append(f'      {l}: {v} ({round(v*100/len(clus_pps), 2)}%)\n')
        eval_str.append('   Succeeding Svara:\n')
        eval_str.append('   ----------------\n')
        for l,v in s_c.items():
            eval_str.append(f'      {l}: {v} ({round(v*100/len(clus_pps), 2)}%)\n')

        winners_prec.append(list(p_c.keys())[np.argmax(p_c.values())])
        winners_suc.append(list(s_c.keys())[np.argmax(s_c.values())])
        winners_precsuc.append(list(ps_c.keys())[np.argmax(ps_c.values())])

    prec_unique = round(len(set(winners_prec))*100/len(winners_prec),2)
    suc_unique = round(len(set(winners_suc))*100/len(winners_suc),2)
    precsuc_unique = round(len(set(winners_precsuc))*100/len(winners_precsuc),2)

    eval_str = [f'Preceeding uniqueness: {prec_unique}%\n'] + eval_str
    eval_str = [f'Succeeding uniqueness: {suc_unique}%\n'] + eval_str
    eval_str = [f'Preceeding/Succeeding uniqueness: {precsuc_unique}%\n'] + eval_str
    eval_str = ['\n'] + eval_str

    with open(path, "a") as f:
        for es in eval_str:
            f.write(es)


import copy

def remove_outliers(clus_labels, X):
    new_clus_labels = copy.deepcopy(clus_labels)
    for clus in set(clus_labels):
        if clus == -1:
            continue
        clindices = np.array([i for i,l in enumerate(clus_labels) if l == clus])
        embs = X[clindices]

        av_dist = []
        for i,e in enumerate(embs):
            avd = []
            for j,k in enumerate(embs):
                if i!=j:
                    avd.append(np.linalg.norm(k-e))
            av_dist.append(np.mean(avd))

        q75, q25 = np.percentile(av_dist, [75 ,25])
        iqr = q75 - q25
        bad_indices = []
        for i in range(len(av_dist)):
            if (av_dist[i] < q25 - 1.5*iqr) or (av_dist[i] > q75 + 1.5*iqr):
                bad_indices.append(i)

        new_clus_labels[clindices[bad_indices]] = -clus

    return new_clus_labels


def split_clusters_on_durations(clus_labels, eps=10):
    new_clus_labels = copy.deepcopy(clus_labels)
    this_ix = [i for i,l in enumerate(all_labels) if l == this_label]
    this_durations = np.array([pp for i,pp in enumerate(all_lengths) if i in this_ix])
    maxi=0
    for clus in set(clus_labels):
        clindices = np.array([i for i,l in enumerate(clus_labels) if l == clus])
        durations = this_durations[clindices]

        dur_clust = DBSCAN(eps=eps, min_samples=1).fit(np.array(durations).reshape(-1, 1))
        dur_labels = dur_clust.labels_
        dur_labels = dur_labels + maxi
        new_clus_labels[clindices] = dur_labels
        maxi = max(dur_labels)
    return new_clus_labels

##################################
### CREATE DATASETS FOR CLUSTERING
##################################
all_static = []
all_indices = []
all_svaras = []
all_plot_paths = []

for i in range(len(all_embds)):
    embds = all_embds[i]
    label = all_labels[i]
    oi = all_oi[i]
    svara = label_lookup[label]
    
    features = svara_data[(svara_data['index']==oi) & (svara_data['svara']==svara)][feature_names].values[0]
    
    pp = plots_paths[
        (plots_paths['svara']==svara) & \
        (plots_paths['index']==oi)]['plot_path'].values[0]

    all_static.append(features)
    all_svaras.append(svara)
    all_plot_paths.append(pp)

all_static = np.array(all_static)
all_embds = np.array(all_embds)


X1 = np.concatenate([all_embds, all_static], axis=1).astype(float)
X2 = all_embds.astype(float)
X3 = all_static.astype(float)
data_names = ['embstat', 'emb', 'stat']

X = X2


################
##### CLUSTERING
################



### SVARA DATA
##############
import shutil
# {0: 'ga', 1: 'ni', 2: 'ri', 3: 'sa', 4: 'pa', 5: 'ma', 6: 'dha'}
params = {
        'sa': {'nn':5,'md':0.1}, #{'nn':5,'md':0.1},
        'ri': {'nn':5,'md':0.1}, #{'nn':6,'md':0.1},
        'ga': {'nn':5,'md':0.1}, #{'nn':10,'md':0.2},
        'ma': {'nn':5,'md':0.1}, #{'nn':7,'md':0.1},
        'pa': {'nn':5,'md':0.1}, #{'nn':5,'md':0.1},
        'dha':{'nn':5,'md':0.1}, #{'nn':5,'md':0.1},
        'ni': {'nn':5,'md':0.1} #{'nn':5,'md':0.1}
    }

# normalise
X_norm = stats.zscore(X, axis=None).astype(float)

inertia_elbows_5 = {
    'sa':6,
    'ri':9,
    'ga':8,
    'ma':8,
    'pa':7,
    'dha':9,
    'ni': 7
}

inertia_elbows_10 = {
    'sa':9,
    'ri':7,
    'ga':7,
    'ma':9,
    'pa':7,
    'dha':10,
    'ni':6
}

isnorm = True
n_neighbours = 5

if n_neighbours == 5:
    inertia_elbows = inertia_elbows_5
else:
    inertia_elbows = inertia_elbows_10

    
for this_label in label_lookup:

    # Get data for this svara
    this_svara = label_lookup[this_label]
    n_clusters = inertia_elbows[this_svara]
    clust_range = range(5, 30)
    X_svara = np.array([x for x,l in zip(X_norm, all_labels) if l == this_label])
    X_lengths = np.array([x for x,l in zip(all_lengths, all_labels) if l == this_label])

    # EMBED
    print(f'Embedding svara {this_svara}')
    reducer = umap.UMAP(
                    n_neighbors=10,
                    min_dist=0.025,
                    learning_rate=1,
                    metric='euclidean')

    X_embedded = reducer.fit_transform(X_svara)

    print(f'Embedding svara to 4 components, {this_svara}')
    reducer = umap.UMAP(
                    n_neighbors=params[this_svara]['nn'],
                    min_dist=params[this_svara]['md'],
                    learning_rate=1,
                    n_components=n_neighbours,
                    metric='euclidean')

    X_embedded4 = reducer.fit_transform(X_svara)
    
    if isnorm:
        X_embedded4 = stats.zscore(X_embedded4, axis=None).astype(float)

    ### KMEANS
    ##########
    print(f'K means for svara: {this_svara}')
    all_scores = []
    for c in clust_range:
        print(f'  n_clusters={c}')
        kmeans = KMeans(n_clusters=c, random_state=0, n_init="auto").fit(X_embedded4)
        clus_labels = kmeans.labels_
        score = kmeans.inertia_#silhouette_score(X_embedded4, clus_labels)
        print(f'  score={score}')
        all_scores.append(score)

    plt.close('all')
    plt.plot(clust_range, all_scores)
    plt.title(this_svara)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(cpath(f'{direc}/clusters_single/{this_svara}_kmeans_silplot.png'))
    plt.close('all')
    high_score = clust_range[np.argmax(all_scores)]
    #print(f'Best score={round(max(all_scores),2)}, n_clusters={high_score}')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_embedded4)
    clus_labels = kmeans.labels_

    new_clus_labels = remove_outliers(clus_labels, X_embedded4)

                    #   ### Output dataframe for visualisation
                    #   ######################################
                    #   print('Outputting embedded data and clustering...')
                    #   for_app = pd.DataFrame(columns=['x', 'y', 'label', 'plot_path'])
                    #   this_pp = np.array([x for x,l in zip(all_plot_paths, all_labels) if l == this_label])
                    #   for i in range(len(X_embedded)):
                    #       pp = this_pp[i]
                    #       label = clus_labels[i]
                    #       d = {
                    #           'x': [X_embedded[i][0]],
                    #           'y': [X_embedded[i][1]],
                    #           'label': [label],
                    #           'plot_path': [pp],
                    #           'audio_path': [pp.replace('.png','.wav')],
                    #       }
                    #       for_app = pd.concat([for_app, pd.DataFrame(d)], ignore_index=True)

                    #   for_app.to_csv(cpath(direc,f'clusters_single/kmeans/embedding_kmeans_{this_svara}.csv'),index=False)
                    #   print('')


    ### Output to folders   
    #####################
    clusdir = cpath(direc,f'clusters_{isnorm}_{n_neighbours}/kmeans/{this_svara}/')
    output_clusters(clusdir, new_clus_labels, this_label)
    evalpath = cpath(direc,f'clusters_{isnorm}_{n_neighbours}/kmeans/{this_svara}/report.txt')
    evaluate_clusters(clus_labels, evalpath)

    ### HDBSCAN
    ###########
#    print(f'HDBSCAN for svara: {this_svara}')
#    clus_labels = cluster(X_svara, algo='hdbscan')
#    clus_labels = remove_outliers(clus_labels, X_svara)
#    print(f'  {len(set(clus_labels))} clusters found')
#    for_app = pd.DataFrame(columns=['x', 'y', 'label', 'plot_path'])
#    this_pp = np.array([x for x,l in zip(all_plot_paths, all_labels) if l == this_label])
#    
#    print('Outputting embedded data and clustering...')
#    for i in range(len(X_embedded)):
#        pp = this_pp[i]
#        label = clus_labels[i]
#        d = {
#            'x': [X_embedded[i][0]],
#            'y': [X_embedded[i][1]],
#            'label': [label],
#            'plot_path': [pp],
#            'audio_path': [pp.replace('.png','.wav')],
#        }
#        for_app = pd.concat([for_app, pd.DataFrame(d)], ignore_index=True)

#    for_app.to_csv(cpath(direc,f'clusters_single/hdbscan/embedding_kmeans_{this_svara}.csv'),index=False)

#    ### Output to folders   
#    #####################
#    clusdir = cpath(direc,f'clusters_single/hdbscan/{this_svara}/')
#    output_clusters(clusdir, clus_labels, this_label)
#    evalpath = cpath(direc,f'clusters_single/hdbscan/{this_svara}/report.txt')
#    #evaluate_clusters(clus_labels, evalpath)

### UMAP ALL
############
import umap
print('Embedding all svaras and outputting...')
reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                learning_rate=1,
                metric='euclidean')

X_embedded = reducer.fit_transform(X_norm)

### Output dataframe for visualisation
######################################
for_app = pd.DataFrame(columns=['x', 'y', 'label', 'plot_path'])

for i in range(len(X_embedded)):
    svara = all_svaras[i]
    pp = all_plot_paths[i]
    label = all_labels[i]
    d = {
        'x': [X_embedded[i][0]],
        'y': [X_embedded[i][1]],
        'label': [label],
        'plot_path': [pp],
        'audio_path': [pp.replace('.png','.wav')],
    }
    for_app = pd.concat([for_app, pd.DataFrame(d)], ignore_index=True)

for_app.to_csv(cpath(direc,f'clusters_single/embedding_umap.csv'),index=False)
