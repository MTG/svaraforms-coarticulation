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


### Feature Data
################
analysis_name = 'bhairaviTransposed'
model_path = '../DeepGRU/models/best/transpose'

direc = f'data/analysis/{analysis_name}'
pldirec = f'data/analysis/{analysis_name}'
raga = 'bhairavi'

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

feature_names = [c for c in data.columns if c not in ['label', 'track', 'svara', 'index', 'timestep']]
contextual_features = ['prec_pitch_range', 'av_prec_pitch', 'min_prec_pitch', 'max_prec_pitch', 'prec_pitch75', 'prec_pitch25', 
                              'succ_pitch_range', 'av_succ_pitch', 'min_succ_pitch', 'max_succ_pitch', 'succ_pitch75', 'succ_pitch25']

# remove contextual features
feature_names = [x for x in feature_names if x not in contextual_features]

featmeans = {k:np.mean(data[k].values) for k in feature_names}
featstds = {k:np.std(data[k].values) for k in feature_names}

### Model
#########
import sys
sys.path.append('../DeepGRU')
from dataset.datafactory import DataFactory
from model import DeepGRU
import torch
from utils.logger import log

dataset_name = 'bhairavi'
seed = 1570254494


# Load model
log.set_dataset_name(dataset_name)
dataset = DataFactory.instantiate(dataset_name, root=f'../DeepGRU/data/{analysis_name}')
hyperparameters = dataset.get_hyperparameter_set()

# Instantiate the model, loss measure and optimizer
model = DeepGRU(dataset.num_features, dataset.num_classes, dataset.static_features.shape[1], with_static=False)

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
model.load_state_dict(state_dict)

_, _ = dataset.get_data_loaders(0, shuffle=True, random_seed=seed+0, normalize_ts=True, normalize_static=True)

avg_ts = dataset.avg_ts[0]
std_ts = dataset.std_ts[0]

avg_static = dataset.avg_static
std_static = dataset.std_static

model.eval()
### Embed time series
#####################
import numpy as np

def seq_to_tensor(seq):
    res = np.reshape(seq, (-1, 1))
    return torch.from_numpy(res).float().detach()

def tensorize_pitch(samp, padding_len=700):

    l = len(samp)
    
    assert l <= padding_len, "sample must be shorter than padding len"
    
    samp = np.concatenate([samp, np.zeros(padding_len - samp.shape[0])])

    samp_t = torch.from_numpy(np.array([seq_to_tensor(samp)]))

    padded = torch.nn.utils.rnn.pad_sequence(samp_t, batch_first=True)

    return padded

all_pitch = np.concatenate([sd['pitch'] for all_sd in svara_dict.values() for sd in all_sd])
pitch_avg = np.mean(all_pitch)
pitch_std = np.std(all_pitch)

# transform time series
tensors = {s:{} for s in svara_dict if s != 'none'}
for svara, sv_sd in svara_dict.items():
    for ni,sd in enumerate(sv_sd):
        if svara == 'none':
            continue
        pitch = sd['pitch']
        # normalize
        pitch = (pitch - pitch_avg)/pitch_std
        tensor = tensorize_pitch(pitch)
        l = torch.from_numpy(np.array([[len(pitch)]]))
        tensors[svara][ni] = (tensor, l)

# embed
import tqdm
from sklearn.decomposition import PCA
import hdbscan

embeddings = {s:{} for s in tensors}
for svara, ts_dict in tensors.items():
    print(f"Embedding {svara}'s")
    for ni,(t,l) in tqdm.tqdm(ts_dict.items()):
        emb = model.embed(t, l).detach().numpy()
        embeddings[svara][ni] = emb[0]


### Combine Embeddings with Static Features
###########################################
from sklearn.cluster import KMeans
import umap
from sklearn.metrics import silhouette_score

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

all_embds = []
all_static = []
all_labels = []
all_indices = []
all_svaras = []
all_plot_paths = []

for i in range(len(svara_data)):
    this_row = svara_data.iloc[i]
    svara = this_row['svara']
    ix = this_row['index']
    label = this_row['label']
    features = this_row[feature_names]
    embds = embeddings[svara][ix]
    
    pp = plots_paths[
        (plots_paths['svara']==svara) & \
        (plots_paths['index']==ix)]['plot_path'].values[0]

    all_embds.append(embds)
    all_static.append(features)
    all_labels.append(label)
    all_indices.append(ix)
    all_svaras.append(svara)
    all_plot_paths.append(pp)

all_embds = np.array(all_embds)
all_static = np.array(all_static)

X1 = np.concatenate([all_embds, all_static], axis=1).astype(float)
X2 = all_embds.astype(float)
X3 = all_static.astype(float)
data_names = ['embstat', 'emb', 'stat']

X = X2

# normalise
X_norm = stats.zscore(X, axis=None).astype(float)

inertia_elbows = {
    'sa':10,
    'ri':12,
    'ga':13,
    'ma':10,
    'pa':14,
    'dha':12,
    'ni': 10
}

def output_clusters(clusdir, clus_labels):
    for clus in set(clus_labels):
        cluscharpath = cpath(clusdir, f'cluster_{clus}/')
        svindices = [i for i,l in enumerate(all_labels) if l == this_label]
        clindices = [i for i,l in enumerate(clus_labels) if l == clus]
        this_data = data.iloc[svindices].reset_index(drop=True).iloc[clindices]
        plots_paths[plots_paths['svara']==this_svara]
        clus_pps = plots_paths[(plots_paths['svara']==this_svara) & (plots_paths['index'].isin(this_data['index']))]['plot_path'].values
        sd_ix = this_data['index'].values.astype(int)
        sd_list = np.array(svara_dict[this_svara])[sd_ix]

        for pp in clus_pps:
            ap = pp.replace('.png', '.wav').replace('_TRANSPOSED','')
            # 2nd option
            shutil.copy(pp, cluscharpath)
            shutil.copy(ap, cluscharpath)

from collections import Counter
def evaluate_clusters(clus_labels, path):
    eval_str = []
    winners_prec = []
    winners_suc = []
    winners_precsuc = []
    for clus in set(clus_labels):
        svindices = [i for i,l in enumerate(all_labels) if l == this_label]
        clindices = [i for i,l in enumerate(clus_labels) if l == clus]
        this_data = data.iloc[svindices].reset_index(drop=True).iloc[clindices]
        eval_str.append('\n')
        eval_str.append(f'Cluster {clus}, population: {len(this_data)}...\n')
        plots_paths[plots_paths['svara']==this_svara]
        clus_pps = plots_paths[(plots_paths['svara']==this_svara) & (plots_paths['index'].isin(this_data['index']))]['plot_path'].values
        sd_ix = this_data['index'].values.astype(int)
        sd_list = np.array(svara_dict[this_svara])[sd_ix]

        prec_suc = [(sd['preceeding_svara'], sd['succeeding_svara']) for sd in sd_list]
        prec_suc_comb = [f'{x[0]}-{x[1]}' for x in prec_suc]
        ps_c = Counter(prec_suc_comb)
        prec = [x[0] for x in prec_suc]
        p_c = Counter(prec)
        suc = [x[1] for x in prec_suc]
        s_c = Counter(suc)
        eval_str.append('   Full Context:\n')
        eval_str.append('   -------------\n')
        for l,v in ps_c.items():
            eval_str.append(f'      {l}: {v} ({round(v*100/len(this_data), 2)}%)\n')
        eval_str.append('   Preceeding Svara:\n')
        eval_str.append('   -----------------\n')
        for l,v in p_c.items():
            eval_str.append(f'      {l}: {v} ({round(v*100/len(this_data), 2)}%)\n')
        eval_str.append('   Succeeding Svara:\n')
        eval_str.append('   ----------------\n')
        for l,v in s_c.items():
            eval_str.append(f'      {l}: {v} ({round(v*100/len(this_data), 2)}%)\n')

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

def remove_outliers(clus_labels):
    new_clus_labels = copy.deepcopy(clus_labels)
    clusters = []
    for clus in set(clus_labels):
        clindices = np.array([i for i,l in enumerate(clus_labels) if l == clus])
        embs = X_embedded4[clindices]

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

        new_clus_labels[clindices[bad_indices]] = -1
    
    return new_clus_labels




### SVARA DATA
##############
import shutil
# {0: 'ga', 1: 'ni', 2: 'ri', 3: 'sa', 4: 'pa', 5: 'ma', 6: 'dha'}
params = {
        'sa': {'nn':5,'md':0.1},
        'ri': {'nn':6,'md':0.1},
        'ga': {'nn':10,'md':0.2},
        'ma': {'nn':7,'md':0.1},
        'pa': {'nn':5,'md':0.1},
        'dha':{'nn':5,'md':0.1},
        'ni': {'nn':5,'md':0.1}
    }
for this_label in label_lookup:

    # Get data for this svara
    this_svara = label_lookup[this_label]
    n_clusters = inertia_elbows[this_svara]
    clust_range = range(8, 30)
    X_svara = np.array([x for x,l in zip(X_norm, all_labels) if l == this_label])

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
                    n_components=5,
                    metric='euclidean')

    X_embedded4 = reducer.fit_transform(X_svara)

    ### KMEANS
    ##########
#    print(f'K means for svara: {this_svara}')
#    all_scores = []
#    for c in clust_range:
#        print(f'  n_clusters={c}')
#        kmeans = KMeans(n_clusters=c, random_state=0, n_init="auto").fit(X_embedded4)
#        clus_labels = kmeans.labels_
#        score = kmeans.inertia_#silhouette_score(X_embedded4, clus_labels)
#        print(f'  score={score}')
#        all_scores.append(score)

#    plt.close('all')
#    plt.plot(clust_range, all_scores)
#    plt.title(this_svara)
#    plt.xlabel('num clusters')
#    plt.ylabel('silhouette score')
#    plt.savefig(cpath(f'{direc}/clusters_single/{this_svara}_kmeans_silplot.png'))
#    plt.close('all')
   # high_score = clust_range[np.argmax(all_scores)]
  #  print(f'Best score={round(max(all_scores),2)}, n_clusters={high_score}')
                    #   kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_embedded4)
                    #   clus_labels = kmeans.labels_

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
                    #   clusdir = cpath(direc,f'clusters_single/kmeans/{this_svara}/')
                    #   output_clusters(clusdir, clus_labels)
                    #   evalpath = cpath(direc,f'clusters_single/kmeans/{this_svara}/report.txt')
                    #   evaluate_clusters(clus_labels, evalpath)

    ### HDBSCAN
    ###########
    print(f'HDBSCAN for svara: {this_svara}')
    clus_labels = cluster(X_svara, algo='hdbscan')
    clus_labels = remove_outliers(clus_labels)
    print(f'  {len(set(clus_labels))} clusters found')
    for_app = pd.DataFrame(columns=['x', 'y', 'label', 'plot_path'])
    this_pp = np.array([x for x,l in zip(all_plot_paths, all_labels) if l == this_label])
    
    print('Outputting embedded data and clustering...')
    for i in range(len(X_embedded)):
        pp = this_pp[i]
        label = clus_labels[i]
        d = {
            'x': [X_embedded[i][0]],
            'y': [X_embedded[i][1]],
            'label': [label],
            'plot_path': [pp],
            'audio_path': [pp.replace('.png','.wav')],
        }
        for_app = pd.concat([for_app, pd.DataFrame(d)], ignore_index=True)

    for_app.to_csv(cpath(direc,f'clusters_single/hdbscan/embedding_kmeans_{this_svara}.csv'),index=False)

    ### Output to folders   
    #####################
    clusdir = cpath(direc,f'clusters_single/hdbscan/{this_svara}/')
    output_clusters(clusdir, clus_labels)
    evalpath = cpath(direc,f'clusters_single/hdbscan/{this_svara}/report.txt')
    evaluate_clusters(clus_labels, evalpath)

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
