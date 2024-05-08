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

for di,X in enumerate([X1, X2, X3]):
    data_name = data_names[di]
    
    # normalise
    X_norm = stats.zscore(X, axis=None).astype(float)

    ### PCA 
    #######
    pca = PCA(n_components=0.999)
    pca.fit(X_norm.T)

    print('PCA variance explained ratios...')
    print(pca.explained_variance_ratio_)

    X_pca = pca.components_.T if data_name != 'stat' else X_norm

    ### SVARA DATA
    ##############
    # {0: 'ga', 1: 'ni', 2: 'ri', 3: 'sa', 4: 'pa', 5: 'ma', 6: 'dha'}
    for this_label in label_lookup:

        # Get data for this svara
        this_svara = label_lookup[this_label]
        clust_range = range(8, 18)
        X_svara_pca = np.array([x for x,l in zip(X_pca, all_labels) if l == this_label])
        X_svara = np.array([x for x,l in zip(X_norm, all_labels) if l == this_label])

        # EMBED
        print(f'Embedding svara {this_svara}')
        reducer = umap.UMAP(
                        n_neighbors=5,
                        min_dist=0.1,
                        learning_rate=1,
                        metric='euclidean')

        X_embedded = reducer.fit_transform(X_svara)

        print(f'Embedding svara to 4 components, {this_svara}')
        reducer = umap.UMAP(
                        n_neighbors=5,
                        min_dist=0.05,
                        learning_rate=1,
                        n_components=5,
                        metric='euclidean')

        X_embedded4 = reducer.fit_transform(X_svara)

        ### KMEANS
        ##########
        print(f'K means for svara: {this_svara}')
        all_scores = []
        for c in clust_range:
            print(f'  n_clusters={c}')
            kmeans = KMeans(n_clusters=c, random_state=0, n_init="auto").fit(X_embedded4)
            clus_labels = kmeans.labels_
            score = silhouette_score(X_embedded4, clus_labels)
            print(f'  score={score}')
            all_scores.append(score)

        plt.close('all')
        plt.plot(clust_range, all_scores)
        plt.title(this_svara)
        plt.xlabel('num clusters')
        plt.ylabel('silhouette score')
        plt.savefig(cpath(f'{direc}/clusters/{data_name}/{this_svara}_kmeans_silplot.png'))
        plt.close('all')
        high_score = clust_range[np.argmax(all_scores)]
        print(f'Best score={round(max(all_scores),2)}, n_clusters={high_score}')
        kmeans = KMeans(n_clusters=high_score, random_state=0, n_init="auto").fit(X_embedded4)
        clus_labels = kmeans.labels_

        ### Output dataframe for visualisation
        ######################################
        print('Outputting embedded data and clustering...')
        for_app = pd.DataFrame(columns=['x', 'y', 'label', 'plot_path'])
        this_pp = np.array([x for x,l in zip(all_plot_paths, all_labels) if l == this_label])
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

        for_app.to_csv(cpath(direc,f'clusters/{data_name}/kmeans/embedding_kmeans_{this_svara}.csv'),index=False)
        print('')


        # Cluster characterization
        for clus in set(clus_labels):
            cluschar_path = cpath(direc,f'clusters/{data_name}/kmeans/{this_svara}/cluster_{clus}/')
            svindices = [i for i,l in enumerate(all_labels) if l == this_label]
            clindices = [i for i,l in enumerate(clus_labels) if l == clus]
            this_data = data.iloc[svindices].reset_index(drop=True).iloc[clindices]
            sd_ix = this_data['index'].values.astype(int)
            sd_list = np.array(svara_dict[this_svara])[sd_ix]

            features = {
                'start': [s['start'] for s in sd_list],
                'duration': [s['duration'] for s in sd_list],
                'preceeding_svara': [s['preceeding_svara'] for s in sd_list],
                'succeeding_svara': [s['succeeding_svara'] for s in sd_list],
                'gamaka': [s['gamaka'] for s in sd_list],
                'transposed': [s['transposed'] for s in sd_list],
                'pitch_range': this_data['pitch_range'].values,
                'av_pitch': this_data['av_pitch'].values,
                'min_pitch': this_data['min_pitch'].values,
                'max_pitch': this_data['max_pitch'].values,
                'pitch25': this_data['pitch25'].values,
                'pitch75': this_data['pitch75'].values,
                'av_first_pitch': this_data['av_first_pitch'].values,
                'av_end_pitch': this_data['av_end_pitch'].values,
                'num_change_points_pitch': this_data['num_change_points_pitch'].values,
                'num_change_points_loudness': this_data['num_change_points_loudness'].values,
                'max_loudness': this_data['max_loudness'].values,
                'min_loudness': this_data['min_loudness'].values,
                'loudness75': this_data['loudness75'].values,
                'loudness25': this_data['loudness25'].values,
                'direction_asc': this_data['direction_asc'].values,
                'direction_desc': this_data['direction_desc'].values
            }

            # timeline 
            timeline_path = cpath(cluschar_path, 'timeline.png')
            plt.close('all')
            plt.figure(tight_layout=True)
            plt.ylim((0,10))
            plt.title('Timeline')
            plt.hlines(1,1,1224)  # Draw a horizontal line
            plt.text(1,1.2,'00:00')
            plt.text(1150,1.2,'20:24')
            plt.eventplot(features['start'], orientation='horizontal', colors='b', linewidth=0.5)
            plt.axis('off')
            fig = plt.gcf()
            canvas = fig.canvas
            canvas.draw() 
            image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            width, height = fig.get_size_inches() * fig.get_dpi() 
            img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            img = img[360:-10,]
            im = Image.fromarray(img)
            im.save(timeline_path)
            plt.close('all')

            # Svara plots
            svara_plots_path = cpath(cluschar_path, 'svara_plots.png')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3))
            d = {
                'sa': sum([1 for s in features['preceeding_svara'] if s == 'sa']),
                'ri': sum([1 for s in features['preceeding_svara'] if s == 'ri']),
                'ga': sum([1 for s in features['preceeding_svara'] if s == 'ga']),
                'ma': sum([1 for s in features['preceeding_svara'] if s == 'ma']),
                'pa': sum([1 for s in features['preceeding_svara'] if s == 'pa']),
                'dha': sum([1 for s in features['preceeding_svara'] if s == 'dha']),
                'ni': sum([1 for s in features['preceeding_svara'] if s == 'ni']),
                'none': sum([1 for s in features['preceeding_svara'] if s == 'silence'])
            }
            ax1.bar(d.keys(), d.values(), color='darkgreen')
            ax1.set_title('Preceeding svara')
            ax1.set_ylabel('count')

            d = {
                'sa': sum([1 for s in features['succeeding_svara'] if s == 'sa']),
                'ri': sum([1 for s in features['succeeding_svara'] if s == 'ri']),
                'ga': sum([1 for s in features['succeeding_svara'] if s == 'ga']),
                'ma': sum([1 for s in features['succeeding_svara'] if s == 'ma']),
                'pa': sum([1 for s in features['succeeding_svara'] if s == 'pa']),
                'dha': sum([1 for s in features['succeeding_svara'] if s == 'dha']),
                'ni': sum([1 for s in features['succeeding_svara'] if s == 'ni']),
                'none': sum([1 for s in features['succeeding_svara'] if s == 'silence'])
            }
            ax2.bar(d.keys(), d.values(), color='darkorange')
            ax2.set_title('Succeeding svara')
            ax2.set_ylabel('count')

            d = {
                'kampita': sum([1 for s in features['gamaka'] if s == 'kampita']),
                'jaaru': sum([1 for s in features['gamaka'] if s == 'jaaru']),
                'none': sum([1 for s in features['gamaka'] if s == 'none'])
            }
            ax3.bar(d.keys(), d.values(), color='darkred')
            ax3.set_title('Gamaka')
            ax3.set_ylabel('count')


            plt.savefig(svara_plots_path)
            plt.close('all')

            # feature plots 
            feature_plots_path = cpath(cluschar_path, 'feature_plots.png')
            featcols = [
                'duration', 'pitch_range', 'av_pitch', 
                'min_pitch', 'max_pitch', 'pitch25', 'pitch75', 'av_first_pitch', 'av_end_pitch', 'num_change_points_pitch', 
                'num_change_points_loudness', 'max_loudness', 'min_loudness', 'loudness75', 'loudness25', 'direction_asc', 
                'direction_desc'
            ]

            d = {k:(np.mean(v)-featmeans[k])/featstds[k] for k,v in features.items() if k in featcols}
            val = {k:np.mean(v) for k,v in features.items() if k in featcols}
            plt.figure(figsize=(12,6), tight_layout=True)
            plt.grid(zorder=0)
            bar = plt.barh(list(d.keys()), list(d.values()), zorder=2, color='purple')
            plt.title('Features')
            plt.xlabel('Z-score')
            ax = plt.gca()
            for bar, v in zip(ax.patches, val.values()):
                t = ax.text(min(d.values()), bar.get_y()+bar.get_height()/2, round(v,2), color = 'white', ha = 'left', va = 'center')
                t.set_bbox(dict(facecolor='black', alpha=0.7))

            plt.savefig(feature_plots_path)
            plt.close('all')


        ### HDBSCAN
        ###########
        print(f'HDBSCAN for svara: {this_svara}')
        clus_labels = cluster(X_embedded4, algo='hdbscan')

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

        for_app.to_csv(cpath(direc,f'clusters/{data_name}/hdbscan/embedding_kmeans_{this_svara}.csv'),index=False)

        # Cluster characterization 
        for clus in set(clus_labels):
            if clus != -1:
                cluschar_path = cpath(direc,f'clusters/{data_name}/hdbscan/{this_svara}/cluster_{clus}/')
            else:
                cluschar_path = cpath(direc,f'clusters/{data_name}/hdbscan/{this_svara}/Noise/')
            svindices = [i for i,l in enumerate(all_labels) if l == this_label]
            clindices = [i for i,l in enumerate(clus_labels) if l == clus]
            this_data = data.iloc[svindices].reset_index(drop=True).iloc[clindices]
            sd_ix = this_data['index'].values.astype(int)
            sd_list = np.array(svara_dict[this_svara])[sd_ix]

            features = {
                'start': [s['start'] for s in sd_list],
                'duration': [s['duration'] for s in sd_list],
                'preceeding_svara': [s['preceeding_svara'] for s in sd_list],
                'succeeding_svara': [s['succeeding_svara'] for s in sd_list],
                'gamaka': [s['gamaka'] for s in sd_list],
                'transposed': [s['transposed'] for s in sd_list],
                'pitch_range': this_data['pitch_range'].values,
                'av_pitch': this_data['av_pitch'].values,
                'min_pitch': this_data['min_pitch'].values,
                'max_pitch': this_data['max_pitch'].values,
                'pitch25': this_data['pitch25'].values,
                'pitch75': this_data['pitch75'].values,
                'av_first_pitch': this_data['av_first_pitch'].values,
                'av_end_pitch': this_data['av_end_pitch'].values,
                'num_change_points_pitch': this_data['num_change_points_pitch'].values,
                'num_change_points_loudness': this_data['num_change_points_loudness'].values,
                'max_loudness': this_data['max_loudness'].values,
                'min_loudness': this_data['min_loudness'].values,
                'loudness75': this_data['loudness75'].values,
                'loudness25': this_data['loudness25'].values,
                'direction_asc': this_data['direction_asc'].values,
                'direction_desc': this_data['direction_desc'].values
            }

            # timeline 
            timeline_path = cpath(cluschar_path, 'timeline.png')
            plt.close('all')
            plt.figure(tight_layout=True)
            plt.ylim((0,10))
            plt.title('Timeline')
            plt.hlines(1,1,1224)  # Draw a horizontal line
            plt.text(1,1.2,'00:00')
            plt.text(1150,1.2,'20:24')
            plt.eventplot(features['start'], orientation='horizontal', colors='b', linewidth=0.5)
            plt.axis('off')
            fig = plt.gcf()
            canvas = fig.canvas
            canvas.draw() 
            image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            width, height = fig.get_size_inches() * fig.get_dpi() 
            img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            img = img[360:-10,]
            im = Image.fromarray(img)
            im.save(timeline_path)
            plt.close('all')

            # Svara plots
            svara_plots_path = cpath(cluschar_path, 'svara_plots.png')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3))
            d = {
                'sa': sum([1 for s in features['preceeding_svara'] if s == 'sa']),
                'ri': sum([1 for s in features['preceeding_svara'] if s == 'ri']),
                'ga': sum([1 for s in features['preceeding_svara'] if s == 'ga']),
                'ma': sum([1 for s in features['preceeding_svara'] if s == 'ma']),
                'pa': sum([1 for s in features['preceeding_svara'] if s == 'pa']),
                'dha': sum([1 for s in features['preceeding_svara'] if s == 'dha']),
                'ni': sum([1 for s in features['preceeding_svara'] if s == 'ni']),
                'none': sum([1 for s in features['preceeding_svara'] if s == 'silence'])
            }
            ax1.bar(d.keys(), d.values(), color='darkgreen')
            ax1.set_title('Preceeding svara')
            ax1.set_ylabel('count')

            d = {
                'sa': sum([1 for s in features['succeeding_svara'] if s == 'sa']),
                'ri': sum([1 for s in features['succeeding_svara'] if s == 'ri']),
                'ga': sum([1 for s in features['succeeding_svara'] if s == 'ga']),
                'ma': sum([1 for s in features['succeeding_svara'] if s == 'ma']),
                'pa': sum([1 for s in features['succeeding_svara'] if s == 'pa']),
                'dha': sum([1 for s in features['succeeding_svara'] if s == 'dha']),
                'ni': sum([1 for s in features['succeeding_svara'] if s == 'ni']),
                'none': sum([1 for s in features['succeeding_svara'] if s == 'silence'])
            }
            ax2.bar(d.keys(), d.values(), color='darkorange')
            ax2.set_title('Succeeding svara')
            ax2.set_ylabel('count')

            d = {
                'kampita': sum([1 for s in features['gamaka'] if s == 'kampita']),
                'jaaru': sum([1 for s in features['gamaka'] if s == 'jaaru']),
                'none': sum([1 for s in features['gamaka'] if s == 'none'])
            }
            ax3.bar(d.keys(), d.values(), color='darkred')
            ax3.set_title('Gamaka')
            ax3.set_ylabel('count')


            plt.savefig(svara_plots_path)
            plt.close('all')

            # feature plots 
            feature_plots_path = cpath(cluschar_path, 'feature_plots.png')
            featcols = [
                'duration', 'pitch_range', 'av_pitch', 
                'min_pitch', 'max_pitch', 'pitch25', 'pitch75', 'av_first_pitch', 'av_end_pitch', 'num_change_points_pitch', 
                'num_change_points_loudness', 'max_loudness', 'min_loudness', 'loudness75', 'loudness25', 'direction_asc', 
                'direction_desc'
            ]

            d = {k:(np.mean(v)-featmeans[k])/featstds[k] for k,v in features.items() if k in featcols}
            val = {k:np.mean(v) for k,v in features.items() if k in featcols}
            plt.figure(figsize=(12,6), tight_layout=True)
            plt.grid(zorder=0)
            bar = plt.barh(list(d.keys()), list(d.values()), zorder=2, color='purple')
            plt.title('Features')
            plt.xlabel('Z-score')
            ax = plt.gca()
            for bar, v in zip(ax.patches, val.values()):
                t = ax.text(min(d.values()), bar.get_y()+bar.get_height()/2, round(v,2), color = 'white', ha = 'left', va = 'center')
                t.set_bbox(dict(facecolor='black', alpha=0.7))

            plt.savefig(feature_plots_path)
            plt.close('all')

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

    for_app.to_csv(cpath(direc,f'clusters/{data_name}/embedding_umap.csv'),index=False)
