import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd 
from sklearn.manifold import TSNE
from scipy import stats

from src.utils import cpath, load_pkl
from src.visualisation import get_plot_kwargs


direc = 'data/analysis/bhairaviFullFeature'
raga = 'bhairavi'

plots_dir = os.path.join(direc, 'plots', '')

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

feature_means = data[feature_names].mean().values
feature_std = data[feature_names].std().values

X = svara_data[feature_names].values
labels = svara_data['label'].values

X_mean = stats.zscore(X, axis=None).astype(float)

X_embedded = TSNE(n_components=2, init='random', perplexity=10).fit_transform(X_mean)

colours = ['red', 'green', 'blue', 'purple', 'black', 'pink', 'yellow', 'orange']

plt.close('all')
plt.figure(figsize=(10,10))
ax = plt.gca()
plt.title('T-SNE on all svaras feature vector')
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap=ListedColormap(colours), s=8, alpha=0.8)
ax.set_facecolor('#e6e6e6')

recs = []
for i in range(0,len(colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=colours[i]))

plt.legend(recs, [label_lookup[l] for l in sorted(set(labels))], loc=4)

plt.savefig(cpath(plots_dir, 'tsne_all.png'))
plt.close('all')



for_app = pd.DataFrame(columns=['x', 'y', 'label', 'plot_path'])

for i in range(len(svara_data)):
	svara = svara_data.iloc[i]['svara']
	ix = svara_data.iloc[i]['index']
	pp = plots_paths[(plots_paths['svara']==svara) & (plots_paths['index']==ix)]['plot_path'].values[0]
	d = {
		'x': X_embedded[i][0],
		'y': X_embedded[i][1],
		'label': labels[i],
		'plot_path': pp
	}
	for_app = for_app.append(d, ignore_index=True)

for_app.to_csv(cpath(direc,'tsne_data.csv'),index=False)
