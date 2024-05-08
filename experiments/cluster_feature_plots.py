# - Feature plots
# 	- number of change points
# 	- preceeding succeeding pitch
# 	- descending/ascending
# 	- duration
# 	- confusion matrix

# The most relevant features for my discussion are:
	# duration, 
	# actual preceding pitch last 10%, 
	# actual succeeding pitch first 10%, 
	# svara name before, svara name after, 
	# number of pitch change points, 
	# overall pitch range,
	# min max pitch

import matplotlib.pyplot as plt

# Get means and stds
feature = 'duration'
feat_i = feature_names.index(feature)
this_ix = [i for i,l in enumerate(all_labels) if l == this_label]
this_static = np.array([pp for i,pp in enumerate(all_static) if i in this_ix])

unique_clusters = set([x for x in new_clus_labels if x != -1])
feat_d = {}
for clus in unique_clusters:
	
	clus_ix = [i for i,x in enumerate(new_clus_labels) if x == clus]
	clus_static = this_static[clus_ix]
	clus_feat = [s[feat_i] for s in clus_static]

	feat_d[clus] = np.mean(clus_feat), np.std(clus_feat), len(clus_ix)

# Plot histograms

# mere numerical grid to plot densities (no statistical significance)
x_ = np.linspace(0.0, max([x[0] for x in feat_d.values()])+max([x[1] for x in feat_d.values()]), 1000)

# estimate mean (mu) and standard deviation (sigma) for each column
plt.close('all')
plt.figure(figsize=(10,5))
# histograms
for clus in unique_clusters:
    plt.plot(x_, stats.norm.pdf(x_, loc=feat_d[clus][0], scale=feat_d[clus][1]), label='Cluster {}'.format(cluster), linewidth=0.75)

plt.title(f'Probability Densities for {feature}')
plt.xlabel(f'{feature}')
plt.savefig('test.png')
plt.close('all')




# Get preceeding and succeeding svara
# Get means and stds
feat_i = feature_names.index(feature)
this_ix = [i for i,l in enumerate(all_labels) if l == this_label]
this_static = np.array([pp for i,pp in enumerate(all_static) if i in this_ix])

unique_clusters = set([x for x in new_clus_labels if x != -1])
feat_d = {}
for clus in unique_clusters:
	
	clus_ix = [i for i,x in enumerate(new_clus_labels) if x == clus]
	clus_static = this_static[clus_ix]
	clus_feat = [s[feat_i] for s in clus_static]

	feat_d[clus] = np.mean(clus_feat), np.std(clus_feat), len(clus_ix)
























# how many in each cluster share 
# a) both the preceding and following svara, and 
# b) either the same preceding or following svara. 
# Also we could look at general direction before and after the svara, which could mean something like 
# c) is the svara preceded by a svara either above or below it 
# d) is the svara followed by a svara that is either above or below it

