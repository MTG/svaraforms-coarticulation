# transpose is current
# with_static is transpose with static features (88% compared to 86.7%)
# context_ablation_norm_ts

# Batch size changed to 80 then 32


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
df['precdir'] = df.apply(lambda y: asc_or_desc(y.prec, y.svara), axis=1)
df['sucdir'] = df.apply(lambda y: asc_or_desc(y.svara, y.suc), axis=1)
df['precsucdir'] = df.apply(lambda y: asc_or_desc(y.prec, y.suc), axis=1)

report_str = ['Context Report for clustering, clusters_True_5']
report_str +=['==============================================\n']

context_d = {s:[] for s in df['cluster']}
for svara in label_lookup.values():
	this_df = df[df['svara']==svara].sort_values(by='cluster')
	clusters = this_df['cluster'].unique()
	report_str += [f'Svara, {svara}']
	report_str += [len(f'Svara, {svara}')*'-']
	for c in clusters:
		that_df = this_df[this_df['cluster']==c]
		n = len(that_df)
		report_str += [f'[{svara}] {c}, population, {n}:']
		
		prec_count = {k:v for k,v in Counter(that_df['prec']).items()}
		prec_perc = {k:round(v*100/n,2) for k,v in sorted(prec_count.items(), key=lambda y:-y[1])}
		prec_count = {k:f'{c} ({prec_perc[k]}%)' for k,c in prec_count.items()}
		for k,v in prec_perc.items():
			if v > 70:
				context_d[c] += [(k, v)]
		report_str += [f'Preceeding:\n    {prec_count}']

		suc_count = {k:v for k,v in Counter(that_df['suc']).items()}
		suc_perc = {k:round(v*100/n,2) for k,v in sorted(suc_count.items(), key=lambda y:-y[1])}
		suc_count = {k:f'{c} ({suc_perc[k]}%)' for k,c in suc_count.items()}
		for k,v in suc_perc.items():
			if v > 70:
				context_d[c] += [(k, v)]

		report_str += [f'Succeedding:\n    {suc_count}']

		precsuc_count = {k:v for k,v in Counter(that_df['precsuc']).items()}
		precsuc_perc = {k:round(v*100/n,2) for k,v in sorted(precsuc_count.items(), key=lambda y:-y[1])}
		precsuc_count = {k:f'{c} ({precsuc_perc[k]}%)' for k,c in precsuc_count.items()}
		for k,v in precsuc_perc.items():
			if v > 70:
				context_d[c] += [(k, v)]

		report_str += [f'Full context:\n    {precsuc_count}']

		suc_count = {k:v for k,v in Counter(that_df['precdir']).items()}
		suc_perc = {k:round(v*100/n,2) for k,v in sorted(suc_count.items(), key=lambda y:-y[1])}
		suc_count = {k:f'{c} ({suc_perc[k]}%)' for k,c in suc_count.items()}
		for k,v in suc_perc.items():
			if v > 70:
				context_d[c] += [(k, v)]

		report_str += [f'Precceeding direction:\n    {suc_count}']

		suc_count = {k:v for k,v in Counter(that_df['sucdir']).items()}
		suc_perc = {k:round(v*100/n,2) for k,v in sorted(suc_count.items(), key=lambda y:-y[1])}
		suc_count = {k:f'{c} ({suc_perc[k]}%)' for k,c in suc_count.items()}
		for k,v in suc_perc.items():
			if v > 70:
				context_d[c] += [(k, v)]

		report_str += [f'Succeedding direction:\n    {suc_count}']

		suc_count = {k:v for k,v in Counter(that_df['precsucdir']).items()}
		suc_perc = {k:round(v*100/n,2) for k,v in sorted(suc_count.items(), key=lambda y:-y[1])}
		suc_count = {k:f'{c} ({suc_perc[k]}%)' for k,c in suc_count.items()}
		for k,v in suc_perc.items():
			if v > 70:
				context_d[c] += [(k, v)]

		report_str += [f'Succeeding/preceeding direction:\n    {suc_count}']

		report_str += ['\n']
	report_str += ['\n']
	report_str += ['\n']

# len([k for k,v in context_d.items() if len(v)==0])

for r in report_str:
	print(r)

with open('report.txt', "a") as f:
    for es in report_str:
        f.write(es)
        f.write('\n')


def asc_or_desc(s1, s2):
	if s2 == 'silence':
		return 'to_silence'
	if s1 == 'silence':
		return 'from_silence'
	if s1 == s2:
		return 'same'

	si = allsvaras.index(s2)
	sort = [allsvaras[(si-3+i) % len(allsvaras)] for i in range(7)]

	si1 = sort.index(s1)
	si2 = sort.index(s2)

	if si2 > si1:
		return 'asc'
	elif si1 > si2:
		return 'desc'

	raise Exception('what?')


from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


for svara in df['svara'].unique():
	print(svara)
	for col in ['suc','prec','precsuc','precdir','sucdir','precsucdir']:
		this_df = df[df['svara']==svara]


		CrosstabResult = pd.crosstab(index=this_df['prec'], columns=this_df['cluster'])

		le = LabelEncoder()
		le.fit(this_df[col].values)

		le.classes_

		x = le.transform(this_df[col].values)

		le = LabelEncoder()
		le.fit(this_df['cluster'].values)

		le.classes_

		y = le.transform(this_df['cluster'].values)

		
		val = mutual_info_classif(x.reshape(-1,1), y, discrete_features=True)[0]
		x_ent = cat_entropy(x)
		y_ent = cat_entropy(y)
		mean_ent = (x_ent + y_ent)/2
		print(f'  {col}: {round(val/mean_ent,3)}')



import scipy
def cat_entropy(x):
	counted = Counter(x)
	counted = np.array([v/sum(counted.values()) for v in counted.values()])
	return scipy.stats.entropy(counted)


import random
from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


for svara in allsvaras:
	this_df = df[df['svara']==svara]

	si = allsvaras.index(svara)

	ticks = ['silence'] + [allsvaras[(si-3+i) % len(allsvaras)] for i in range(7)]

	this_df['yloc'] = this_df['prec'].apply(lambda y: ticks.index(y)+random.uniform(0.1, 0.9))
	this_df['xloc'] = this_df['suc'].apply(lambda y: ticks.index(y)+random.uniform(0.1, 0.9))


	n_labels = this_df['cluster'].nunique()

	colorsl = ['#000000','#006400','#ff0000','#ffd700','#0000cd','#00ff00','#00ffff','#1e90ff','#ff69b4']
	random.shuffle(colorsl)
	clus_label_lookup = {x:i for i,x in enumerate(this_df['cluster'].unique())}
	clabels = this_df['cluster'].apply(lambda y: clus_label_lookup[y]).values

	plt.close('all')
	plt.figure(figsize=(10,10))
	ax = plt.gca()

	plt.xticks(ticks=[x+0.5 for x in range(len(ticks))], labels=ticks)
	plt.yticks(ticks=[x+0.5 for x in range(len(ticks))], labels=ticks)
	plt.xlim((0,len(ticks)))
	plt.ylim((0,len(ticks)))

	plt.axhline(4, linestyle='--', linewidth=0.7, color='red')
	plt.axhline(5, linestyle='--', linewidth=0.7, color='red')
	plt.text(0.1, 5.1, 'Ascending', color='red')
	plt.text(len(ticks)-0.9, 3.8, 'Descending', color='red')

	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.yaxis.set_minor_locator(MultipleLocator(1))
	ax.grid(True, which='minor')
	plt.minorticks_on

	for i,c in enumerate(sorted(this_df['cluster'].unique())):
		plt.scatter(this_df[this_df['cluster']==c]['xloc'], this_df[this_df['cluster']==c]['yloc'], label=c.replace(f'{svara}_','').replace('_',' ').capitalize(), s=15, alpha=0.75, color=colorsl[i])

	plt.legend()
	plt.title(svara)
	plt.ylabel('Preceeding svara')
	plt.xlabel('Succeedding svara')
	plt.savefig(f'{svara}_context.png')
	plt.close('all')
