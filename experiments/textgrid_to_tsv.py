### list of annontations files

### Load each, tonic normalise pitch extraction

### 

import pandas as pd
import textgrid

inpath = 'data/annotation/chintayama_kanda.TextGrid'
outpath = 'data/annotation/chintayama_kanda.txt'

tg = textgrid.TextGrid.fromFile(inpath)
svaras = tg[0]

df = pd.DataFrame(columns=['level', 't1', 't2', 'duration', 'annotation'])
for row in svaras:
	t1 = row.minTime
	t2 = row.maxTime
	annotation = row.mark
	duration = t2-t1
	level = 'svara'
	df = df.append({
		'level':level,
		't1':t1,
		't2':t2,
		'duration':duration,
		'annotation':annotation,
		}, ignore_index=True)

df.to_csv(outpath, sep='\t', header=False)