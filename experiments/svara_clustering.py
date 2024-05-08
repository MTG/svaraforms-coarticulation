%load_ext autoreload
%autoreload 2

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

from numpy.linalg import norm
from numpy import dot

from src.dtw import dtw
from src.tools import compute_novelty_spectrum, get_loudness, interpolate_below_length, get_derivative, chaincode
from src.utils import load_audacity_annotations, load_elan_annotations, cpath,  write_pkl, write_pitch_track, load_pitch_track, read_txt, load_pkl
from src.visualisation import get_plot_kwargs, plot_subsequence, get_arohana_avarohana
from src.pitch import pitch_seq_to_cents
from src.svara import get_svara_dict, get_unique_svaras, pairwise_distances_to_file
from scipy.signal import savgol_filter
from src.clustering import duration_clustering, cadence_clustering, hier_clustering

out_dir = cpath('data', 'short_test')

track = 'kamakshi'
raga = 'bhairavi'


data_home = '/Users/thomasnuttall/code/mir_datasets/saraga1.5_carnatic/'
saraga = mirdata.initialize('saraga_carnatic', data_home=data_home)

#carnatic = compiam.load_dataset('saraga_carnatic')
tracks = saraga.load_tracks()


###########
# Get paths
###########
mridangam_left_path = tracks[track].audio_mridangam_left_path
mridangam_right_path = tracks[track].audio_mridangam_right_path
#vocal_path = tracks[track].audio_vocal_path
tonic_path = tracks[track].ctonic_path
sections_path = tracks[track].sections_path
tempo_path = tracks[track].tempo_path
annotations_path = os.path.join('data', 'annotation', f'{track}.txt')

vocal_path = 'data/audio/Kamakshi.mp3'

###########
# Load Data
###########
sr = 44100
mridangam_left, _ = librosa.load(mridangam_left_path, sr=sr)
mridangam_right, _ = librosa.load(mridangam_right_path, sr=sr)
vocal, _ = librosa.load(vocal_path, sr=sr)

sections = pd.read_csv(sections_path, sep='\t', header=None)
sections.columns = ['start', 'group', 'duration', 'section']
del sections['group']

tempo = read_txt(tempo_path) #tempo, matra_interval, sama_interval, matras_per_cycle, start_time, end_time
tonic = read_txt(tonic_path)
tonic = float(tonic[0].replace('\n',''))
plot_kwargs = get_plot_kwargs(raga, tonic)
yticks_dict = plot_kwargs['yticks_dict']
aro, avaro = get_arohana_avarohana(raga)

# Load annotations
#annotations = load_audacity_annotations(annotations_path)

annotations = load_elan_annotations(annotations_path)

##################
# Extract Features
##################
# Pitch track -> change points
ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
pitch_track_path = cpath('data', 'pitch_tracks', f'{track}.tsv')
#ftanet_pitch_track = ftanet_carnatic.predict(vocal_path,hop_size=30)
#write_pitch_track(ftanet_pitch_track, pitch_track_path, sep='\t')
ftanet_pitch_track = load_pitch_track(pitch_track_path)


pitch = ftanet_pitch_track[:,1]
time = ftanet_pitch_track[:,0]
timestep = time[3]-time[2]

pitch = interpolate_below_length(pitch, 0, (250*0.001/timestep))
null_ind = pitch==0

pitch[pitch<50]=0
pitch[null_ind]=0

pitch_cents = pitch_seq_to_cents(pitch, tonic)


#####################
# Get Svaras Features
#####################
svara_dict_path = cpath(out_dir, 'data', 'svara_dict', f'{track}.csv')
unique_svaras = get_unique_svaras(annotations)

svara_dict = get_svara_dict(annotations, pitch_cents, timestep, track, path=None)


#####
# DTW
#####
distance_paths = {}
for svara in unique_svaras:
    print(f'Computing distances for {svara}')
    all_svaras = svara_dict[svara]
    all_ix = list(range(len(all_svaras)))
    dtw_distances_path = cpath(out_dir, 'data', 'dtw_distances', f'{track}', f'{svara}.csv')
    pairwise_distances_to_file(all_ix, all_svaras, dtw_distances_path, mean_norm=True)
    distance_paths[svara] = dtw_distances_path


#############
## Clustering
#############
wl = round(145*0.001/timestep)
wl = wl if not wl%2 == 0 else wl+1

min_samples = 1 # duration min samp
eps = 0.05 # duration epsilon

t = 1 # hierarchical clustering t
min_in_group = 1 # min in group for hierarchical

plot = False # plot final clusters?

cluster_dict = {}
for svara, sd in svara_dict.items():
    print(f'Duration clustering, {svara}')
    cluster_dict[svara] = duration_clustering(sd, eps=0.05)
    print(f'    {len(cluster_dict[svara])} clusters')

# Force Sa and Pa to be uniform
for svara,c_val in [('sa',0), ('pa',700)]:
    print(f'Removing gamaka from {svara}')
    sd = cluster_dict[svara]
    new_sd = []
    for s1 in sd:
        new_s1 = []
        for n,s2 in s1:
            this_dict = {
                'pitch': np.array([c_val for s in s2['pitch']]),
                'track': s2['track'],
                'start': s2['start'],
                'end': s2['end'],
                'duration': s2['duration'],
                'annotation_index': s2['annotation_index'],
                'preceeding_svara': s2['preceeding_svara'],
                'succeeding_svara': s2['succeeding_svara'],
                'gamaka': 'none'
            }
            new_s1.append((n, this_dict))
        new_sd.append(new_s1)
    cluster_dict[svara] = new_sd


for svara, clusters in cluster_dict.items():
    if svara in ['sa','pa']:
        continue
    print(f'Cadence clustering, {svara}')
    new_clusters = []
    for cluster in clusters:
        cadclust = cadence_clustering(cluster, svara)
        new_clusters += cadclust
    cluster_dict[svara] = new_clusters
    print(f'    {len(cluster_dict[svara])} clusters')


for svara, clusters in cluster_dict.items():
    if svara in ['sa','pa']:
        continue
    print(f'Hierarchical clustering, {svara}')
    distance_path = distance_paths[svara]
    distances = pd.read_csv(distance_path)
    distances_flip = distances.copy()
    distances_flip.columns = ['index2', 'index1', 'distance']
    distances = pd.concat([distances, distances_flip])
    new_clusters = []
    for cluster in clusters:
        hier = hier_clustering(cluster, distances, t=t, min_in_group=min_in_group)
        if hier: # hier can return empty array if min_in_group > 1
            new_clusters += hier
    cluster_dict[svara] = new_clusters
    print(f'    {len(cluster_dict[svara])} clusters')


if plot:
    for svara, clusters in cluster_dict.items():
        print(f'Plotting {svara} cluster...')
        for i,cluster in tqdm.tqdm(list(enumerate(clusters))):
            for j,sd in cluster:
                p = sd['pitch']
                gamaka = sd['gamaka']
                t = [x*timestep for x in range(len(p))]
                
                plt.plot(t,p)
                plt.xlabel('Time (s)')
                plt.ylabel('Pitch (cents)')
                path = cpath(out_dir, 'plots', f'{track}', 'clustering', f'{svara}', f'cluster_{i}', f'{gamaka}_{j}.png')
                plt.savefig(path)
                plt.close('all')


#########################
## Pick cluster candidate
#########################
final_clusters = {}
for svara, clusters in cluster_dict.items():
    print(f'Reducing {svara} clusters to 1 candidate')
    clusts = []
    for c in clusters:
        clusts.append(random.choice(c)[1])
    final_clusters[svara] = clusts


########################
## Get Distance profiles
########################
sample = 30

if sample:
    sample_pitch_cents = pitch_cents[:round(sample/timestep)]
else:
    sample_pitch_cents = pitch_cents

distance_profiles = {}
for svara, clusters in final_clusters.items():
    print(f'Computing {svara} distance profile')
    distance_profiles[svara] = {i:[] for i in range(len(clusters))}
    for c, s in enumerate(clusters):
        print(f'Cluster {c}')
        unit = s['pitch']
        for i in tqdm.tqdm(list(range(len(sample_pitch_cents)))):
            target = sample_pitch_cents[i:i+len(unit)]
            if len(target) < wl:
                break
            target = savgol_filter(target, polyorder=2, window_length=wl, mode='interp')

            if np.isnan(target).any():
                distance_profiles[svara][c].append(np.Inf)
                continue

            pi = len(target)
            pj = len(unit)
            l_longest = max([pi, pj])

            path, dtw_val = dtw(target, unit, radius=round(l_longest*0.05))
            l = len(path)
            dtw_norm = dtw_val/l
            distance_profiles[svara][c].append(dtw_norm)

sample_str = f"_{sample}" if sample else ""

dp_path = cpath(out_dir, 'data', 'distance_profiles', f'{track}{sample_str}.pkl')
write_pkl(distance_profiles, dp_path)

dp_path = cpath(out_dir, 'data', 'distance_profiles', f'{track}{sample_str}.pkl')
distance_profiles = load_pkl(dp_path)

###########
## Annotate
###########
annot_dist_thresh = 5

occurences = []
distances = []
lengths = []
labels = []
gamakas = []
for s,ap in distance_profiles.items():
    print(f'Annotating svara, {s}')
    for ix,i in enumerate(ap):
        gamaka = svara_dict[s][ix]['gamaka']
        dp = np.array(ap[i]).copy()

        max_dist = 0
        while max_dist < annot_dist_thresh:
            
            l = len(final_clusters[s][i]['pitch'])
            
            ix = dp.argmin()
            dist = dp[ix]
            
            dp[max(0,int(ix-l)):min(len(dp),int(ix+l))] = np.Inf

            distances.append(dist)
            lengths.append(l)
            occurences.append(ix)
            labels.append(s)
            gamakas.append(gamaka)

            max_dist = dist

occurences = np.array(occurences)
distances = np.array(distances)
lengths = np.array(lengths)
labels = np.array(labels)
gamakas = np.array(gamakas)

###########
## Clean Up
###########
def reduce_labels(occurences, labels, lengths, distances, gamakas, timestep, ovl=0.5, chunk_size_seconds=0.1):
    chunk_size = round(chunk_size_seconds/timestep)
    chunkovl = chunk_size*ovl
    starts = occurences
    ends = starts + lengths
    l_track = ends.max()

    occs = []
    lens = []
    gams = []
    dists = []
    labs = []
    for i1 in np.arange(0, l_track, chunk_size):
        i2 = i1 + chunk_size

        # start before end during
        sbed = ((starts <= i1) & (ends >= i1) & (ends <= i2))
        # occupy at least <ovl> of the chunk?
        oalo1 = sbed & (ends - i1 > chunkovl)
        
        # start during end after
        sdea = ((starts >= i1) & (starts <= i2) & (ends >= i2))
        # occupy at least <ovl> of the chunk?
        oalo2 = sdea & (abs(starts - i2) > chunkovl)

        # start during end during
        sded = ((starts >= i1) & (starts <= i2) & (ends >= i1) & (ends <= i2))
        # occupy at least <ovl> of the chunk?
        oalo3 = sded & (starts - ends > chunkovl)

        # start before end after
        sbea = ((starts <= i1) & (ends >= i2))

        # occupy at least <ovl> of the chunk?
        oalo = oalo1 | oalo2 | oalo2

        all_options = np.where(sbed | sdea | sded | sbea | oalo)[0]

        if len(all_options) > 0:
            # For chunk select svara with lowest distance
            winner = np.argmin(distances[all_options])
            winner_ix = all_options[winner]

            occs.append(i1)
            lens.append(chunk_size)
            gams.append(gamakas[winner_ix])
            dists.append(distances[winner_ix])
            labs.append(labels[winner_ix])

    return np.array(occs), np.array(labs), np.array(lens), np.array(dists), np.array(gams)


def join_neighbouring_svaras(occurences, labels, lengths, distances, gamakas, include_gamaka=True):
    # make sure in chronological order
    ix = sorted(range(len(occurences)), key=lambda y: occurences[y])
    occs = occurences[ix]
    lens = lengths[ix]
    dist = distances[ix]
    gams = gamakas[ix]
    labs = labels[ix]

    batches = [[0]]
    for i,_ in enumerate(occurences[1:],1):
        
        o1 = occurences[i]
        o2 = o1 + lens[i]
        ol = labs[i]
        og = gams[i]

        p1 = occurences[i-1]
        p2 = p1 + lens[i-1]
        pl = labs[i-1]
        pg = gams[i-1]

        gcheck = (pg == og) if include_gamaka else True

        if (p2 == o1) and (pl == ol) and (pg == og) and gcheck:
            # append to existing batch
            batches[-1].append(i)
        else:
            # create new batch
            batches.append([i])
    

    j_occs = []
    j_lens = []
    j_dist = []
    j_gams = []
    j_labs = []
    for batch in batches:
        j_occs.append(occs[batch].min())
        j_lens.append(lens[batch].sum())
        j_dist.append(dist[batch].mean())
        j_gams.append(gams[batch][0])
        j_labs.append(labs[batch][0])

    return np.array(j_occs), np.array(j_labs), np.array(j_lens), np.array(j_dist), np.array(j_gams)


def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found

        
occs, labs, lens, dists, gams = reduce_labels(occurences, labels, lengths, distances, gamakas, timestep, chunk_size_seconds=0.05)
occs, labs, lens, dists, gams = join_neighbouring_svaras(occs, labs, lens, dists, gams)

gams = [gams[i] if l not in ['sa','pa'] else 'none' for i,l in enumerate(labs)]
#occs, labs, lens, dists, gams = occurences, labels, lengths, distances, gamakas


# Check arohana and avorahana errors 
def check_trio_error(trio, aro, avaro, n=1):

    succ = trio[1:]
    prec = trio[:2]

    # successive svara does not obey either arohana or avarahana
    c1 = search_sequence_numpy(aro, succ)
    c2 = search_sequence_numpy(avaro, succ)
    c3 = succ[0] == succ[1] # same svara
    if not any([len(c1), len(c2), c3]):
        return True
    
    # preceeding svara does not obey either arohana or avorahana
    c4 = search_sequence_numpy(aro, prec)
    c5 = search_sequence_numpy(avaro, prec)
    c6 = prec[0] == prec[1] # same svara
    if not any([len(c4), len(c5), c6]):
        return True

    return False


def replace_svara_txt(s):
    if 'S' in s:
        return 'sa'
    if 'R' in s:
        return 'ri'
    if 'G' in s:
        return 'ga'
    if 'M' in s:
        return 'ma'
    if 'P' in s:
        return 'pa'
    if 'D' in s:
        return 'dha'
    if 'N' in s:
        return 'ni'

aro_txt = [replace_svara_txt(x) for x in aro]
avaro_txt = [replace_svara_txt(x) for x in avaro]

# Check errors
errors = []
for i,_ in enumerate(occs[1:-1],1):
    trio_ix = [i-1, i, i-2]
    
    ls = labs[trio_ix]
    gs = gams[trio_ix]

    e = check_trio_error(ls, aro_txt, avaro_txt)

    if e:
        print(ls)
        errors.append(i)



# Check ornamented Sa and Pa
errors = []
for i,_ in enumerate(occs):
    l = labs[i]
    g = gams[i]
    
    if l in ['sa', 'pa']:
        



#########
## Export
#########
starts = [o*timestep for o in occs]
ends   = [starts[i]+(lens[i]*timestep) for i in range(len(starts))]
transcription = pd.DataFrame({
        'start':starts,
        'end':ends,
        'label':[f"{l} ({g})" for l,g in zip(labs, gams)],
    }).sort_values(by='start')

trans_path = cpath(out_dir, 'data', 'transcription', f'{track}.txt')
transcription.to_csv(trans_path, index=False, header=False, sep='\t')

#  - Clean up
#      - dtw distances between them all to trim
#      - remove silences
#  - Characterize svaras
#      - cluster into profiles based on features
#      - what are the distinguishing characteristics
#      - Markov chains
#  - Query track with MP style approach to identify other occurrences
#  - Clean up annotations with raga grammar
#  - Clean annotations with onsets from spectral flux?
#  - Which characteristics are due to the pronunciation and which due to the melody?
#  - Evaluate somehow?
