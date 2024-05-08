import os

import pandas as pd 
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from src.utils import cpath, load_pkl, myround
from src.visualisation import get_plot_kwargs

direc = 'data/analysis/bhairaviTransposedIndex'
raga = 'bhairavi'

plots_dir = os.path.join(direc, 'plots', '')

features_path = os.path.join(direc, 'data.csv')
svara_dict_path = os.path.join(direc, 'svara_dict.pkl')
label_lookup_path = os.path.join(direc, 'label_lookup.pkl')

data = pd.read_csv(features_path)
svara_dict = load_pkl(svara_dict_path)
label_lookup = load_pkl(label_lookup_path)

data['av_start_pitch'] = data['av_first_pitch']

feature_names = ['duration', 'av_pitch', 'min_pitch', 'max_pitch', 'pitch_range', 'av_start_pitch', 'av_end_pitch', 'num_change_points_pitch']
feature_names.reverse()


plot_path_df = pd.DataFrame(columns=['svara', 'index', 'plot_path'])

sr = 44100
vocal_path = os.path.join('data','audio', f'kamakshi.mp3')
vocal, _ = librosa.load(vocal_path, sr=sr)

###############################
### PLOTTING PITCH AND FEATURES
###############################
for svara, sv_sd in svara_dict.items():
    feature_means = data[data['svara']==svara][feature_names].mean().values
    feature_std = data[data['svara']==svara][feature_names].std().values
    for ni,sd in enumerate(sv_sd):

        ## Plot svara
        tonic = sd['tonic']
        pitch = sd['pitch']
        start = sd['start']
        end = sd['end']
        track = sd['track']
        timestep = sd['timestep']           
        succ = sd['succeeding_svara']
        succ = succ if succ else 'none'
        prec = sd['preceeding_svara']
        prec = prec if prec else 'none'
        transposed = sd['transposed']

        if track == 'chintayama_kamnda':
            continue
        
        ran = max(pitch)-min(pitch)
        
        # exclude incorrectly extracted time series
        if ran > 1000:
            continue

        time = [x*timestep+start for x in range(len(pitch))]

        plot_kwargs = get_plot_kwargs(raga, tonic, cents=True)
        yticks_dict = plot_kwargs['yticks_dict']

        yticks_dict = {k:v for k,v in yticks_dict.items() if v<=myround(max(pitch),100)+200 and v>=myround(min(pitch))-200}
        tick_names = list(yticks_dict.keys())
        tick_loc = [p for p in yticks_dict.values()]

        plt.close('all')

        fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(10,9))

        # Pitch
        t_str1 = ' (transposed)' if transposed else ''
        axs[0].plot(time, pitch, color='darkgreen')
        axs[0].set_title(f'[{prec}] {svara} [{succ}]{t_str1}')
        axs[0].grid()

        axs[0].set_yticks(tick_loc)
        axs[0].set_yticklabels(tick_names)

        axs[0].set_facecolor('#f2f2f2')

        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Pitch (cents)')

        # Features
        features = data[(data['svara']==svara) & (data['index']==ni)][feature_names]
        indx = features.index[0]
        labels = features.columns
        vals = features.values[0]
        zvals = (vals-feature_means)/feature_std
        axs[1].grid(zorder=0)
        axs[1].set_xlabel('z-score [(value - mean)/std]')
        bar = axs[1].barh(labels, zvals, zorder=3, color='#CC5500')
        for bar, v in zip(axs[1].patches, vals):
            fname = feature_names[list(vals).index(v)]
            if 'pitch' in  fname and 'change' not in fname:
                text_ = f'{round(v,2)} Cents'
            elif fname == 'duration':
                text_ = f'{round(v,2)} s'
            else:
                text_ = round(v,2)

            t = axs[1].text(min(zvals)+0.1, bar.get_y()+bar.get_height()/2, text_ , color = 'white', ha = 'left', va = 'center')
            t.set_bbox(dict(facecolor='black', alpha=0.7))

        t_str = '_TRANSPOSED' if transposed else ''
        path = cpath(plots_dir, svara, f'{prec}_{ni}_{succ}{t_str}.png')
        plot_path_df = plot_path_df.append({'svara':svara,'index':ni,'plot_path':path}, ignore_index=True)
        plt.savefig(path)
        plt.close('all')
        
        audio_path = cpath(plots_dir, svara, f'{prec}_{ni}_{succ}.wav')
        y = vocal[round(int(start*sr)):round(int(end*sr))]
        sf.write(audio_path, y, sr)


plot_path_df.to_csv(cpath(plots_dir, 'plot_paths.csv'), index=False)

###########################
### PLOTTING SVARA FEATURES
###########################
plt.close('all')
for svara in svara_dict:
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    features = data[(data['svara']==svara)][feature_names].mean()
    labels = features.index
    vals = features.values
    zvals = (vals-feature_means)/feature_std
    plt.title(svara)
    ax.grid(zorder=0)
    ax.set_xlabel('z-score [(value - mean)/std]')
    ax.barh(labels, zvals, zorder=3, color='#CC5500')
    path = cpath(plots_dir, svara, f'0_all_features.png')
    plt.savefig(path)
    plt.close('all')
