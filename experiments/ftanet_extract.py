import compiam
from src.utils import write_pitch_track
vocal_path = "/Users/thomasnuttall/Downloads/bageshri.mp3"
ftanet_carnatic = compiam.load_model("melody:ftanet-carnatic")
pitch_track_path = '/Users/thomasnuttall/Downloads/bageshri.tsv'
ftanet_pitch_track = ftanet_carnatic.predict(vocal_path,hop_size=30)

pitch = ftanet_pitch_track[:,1]
time = ftanet_pitch_track[:,0]
timestep = time[3]-time[2]

pitch = interpolate_below_length(pitch, 0, (250*0.001/timestep))
null_ind = pitch==0

pitch[pitch<50]=0
pitch[null_ind]=0

wl = round(0.145/timestep)
wl = wl if not wl%2 == 0 else wl+1
pitch = savgol_filter(pitch, polyorder=2, window_length=wl, mode='interp')
pitch = savgol_filter(pitch, polyorder=2, window_length=wl, mode='interp')


ftanet_pitch_track = np.array(list(zip(time, pitch)))
write_pitch_track(ftanet_pitch_track, pitch_track_path, sep='\t')

