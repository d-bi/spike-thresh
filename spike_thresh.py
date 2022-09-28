from open_ephys.analysis import Session
import numpy as np
import json
import sys

directory = 'C:\\Users\\Caltech University\\Desktop\\spike-thresh\\2022-09-07_14-15-37'
session = Session(directory)

# parse json info
with open(session.recordnodes[0].recordings[0].directory+'\\structure.oebin') as f:
    j = json.load(f)
cts_meta = j['continuous'][0]
n_chan = cts_meta['num_channels']
bit_volts = []
for ch in cts_meta['channels']:
    bit_volts.append(ch['bit_volts'])

# get raw data
samples = session.recordnodes[0].recordings[0].continuous[0].samples
timestamps = session.recordnodes[0].recordings[0].continuous[0].timestamps
metadata = session.recordnodes[0].recordings[0].continuous[0].metadata

# unit conversion and statistics
samples_mv = samples*np.array(bit_volts)
means = np.mean(samples_mv, axis=0)
stds = np.std(samples_mv, axis=0)
meds = np.median(samples_mv, axis=0)

# spike thresholding
spike_channels = []
spike_timestamps = []
for ch_idx in range(16):
    ch_samples_mv = samples_mv[:,ch_idx]
    ch_mean = means[ch_idx]
    ch_std = stds[ch_idx]
    
    thr_exceeded_timestamps = np.asarray((np.abs(ch_samples_mv - ch_mean) > 4 * ch_std)).nonzero()[0]
    spike_events = []
    spike_peaks = []
    t_peak = 0
    for i in range(len(thr_exceeded_timestamps)):
        t = thr_exceeded_timestamps[i]
        if t < t_peak:
            continue
        spike_events.append(t)
        if ch_samples_mv[t] < ch_samples_mv[t+1]: # assume that spike voltages decrease initially (only record falling stage)
            continue
        peak_offset = np.min(np.where(np.diff(ch_samples_mv[t:t+1000])>0)[0]) # assumes that a spike does not exceed order of 1000 samples
        spike_peaks.append(t+peak_offset)
        t_peak = spike_peaks[-1]
        
        spike_channels.append(ch_idx)
        spike_timestamps.append(t+peak_offset)

# write to disk
np.save('spike_channels.npy', np.array(spike_channels))
np.save('spike_timestamps.npy', np.array(spike_timestamps))
