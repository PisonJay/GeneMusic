#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import wavfile

def get_random_weight():
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    weight = [  2, 0.1,   1, 0.1,   2,   1, 0.1,   2, 0.1,   1.5, 0.1,   2]
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]

    note_weight = dict(zip(keys, [np.exp(-3 * np.abs((n//12) - 3)) * weight[n % 12] for n in range(len(keys))]))
    note_weight[''] = 0.1
    return note_weight

def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    
    note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0 # stop
    return note_freqs

# generating original wave
def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    t = np.linspace(0, duration, int(sample_rate*duration))
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave

# apply overtunes to the key
def apply_overtones(frequency, duration, factor, sample_rate=44100, amplitude=4096):
    assert abs(1-sum(factor)) < 1e-8
    
    frequencies = np.minimum(np.array([frequency*(x+1) for x in range(len(factor))]), sample_rate//2) # k-th overtune with freq. k*f
    amplitudes = np.array([amplitude*x for x in factor])
    
    fundamental = get_sine_wave(frequencies[0], duration, sample_rate, amplitudes[0])
    for i in range(1, len(factor)):
        overtone = get_sine_wave(frequencies[i], duration, sample_rate, amplitudes[i])
        fundamental += overtone
    return fundamental

# weighted amplitude according to piano
def get_adsr_weights(frequency, duration, length, decay, sustain_level, sample_rate=44100):
    assert abs(sum(length)-1) < 1e-8
    assert len(length) ==len(decay) == 4

    intervals = int(duration*frequency)

    if (intervals == 0):
        return [0.0]

    len_A = np.maximum(int(intervals*length[0]),1)
    len_D = np.maximum(int(intervals*length[1]),1)
    len_S = np.maximum(int(intervals*length[2]),1)
    len_R = np.maximum(int(intervals*length[3]),1)
    
    decay_A = decay[0]
    decay_D = decay[1]
    decay_S = decay[2]
    decay_R = decay[3]
    
    A = 1/np.array([(1-decay_A)**n for n in range(len_A)])
    A = A/np.nanmax(A)
    D = np.array([(1-decay_D)**n for n in range(len_D)])
    D = D*(1-sustain_level)+sustain_level
    S = np.array([(1-decay_S)**n for n in range(len_S)])
    S = S*sustain_level
    R = np.array([(1-decay_R)**n for n in range(len_R)])
    R = R*S[-1]
    
    weights = np.concatenate((A,D,S,R))
    smoothing = np.array([0.1*(1-0.1)**n for n in range(5)])
    smoothing = smoothing/np.nansum(smoothing)
    weights = np.convolve(weights, smoothing, mode='same')
    
    weights = np.repeat(weights, int(sample_rate*duration/intervals))
    tail = int(sample_rate*duration-weights.shape[0])
    if tail > 0:
        weights = np.concatenate((weights, weights[-1]-weights[-1]/tail*np.arange(tail)))
    return weights

def apply_pedal(note_values, bar_value):
    assert sum(note_values) % bar_value == 0
    new_values = []
    start = 0
    while True:
        cum_value = np.cumsum(np.array(note_values[start:]))
        end = np.where(cum_value == bar_value)[0][0]
        if end == 0:
            new_values += [note_values[start]]
        else:
            this_bar = np.array(note_values[start:start+end+1])
            new_values += [bar_value-np.sum(this_bar[:i]) for i in range(len(this_bar))]
        start += end+1
        if start == len(note_values):
            break
    return new_values

def get_song_data(music_notes, note_values, bar_value, factor, length,
                  decay, sustain_level, sample_rate=44100, amplitude=4096):
    note_freqs = get_piano_notes()
    frequencies = [note_freqs[note] for note in music_notes]
    new_values = apply_pedal(note_values, bar_value)
    duration = int(sum(note_values)*sample_rate)
    end_idx = np.cumsum(np.array(note_values)*sample_rate).astype(int)
    start_idx = np.concatenate(([0], end_idx[:-1]))
    end_idx = np.array([start_idx[i]+new_values[i]*sample_rate for i in range(len(new_values))]).astype(int)
    
    song = np.zeros((duration,))
    for i in range(len(music_notes)):
        this_note = apply_overtones(frequencies[i], new_values[i], factor)
        weights = get_adsr_weights(frequencies[i], new_values[i], length, 
                                   decay, sustain_level)
        song[start_idx[i]:end_idx[i]] += this_note*weights

    song = song*(amplitude/np.max(song))
    return song

# generating music and store into <path>
def gen(note, duration, path):
    duration = np.array(list(map(lambda x : 4 / x, duration)))
    song_data_r = get_song_data(note, duration, 4,
                                [0.73, 0.16, 0.06, 0.01, 0.02, 0.01, 0.01], 
                                [0.01, 0.29, 0.6, 0.1],
                                [0.05, 0.02, 0.005, 0.1], 0.1)
    song_data_l = get_song_data(note, duration, 4, 
                                [0.68, 0.26, 0.03, 0., 0.03],
                                [0.01, 0.6, 0.29, 0.1],
                                [0.05,0.02,0.005,0.1], 0.1)

    song_data = song_data_r + song_data_l
    song_data = song_data * (4096 / np.max(song_data))
    wavfile.write(path, 44100, song_data.astype(np.int16))

if __name__ == '__main__':
    # note = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
    #         'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
    #         'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
    #         'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',]
    # dura = [8] * 32
    # note = ['E4', 'g4', 'c4', 'D4', 'f4', 'E4', 'c4', 'B3', 'E4', 'd4', 'c4', 'g4', 'D4', 'A3', 'A2', 'E7', 'c4']
    # dura = [4, 2, 4, 4, 4, 8, 4, 16, 16, 4, 2, 16, 16, 8, 4, 2, 4]
    # note = ['A3', 'g4', 'A3', 'A3', 'g4', 'D4', 'g4', 'c4', 'E4', 'E4', 'A3', 'B3', 'g4', 'A3', 'A3', 'c4', 'A2', 'g4', 'D4', 'A3', 'f4', 'f4', 'f4', 'g4', 'g4', 'c4', 'A3', 'g4', 'F4', 'B3', 'g4']
    # dura = [4, 16, 16, 16, 16, 16, 16, 8, 16, 16, 8, 4, 16, 16, 8, 4, 8, 8, 4, 8, 4, 8, 8, 8, 4, 16, 16, 8, 8, 8, 4]
    # note = ['A3', 'g4', 'G4', 'A3', 'A3', 'A3', 'c4', 'E4', 'B3', 'g4', 'B3', 'B3', 'c4', 'c4', 'c4', 'g4', 'g5', 'A4', 'B3', 'C4', 'B3']
    # dura = [4, 8, 4, 16, 16, 4, 8, 4, 8, 4, 8, 8, 4, 16, 16, 8, 2, 4, 8, 4, 4, 8]
    # note = ['A3', 'E4', 'E4', 'c4', 'a3', 'g4', 'g4', 'B2', 'g4', 'g4', 'E3', 'D4', 'c4', 'A3', 'c4', 'B3', 'D4', 'E4', 'B3', 'g4']
    # dura = [4, 8, 8, 4, 8, 8, 4, 8, 4, 8, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4]
    # note = ['E4', 'A3', 'f4', 'g4', 'A3', 'c4', 'g4', 'c4', 'A3', 'D4', 'c4', 'g4', 'D4', 'f4', 'A2', 'D4', 'E4', 'A3']
    # dura = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4, 8, 8]
    # note = ['B3', 'D4', 'f4', 'E4', 'E4', 'E4', 'f4', 'B4', 'A3', 'A3', 'f4', 'B3', 'g4', 'D4', 'c4', 'E4', 'E3', 'A4', 'E4', 'G4', 'D4', 'E4', 'E4', 'E4']
    # dura = [8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 8, 8, 16, 16, 8, 4, 4, 4, 4, 4, 8, 4, 8, 4]
    # note = ['A3', 'A4', 'A3', 'c4', 'c4', 'B4', 'D4', 'E4', 'D4', 'A3', 'D4', 'E4', 'c4', 'g4', 'c4', 'A4', 'D4']
    # dura = [8, 8, 2, 8, 8, 4, 4, 4, 8, 8, 4, 8, 4, 8, 4, 4, 2, 8, 8]
    
    # note = ['A3', 'E4', 'A3', '', 'c4', 'A3', 'f4', 'E4', '', 'E4']
    # dura = [2, 4, 4, 2, 4, 8, 8, 1, 4, 2, 8, 8]
    # note = ['D4', 'D4', 'A3', 'c4', 'B3', 'E4', 'B3', 'E4', 'A3', 'E4', 'A3', 'E4', 'E4', 'B3', 'E4', 'A3', 'E4', 'A3']
    # dura = [4, 4, 4, 16, 16, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    # note = ['E4', 'A3', 'E4', 'A3', 'E4', 'A4', 'A3', 'D4', 'A3', 'A3', 'A3', 'E4', 'A3', 'E4', 'E4', 'A3', 'E4', 'A3', 'c3', 'D4']
    # dura = [4, 8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 8, 8, 8, 8, 4]
    # note = ['A3', 'E4', 'E4', 'f4', 'A3', 'D4', 'A3', 'D4', 'A3', 'E4', 'E4', 'B3', 'B3', 'E4', 'A3', 'A3', 'D4', 'A4', 'D4']
    # dura = [4, 4, 4, 16, 16, 8, 4, 4, 4, 16, 16, 8, 4, 4, 4, 4, 4, 4, 4, 8, 8]
    # note = ['E4', 'f4', 'E4', 'A4', 'D4', 'B3', 'E4', 'A3', 'E4', 'B3', 'E4', 'E4', 'A3', 'g4', 'B3', 'E4', 'E4', 'A3', 'E4', 'A3']
    # dura = [4, 16, 16, 4, 8, 4, 4, 4, 2, 4, 4, 4, 16, 16, 8, 4, 4, 4, 8, 8]
    # note = ['B3', 'A3', 'E4', 'A3', 'D4', 'D4', 'A3', 'A3', 'D4', 'A3', 'E4', 'A3', 'E4', 'E4', 'A3', 'E4', 'D4', 'E4', 'B3']
    # dura = [4, 8, 4, 8, 4, 4, 4, 4, 4, 4, 8, 8, 4, 8, 8, 4, 8, 4, 8, 4]

    #长度比重放大
    # note = ['E4', 'A3', 'E4', 'E4', 'E4', 'A3', 'A3', 'D4', 'E4', 'A3', 'D4', 'E4', 'A3', 'f4', 'E4', 'G4', 'F4', 'B3', 'E4', 'B3', 'A3', 'D4', 'B3', 'B3', 'E4']
    # dura = [4, 8, 8, 4, 8, 8, 4, 16, 16, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8]
    # note = ['A3', 'A3', 'A3', 'A3', 'A3', 'D4', 'A3', 'D4', 'D4', 'A3', 'f4', 'E4', 'E4', 'f4', 'f4', 'A3', 'D4', 'E4', 'D4', 'E4', 'f4', 'B3', 'E4', 'A3', 'D4', 'A3', 'E4', 'E4', 'A3']
    # dura = [8, 8, 16, 16, 4, 16, 16, 8, 8, 4, 8, 8, 4, 16, 16, 16, 16, 8, 8, 8, 8, 4, 8, 16, 16, 4, 8, 4, 8, 4]
    # note = ['E4', 'B4', 'E4', 'E5', 'A4', 'A3', 'E4', 'g4', 'A4', 'E4', 'A3', 'E4', 'B3', 'D4', 'A3', 'D4', 'A3', 'g3', 'E4', 'A3', 'D4']
    # dura = [4, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 4, 16, 16, 16, 16, 8, 4, 4, 4, 4]
    # gen(note, dura, './test.wav')

    # 和谐比重放大
    # note = ['E5', 'E4', 'E4', 'A3', 'A3', 'E4', 'B3', 'E4', 'A3', 'A3', 'E4', 'E4', 'A3', 'D4', 'A3', 'E4', 'E4', 'c4', 'A4', 'g4', 'A3', 'A3', 'A3']
    # dura = [8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 4, 2, 4, 8, 8, 4, 16, 16, 4, 8, 4]
    # note = ['A3', 'c4', 'A3', 'E4', 'A3', 'E4', 'E4', 'A3', 'A3', 'A3', 'A3', 'E4', 'E4', 'A4', 'E4', 'A3', 'E4', 'A3', 'E4', 'A3', 'D4', 'A3', 'E4']
    # dura = [4, 8, 4, 8, 4, 4, 8, 4, 8, 4, 4, 8, 4, 8, 4, 4, 16, 16, 8, 8, 8, 8, 8]
    # note = ['A3', 'D4', 'A3', 'A3', 'E4', 'B3', 'E4', 'a3', 'D4', 'A3', 'E4', 'g4', 'B3', 'E4', 'B3', 'g4', 'D4', 'A3', 'E4', 'A3', 'A3', 'E4', 'A3']
    # dura = [4, 4, 4, 8, 8, 4, 16, 16, 8, 8, 8, 8, 8, 4, 8, 16, 16, 4, 4, 4, 4, 4, 4]
    note = ['B3', 'E4', 'A3', 'D4', 'B3', 'f4', 'A3', 'E4', 'A3', 'D4', 'A3', 'E4', 'A3', 'A3', 'c4', 'E4', 'B3', 'E4', 'A3', 'A3', 'E4']
    dura = [4, 8, 4, 8, 16, 16, 8, 4, 16, 16, 4, 8, 4, 4, 8, 8, 4, 4, 4, 4, 4, 4]


    # updated fitness function
    note = ['E4', 'D4', 'D4', 'D4', 'E4', 'f4', 'g4', 'A3', 'c4', 'g4', 'g3', 'A3', 'B3', 'A3', 'E4', 'A3', 'D4', 'f4', 'A3', 'E4']
    dura = [4, 4, 4, 8, 8, 4, 8, 4, 8, 8, 8, 4, 8, 8, 4, 4, 4, 4, 4, 4]
    note = ['c4', 'E4', 'E4', 'D4', 'E4', 'D4', 'D4', 'A4', 'A3', 'E4', 'D4', 'A3', 'E4', 'E4', 'D4', 'f4', 'g4', 'a3', 'E4', 'D4', 'g4', 'B3', 'f4', 'g4', 'E4', 'E4', 'g4']
    dura = [8, 8, 8, 4, 8, 8, 16, 16, 4, 8, 8, 8, 8, 8, 8, 4, 8, 8, 4, 8, 16, 16, 4, 16, 16, 8, 4, 8, 8]
    note = ['c4', 'B3', 'A3', 'f3', 'c4', 'B3', 'A3', 'c4', 'c4', 'g4', 'c4', 'B3', 'g4', 'A4', 'g4', 'g4', 'f4', 'c4', 'E4', 'c4', 'c4', 'B3', 'c4', 'f4', 'c4', 'E4']
    dura = [4, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 8, 8, 8, 8, 4, 16, 16, 4]
    note = ['g4', 'c4', 'E4', 'E4', 'E4', 'c4', 'g4', 'D4', 'D4', 'f4', 'g4', 'A3', 'A3', 'A3', 'A3', 'A3', 'C4', 'E4', 'D4', 'c4', 'A3', 'f4', 'c4', 'D4', 'f4', 'a3']
    dura = [8, 8, 8, 8, 8, 8, 4, 4, 16, 16, 8, 4, 4, 4, 8, 8, 8, 16, 16, 4, 4, 8, 8, 4, 8, 8]


    # updated random function in mutation
    note = ['A3', 'E4', 'E4', 'E4', 'A4', 'D4', 'c4', 'E4', 'A3', 'f4', 'B3', 'E3', 'E4', 'A4', 'E4', 'A3', 'E4', 'A3', 'E3', 'B3', 'E4', 'B3', 'D4']
    dura = [4, 8, 4, 8, 4, 8, 8, 8, 4, 8, 8, 8, 4, 8, 8, 4, 8, 8, 4, 4, 8, 8, 8, 8]

    # merged fitness function
    # 4383 @ 21 : 0.00019535002705889568
    # note = ['E4', 'E4', 'A3', 'D4', 'F4', 'E4', 'c4', 'B3', 'A3', 'D4', 'G4', 'g4', 'd4', 'c5', 'E4', 'B3', 'f4', 'B3', 'A3', 'D4', 'A3', 'E4', 'E5']
    # dura = [8, 8, 4, 4, 8, 8, 4, 8, 4, 8, 4, 16, 16, 8, 8, 8, 8, 8, 4, 8, 8, 4, 4, 8, 8]
    note = ['D4', 'A3', 'B3', 'c4', 'A3', 'E4', 'E4', 'A3', 'C4', 'D4', 'c4', 'B3', 'A3', 'A3', 'B3', 'c4', 'c4', 'E4']
    dura = [4, 4, 4, 16, 16, 8, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4, 2, 4]
    note = ['A4', 'D4', 'A3', 'f4', 'D4', 'g4', 'f4', 'A3', 'F4', 'g4', 'g4', 'A3', 'D4', 'A3', 'F4', 'A3', 'A3', 'c3', 'a3', 'E4', 'D4', 'A3', 'A4', 'A3']
    dura = [4, 8, 4, 8, 8, 16, 16, 4, 4, 8, 8, 4, 8, 8, 4, 4, 8, 8, 4, 8, 8, 8, 8, 4]
    note = ['A3', 'E4', 'c4', 'f4', 'A3', 'A3', 'B3', 'g4', 'B3', 'E4', 'D4', 'A3', 'f4', 'D4', 'A3', 'f4', 'c4', 'c4', 'D4']
    note = ['A3', 'E4', 'g4', 'D4', 'E4', 'A3', 'g4', 'A3', 'E4', 'G4', 'B3', 'c4', 'D4', 'A3', 'E4', 'B3', 'A3', 'A4', 'c4', 'D4', 'A3']
    dura = [4, 8, 8, 8, 8, 8, 8, 4, 8, 4, 8, 4, 4, 8, 4, 8, 4, 4, 4, 4, 4]
    note = ['A3', 'B3', 'f4', 'E4', 'A3', 'E4', 'd4', 'E4', 'c4', 'A3', 'E4', 'f5', 'B3', 'B3', 'D4', 'f4', 'D4', 'g4', 'A3', 'c3', 'c4', 'E4']
    dura = [4, 8, 8, 4, 8, 16, 16, 4, 8, 4, 8, 8, 8, 4, 8, 8, 4, 4, 4, 8, 8, 4, 8, 8]
    note = ['E4', 'E4', 'D4', 'D4', 'g4', 'c4', 'E4', 'D4', 'D4', 'D4', 'f4', 'E4', 'g4', 'B3', 'A3', 'E4', 'c4', 'E4', 'g4', 'E4', 'C4', 'E4']
    dura = [4, 8, 8, 16, 16, 8, 8, 8, 4, 8, 4, 8, 4, 4, 4, 4, 4, 4, 8, 8, 4, 4]
    note = ['D4', 'E4', 'A3', 'E4', 'D4', 'D5', 'E5', 'E4', 'A3', 'B3', 'B3', 'D4', 'E4', 'f4', 'g4', 'B3', 'D4', 'B3', 'D4']
    dura = [4, 8, 8, 2, 4, 8, 8, 2, 4, 4, 4, 8, 8, 8, 8, 4, 8, 8, 4]
    note = ['E4', 'g4', 'B3', 'A3', 'F4', 'A3', 'D4', 'F4', 'A3', 'B3', 'g4', 'E4', 'c4', 'B3', 'G4', 'F4', 'g3', 'C4', 'A3', 'E4', 'G3', 'A3', 'f4', 'D4']
    dura = [4, 8, 8, 4, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 8, 4]
    print (len(note), len(dura))

    gen(note, dura, './test.wav')
