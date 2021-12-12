#from typing import Final
import numpy as np
import math

tone=['C', 'D', 'E', 'F', 'G', 'A', 'B'] #C大调
tottone=['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] #C大调

totnote = []

def get_piano_notes():
    global totnote

    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    global totnote
    totnote = [x+str(y) for y in range(0,9) for x in octave][start:end+1]
    
    note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0 # stop
    return note_freqs

note_fre=get_piano_notes()

class Fitness:
    def __init__(self, ref_note, ref_dura):
        self.ref_u = []
        self.ref_s = []
        length = len(ref_note)
        t = 0
        tmp = []
        for i in range(length):
            d = 16//ref_dura[i]
            if ref_note[i] != '':
                tmp += [note_fre[ref_note[i]]] * d
            t += d
            if t % 16 == 0:
                self.ref_u.append(np.mean(tmp))
                self.ref_s.append(np.std(tmp))
                tmp = []
    
    def compute(self, mus):
        tmp_u = []
        tmp_s = []
        length = len(mus.note)
        bad_tones = 0
        for i in range(4):
            num = 0
            tmp = []
            for j in range(length):
                num += 16 // mus.dura[j]
                if (mus.note[j]==''):
                    continue
                if (num > i*16 and num <= (i+1)*16):
                    if (tone.count( mus.note[j][0] ) == 0): # 不符合音调的音符数
                        bad_tones = bad_tones + 16 // mus.dura[j]
                    for k in range(16 // mus.dura[j]):
                        tmp.append( note_fre[mus.note[j]] )
        
        res_value = 10 / length
        count_table = np.array([0] * 12)

        for i in range(length-1):
            if (mus.note[i] == '' or mus.note[i+1] == '' ):
                res_value *= 1.3
                continue

            l = note_fre[mus.note[i]]
            r = note_fre[mus.note[i+1]]

            if (l < r):
                __temp = l
                l = r
                r = __temp

            l /= r
            if (l > 2 + 1e-7):
                return 1000

            __diff = [(abs(l*q - round(l*q)), q) for q in range(2, 10)]
            __diff.sort()
            count_table[__diff[0][1]] += (16 / mus.dura[i] + 16 / mus.dura[i+1]) / 32
        
        res_value += sum(count_table != 0) / length

        for i in range(12):
            res_value += np.exp(count_table[i] / length * 2)
        
        # print("fit loss = %f" % (res_value))
        return res_value

def GenFitnessFunc(ref_sample):
    return Fitness(ref_sample.note, ref_sample.dura).compute

# class NoteSample:
#     sort_key = lambda x: x.score

#     def __init__(self, notes, duration, score = 0, generation = None, id = None):
#         self.note = notes
#         self.dura = duration

# ff = NoteSample(['D4', 'f4', 'E4', 'f4', 'g4', 'A3', 'g4', 'f4', 'c4', 'f4', 'E5', 'f5', 'f4', 'g4', 'E4', 'c4', 'c4', 'E3'],[4, 4, 4, 8, 8, 4, 4, 4, 8, 8, 4, 4, 4, 4, 2, 4, 8, 8])
# gg = Fitness(['D4', 'f4', 'E4', 'f4', 'g4', 'A3', 'g4', 'f4', 'c4', 'f4', 'E5', 'f5', 'f4', 'g4', 'E4', 'c4', 'c4', 'E3'],[4, 4, 4, 8, 8, 4, 4, 4, 8, 8, 4, 4, 4, 4, 2, 4, 8, 8])
# print(gg.compute(ff))
