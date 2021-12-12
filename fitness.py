#from typing import Final
import numpy as np
import math

tone=['C', 'D', 'E', 'F', 'G', 'A', 'B'] #C大调
tottone=['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] #C大调

ref_note = ['D4', 'E4', 'F4', 'G4', 'A4', 'D5', 'c5', 'A4', 'E4', 'G4', 'f4', 'D4', 'C5', '', 'B4', 'A4', 'a4', 'G4', 'E4', 'G4', 'B4', 'D4', 'c4', 'A4', 'D4', 'G4', 'F4', 'E4', 'D4', 'c4', 'D4', 'E4', 'f4', 'g4', 'A4', 'B4']
ref_dura = [16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16 ]
totnote = []

def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    global totnote
    totnote = [x+str(y) for y in range(0,9) for x in octave]
    print(totnote)
    
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
            # val = 0
            # for i in range(len(tmp)):
            #     val = val + tmp[i]
            # tmp_u.append(val / len(tmp)) # 每小节均值
    
            # val_s = 0
            # for i in range(len(tmp)):
            #     val_s = (val - tmp[i])**2
            # tmp_s.append(val_s / len(tmp)) # 每小节方差

            # if tmp:
            #     tmp_u.append(np.mean(tmp))
            #     tmp_s.append(np.std(tmp))
            # else:
            #     tmp_u.append(0)
            #     tmp_s.append(0)
        
        fit_value = 0
        for i in range(length-1):
            if (mus.note[i] == '' or mus.note[i+1] == '' ):
                fit_value = fit_value + 10000
                continue
            if (tone.count( mus.note[i][0] ) == 0 or tone.count( mus.note[i+1][0] ) == 0):
                continue

            l,r = mus.note[i],mus.note[i+1]
            degree,Semitone = 0,0
            diff = totnote.index(l) -  totnote.index(r)
            if ( abs(diff) > 12):
                return 1e100
            elif abs(diff) == 12 or diff == 0:
                fit_value = fit_value + 20 * mus.dura[i]*mus.dura[i+1]
            else:
                if tone.index(l[0]) < tone.index(r[0]):
                    degree = tone.index(r[0]) - tone.index(l[0]) + 1
                    Semitone = tottone.index(r[0]) - tottone.index(l[0])
                else:
                    degree = 8 + tone.index(r[0]) - tone.index(l[0])
                    Semitone = 12 + tottone.index(r[0]) - tottone.index(l[0])
                if ((degree == 4 and Semitone == 5) or (degree == 5 and Semitone == 7)):
                    fit_value = fit_value + 2 * ( 16 / mus.dura[i] ) * ( 16 / mus.dura[i+1] )
                elif (degree == 3 and (Semitone == 3 or Semitone == 4)):
                    fit_value = fit_value + 200 * ( 16 / mus.dura[i] ) * ( 16 / mus.dura[i+1] )
                elif (degree == 6 and (Semitone == 8 or Semitone == 9)):
                    fit_value = fit_value + 200 * ( 16 / mus.dura[i] ) * ( 16 / mus.dura[i+1] )
                else:
                    fit_value = fit_value + 500 * ( 16 / mus.dura[i] ) * ( 16 / mus.dura[i+1] )

        # for i in range(4):
        #     fit_value = fit_value + math.fabs(tmp_u[i] - self.ref_u[i]) # * 10 # 均值权重*10
        # for i in range(4):
        #     fit_value = fit_value + math.fabs(tmp_s[i] - self.ref_s[i]) # 方差权重 1
        
        fit_value = fit_value + (bad_tones**2) * 100 # (1 / bad_tones if bad_tones != 0 else 9999)
        
        return fit_value / ((len(mus.note)) ** 3)

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