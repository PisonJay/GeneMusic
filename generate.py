from data import sample_dura, sample_note, gen_rhythm
import random
import numpy as np
import play
from fitness import GenFitnessFunc, totnote
MIN_DURATION = 16
MAX_N_GENES = 2560
VALID_DURATION = {1, 2, 4, 8, 16}

def random_tone_between(A, B):
    if B == '':
        A = B
    if A == '':
        B = A
    if A == '' and B == '':
        mean = totnote.index('C3')
        sigma = 0
    else:
        A = totnote.index(A)
        B = totnote.index(B)
    
        mean = (A+B)/2
        sigma = abs(B-A)/2
    # octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    weight = [  2, 0.1,   1, 0.1,   2,   1, 0.1,   2, 0.1,   1.5, 0.1,   2]
    keys = totnote
    note_weight = dict(zip(keys,[np.exp(-3 * np.abs((n-mean)/(12+sigma))) * weight[(n + 9)%12] for n in range(len(keys))]))
    note_weight[''] = 0.1
    keys = list(note_weight.keys())
    note_weight = np.array(list(note_weight.values()))
    note_weight /= sum(note_weight)
    return np.random.choice(keys, 1, p = note_weight)[0]
    
class NoteSample:
    sort_key = lambda x: x.score

    def __init__(self, notes, duration, score = 0, generation = None, id = None):
        self.note = notes
        self.dura = duration
        self.score = score
        self.generation = generation
        self.id = id
    
    def log(self):
        print(f'{self.id} @ {self.generation} : {self.score}')
        print(self.note)
        print(self.dura)

def duration_to_map(dura):
    ret = {0:0}
    t = 0
    i = 0
    for d in dura:
        t += MIN_DURATION//d
        i += 1
        ret[t] = i
    return ret

def rand_crossover(par1, par2):
    durmap1 = duration_to_map(par1.dura)
    durmap2 = duration_to_map(par2.dura)
    keys = durmap1.keys() & durmap2.keys()
    split = random.sample(list(keys)[1:-1], 1)[0]
    child_notes_1 = par1.note[:durmap1[split]] + par2.note[durmap2[split]:]
    child_dur_1 = par1.dura[:durmap1[split]] + par2.dura[durmap2[split]:]

    child_notes_2 =  par2.note[:durmap2[split]] + par1.note[durmap1[split]:]
    child_dur_2 =  par2.dura[:durmap2[split]] + par1.dura[durmap1[split]:]
    return NoteSample(child_notes_1, child_dur_1), NoteSample(child_notes_2, child_dur_2)
        
def split_note(note_sample):
    idx = np.random.choice(range(len(note_sample.note)))
    if note_sample.dura[idx] != 16:
        cur = note_sample.note[idx]
        L = note_sample.note[idx-1] if idx > 0 else cur
        R = cur
        note_sample.note.insert(idx, random_tone_between(L, R))
        note_sample.dura[idx] *= 2
        note_sample.dura.insert(idx, note_sample.dura[idx])

def idx2offset(dura, id):
    ret = 0
    for i in range(id):
        ret += MIN_DURATION//dura[i]
    return ret

def merge_note(note_sample):
    idx = np.random.choice(range(len(note_sample.note)))
    end = idx + 1
    d = MIN_DURATION//note_sample.dura[idx]

    while end < len(note_sample.note):
        d += MIN_DURATION//note_sample.dura[end]
        if 16 % d == 0:
            break
        end += 1

    if end < len(note_sample.note) and idx2offset(note_sample.dura,idx)//16 == idx2offset(note_sample.dura, end)//16:
        i = idx
        while i < end:
            del note_sample.dura[idx]
            del note_sample.note[idx]
            i += 1
        note_sample.dura[idx] = MIN_DURATION//d

        cur = note_sample.note[idx]
        L = note_sample.note[idx-1] if idx > 0 else cur
        R = note_sample.note[idx+1] if idx < len(note_sample.note) - 1 else cur
        note_sample.note[idx] = random_tone_between(L, R)

def change_tone(note_sample):
    idx = np.random.choice(range(len(note_sample.note)))
    cur = note_sample.note[idx]
    L = note_sample.note[idx-1] if idx > 0 else cur
    R = note_sample.note[idx+1] if idx < len(note_sample.note) - 1 else cur
    note_sample.note[idx] = random_tone_between(L, R)


def swap_tone(note_sample):
    ids = np.random.choice(range(len(note_sample.note)),2, replace=False)
    t = note_sample.note[ids[0]]
    note_sample.note[ids[0]] = note_sample.note[ids[1]]
    note_sample.note[ids[1]] = t

def null(xxx):
    pass

def mutate(note_sample):
    choices = [split_note, merge_note, change_tone, swap_tone, null]
    p = [0.1, 0.1, 0.2, 0.1, 0.9]
    p = np.array(p)/np.sum(p)
    idx = np.random.choice(range(len(choices)), p = p)
    choices[idx](note_sample)


def crossover_n_mutation(note_samples):
    buffer = []
    while note_samples:
        l = len(note_samples)
        par1_index = random.randint(0,l-1)
        par2_index = random.randint(0,l-2)
        par1 = note_samples[par1_index]
        del note_samples[par1_index]
        par2 = note_samples[par2_index]
        del note_samples[par2_index]
        
        child1, child2 = rand_crossover(par1, par2)
        mutate(child1)
        mutate(child2)
        mutate(par1)
        mutate(par2)

        buffer += [child1, child2, par1, par2]
    return buffer

def fitness_filter(note_samples, fitness_func):
    p = []
    for i in note_samples:
        i.score = 1/( fitness_func(i) + 0.1 )
        p += [i.score]
    p = np.array(p)
    p /= np.sum(p)
    samples = np.random.choice(note_samples, min(len(note_samples), MAX_N_GENES), replace=False, p=p)
    return list(samples)

def log(samples, headline = 'results:'):
    print(headline)
    for i in sorted(samples, key=NoteSample.sort_key):
        i.log()


def GeneticAlgo(note_samples, n_iter, fitness_func, should_log = None):
    iter = 0
    if should_log == None:
        should_log = lambda x : x % 10 == 0
    samples = note_samples
    while iter < n_iter:
        samples = fitness_filter(samples, fitness_func)
        samples = crossover_n_mutation(samples)
        for i in range(len(samples)):
            samples[i].generation = iter + 1
            samples[i].id = i

        if should_log(iter):
            log(samples, f'Results for #{iter}:')
        iter += 1

def genInitialSamples(n):
    note, dura = gen_rhythm(n)
    ret = []
    for i in range(n):
        ret.append(NoteSample(note[i], dura[i]))
    return ret







if __name__ == "__main__":
    print(totnote)
    N_INITIAL_SAMPLES = 600
    samples = genInitialSamples(N_INITIAL_SAMPLES)
    refSample = NoteSample(sample_note[0], sample_dura[0])
    GeneticAlgo(samples, 21, fitness_func=GenFitnessFunc(refSample))