import play
import random
import string
import numpy as np

from music21 import *
from os import sys, walk

def fix(freq):
    notes = play.get_piano_notes()
    res = ''
    for x in notes.keys():
        if (abs(notes[x]-freq) < abs(notes[res]-freq)):
            res = x
    return res

def trans(dura_type):
    if (dura_type == 'whole'):
        return 32
    elif (dura_type == 'half'):
        return 16
    elif (dura_type == 'quarter'):
        return 8
    elif (dura_type == 'eighth'):
        return 4
    elif (dura_type == '16th'):
        return 2
    elif (dura_type == '32nd'):
        return 1
    print (dura_type)
    return None

sys.stdout = open("data.txt", "w")

for (_, __, file) in walk('./xml'):
    for s in file:
        if (s[0] == '.'):
            continue

        b = converter.parse('./xml/' + s)
        TS = b.recurse().getElementsByClass('TimeSignature')[0]

        if (TS.ratioString != '4/4'):
            continue

        cases = 6
        __collect = []
        print ('Loading from ' + s)

        while (cases > 0):
            flag = True
            __note = []
            __dura = []

            PN = random.randint(0, 1)
            MN = random.randint(0, len(b.getElementsByClass(stream.Part)[PN].getElementsByClass(stream.Measure)) - 5)
            
            for m in range(MN, MN + 4):
                total = 0
                for n in b.getElementsByClass(stream.Part)[PN].getElementsByClass(stream.Measure)[m]:

                    interval = 0
                    if (type(n) == note.Note):
                        __note.append(fix(n.pitch.frequency))
                        interval = trans(n.duration.type)
                        __dura.append(32 // interval)
                    elif (type(n) == note.Rest):
                        __note.append('')
                        interval = trans(n.duration.type)
                        __dura.append(32 // interval)
                    elif (n.duration.type != 'zero'):
                        flag = False
                    total += interval
                
                if ((not flag)):
                    break
                if (total != 32):
                    __note.append('')
                    __dura.append(32 / (32 - total))

            if (not flag):
                continue

            cases -= 1
            __collect.append((__note, __dura))

        for d in range(2):
            for i in range(len(__collect)):
                print (__collect[i][d], ',')
            print ()
            print ()
