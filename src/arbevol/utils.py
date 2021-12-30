'''
arbevol.
Miscellaneous utility functions.
Michael Gasser
gasser@indiana.edu
'''

VERBOSE = True

LUCE_EPS = 0.01

import os, tarfile, math, random, sys, getopt, re
import numpy as np
from functools import reduce

def l2a(lst):
    """
    Convert a list to a numpy array.
    """
    return np.array(lst)

## DICTS

def dict_depth(dct):
    '''The depth of a dictionary.'''
    if not isinstance(dct, dict):
        return 0
    else:
        return 1 + max([dict_depth(v) for v in list(dct.values())])

## SEQUENCES AND SETS

def make_list(lnth, random = False, dflt = 0, binary = False):
    '''A list of length lnth with val in each position.'''
    return [ran(binary) if random else dflt for x in range(lnth)]

def seqs2seq(seq):
    '''Returns a sequence with the subsequences in seq appended to each other.'''
    return reduce(lambda x,y: x+y, seq)

def seq2str(seq):
    '''A string composed of the print representations of the elements of the sequence.'''
    if isinstance(seq, list):
        s = '[ '
    else:
        s = '( '
    for x in seq:
        s += x.__str__() + ' '
    if isinstance(seq, list):
        s += ']'
    else:
        s += ')'
    return s

def indices(x, seq):
    '''List of indices in seq where x is found.'''
    return [pos for pos, item in enumerate(seq) if item == x]

def distance2(vec1, vec2):
    '''Square of the euclidian distance between the vectors.'''
    return sum([(x - y)**2 for x,y in zip(vec1, vec2)])

def distance(vec1, vec2):
    '''Euclidian distance between the vectors.'''
    return math.sqrt(distance2(vec1, vec2))

def intensity(pos1, pos2, maximum=1.0, minimum=.01, magnitude=1):
    '''Intensity of an event at pos2 as viewed from pos1.'''
    # Squared euclidian distance from pos1 to pos2
    dist2 = distance2(pos1, pos2)
    if dist2 == 0.0:
        # Maximum intensity at 0 distance
        return maximum
    else:
        # Otherwise min of minimum/dist2 and maximum
        return min([magnitude * (minimum / dist2), maximum])

def ran_bin_vec(length):
    '''A vector of len length with random 0s and 1s.'''
    return [random.randrange(2) for x in range(length)]

def union(sets):
    '''The union of multiple sets in a sequence.'''
    return reduce(lambda x, y: x | y, sets)

def intersection(sets):
    '''The intersection of multiple sets in a sequence.'''
    return reduce(lambda x, y: x & y, sets)

def match_upto(seq1, seq2, upto):
    '''Does the sequences agree up to upto?'''
    return seq1[:upto] == seq2[:upto]

def first(pred, seq):
    '''Returns the first element of seq for with pred is true.'''
    for x in seq:
        if pred(x):
            return x
    return False

def some(pred, seq):
    '''Returns the first successful application of pred to elements in seq.'''
    for x in seq:
        px = pred(x)
        if px:
            return px
    return False

def which(pred, seq):
    '''Returns the first element in the sequence or iterator fo which the pred is True.'''
    for x in seq:
        px = pred(x)
        if px:
            return x
    return False

def every(pred, seq):
    '''True if every member of the sequence or iterator satisfies the predicate.'''
    for x in seq:
        if not pred(x):
            return False
    return True

def reduce_lists(lists):
    '''Flatten a list of lists (doesn't mutate lists).'''
    return reduce(lambda x, y: x + y, lists)

def remove_dups(seq):
    '''Make a copy of seq with no duplicate elements.'''
    copy = seq[:]
    for p,x in enumerate(seq):
        if x in seq[p+1:]:
            copy.remove(x)
    return copy

def hamdist(s1, s2):
    """Hamming distance between two sequences."""
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

## MATH

def ran(continuous = True):
    '''A random number.'''
    return random.random() if continuous else random.randint(0, 1)

def normalize(vector):
    '''Make the vector (really a list) length 1.0.'''
    total = math.sqrt(sum([x**2 for x in vector]))
    for i in range(len(vector)):
        vector[i] /= total

def dot_product(v1, v2):
    '''Dot product of the two vectors.'''
    return sum([x1 * x2 for x1, x2 in zip(v1, v2)])

def pairwise_product(v1, v2):
    '''Pairwise product of the two vectors. Doesn't check whether they're the same length.'''
    return [x1 * x2 for x1, x2 in zip(v1, v2)]

def threshold(inp, thresh, min_val, max_val):
    '''Simple threshold function.'''
    if inp >= thresh:
        return max_val
    else:
        return min_val

# Error functions

def quadratic(out, target):
    error = target - out
    return 0.5 * error * error

def quadratic_slope(out, target):
    return target - out

def cross_entropy(out, target):
    return target * math.log(out) + (1.0 - target) * math.log(1.0 - out)

# We don't actually need this, since we assume the activation function
# is sigmoid (and there are problems with division by 0 anyway).
def crossent_slope(out, target):
    return (out - target) / (out * (1.0 - out))

# Activation functions

def linear(inp, thresh=None, gain=None, slope=1.0):
    return slope * inp

def linear_slope(inp, slope=1.0):
    return slope

def sigmoid(inp, thresh, gain, var=None):
    '''Sigmoid function (0 < y < 1) with threshold and gain.'''
    return 1.0 / (1.0 + math.exp(gain * (-inp + thresh)))

def sigmoid_slope(x, var=None):
    '''Slope of the sigmoid with output x.'''
    return x * (1.0 - x)

def tanh(inp, thresh, gain, var=None):
    '''Hyperbolic tangent with threshold and gain.'''
    return math.tanh(gain * (inp + thresh))

def tanh_slope(x, var=None):
    '''Slope of tanh with output x.'''
    return 1.0 - x * x

def relu(x, thresh=None, gain=None, var=None):
    '''x if x>=0, else 0.'''
    return x if x >= 0 else 0

def relu_slope(x, var=None):
    '''1 if x>=0, else 0.'''
    return 1.0 if x >= 0 else 0.0

def leaky_relu(x, thresh=None, gain=None, var=0.01):
    '''Var is the slope of the curve for x < 0.'''
    return x if x >= 0 else -var

def leaky_relu_slope(x, var=0.01):
    '''Var is the slope of the curve for x < 0.'''
    return 1 if x >= 0 else var

def same_order(s1, s2):
    '''Are the elements in two lists in the same order?'''
    curr_pos = 0
    for e1 in s1:
        if e1 in s2:
            next_pos = s2.index(e1)
            if next_pos >= curr_pos:
                curr_pos = next_pos
            else:
                return False
    return True

def get_point_dist(x1, y1, x2, y2):
    '''Integer Euclidian distance between points x1,y1 and x2,y2.'''
    xdiff = x1 - x2
    ydiff = y1 - y2
    return int(math.sqrt(xdiff * xdiff + ydiff * ydiff))

def get_point_angle(x1, y1, x2, y2):
    '''Angle in integer degrees between points x1,y1 and x2,y2.'''
    xdiff = x2 - x1
    ydiff = y1 - y2
    if xdiff != 0:
        tang = float(ydiff) / xdiff
    else:
        tang = 1000.0
    ang = round(math.degrees(math.atan(tang)))
    if xdiff < 0:
        ang += 180
    if ang < 0:
        ang += 360
    return int(ang)

def get_endpoint(x1, y1, angle, dist):
    '''Coordinates of end of line that starts at x1,y1 at given angle and dist.'''
    return x1 + dist * math.cos(math.radians(360 - angle)), \
           y1 + dist * math.sin(math.radians(360 - angle))

def rms(values):
    '''Root mean square of list of values.'''
    if values:
        return math.sqrt(sum([v * v for v in values]) / len(values))

## COLOR

def complement(rgb):
    '''Complement of a color in the form of a list of rgb ints.'''
    return (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])

def rgb2color(rgb):
    '''Convert a list of RGB values to a hex color string.'''
    return "#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2])

##def darken(rgb, amount):
##    return (rgb[0] - int(rgb[0] * amount),
##            rgb[1] - int(rgb[1] * amount),
##            rgb[2] - int(rgb[2] * amount))

## ARGMAX, LUCE CHOICE

def luce_choice(seq):
    '''Choose index of value in seq, treating value as probabilistic weight.'''
    # Something is non-zero.
    if some(lambda x: x != 0.0, seq):
        # Some elements may be negative, so raise them all by the minimum + LUCE_EPS
        minseq = min(seq)
        minseq = -minseq if minseq < 0 else 0.0
        incr = minseq + LUCE_EPS
        for i in range(len(seq)):
            seq[i] += incr
        total = float(sum(seq))
        ran = random.random()
        scaled_total = 0.0
        for index, elem in enumerate(seq):
            scaled_total += elem / total
            if ran < scaled_total:
                return index
    # All values are 0 or somehow we failed, pick a random position
    return random.randint(0, len(seq) - 1)

def exp_luce_choice(seq, mult=1.0):
    '''Choose index of value in seq, treating value as probabilistic weight.'''
    exp_seq = [math.exp(x * mult) for x in seq]
    total = sum(exp_seq)
    if total:
        ran = random.random()
        scaled_total = 0.0
        for index, elem in enumerate(exp_seq):
            scaled_total += elem / total
            if ran < scaled_total:
                return index
    # All values are 0(??) or somehow we failed; pick a random position
    return random.randint(0, len(seq) - 1)

def argmax(seq, func):
    '''Element of seq with the highest value for func.'''
    best = seq[0]
    best_score = func(best)
    for x in seq:
        x_score = func(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return best

def argmax_index(seq, func):
    '''Index of element of seq with the highest value for func.'''
    best_score = func(seq[0])
    best_index = 0
    for i, x in enumerate(seq):
        x_score = func(x)
        if x_score > best_score:
            best_score, best_index = x_score, i
    return best_index

def argmax2(seq, func):
    '''Element and value for seq with the highest value for func.'''
    best = seq[0]
    best_score = func(best)
    for x in seq:
        x_score = func(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return (best, best_score)

def longest(*seqs):
    '''The longest sequence in seqs.'''
    return argmax(seqs, lambda x: len(x))

## FILES, DIRECTORIES

def tarpy(direc = '.', filename = 'files', others = ()):
    '''Make gzipped tar archive called filename for .py files in direc and others.'''
    tar = tarfile.open(filename + ".tgz", "w:gz")
    for fl in [f for f in os.listdir(direc) if f[-3:] == '.py']:
        tar.add(fl)
    for o in others:
        tar.add(o)
    tar.close()

# To change directories, do
# >>> os.chdir(path)
#
# To list files in a directory, do
# >>> os.listdir(path)
# (Use '.' for current directory.)
#
# For the current directory, do
# >>> os.getcwd()
