'''
arbevol.
Pattern and sequence generators for perception and action.
'''

from utils import *
# import numpy as np

# Pattern elements
DONT_CARE = -100.0

# Noise to add to patterns
NOISE = .05
DEFAULT_ACT = 0.0

### Some utility functions used in creating patterns and sequences

def gen_random_sequence(length=10, array=True):
    l = [gen_random_value(False, True) for x in range(length)]
    if array:
        return l2a(l)
    return l

def remember_sequence(length=10, distance=1, array=True):
    seq = gen_random_sequence(length=length, array=False)
    target = seq[-distance:] + seq[:-distance]
#    target = [seq[-1]] + seq[:-1]
    return [[[s], [t]] for s, t in zip(seq, target)]

def gen_random_value(continuous = True, bipolar = False):
    '''A random value for a pattern.'''
    val = ran(continuous)
    if bipolar:
        val = 2 * val - 1
    return val

def string2value(string, continuous=True, target=False, bipolar=False, squeeze=True):
    '''Convert a string to a pattern value.'''
    if string == DONT_CARE:
        return DONT_CARE
    else:
        val = float(string) if continuous else int(string)
        if squeeze:
            if val == 1:
                val = .9
            elif val == 0:
                val = .1
        return val

def string2number(string, continuous = True, bipolar = False):
    if string == DONT_CARE:
        return gen_random_value(continuous, bipolar)
    else:
        print("DON'T KNOW HOW TO CONVERT", string)

def string2numberlist(string, bipolar=True, array=True):
    l = [(1 if x=='1' else (-1 if x == '0' else 0)) for x in string]
    if array:
        return l2a(l)
    return l

def stringpair2pattern(pair, array=True):
    return [string2numberlist(pair[0], array=array),
            string2numberlist(pair[1], array=array)]

def stringpairlist2sequence(l, array=True):
    return [stringpair2pattern(p, array=array) for p in l]

def smooth(x):
    """Make x (a target value) avoid extremes."""
    if x == 1:
        return .9
    elif x == 0:
        return .1
    elif x == -1:
        return -.9
    else:
        return x

def smooth_pattern(p):
    for i, x in enumerate(p):
        p[i] = smooth(x)

class PatGen:
    '''A generator for patterns (lists).'''

    def __init__(self,
                 n,                                   # Length of input patterns
                 function = None,                     # Explicit function to call on each step
                 target = 0,                          # Length of target; if 0, no target
                 noise = 0.0, targ_noise = 0.0,       # Noise rates for input and target
                 ran_prob = 0.0,                      # Probability of a random pattern
                 squeeze = True,                      # Whether to keep targets away from asymptotes
                 continuous = True, bipolar = False,  # Kinds of input pattern values
                 default = DEFAULT_ACT,               # Default pattern value
                 filename = None,                     # File to read in patterns from
                 patterns = [],                       # Explicit list or array of patterns
                 sequential = False,                  # Whether patterns must appear in order
                 smooth = True,                       # Whether to smooth targets
                 check = True,                        # Whether to check for errors as patters are created
                 random_choice = True,                # Whether to choose patterns randomly
                 targets = None,                      # Function to generate targets
                 array = True):                       # Whether to create arrays rather than lists
        '''Set pattern parameters and create patterns from patterns list or file.'''
        self.n = n
        self.function = function
        self.random = random
        self.noise = noise
        self.smooth = smooth
        self.targ_noise = targ_noise
        self.continuous = continuous
        self.bipolar = bipolar
        self.ran_prob = ran_prob
        self.squeeze = squeeze
        self.min = -1 if bipolar else 0
        self.check = check
        self.sequential = sequential
        # Length of target pattern
        self.target = target
        # The function that generates the pattern
        self.gen_value = lambda: self.gen()
        self.array = array
        self.random_choice = random_choice
        # only relevant if random_choice is False; keeps track of current pattern
        self.pindex = 0
        self.targets = targets
        if filename:
            self.read_patterns(filename)
        else:
            self.patterns = patterns
        if self.target and self.smooth:
            for i, t in self.patterns:
                smooth_pattern(t)
        if sequential:
            self.init_iterator()
        if check:
            self.check_patterns()

    def check_patterns(self):
        '''Check to see whether patterns are the right length.'''
        for p in self.patterns:
            inp = p[0] if self.target else p
            if len(inp) > self.n:
                print('WARNING: input pattern', inp, 'is too long!')
            elif len(inp) > self.n:
                print('WARNING: input pattern', inp, 'is too short!')
            if self.target:
                target = p[1]
                if len(target) > self.target:
                    print('WARNING: target pattern', target, 'is too long!')
                elif len(target) > self.target:
                    print('WARNING: target pattern', target, 'is too short!')

    def init_iterator(self):
        self.iterator = iter(self.patterns)

    def string2pattern(self, string_list, target):
        l = [string2value(s, continuous=self.continuous, target=target, bipolar=self.bipolar, squeeze=self.squeeze) \
                for s in string_list]
        if self.array:
            return l2a(l)
        return l

    def read_patterns(self, filename):
        '''Read the patterns in from a file, ignoring blank lines.'''
        try:
            pat_file = open(filename)
            self.patterns = []
            inp = True
            for line in pat_file:
                pattern = self.string2pattern(line.split(), (not inp and self.target))
                if not self.target:
                    self.patterns.append(pattern)
                elif inp:
                    pat = [pattern]
                    inp = False
                else:
                    pat.append(pattern)
                    self.patterns.append(pat)
                    inp = True

            pat_file.close()
        except IOError:
            print('No file named', filename)

    def gen_random(self):
        '''A random pattern.'''
        l = [gen_random_value(self.continuous, self.bipolar) for x in range(self.n)]
        if self.array:
            return l2a(l)
        return l

    def next(self, repeat=True):
        '''For sequential patterns, get the next one, resetting the iterator if at end.'''
        try:
            pattern = next(self.iterator)
            return pattern
        except StopIteration:
#            print 'End of sequence'
            self.init_iterator()
            return next(self.iterator) if repeat else False

    def gen(self, repeat=True):
        '''
        Generate a pattern. Returns pattern followed by index, representing
        whether this is the end of an epoch.
        '''
        # Call the function
        if self.function:
            return self.function()
        # Generate a random pattern
        elif (self.ran_prob > 0.0 and random.random() < self.ran_prob) or not self.patterns:
            return self.gen_random(), 0
        # Select from one of the predefined patterns
        else:
            if self.sequential:
                pattern = self.next(repeat)
            elif not self.random_choice:
                pattern = self.patterns[self.pindex]
                self.pindex = (self.pindex + 1) % len(self.patterns)
            else:
                pattern = random.choice(self.patterns)
            return self.realize(pattern), 0

    def realize(self, pattern):
        '''
        Return copy of pattern, noisified and with DONT_CAREs filled in
        (unless target).
        '''
        inp = self.pattern_input(pattern)
        if self.noise > 0:
            self.noisify(inp)
            self.convert(inp)
        if self.target:
            targ = self.pattern_target(pattern)
            if self.targ_noise > 0:
                self.noisify(targ)
            return inp, targ
        return inp

    def convert(self, ls):
        '''Convert all non-numeric characters to numeric ones.'''
        for i, x in enumerate(ls):
            if isinstance(x, str):
                ls[i] = string2number(x, continuous=self.continuous, bipolar=self.bipolar)

    def pattern_input(self, pattern):
        '''A copy of the input part of the pattern.'''
        return pattern[0][:] if self.target else pattern[:]

    def pattern_target(self, pattern):
        '''
        A copy of the target part of the pattern; does not check whether this one.
        '''
        return pattern[1][:]

    def noisify(self, pat):
        '''Add noise to elements of pat.'''
        for i in range(len(pat)):
            if isinstance(i, str):
                # Don't do anything if there's a string here
                pass
            elif self.continuous:
                change = random.random() * self.noise
                if random.random() < .5:
                    change = -change
                    pat[i] = max(self.min, min(1.0, pat[i] + change))
                elif random.random() < self.noise:
                    if self.bipolar:
                        vals = [-1, 0, 1]
                        vals.remove(pat[i])
                        pat[i] = random.choice(vals)
                    else:
                        pat[i] = 0 if pat[i] else 1

    def get_targets(self):
        """
        The list of target patterns.
        """
        if self.targets:
            return self.targets()
        elif self.patterns:
            return [p[1] for p in self.patterns]
        return None

    def get_nearest(self, pattern, target, verbose=0):
        """
        Find the nearest pattern among patterns to patterns,
        returning that pattern and True if it's target, False if it's not.
        """
        targets = self.get_targets()
        if not targets:
            print("No targest for get_nearest!")
            return None, False
        nearest = nearest_array(pattern, targets)
#        if verbose:
#            print("** nearest to {}:\n{}".format(pattern, nearest))
#            print("** {}".format(nearest is target))
        return nearest, nearest is target

    def __call__(self):
        '''What to do when the object is called.'''
        return self.gen(True)

class SeqGen:
    '''Generator for sequences of patterns.'''

    def __init__(self, pat_length, target = 0,   # No target if length = 0
                 noise = NOISE, targ_noise = NOISE,
                 ran_prob = .1, continuous = True, bipolar = False, default = DEFAULT_ACT,
                 # Sequences from file or lists passed directly to constructor
                 filename = None, sequences = [],
                 smooth=True,
                 check=True,
                 array=True):
        self.pat_length = pat_length
        self.target = target
        self.random = random
        self.noise = noise
        self.targ_noise = targ_noise
        self.continuous = continuous
        self.bipolar = bipolar
        self.smooth = smooth
        self.ran_prob = ran_prob
        self.min = -1 if bipolar else 0
        self.check = check
        self.gen_value = lambda: self.gen()
        self.array = array
        if filename:
            self.read_sequences(filename)
        else:
            self.sequences = [self.make_pat_gen(s) for s in sequences]
        self.init_iterator()
#        if check:
#            for s in self.sequences:
#                s.check_patterns()

    def make_pat_gen(self, patterns):
        '''Create a pattern generator for one sequence, given a list of patterns.'''
        return PatGen(n=self.pat_length, target=self.target,
                      noise=self.noise, ran_prob=self.ran_prob,
                      continuous=self.continuous, bipolar=self.bipolar,
                      patterns=patterns, sequential=True, smooth=self.smooth,
                      check=self.check,
                      array=self.array)

    def init_iterator(self):
        '''Pick a sequence and initialize its iterator.'''
        # Pick a random sequence
        self.sequence=random.choice(self.sequences)
        self.sequence.init_iterator()

    def gen(self):
        '''Generate another pattern.'''
        next=self.sequence.next(repeat = False)
        if not next:
            # End of current sequence; go to a new one
            self.init_iterator()
            return self.sequence.next(repeat=False), 1
        else:
            return next, 0

    def read_sequences(self, filename):
        '''Read in sequences from file, treating blank lines as separators.'''
        try:
            self.sequences = []
            seq_file = open(filename)
            pats = []
            inp = True
            for line in seq_file:
                split = line.split()
                if not split:
                    # Line is empty, end current seq
                    if not inp and self.target:
                        print('ODD NUMBER OF LINES IN SEQUENCE, MISSING TARGET')
                        pats = pats[:-1]
                    self.sequences.append(self.make_pat_gen(pats))
                    pats = []
                else:
                    pattern = [string2value(s, continuous = self.continuous, bipolar = self.bipolar,
                                           target = (not inp and self.target)) \
                               for s in split]
                    if self.array:
                        pattern = l2a(pattern)
                    if not self.target:
                        pats.append(pattern)
                    elif inp:
                        pat = [pattern]
                        inp = False
                    else:
                        pat.append(pattern)
                        pats.append(pat)
                        inp = True
            # End the last sequence
            if pats:
                self.sequences.append(self.make_pat_gen(pats))

            seq_file.close()

        except IOError:
            print('No file named', filename)

    def __call__(self):
        '''What to do when the object is called.'''
        return self.gen()

# Some examples of pattern generators

# XOR
xor_sig = PatGen(2, target=1, continuous=False,
                patterns = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]])


xor_bin = PatGen(2, target=1, continuous=False,
                patterns = [[[-1, -1], [-1]], [[-1, 1], [1]], [[1, -1], [1]], [[1, 1], [-1]]])

# OR
or_bin = PatGen(2, target=1, continuous=False, random_choice=False,
               patterns = [[[1, 1], [1]], [[-1, 1], [1]], [[1, -1], [1]], [[-1, -1], [-1]]])

or_bin2 = PatGen(2, target=1, continuous=False, random_choice=False,
               patterns = [[[1, 1], [1]], [[-1, 1], [1]], [[1, -1], [1]], [[-1, -1], [-1]]])

or_pos = PatGen(2, target=1, continuous=False,
               patterns = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [1]]])

# 2 binary categories, length 4
cl_4_2_pg = PatGen(4, patterns = [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]])

# Auto-associative local patterns of length 4
aa_4_pg = PatGen(4, patterns = [ [[1, 0, 0, 0], [1, 0, 0, 0]],
                                 [[0, 1, 0, 0], [0, 1, 0, 0]],
                                 [[0, 0, 1, 0], [0, 0, 1, 0]],
                                 [[0, 0, 0, 1], [0, 0, 0, 1]] ])

# Auto-associative local patterns of length 6
aa_6_pg = PatGen(6, patterns = [ [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
                                 [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
                                 [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
                                 [[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]],
                                 [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]],
                                 [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]] ])

def gen_gauss():
    '''A number between 0 and 1 with a strong tendency to be near .5.'''
    return max([min([random.gauss(.5, .2), 1.0]), 0.0])

def gen_intensities(points, gauss = False):
    '''Distances of a random point in 2-space from each of points.'''
    if gauss:
        position = gen_gauss(), gen_gauss()
    else:
        position = random.random(), random.random()
    return [intensity(position, point, minimum = .05) for point in points]

# 2 random inputs
unit_square_pg = PatGen(2)

# 2 random inputs clustering around .5
unit_square_gauss_pg = PatGen(2, lambda: [gen_gauss(), gen_gauss()])

# "Intensities" of random points in 2-space from two "lights"
lights37_pg = PatGen(2, function = lambda: gen_intensities(((.3, .3), (.7, .7))))

lights01_pg = PatGen(2, function = lambda: gen_intensities(((0, 0), (1, 1))))

# "Intensities" of random points clustered around .5, .5 from two "lights"
lights37_gauss_pg = PatGen(2, function = lambda: gen_intensities(((.3, .3), (.7, .7)), True))

lights01_gauss_pg = PatGen(2, function = lambda: gen_intensities(((0, 0), (1, 1)), True))

# Sequence generators

aaabbb_in = SeqGen(2, target = 2,
                      sequences = [ [ [ [1, 0], [1, 0] ],
                                      [ [1, 0], [1, 0] ],
                                      [ [1, 0], [0, 1] ],
                                      [ [0, 1], [0, 1] ],
                                      [ [0, 1], [0, 1] ],
                                      [ [0, 1], [1, 0] ]
                                      ]
                                    ])

aaabbb_out = SeqGen(1, target = 2,
                       sequences = [ [ [ [1], [1, 0] ],
                                       [ [1], [0, 1] ]
                                       ]
                                     ])

predict6_bin = SeqGen(6, target=6,
                  sequences=[ [ [ [1, -1, -1, -1, -1, -1], [-1, 1, -1, -1, -1, -1] ],
                                [ [-1, 1, -1, -1, -1, -1], [-1, -1, 1, -1, -1, -1] ],
                                [ [-1, -1, 1, -1, -1, -1], [-1, -1, -1, 1, -1, -1] ],
                                [ [-1, -1, -1, 1, -1, -1], [-1, -1, -1, -1, 1, -1] ],
                                [ [-1, -1, -1, -1, 1, -1], [-1, -1, -1, -1, -1, 1] ],
                                [ [-1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1] ]
                                ]
                                ])

classify5_bin = SeqGen(5, target=1,
                  sequences=[ [ [ [1, -1, -1, -1, -1], [1] ],
                                [ [1, -1, -1, -1, -1], [1] ],
                                [ [-1, 1, -1, -1, -1], [1] ],
                                [ [-1, 1, -1, -1, -1], [1] ],
                                [ [-1, -1, 1, -1, -1], [1] ],
                                [ [-1, -1, 1, -1, -1], [1] ],
                                [ [-1, -1, -1, 1, -1], [1] ],
                                [ [-1, -1, -1, 1, -1], [1] ],
                                [ [-1, -1, -1, -1, 1], [1] ],
                                [ [-1, -1, -1, -1, 1], [1] ] ],
                              [ [ [1, -1, -1, -1, -1], [-1] ],
                                [ [-1, 1, -1, -1, -1], [-1] ],
                                [ [-1, -1, 1, -1, -1], [-1] ],
                                [ [-1, -1, -1, 1, -1], [-1] ],
                                [ [-1, -1, -1, -1, 1], [-1] ] ],
                                ])

predict6 = SeqGen(6, target=6,
                  sequences=[ [ [ [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0] ],
                                [ [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0] ],
                                [ [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0] ],
                                [ [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0] ],
                                [ [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1] ],
                                [ [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0] ]
                                ]
                                ])

backforth = SeqGen(4, target=1,
                   sequences= [ [
                                  # forward
                                  [ [1, -1, -1, -1], [1] ],
                                  [ [-1, 1, -1, -1], [1] ],
                                  [ [-1, -1, 1, -1], [1] ],
                                  [ [-1, -1, -1, 1], [1] ] ],
                                [
                                  # backward
                                  [ [-1, -1, -1, 1], [-1] ],
                                  [ [-1, -1, 1, -1], [-1] ],
                                  [ [-1, 1, -1, -1], [-1] ],
                                  [ [1, -1, -1, -1], [-1] ] ],
                                 ])

updown = SeqGen(3, target=3,
                sequences = [ [ [ [1, -1, -1], [-1, 1, -1] ],
                                [ [-1, 1, -1], [-1, -1, 1] ],
                                [ [-1, -1, 1], [-1, 1, -1] ],
                                [ [-1, 1, -1], [1, -1, -1] ] ] ])

# classify a sequence by whether the first element is 1 or -1
seqclass = SeqGen(4, target=1,
                  sequences=[ [ [ [1, -1, -1, -1], [-1] ],
                                [ [-1, 1, -1, -1], [-1] ],
                                [ [-1, -1, 1, -1], [-1] ],
                                [ [-1, -1, -1, 1], [+1] ] ],
                              [ [ [-1, -1, -1, -1], [-1] ],
                                [ [-1, 1, -1, -1], [-1] ],
                                [ [-1, -1, 1, -1], [-1] ],
                                [ [-1, -1, -1, 1], [-1] ] ]
                            ])

class3s = [[("100", "00"),
           ("010", "00"),
           ("001", "10")],
          [("100", "00"),
           ("001", "00"),
           ("100", "10")],
          [("100", "00"),
           ("001", "00"),
           ("010", "10")],
          [("010", "00"),
           ("100", "00"),
           ("010", "01")],
          [("010", "00"),
           ("010", "00"),
           ("100", "01")],
          [("010", "00"),
           ("010", "00"),
           ("001", "01")]]

class2s = [[("10", "0"),
            ("01", "1")],
           [("10", "0"),
            ("10", "1")],
           [("01", "0"),
            ("01", "0")],
           [("01", "0"),
            ("10", "0")]]

class3 = SeqGen(3, target=2,
                sequences=[stringpairlist2sequence(x) for x in class3s])

class2 = SeqGen(2, target=1,
                sequences=[stringpairlist2sequence(x) for x in class2s])

counter_s = [[("1", "100"),
              ("1", "010"),
              ("1", "001"),
              ("0", "000")],
             [("1", "100"),
              ("1", "010"),
              ("0", "000")],
             [("1", "100"),
              ("0", "000")]]

counter = SeqGen(1, target=3,
                 sequences=[stringpairlist2sequence(x) for x in counter_s])


feat0_s = \
 [[ ["10", "0"],
    ["01", "0"],
    ["00", "1"] ],
  [ ["10", "0"],
    ["00", "0"],
    ["01", "1"] ],
  [ ["10", "0"],
    ["01", "0"],
    ["01", "1"] ],
  [ ["10", "0"],
    ["00", "0"],
    ["00", "1"] ],
  [ ["00", "0"],
    ["01", "0"],
    ["00", "0"] ],
  [ ["00", "0"],
    ["00", "0"],
    ["01", "1"] ],
  [ ["00", "0"],
    ["01", "0"],
    ["01", "1"] ],
  [ ["00", "0"],
    ["00", "0"],
    ["00", "0"] ] ]

feat0 = SeqGen(2, target=1,
               sequences=[stringpairlist2sequence(x) for x in feat0_s])

# repeat first element after two intervening irrelevant elements
intervene_s = \
  [ # starting with 1
   [ ["1", "0"],
     ["0", "0"],
     ["1", "1"],
     ["0", "1"] ],
   [ ["1", "0"],
     ["0", "0"],
     ["0", "0"],
     ["0", "1"] ],
   [ ["1", "0"],
     ["1", "1"],
     ["0", "0"],
     ["0", "1"] ],
   [ ["1", "0"],
     ["1", "1"],
     ["1", "1"],
     ["0", "1"] ],
   # starting with 0
   [ ["0", "0"],
     ["0", "0"],
     ["1", "1"],
     ["0", "0"] ],
   [ ["0", "0"],
     ["0", "0"],
     ["0", "0"],
     ["0", "0"] ],
   [ ["0", "0"],
     ["1", "1"],
     ["0", "0"],
     ["0", "0"] ],
   [ ["0", "0"],
     ["1", "1"],
     ["1", "1"],
     ["0", "0"] ]
     ]

intervene = SeqGen(1, target=1,
                   sequences=[stringpairlist2sequence(x) for x in intervene_s])

# repeat first element after one intervening irrelevant element
intervene1_s = \
  [ # starting with 1
   [ ["1", "0"],
     ["1", "1"],
     ["0", "1"] ],
   [ ["1", "0"],
     ["0", "0"],
     ["0", "1"] ],
   # starting with 0
   [ ["0", "0"],
     ["0", "0"],
     ["0", "0"] ],
   [ ["0", "0"],
     ["1", "1"],
     ["0", "0"] ],
     ]

intervene1 = SeqGen(1, target=1,
                   sequences=[stringpairlist2sequence(x) for x in intervene1_s])

remem1 = SeqGen(1, target=1,
                sequences = [remember_sequence(50, 1)])

remem2 = SeqGen(1, target=1,
                sequences = [remember_sequence(50, 2)])

gru_test1_pats = SeqGen(2, target=0,
                       sequences = [  # on
                                    [ [+1, 0],
                                      [-1, 0],
                                      [-1, 0],
                                      [-1, 0],
                                      [-1, 1]],
                                    [ [-1, 0],
                                      [-1, 0],
                                      [-1, 0],
                                      [-1, 0],
                                      [-1, 1]] ])

gru_train1 = SeqGen(2, target=1,
                    sequences = [
                                 [ [[+1, 0], [0]],
                                   [[-1, 0], [0]],
                                   [[-1, 0], [0]],
                                   [[-1, 1], [1]]],
                                 [ [[-1, 0], [0]],
                                   [[-1, 0], [0]],
                                   [[-1, 0], [0]],
                                   [[-1, 1], [-1]]] ])

seqgen = SeqGen(3, target=3,
                sequences = [
                              [ [[+1, -1, -1], [+1, -1, -1]],
                                [[+1, -1, -1], [-1, +1, -1]],
                                [[+1, -1, -1], [-1, -1, +1]] ],
                              [ [[-1, +1, -1], [-1, -1, +1]],
                                [[-1, +1, -1], [-1, +1, -1]],
                                [[-1, +1, -1], [+1, -1, -1]] ],
                              [ [[-1, -1, +1], [-1, +1, -1]],
                                [[-1, -1, +1], [+1, -1, -1]],
                                [[-1, -1, +1], [-1, -1, +1]] ]
                                ])
