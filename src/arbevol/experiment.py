#!/usr/bin/env python3

'''
arbevol.

main.py <experiment>
   runs a neural network experiment.

A Network consists of two or more Layers of units.

An Experiment has a Network and one more "conditions", each with a different Pattern or
Sequence Generator.

A pattern is a list of numbers for the input layer, and for supervised learning, a list
of target numbers for each output layer.

When an Experiment is run, it records error statistics.

When an Experiment is created, it is added to the dict EXPERIMENTS.

Assuming you are running this in a version of Python greater than or equal to 3.0
and this file is executable, you can run the program with graphics by typing
./main.py <exp>
where "exp" is the name of one of the Experiments.  If it doesn't match any name,
the first Experiment in EXPERIMENTS is loaded.
'''

from pattern import *
from utils import *
from network_view import *
from recur import *
import time

# Dict of Experiments
EXPERIMENTS = {}

class Experiment:
    '''A Network with a set of conditions, each with a separate pattern generator.'''

    def __init__(self, name, network, conditions = None, training=True, verbose=0):
        '''Initialize name, network, learning rate, conditions, error; add to EXPERIMENTS.'''
        self.network = network
        self.name = name
        self.errors = []
        self.trials = 0
        # Index of current pattern generator
        self.condition = 0
        self.current_error = 0.0
        self.conditions = conditions
        self.pat_gen = self.conditions[0]
        self.verbose = verbose
        self.training = training
        EXPERIMENTS[name] = self

    def step(self, train=True, show_error=False, show_act=False, verbose=0):
        '''
        Run the Experiment on one pattern, return the target pattern, error.
        '''
        # The next pattern and an integer indicating whether this is the last pattern
        # in a sequence (1) and/or the last pattern or sequence in an epoch (2)
        pat, seqfirst = self.pat_gen()
        error = self.network.step(pat, train, show_act, seqfirst=seqfirst, verbose=verbose)
        if self.training and train:
            self.errors.append(error[0])
            self.trials += 1
        if show_act and self.training:
            self.show_target(pat)
        if self.training and show_error:
            print('Pat error'.ljust(12), end=' ')
            print('%.3f' % error[0])
        if verbose and seqfirst:
            print("Beginning of new sequence")
        return pat, error[0], error[1]

    def run(self, n, train=True):
        '''
        Run the Experiment on n patterns, training it and incrementing trials
        if train is True.
        '''
        self.current_error = 0.0
        trial0 = self.trials
        for i in range(n):
            pat_err_win = self.step(train, show_error = False)
            self.current_error += pat_err_win[1]
        print(self.trials, 'trials')
        print('Run error:', end=' ')
        print('%.3f' % (self.current_error / n))

    def test(self, n):
        '''Test the network on n patterns.'''
        self.current_error = 0.0
        for i in range(n):
            pat_err_win = self.step(False, show_error=True, show_act=True)
            self.current_error += pat_err_win[1]
        print('Run error'.ljust(12), end=' ')
        print('%.3f' % (self.current_error / n))

    def reinit(self):
        '''Start from 0 trials and reinitalize weights in network.'''
        self.network.reinit()
        self.trials = 0

    def next_condition(self):
        '''Change to the next condition (if there is more than one).'''
        if len(self.conditions) > 1:
            self.condition = (self.condition + 1) % len(self.conditions)
            self.pat_gen = self.conditions[self.condition]
            self.reinit()
            print('Changing to experiment condition', self.condition)

    def show(self):
        '''Print activations for output layer of network.'''
        self.network.show()

    def show_target(self, pattern):
        '''Print the target for the pattern.'''
        print('['.rjust(12), end=' ')
        for v in pattern[1]:
            print("{: .3f}".format(v), end=' ')
        print(']')

    def show_weights(self):
        self.network.show_weights()

    def display(self):
        '''Create the graphical GUI for the Experiment.'''
        root = Tk()
        NNFrame(root, self).mainloop()

#################
### Some examples
#################

### OR, no hidden layer
or_bin_x = Experiment('or1', Network('or_pa', layers = [Layer('in', 2, array=False),
                                                        Layer('out', 1, array=False)],
                      array=False),
                      conditions = [or_bin])

or_bin_xa = Experiment('or1a', Network('or_pa', layers = [Layer('in', 2), Layer('out', 1)]),
                       conditions = [or_bin2])
# copy the weights from the list network to the array network
or_bin_xa.network.assign_weights(or_bin_x.network.layers[1].weights[0])

or_sig_x = Experiment('or2', Network('or_pa', layers = [Layer('in', 2, bipolar=False),
                                                       Layer('out', 1, bipolar=False)]),
                   conditions = [or_pos])

### XOR, hidden layer
xor_sig_x = Experiment('xor1',
                        Network('xor_pa',
                                layers = [Layer('in', 2),
                                          Layer('hid', 2, lr = .2, bipolar=False),
                                          Layer('out', 1, lr = .2, bipolar=False)]),
                        conditions = [xor_sig])

xor_bin_x = Experiment('xor2',
                        Network('xor_bin',
                                layers = [Layer('in', 2),
                                          Layer('hid', 2, lr = .2),
                                          Layer('out', 1, lr = .2)]),
                        conditions = [xor_bin])

## Auto-association
# easy
aa_42_exp = Experiment('aa42',
                       Network('aa', layers = [Layer('in', 4, array=False),
                                               Layer('hid', 2, lr=.2, bipolar=False, array=False),
                                               Layer('out', 4, lr=.2, bipolar=False, array=False)],
                               array=False),
                       conditions = [aa_4_pg])

aa_42_exp_a = Experiment('aa42a',
                       Network('aa', layers = [Layer('in', 4, array=True),
                                               Layer('hid', 2, lr=.2, bipolar=False, array=True),
                                               Layer('out', 4, lr=.2, bipolar=False, array=True)],
                               array=True),
                       conditions = [aa_4_pg])

# still failing after 1,000,000
aa_41_exp = Experiment('aa41',
                       Network('aa', layers = [Layer('in', 4),
                                               Layer('hid', 1, lr=.005, bipolar=False),
                                               Layer('out', 4, lr=.005, bipolar=False)]),
                       conditions = [aa_4_pg])

# 200,000
aa_6_exp = Experiment('aa6',
                      Network('aa', layers = [Layer('in', 6),
                                              Layer('hid', 2, lr=.01, bipolar=False),
                                              Layer('out', 6, lr=.01, bipolar=False)]),
                      conditions = [aa_6_pg])

## Simple tanh test
#tanh_x = Experiment('tanh_x',
#                    Network('tanh',
#                            layers = [Layer('in', 2),
#                                      Layer('out', 1)]),
#                    conditions = [PatGen(4, patterns = [ [[1,1],[1]], [[1,-1],[1]], [[-1,1],[1]], [[-1,-1],[-1]] ])])

## Simple relu test
relu_or_x = Experiment('relu_or_x',
                    Network('relu',
                            layers = [Layer('in', 2, linear=True, rectified=True),
                                      Layer('out', 1, linear=True, rectified=True)]),
                    conditions = [PatGen(4, patterns = [ [[1,1],[1]], [[1,0],[0]], [[0,1],[0]], [[0,0],[0]] ])],
                    verbose=0)

## Leaky relu test
leaky_and_x = Experiment('leaky_and_x',
                         Network('leaky',
                                 layers = [Layer('in', 2, linear=True, rectified=True, leaky=True),
                                           Layer('out', 1, linear=True, rectified=True, leaky=True)]),
                         conditions = [PatGen(4, patterns = [ [[1,1],[1]], [[1,0],[1]], [[0,1],[1]], [[0,0],[0]] ])],
                         verbose=0)

srn_leaky0 = RecurrentNetwork('srn_leaky0',
                              layers=[Layer('in', 6, linear=True, rectified=True, leaky=True),
                                      Layer('hid', 3, lr=.1, linear=True, rectified=True, leaky=True),
                                      Layer('out', 6, lr=.1, linear=True, rectified=True, leaky=True)])

srn_leaky0_x = Experiment('srn_leaky0',
                          srn_leaky0,
                          conditions=[predict6])

## Recurrent network experiments

# Recurrence not required
srn0 = RecurrentNetwork('srn0',
                        layers=[Layer('in', 6),
                                Layer('hid', 3, lr=.1),
                                Layer('out', 6, lr=.1)])

gru0 = RecurrentNetwork('srn0',
                        layers=[Layer('in', 6),
                                GRULayer('hid', 3, lr=.1),
                                Layer('out', 6, lr=.1)])

srn0_x = Experiment('srn0',
                    srn0,
                    conditions=[predict6_bin])

gru0_x = Experiment('srn0',
                    gru0,
                    conditions=[predict6_bin])

# 150,000
intervene_x = Experiment("Intervene",
                         RecurrentNetwork('intervene',
                                          layers=[Layer('in', 1),
                                                  Layer('hid', 4, lr=.01),
                                                  Layer('out', 1, lr=.01)]),
                         conditions=[intervene])

# 140,000
intervene_gru = Experiment("Intervene",
                           RecurrentNetwork('intervene',
                                            layers=[Layer('in', 1),
                                                    GRULayer('hid', 4, lr=.01),
                                                    Layer('out', 1, lr=.01)]),
                           conditions=[intervene])

srn_class = RecurrentNetwork('srn_class',
                             layers=[Layer('in', 5),
                                     Layer('hid', 4, lr=.001),
                                     Layer('out', 1, lr=.001)])

gru_class = RecurrentNetwork('srn_class',
                             layers=[Layer('in', 5),
                                     GRULayer('hid', 4, lr=.001),
                                     Layer('out', 1, lr=.001)])

# at 400,000 no progress
srn_class_x = Experiment('srn_class_x',
                         srn_class,
                         conditions=[classify5_bin])

# 360,000
gru_class_x = Experiment('gru_class_x',
                         gru_class,
                         conditions=[classify5_bin])

# :-(
updown_x = Experiment("Up and Down",
                      RecurrentNetwork('3x3', reinit_seq=False,
                                       layers=[Layer('in', 3),
                                               Layer('hid', 5, lr=.001),
                                               Layer('out', 3, lr=.001)]),
                      conditions=[updown])

# 550,000!
gruupdown_x = Experiment("Up and Down",
                      RecurrentNetwork('3x3', reinit_seq=False,
                                       layers=[Layer('in', 3),
                                               GRULayer('hid', 5, lr=.001),
                                               Layer('out', 3, lr=.001)]),
                      conditions=[updown])

## Recurrence required, succeed at least sometimes

# takes a long time (300,000) to get the target's sign right. error hovers around .3
remem1_x = Experiment("Remember Last",
                        RecurrentNetwork('lasttarg', reinit_seq=False,
                                         layers=[Layer('in', 1),
                                                 Layer('hid', 5, lr=.001),
                                                 Layer('out', 1, lr=.001)]),
                        conditions=[remem1])

# succeeded at 400,000 pattern updates
remem2_x = Experiment("Remember 2 Back",
                        RecurrentNetwork('lasttarg', reinit_seq=False,
                                         layers=[Layer('in', 1),
                                                 Layer('hid', 8, lr=.001),
                                                 Layer('out', 1, lr=.001)]),
                        conditions=[remem2])

# succeeds at 300,000
gru_remem2_x = Experiment("Remember -2",
                        RecurrentNetwork('remem -2', reinit_seq=False,
                                         layers=[Layer('in', 1),
                                                 GRULayer('hid', 6, lr=.001),
                                                 Layer('out', 1, lr=.001)]),
                        conditions=[remem2])

intervene1_x = Experiment("Intervene1",
                         RecurrentNetwork('intervene1',
                                          layers=[Layer('in', 1),
                                                  Layer('hid', 4, lr=.001),
                                                  Layer('out', 1, lr=.001)]),
                         conditions=[intervene1])

intervene1_gru = Experiment("Intervene1",
                         RecurrentNetwork('intervene1',
                                          layers=[Layer('in', 1),
                                                  GRULayer('hid', 4, lr=.01),
                                                  Layer('out', 1, lr=.001)]),
                         conditions=[intervene1])

remem1_gru = Experiment("Remember Last (GRU)",
                        RecurrentNetwork('remem1', reinit_seq=False,
                                         layers=[Layer('in', 1),
                                                 GRULayer('hid', 4, lr=.02),
                                                 Layer('out', 1, lr=.02)]),
                        conditions=[remem1])

# 200,000
feature_x = Experiment("Remember Feature",
                       RecurrentNetwork('feature',
                                        layers=[Layer('in', 2),
                                                Layer('hid', 4, lr=.001),
                                                Layer('out', 1, lr=.001)]),
                       conditions=[feat0])

# 480,000!
grufeature_x = Experiment("Remember Feature",
                       RecurrentNetwork('feature',
                                        layers=[Layer('in', 2),
                                                GRULayer('hid', 4, lr=.001),
                                                Layer('out', 1, lr=.001)]),
                       conditions=[feat0])

# 140,000
counter_x = Experiment("Counter",
                       RecurrentNetwork('srnseq', reinit_seq=False,
                                        layers=[Layer('in', 1),
                                                Layer('hid', 3, lr=.001),
                                                Layer('out', 3, lr=.001)]),
                       conditions=[counter])

grucounter_x = Experiment("Counter",
                       RecurrentNetwork('srnseq', reinit_seq=False,
                                        layers=[Layer('in', 1),
                                                GRULayer('hid', 3, lr=.001),
                                                Layer('out', 3, lr=.001)]),
                       conditions=[counter])

class2_x = Experiment('class2_x',
                      RecurrentNetwork('class2',
                                       layers=[Layer('in', 2),
                                               Layer('hid', 4, lr=.005),
                                               Layer('out', 1, lr=.005)]),
                      conditions=[class2])

# 450,000
class3_x = Experiment('class3_x',
                      RecurrentNetwork('class3',
                                       layers=[Layer('in', 3),
                                               Layer('hid', 4, lr=.001),
                                               Layer('out', 2, lr=.001)]),
                      conditions=[class3])

# 320,000 (H smaller)
gruclass3_x = Experiment('gruclass_x',
                   RecurrentNetwork('gru',
                                    layers=[Layer('in', 3),
                                            GRULayer('hid', 3, lr=.001),
                                            Layer('out', 2, lr=.001)]),
                   conditions=[class3])

# 150,000
backforth_x = Experiment('backforth_x',
                         RecurrentNetwork('4x1',
                                          layers=[Layer('in', 4),
                                                 Layer('hid', 4, lr=.001),
                                                 Layer('out', 1, lr=.001)]),
                         conditions=[backforth])

grubackforth_x = Experiment('backforth_x',
                         RecurrentNetwork('4x1',
                                          layers=[Layer('in', 4),
                                                 GRULayer('hid', 4, lr=.001),
                                                 Layer('out', 1, lr=.001)]),
                         conditions=[backforth])

# FAILS
srnseq_x = Experiment('srnseq',
                      RecurrentNetwork('srnseq',
                                       layers=[Layer('in', 4),
                                               Layer('hid', 4, lr=.001),
                                               Layer('out', 1, lr=.001)]),
                      conditions=[seqclass])

# 800,000
gruseq_x = Experiment('gruseq_x',
                   RecurrentNetwork('gru',
                                    layers=[Layer('in', 4),
                                            GRULayer('hid', 5, lr=.001),
                                            Layer('out', 1, lr=.001)]),
                                    conditions=[seqclass])

## Sequence generation learning

# requires about 200,000 updates
seqgen_x = Experiment('seqgen',
                      RecurrentNetwork('seqgen',
                                       layers=[Layer('in', 3),
                                               Layer('hid', 4, lr=.003, weight_range=1.0),
                                               Layer('out', 3, lr=.003, weight_range=1.0)],
                                       source='o'),
                      conditions=[seqgen])

aaabbb_in_x = Experiment('aaabbb in',
                      RecurrentNetwork('aaabbb in', reinit_seq=False,
                                       layers=[Layer('in', 2),
                                               Layer('hid', 4, lr=.001),
                                               Layer('out', 2, lr=.001)]),
                      conditions=[aaabbb_in])

# This hard-wired network remembers a feature on the first time step for three
# more steps, recalling it when a second remember feature is turned on
gru_test1_x = Experiment('GRU test1',
                         RecurrentNetwork('gru1', supervised=False,
                                          layers=[Layer('in', 2),
                                                  GRULayer('hid', 2,
                                                           spec_weights=[[6.0, 0.0,   9.0, 0.0,   -1.0],
                                                                         [0.0, 3.0,   0.0, 0.0,   -2.0]],
                                                           update_spec_weights=[[4.0, 0.0,   0.0, 0.0,   2.0],
                                                                                [0.0, 0.0,   0.0, 0.0,   0.0]]),
                                                  Layer('out', 1, spec_weights=[[2.0, 2.0,  0.0]])]),
                         training=False,
                         conditions=[gru_test1_pats])

# 450,000; doesn't use reset units at all
gru_train1_x = Experiment('GRU train1',
                           RecurrentNetwork('gru1',
                                          layers=[Layer('in', 2),
                                                  GRULayer('hid', 3, lr=.01),
                                                  Layer('out', 1, lr=.01)]),
                           conditions=[gru_train1])

RECUR_X = [aaabbb_in_x, seqgen_x, gruseq_x, gru_class_x, remem1_x, gru_remem2_x]
