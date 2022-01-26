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

    def __init__(self, name, network, conditions=None, training=True,
                 test_nearest=False, record_only_hits=True, verbose=0):
        '''
        Initialize name, network, learning rate, conditions, error;
        add to EXPERIMENTS.
        '''
        self.network = network
        self.name = name
        self.errors = []
        self.trials = 0
        # Index of current pattern generator
        self.condition = 0
        self.current_error = 0.0
        self.conditions = conditions
#        self.pat_gen = self.conditions[0][0]
#        self.test_pat_gen = self.conditions[0][-1]
        self.verbose = verbose
        self.training = training
        self.test_nearest = test_nearest
        # Record only hits
        self.record_only_hits = record_only_hits
        self.time = time.strftime("%y%m%d.%H%M")
        EXPERIMENTS[name] = self

    def __repr__(self):
        return "{}:{}".format(self.name, self.time)

    def step(self, train=True, patgen=None, show_error=False, show_act=False,
             input_layer=0, output_layer=-1,
             pg_kind='full', lr=None, index=-1, verbose=0):
        '''
        Run the Experiment on one pattern, return the target pattern, error.
        '''
        # The next pattern and an integer indicating whether this is the last pattern
        # in a sequence (1) and/or the last pattern or sequence in an epoch (2)
        if not patgen:
            patgen = self.get_patgen(train=train, kind=pg_kind)
        pat, seqfirst = patgen(index=index)
        if not pat:
            return pat, 0.0, 0,0
        error = \
        self.network.step(pat, train=train, seqfirst=seqfirst, lr=lr,
                          input_layer=input_layer, output_layer=output_layer,
                          show_act=show_act, verbose=verbose)
        if self.training and train:
            self.errors.append(error[0])
            self.trials += 1
        if show_act and self.training:
            self.show_target(pat)
        if self.training and show_error:
            print('Pat error'.ljust(12), end=' ')
            print('{: .3f}'.format(error[0]))
        if verbose and seqfirst:
            print("Beginning of new sequence")
        return pat, error[0], error[1]

    def run(self, n=5000, train=True, lr=None,
            input_layer=0, output_layer=-1,
            show_act=False,
            test_every=1000,
            error_thresh=0.02, error_change_thresh=-0.05, miss_thresh=0.0):
        '''
        Run the Experiment on n patterns, training it and incrementing trials
        if train is True.
        '''
        self.current_error = 0.0
        trial0 = self.trials
        patgen = self.get_patgen(train=train)
        test_error = 10.0
        miss_error = 0.0
        misses = None
        record = None
        trials = 0
        for i in range(n):
            trials += 1
            pat_err_win = self.step(train, patgen=patgen, lr=lr,
                                    input_layer=input_layer, output_layer=output_layer,
                                    show_act=show_act, show_error=False)
            self.current_error += pat_err_win[1]
            if test_every and i % test_every == 0:
                print("Testing at {} trials: ".format(i), end=' ')
                old_test_error = test_error
                test_error, miss_error, misses, record = self.test_all()
                error_change = old_test_error - test_error
                if test_error <= error_thresh or \
                    miss_error <= miss_thresh or \
                    error_change < error_change_thresh:
                    break
        error = self.current_error / trials
        print("\n{} TRIALS".format(self.trials))
        print('RUN ERROR: {:.3f}'.format(error))
        if test_every:
            print('TEST ERROR: {:.3f}'.format(test_error))
            if self.test_nearest:
                print('MISS ERROR: {:.3f}'.format(miss_error))
        return error, test_error, miss_error

    def test_all(self, pg_kind='full', reps=1,
                 input_layer=0, output_layer=-1,
                 record=False, verbose=0):
        '''Test the network on all patterns reps times.'''
        current_error = 0.0
        nearest_misses = 0
        nearest_error = 0.0
        patgen = self.get_patgen(train=False, kind=pg_kind)
        pindex_errors = {}
        recorded = {}
        for repetition in range(reps):
            pindex = 0
            while True:
                pat_err_win = self.step(False, patgen=patgen, index=pindex,
                                        input_layer=input_layer, output_layer=output_layer,
                                        show_error=verbose>0, show_act=verbose>0)
                if not pat_err_win[0]:
                    # step() returns empty pattern for this pindex
                    break
                current_error += pat_err_win[1]
                target = pat_err_win[0][1]
                nearest_misses = \
                self.do_nearest(pindex=pindex, patgen=patgen, target=target,
                                pindex_errors=pindex_errors, record=record,
                                recorded=recorded, nearest_misses=nearest_misses,
                                verbose=0)
                if verbose:
                    print()
                pindex += 1

        nitems = pindex * reps
#        print("N items: {}".format(nitems))
        # At this point pindex is number of test patterns
        run_error = current_error / nitems
        print('Run error: {: .3f}'.format(run_error), end=' ')
        if self.test_nearest:
            nearest_error = nearest_misses / nitems
#            for i in pindex_errors:
#                pindex_errors[i] /= reps
            print("Miss error: {: .2f}".format(nearest_error))
        else:
            print()
        if record:
            for i in recorded:
                recorded[i] /= reps
                recorded[i] = np.round_(recorded[i], 2)
        return run_error, nearest_error, pindex_errors, recorded

    def do_nearest(self, pindex=-1, patgen=None, target=None,
                   pindex_errors=None, record=None, recorded=None, nearest_misses=0,
                   verbose=0):
        if self.test_nearest:
            x, nrst, nrst_i = self.nearest(target, patgen, verbose=verbose)
            hit = nrst_i == pindex
            if not hit:
                nearest_misses += 1
                # Assumes reps=1 or only records the last miss
                pindex_errors[pindex] = nrst_i
                if verbose:
                    print("Missed target, found {}".format(nrst_i))
            elif verbose:
                print("Hit target")
            if record != None and pindex >= 0 and (hit or not self.record_only_hits):
                layer_index = record if isinstance(record, int) else -1
                net_out = self.network.layers[layer_index].activations
                record_out = np.array([a for a, t in zip(net_out, target) if t != NO_TARGET])
                if pindex not in recorded:
                    recorded[pindex] = record_out
                else:
                    recorded[pindex] += record_out
            return nearest_misses

    def test(self, n, pg_kind='full', verbose=1):
        '''Test the network on n patterns.'''
        current_error = 0.0
        # nearest_misses = 0
        # nearest_error = 0.0
        patgen = self.get_patgen(train=False, kind=pg_kind)
        for i in range(n):
            pat_err_win = self.step(False, patgen=patgen, show_error=verbose>0, show_act=verbose>0)
            current_error += pat_err_win[1]
            # if self.test_nearest:
            #     target = pat_err_win[0][1]
            #     hit, nrst, nrst_i = self.nearest(target, patgen, verbose=verbose)
            #     if not hit:
            #         nearest_misses += 1
            #         if verbose:
            #             print("Missed target category, found {}".format(nrst_i))
            #     elif verbose:
            #         print("Hit target category")
            if verbose:
                print()
        run_error = current_error / n
        print('TEST RUN ERROR'.ljust(12), end=' ')
        print("{: .3f}".format(run_error))
        # if self.test_nearest:
        #     nearest_error = nearest_misses / n
        #     print("TEST MISS ERROR: {: .2f}".format(nearest_error))
        return run_error # , nearest_error

    def nearest(self, target, patgen, network=None, verbose=0):
        """
        Whether the current output is closer to target than any other pattern.
        """
        network = network or self.network
        output = network.get_output()
#        patgen = self.test_pat_gen if test else self.pat_gen
        correct, nearest, nearest_index = \
          patgen.get_nearest(output, target, verbose=verbose)
        return correct, nearest, nearest_index

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

    def get_patgen(self, train=True, kind=''):
        """
        Get the current pattern generation for training or testing.
        kind is a string that is used if there is a dict of different
        PatGens.
        """
        condition = self.conditions[self.condition]
        # Condition is a pair: training, testing
        patgen = condition[0] if train else condition[1]
        if isinstance(patgen, dict):
            if kind:
                return patgen[kind]
            else:
                # kind not specified, get first patgen found
                return list(patgen.values())[0]
        # only one patgen specified
        return patgen

    def show(self):
        '''Print activations for output layer of network.'''
        self.network.show()

    def show_target(self, pattern):
        '''Print the target for the pattern.'''
        print('['.rjust(12), end=' ')
        for v in pattern[1]:
            if v == DONT_CARE:
                print(" X   ", end=' ')
            else:
                print("{: .2f}".format(v), end=' ')
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
                      conditions = [[or_bin]])

or_bin_xa = Experiment('or1a', Network('or_pa', layers = [Layer('in', 2), Layer('out', 1)]),
                       conditions = [[or_bin2]])
# copy the weights from the list network to the array network
or_bin_xa.network.assign_weights(or_bin_x.network.layers[1].weights[0])

or_sig_x = Experiment('or2', Network('or_pa', layers = [Layer('in', 2, bipolar=False),
                                                       Layer('out', 1, bipolar=False)]),
                   conditions = [[or_pos]])

### XOR, hidden layer
xor_sig_x = Experiment('xor1',
                        Network('xor_pa',
                                layers = [Layer('in', 2),
                                          Layer('hid', 2, lr = .2, bipolar=False),
                                          Layer('out', 1, lr = .2, bipolar=False)]),
                        conditions = [[xor_sig]])

xor_bin_x = Experiment('xor2',
                        Network('xor_bin',
                                layers = [Layer('in', 2),
                                          Layer('hid', 2, lr = .2),
                                          Layer('out', 1, lr = .2)]),
                        conditions = [[xor_bin]])

## Auto-association
# easy
aa_42_exp = Experiment('aa42',
                       Network('aa', layers = [Layer('in', 4, array=False),
                                               Layer('hid', 2, lr=.2, bipolar=False, array=False),
                                               Layer('out', 4, lr=.2, bipolar=False, array=False)],
                               array=False),
                       conditions = [[aa_4_pg]])

aa_42_exp_a = Experiment('aa42a',
                       Network('aa', layers = [Layer('in', 4, array=True),
                                               Layer('hid', 2, lr=.2, bipolar=False, array=True),
                                               Layer('out', 4, lr=.2, bipolar=False, array=True)],
                               array=True),
                       conditions = [[aa_4_pg]])

# still failing after 1,000,000
aa_41_exp = Experiment('aa41',
                       Network('aa', layers = [Layer('in', 4),
                                               Layer('hid', 1, lr=.005, bipolar=False),
                                               Layer('out', 4, lr=.005, bipolar=False)]),
                       conditions = [[aa_4_pg]])

# 200,000
aa_6_exp = Experiment('aa6',
                      Network('aa', layers = [Layer('in', 6),
                                              Layer('hid', 2, lr=.01, bipolar=False),
                                              Layer('out', 6, lr=.01, bipolar=False)]),
                      conditions = [[aa_6_pg]])

### Sequential

# srn_leaky0 = RecurrentNetwork('srn_leaky0',
#                               layers=[Layer('in', 6, linear=True, rectified=True, leaky=True),
#                                       Layer('hid', 3, lr=.1, linear=True, rectified=True, leaky=True),
#                                       Layer('out', 6, lr=.1, linear=True, rectified=True, leaky=True)])
#
# srn_leaky0_x = Experiment('srn_leaky0',
#                           srn_leaky0,
#                           conditions=[predict6])
#
# ## Recurrent network experiments
#
# # Recurrence not required
# srn0 = RecurrentNetwork('srn0',
#                         layers=[Layer('in', 6),
#                                 Layer('hid', 3, lr=.1),
#                                 Layer('out', 6, lr=.1)])
#
# gru0 = RecurrentNetwork('srn0',
#                         layers=[Layer('in', 6),
#                                 GRULayer('hid', 3, lr=.1),
#                                 Layer('out', 6, lr=.1)])
#
# srn0_x = Experiment('srn0',
#                     srn0,
#                     conditions=[predict6_bin])
#
# gru0_x = Experiment('srn0',
#                     gru0,
#                     conditions=[predict6_bin])
#
# # 150,000
# intervene_x = Experiment("Intervene",
#                          RecurrentNetwork('intervene',
#                                           layers=[Layer('in', 1),
#                                                   Layer('hid', 4, lr=.01),
#                                                   Layer('out', 1, lr=.01)]),
#                          conditions=[intervene])
#
# # 140,000
# intervene_gru = Experiment("Intervene",
#                            RecurrentNetwork('intervene',
#                                             layers=[Layer('in', 1),
#                                                     GRULayer('hid', 4, lr=.01),
#                                                     Layer('out', 1, lr=.01)]),
#                            conditions=[intervene])
#
# srn_class = RecurrentNetwork('srn_class',
#                              layers=[Layer('in', 5),
#                                      Layer('hid', 4, lr=.001),
#                                      Layer('out', 1, lr=.001)])
#
# gru_class = RecurrentNetwork('srn_class',
#                              layers=[Layer('in', 5),
#                                      GRULayer('hid', 4, lr=.001),
#                                      Layer('out', 1, lr=.001)])
#
# # at 400,000 no progress
# srn_class_x = Experiment('srn_class_x',
#                          srn_class,
#                          conditions=[classify5_bin])
#
# # 360,000
# gru_class_x = Experiment('gru_class_x',
#                          gru_class,
#                          conditions=[classify5_bin])
#
# # :-(
# updown_x = Experiment("Up and Down",
#                       RecurrentNetwork('3x3', reinit_seq=False,
#                                        layers=[Layer('in', 3),
#                                                Layer('hid', 5, lr=.001),
#                                                Layer('out', 3, lr=.001)]),
#                       conditions=[updown])
#
# # 550,000!
# gruupdown_x = Experiment("Up and Down",
#                       RecurrentNetwork('3x3', reinit_seq=False,
#                                        layers=[Layer('in', 3),
#                                                GRULayer('hid', 5, lr=.001),
#                                                Layer('out', 3, lr=.001)]),
#                       conditions=[updown])
#
# ## Recurrence required, succeed at least sometimes
#
# # takes a long time (300,000) to get the target's sign right. error hovers around .3
# remem1_x = Experiment("Remember Last",
#                         RecurrentNetwork('lasttarg', reinit_seq=False,
#                                          layers=[Layer('in', 1),
#                                                  Layer('hid', 5, lr=.001),
#                                                  Layer('out', 1, lr=.001)]),
#                         conditions=[remem1])
#
# # succeeded at 400,000 pattern updates
# remem2_x = Experiment("Remember 2 Back",
#                         RecurrentNetwork('lasttarg', reinit_seq=False,
#                                          layers=[Layer('in', 1),
#                                                  Layer('hid', 8, lr=.001),
#                                                  Layer('out', 1, lr=.001)]),
#                         conditions=[remem2])
#
# # succeeds at 300,000
# gru_remem2_x = Experiment("Remember -2",
#                         RecurrentNetwork('remem -2', reinit_seq=False,
#                                          layers=[Layer('in', 1),
#                                                  GRULayer('hid', 6, lr=.001),
#                                                  Layer('out', 1, lr=.001)]),
#                         conditions=[remem2])
#
# intervene1_x = Experiment("Intervene1",
#                          RecurrentNetwork('intervene1',
#                                           layers=[Layer('in', 1),
#                                                   Layer('hid', 4, lr=.001),
#                                                   Layer('out', 1, lr=.001)]),
#                          conditions=[intervene1])
#
# intervene1_gru = Experiment("Intervene1",
#                          RecurrentNetwork('intervene1',
#                                           layers=[Layer('in', 1),
#                                                   GRULayer('hid', 4, lr=.01),
#                                                   Layer('out', 1, lr=.001)]),
#                          conditions=[intervene1])
#
# remem1_gru = Experiment("Remember Last (GRU)",
#                         RecurrentNetwork('remem1', reinit_seq=False,
#                                          layers=[Layer('in', 1),
#                                                  GRULayer('hid', 4, lr=.02),
#                                                  Layer('out', 1, lr=.02)]),
#                         conditions=[remem1])
#
# # 200,000
# feature_x = Experiment("Remember Feature",
#                        RecurrentNetwork('feature',
#                                         layers=[Layer('in', 2),
#                                                 Layer('hid', 4, lr=.001),
#                                                 Layer('out', 1, lr=.001)]),
#                        conditions=[feat0])
#
# # 480,000!
# grufeature_x = Experiment("Remember Feature",
#                        RecurrentNetwork('feature',
#                                         layers=[Layer('in', 2),
#                                                 GRULayer('hid', 4, lr=.001),
#                                                 Layer('out', 1, lr=.001)]),
#                        conditions=[feat0])
#
# # 140,000
# counter_x = Experiment("Counter",
#                        RecurrentNetwork('srnseq', reinit_seq=False,
#                                         layers=[Layer('in', 1),
#                                                 Layer('hid', 3, lr=.001),
#                                                 Layer('out', 3, lr=.001)]),
#                        conditions=[counter])
#
# grucounter_x = Experiment("Counter",
#                        RecurrentNetwork('srnseq', reinit_seq=False,
#                                         layers=[Layer('in', 1),
#                                                 GRULayer('hid', 3, lr=.001),
#                                                 Layer('out', 3, lr=.001)]),
#                        conditions=[counter])
#
# class2_x = Experiment('class2_x',
#                       RecurrentNetwork('class2',
#                                        layers=[Layer('in', 2),
#                                                Layer('hid', 4, lr=.005),
#                                                Layer('out', 1, lr=.005)]),
#                       conditions=[class2])
#
# # 450,000
# class3_x = Experiment('class3_x',
#                       RecurrentNetwork('class3',
#                                        layers=[Layer('in', 3),
#                                                Layer('hid', 4, lr=.001),
#                                                Layer('out', 2, lr=.001)]),
#                       conditions=[class3])
#
# # 320,000 (H smaller)
# gruclass3_x = Experiment('gruclass_x',
#                    RecurrentNetwork('gru',
#                                     layers=[Layer('in', 3),
#                                             GRULayer('hid', 3, lr=.001),
#                                             Layer('out', 2, lr=.001)]),
#                    conditions=[class3])
#
# # 150,000
# backforth_x = Experiment('backforth_x',
#                          RecurrentNetwork('4x1',
#                                           layers=[Layer('in', 4),
#                                                  Layer('hid', 4, lr=.001),
#                                                  Layer('out', 1, lr=.001)]),
#                          conditions=[backforth])
#
# grubackforth_x = Experiment('backforth_x',
#                          RecurrentNetwork('4x1',
#                                           layers=[Layer('in', 4),
#                                                  GRULayer('hid', 4, lr=.001),
#                                                  Layer('out', 1, lr=.001)]),
#                          conditions=[backforth])
#
# # FAILS
# srnseq_x = Experiment('srnseq',
#                       RecurrentNetwork('srnseq',
#                                        layers=[Layer('in', 4),
#                                                Layer('hid', 4, lr=.001),
#                                                Layer('out', 1, lr=.001)]),
#                       conditions=[seqclass])
#
# # 800,000
# gruseq_x = Experiment('gruseq_x',
#                    RecurrentNetwork('gru',
#                                     layers=[Layer('in', 4),
#                                             GRULayer('hid', 5, lr=.001),
#                                             Layer('out', 1, lr=.001)]),
#                                     conditions=[seqclass])
#
# ## Sequence generation learning
#
# # requires about 200,000 updates
# seqgen_x = Experiment('seqgen',
#                       RecurrentNetwork('seqgen',
#                                        layers=[Layer('in', 3),
#                                                Layer('hid', 4, lr=.003, weight_range=1.0),
#                                                Layer('out', 3, lr=.003, weight_range=1.0)],
#                                        source='o'),
#                       conditions=[seqgen])
#
# aaabbb_in_x = Experiment('aaabbb in',
#                       RecurrentNetwork('aaabbb in', reinit_seq=False,
#                                        layers=[Layer('in', 2),
#                                                Layer('hid', 4, lr=.001),
#                                                Layer('out', 2, lr=.001)]),
#                       conditions=[aaabbb_in])
#
# # This hard-wired network remembers a feature on the first time step for three
# # more steps, recalling it when a second remember feature is turned on
# gru_test1_x = Experiment('GRU test1',
#                          RecurrentNetwork('gru1', supervised=False,
#                                           layers=[Layer('in', 2),
#                                                   GRULayer('hid', 2,
#                                                            spec_weights=[[6.0, 0.0,   9.0, 0.0,   -1.0],
#                                                                          [0.0, 3.0,   0.0, 0.0,   -2.0]],
#                                                            update_spec_weights=[[4.0, 0.0,   0.0, 0.0,   2.0],
#                                                                                 [0.0, 0.0,   0.0, 0.0,   0.0]]),
#                                                   Layer('out', 1, spec_weights=[[2.0, 2.0,  0.0]])]),
#                          training=False,
#                          conditions=[gru_test1_pats])
#
# # 450,000; doesn't use reset units at all
# gru_train1_x = Experiment('GRU train1',
#                            RecurrentNetwork('gru1',
#                                           layers=[Layer('in', 2),
#                                                   GRULayer('hid', 3, lr=.01),
#                                                   Layer('out', 1, lr=.01)]),
#                            conditions=[gru_train1])
#
# RECUR_X = [aaabbb_in_x, seqgen_x, gruseq_x, gru_class_x, remem1_x, gru_remem2_x]
