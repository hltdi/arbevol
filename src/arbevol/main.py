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

from population import *

print("Welcome to arbevol, a program for investigating the learning of evolution of words.")

def make_pops(nmeanings=100, nhid=20, mlength=6, flength=6, noise=0.1,
              compprob=0.5, prodprob=0.0, mvalues=4):
    iconpop = Population(2, iconic=True, nmeanings=nmeanings, nhid=nhid,
                         mlength=mlength, flength=flength, noise=noise,
                         compprob=compprob, prodprob=prodprob, mvalues=mvalues)
    arbpop = Population(2, iconic=False, nmeanings=nmeanings, nhid=nhid,
                        mlength=mlength, flength=flength, noise=noise,
                        compprob=compprob, prodprob=prodprob, mvalues=mvalues)
    return iconpop, arbpop


def make_exp(nmeanings=25, nhid=10, mlength=6, flength=6, iconpop=None, arbpop=None,
             noise=0.2, compprob=0.5, prodprob=0.0, mvalues=4):
    if not iconpop:
        iconpop, arbpop = make_pops(nmeanings=nmeanings, nhid=nhid,
                                    mlength=mlength, flength=flength, noise=noise,
                                    compprob=compprob, prodprob=prodprob, mvalues=mvalues)
    iconpers = iconpop[1]
    arbpers = arbpop[1]
    iconexp = iconpers.make_experiment()
    arbexp = arbpers.make_experiment()
    return iconexp, arbexp

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Specify an experiment or a world file
        # Do graphics
        root = Tk()
        # Find an Experiment with the name passed as the first argument
        experiment = EXPERIMENTS.get(sys.argv[1], None)
        if not experiment:
            if len(EXPERIMENTS) > 0:
                experiment = list(EXPERIMENTS.values())[0]
        if experiment:
            experiment.display()
#    else:
#        print('You need to specify an experiment')
