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

from experiment import *

print("Welcome to መረበኛ, a simple neural network visualization tool.")

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
