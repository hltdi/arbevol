"""
arbevol: Population of Persons.
"""

from lex import *

class Population(list):

    def __init__(self, size, iconic=True, init_teachers=0.1,
                 mlength=6, flength=6, nhid=20, nmeanings=10, noise=0.1,
                 mvalues=4,
                 compprob=0.0, prodprob=0.0):
        self.size = size
        self.mlength = mlength
        self.flength = flength
        self.mvalues = mvalues
        self.nhid = nhid
        self.iconic = iconic
        self.nmeanings = nmeanings
        self.noise = noise
        self.compprob = compprob
        self.prodprob = prodprob
        self.make_environment()
        for i in range(size):
            self.add()

    def add(self):
        self.append(Person(self, teacher=not self))

    def make_environment(self):
        """
        Create an environment of Meanings. ADD CLUSTERING PARAMETERS LATER.
        """
        e = Environment(self.mvalues, self.mlength, nmeanings=self.nmeanings)
        self.environment = e

class Person:

    n=0

    def __init__(self, population, teacher=True):
        self.id = Person.n
        self.teacher = teacher
        self.population = population
        self.noise = population.noise
        Person.n += 1
        if teacher:
            self.network = None
        else:
            self.network = self.make_network(population.mlength, population.flength, population.nhid)
        # Create an initial lexicon for teachers
        if teacher:
            self.lexicon = \
            self.make_lexicon(iconic=population.iconic,
                              compprob=population.compprob,
                              prodprob=population.prodprob)
        else:
            self.lexicon = None

    def __repr__(self):
        return ":-){}".format(self.id)

    def make_network(self, mlength, flength, nhid):
        return \
        Network(str(self.id),
                layers = [Layer('in', mlength+flength),
                          Layer('hid', nhid),
                          Layer('out', mlength+flength)])

    def make_lexicon(self, iconic=True, compprob=0.0, prodprob=0.0):
        return \
        Lexicon(self.id, self.population.environment, self.population.flength,
                noise=self.noise, iconic=iconic,
                compprob=compprob, prodprob=prodprob)

    def make_experiment(self, name='lex_exp'):
        return Experiment(name, network=self.network,
                          conditions=self.lexicon.exp_conds,
                          test_nearest=True)

    def teach(self, student, comptrain=1.0, prodtrain=0.0):
        """
        Present training patterns to Person student.
        """
        name = 'Teaching'
        conditions = self.lexicon.spec_exp_conditions(comptrain, prodtrain)
        exp = Experiment(name, network=student.network,
                         conditions=conditions,
                         test_nearest=True)
        return exp
