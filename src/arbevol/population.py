"""
arbevol: Population of Persons.
"""

from lex import *

class Population(list):

    def __init__(self, size, iconic=True, init_teachers=0.1,
                 mlength=6, flength=6, nhid=20, nmeanings=10, noise=0.1,
                 mvalues=4,
                 compprob=0.0, prodprob=0.0):
        print("Creating population of size {}".format(size))
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

    def __init__(self, population, teacher=True, newarb=0.1):
        self.id = Person.n
        self.teacher = teacher
        self.population = population
        self.noise = population.noise
        self.newarb = newarb
        Person.n += 1
#        if teacher:
#            self.network = None
#        else:
#        self.network = self.make_network(population.mlength, population.flength, population.nhid)
        self.networks = self.make_networks(population.mlength, population.flength, population.nhid)
        self.joint_network = self.make_joint_network()
        # Create an initial lexicon for teachers
        if teacher:
            self.lexicon = \
            self.make_lexicon(iconic=population.iconic,
                              compprob=population.compprob,
                              prodprob=population.prodprob)
            self.compexp, self.prodexp = self.make_experiments()
            self.jointexp = self.make_joint_experiment()
        else:
            self.lexicon = None

    def __repr__(self):
        return "😀{}".format(self.id)

    def make_networks(self, mlength, flength, nhid):
        return \
        [Network("C{}".format(self.id),
                 layers = [Layer('formin', flength),
                           Layer('chid', nhid),
                           Layer('meanout', mlength)]),
         Network("P{}".format(self.id),
                 layers = [Layer('meanin', mlength),
                           Layer('phid', nhid),
                           Layer('formout', flength)])]

    # def make_network(self, mlength, flength, nhid):
    #     return \
    #     Network(str(self.id),
    #             layers = [Layer('in', mlength+flength),
    #                       Layer('hid', nhid),
    #                       Layer('out', mlength+flength)])
    #
    def make_joint_network(self):
        return Network.join(self.networks[1], self.networks[0])

    def make_joint_experiment(self):
        jointpatfunc = self.lexicon.make_joint_patfunc()
        return \
        Experiment("PC{}".format(self.id),
                   network=self.joint_network,
                   test_nearest=True,
                   conditions=[[jointpatfunc, jointpatfunc]])

    def make_lexicon(self, iconic=True, compprob=0.0, prodprob=0.0):
        return \
        Lexicon(self.id, self.population.environment, self.population.flength,
                nlex=self.population.nmeanings,
                noise=self.noise, iconic=iconic,
                compprob=compprob, prodprob=prodprob)

    # def make_experiment(self, name='lex_exp'):
    #     return Experiment(name, network=self.network,
    #                       conditions=self.lexicon.exp_conds,
    #                       test_nearest=True)

    def make_experiments(self):
        comppatfunc = self.lexicon.make_comp_patfunc()
        prodpatfunc = self.lexicon.make_prod_patfunc()
        comp = Experiment("CE{}".format(self.id), network=self.networks[0],
                          test_nearest=True,
                          conditions=[[comppatfunc, comppatfunc]])
        prod = Experiment("PE{}".format(self.id), network=self.networks[1],
                          test_nearest=True,
                          conditions=[[prodpatfunc, prodpatfunc]])
        return comp, prod

    def address(self, addressee):
        """
        Try all of self's Lex forms on addressee, comparing addressee's output
        meanings to input meanings to self.
        """

    def self_teach(self, trials_per_lex=200, joint_trials_per_lex=100):
        """
        Train on the Person's own Lexicon.
        """
        nlex = self.population.nmeanings
        print("{} teaching itself...".format(self))
        # Train comprehension network
        print("\nTRAINING COMPREHENSION NETWORK")
        self.compexp.run(trials_per_lex * nlex)
        # Train production network
        print("\nTRAINING PRODUCTION NETWORK")
        self.networks[1].reconnect(self.joint_network)
        self.prodexp.run(trials_per_lex * nlex)
        # Train joint network
        print("\nTRAINING JOINT NETWORK")
        self.joint_network.reconnect(self.networks[1])
        self.jointexp.run(joint_trials_per_lex * nlex, lr=0.01)
        # Save successful form representations from joint network in lexicon
        print("\nUPDATING LEXICON FORMS")
        run_err, miss_err, index_err, form_dict = \
        self.jointexp.test_all(record=2)
        self.lexicon.update_forms(form_dict)

    def make_patfunc(self, patterns, target_func):
        '''
        Make a pattern generation function for training a student on
        comprehension or production.
        '''
        nlex = self.population.nmeanings
        def patfunc(index=-1):
            if index < 0:
                input, target = random.choice(patterns)
            elif index >= nlex:
                return False, 0
            else:
                input, target = patterns[index]
            return [input, target], 0
        return PatGen(nlex, function=patfunc, targets=target_func)

    def make_joint_TS_networks(self, student):
        network1 = Network.join(student.networks[1], self.networks[0])
        network2 = Network.join(self.networks[1], student.networks[0])
#        for teachout in network1.layers[]

    def teach(self, student, trials_per_lex=400, joint_trials_per_lex=200):
        """
        Train student on self's current lexicon.
        """
        print("{} TEACHING {}...".format(self, student))
        # patterns, patfuncs, and experiments for student
        comppats = [l.make_comprehension_IT(simple=True) for l in self.lexicon.entries]
        prodpats = [l.make_production_IT(simple=True) for l in self.lexicon.entries]
        jointpats = [[l.get_meaning(), l.get_meaning()] for l in self.lexicon.entries]
        meaningtargfunc = lambda: [e[0] for e in self.lexicon.entries]
        formtargfunc = lambda: [e[1] for e in self.lexicon.entries]
        comppatfunc = self.make_patfunc(comppats, meaningtargfunc)
        prodpatfunc = self.make_patfunc(prodpats, formtargfunc)
        jointpatfunc = self.make_patfunc(jointpats, meaningtargfunc)
        compexp = Experiment("CE{}->{}".format(self.id, student.id),
                             network=student.networks[0],
                             test_nearest=True,
                             conditions=[[comppatfunc, comppatfunc]])
        prodexp = Experiment("PE{}->{}".format(self.id, student.id),
                             network=student.networks[0],
                             test_nearest=True,
                             conditions=[[prodpatfunc, prodpatfunc]])
        jointexp = Experiment("JE{}->{}".format(self.id, student.id),
                              network=student.joint_network,
                              test_nearest=True,
                              conditions=[[jointpatfunc, jointpatfunc]])
        nlex = self.population.nmeanings
        # Train comprehension network
        print("\nTRAINING COMPREHENSION NETWORK")
        compexp.run(trials_per_lex * nlex)
        # Train production network
        print("\nTRAINING PRODUCTION NETWORK")
        student.networks[1].reconnect(student.joint_network)
        prodexp.run(trials_per_lex * nlex)
        # Train joint network
        print("\nTRAINING JOINT NETWORK")
        student.joint_network.reconnect(student.networks[1])
        jointexp.run(joint_trials_per_lex * nlex, lr=0.01)
        # Save successful form representations from joint network in lexicon
        print("\nUPDATING LEXICON FORMS")
        run_err, miss_err, index_err, form_dict = jointexp.test_all(record=2)
#        self.lexicon.update_forms(form_dict)

        return compexp, prodexp, jointexp
#        jointnetworks = self.make_joint_TS_networks(student)
        # jointexp1 = \
        # Experiment("P{}->C{}E".format(self.id, student.id),
        #            network=student.joint_network,
        #            test_nearest=True,
        #            conditions=[[jointpatfunc, jointpatfunc]])
        # name = '{}->{}_T'.format(self, student)
        # conditions = self.lexicon.spec_exp_conditions(comptrain, prodtrain)
        # exp = Experiment(name, network=student.network,
        #                  conditions=conditions,
        #                  test_nearest=True)
        # # train student
        # current_error = 100.0
        # total_trials = 0
        # print("Training...")
        # while total_trials < trial_thresh:
        #     new_error = exp.run(ntrials)
        #     change = current_error - new_error
        #     if new_error < error_thresh or change < change_thresh:
        #         break
        #     current_error = new_error
        #     total_trials += ntrials
        # if make_lex:
        #     print("Creating new lexicon...")
        #     # create new lexicon for student
        #     lexicon = Lexicon.from_network(exp, self.lexicon.meanings, student)
        #     student.lexicon = lexicon
        #
        # return self, student, exp
