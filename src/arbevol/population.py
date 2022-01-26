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
        print("Creating population of size {} with lexicon of size {}".format(size, nmeanings))
        for i in range(size):
            self.add()
        # Initialize Teacher

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
            self.compexp, self.prodexp, self.jointexp = \
            self.create_experiments(self, self)
            self.init()
#            self.compexp, self.prodexp = self.make_experiments()
#            self.jointexp = self.make_joint_experiment()
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
        Experiment("PCE{}".format(self.id),
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

    def create_experiments(self, teacher, student):
        # Teacher's lexicon
        lexicon = teacher.lexicon
        entries = lexicon.entries
        # Student's networks
        compnet = student.networks[0]
        prodnet = student.networks[1]
        jointnet = student.joint_network
        # Patterns and pattern generation functions
        comppats = lexicon.comppats #[l.make_comprehension_IT(simple=True) for l in entries]
        prodpats = lexicon.prodpats # [l.make_production_IT(simple=True) for l in entries]
        jointpats = lexicon.jointpats # [[l.get_meaning(), l.get_meaning()] for l in entries]
        meaningtargfunc = lambda: [e[0] for e in entries]
        formtargfunc = lambda: [e[1] for e in entries]
        comppatfunc = self.make_patfunc(comppats, meaningtargfunc)
        prodpatfunc = self.make_patfunc(prodpats, formtargfunc)
        jointpatfunc = self.make_patfunc(jointpats, meaningtargfunc)
        comp = \
        Experiment("C{}".format(student.id), network=compnet,
                   test_nearest=True, conditions=[[comppatfunc, comppatfunc]])
        prod = \
        Experiment("P{}".format(student.id), network=prodnet,
                   test_nearest=True, conditions=[[prodpatfunc, prodpatfunc]])
        joint = \
        Experiment("PC{}".format(self.id), network=jointnet,
                   test_nearest=True, conditions=[[jointpatfunc, jointpatfunc]])
        return comp, prod, joint

    # def address(self, addressee):
    #     """
    #     Try all of self's Lex forms on addressee, comparing addressee's output
    #     meanings to input meanings to self.
    #     """

    def init(self, trials_per_lex=300, joint_trials_per_lex=200):
        """
        Train on the Person's own Lexicon.
        """
        nlex = self.population.nmeanings
        print("{} initializing themself...".format(self))
        # Train comprehension network
        print("\nTRAINING COMPREHENSION NETWORK")
        self.compexp.run(trials_per_lex * nlex, lr=0.1)
        # Train production network
        print("\nTRAINING PRODUCTION NETWORK")
        self.networks[1].reconnect()
        self.prodexp.run(trials_per_lex * nlex, lr=0.1)
        # Train joint network
        print("\nTRAINING JOINT NETWORK")
        self.run_joint(self.jointexp, joint_trials_per_lex * nlex, lr=0.01)
        # Save successful form representations from joint network in lexicon
        print("\nUPDATING LEXICON FORMS")
        run_err, miss_err, index_err, form_dict = \
        self.jointexp.test_all(record=2)
        self.lexicon.update_forms(form_dict, index_err)
        if not index_err:
            return
        # At least one target category error, so retrain

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

    # def make_joint_TS_networks(self, student):
    #     """
    #     Make networks combining this (teacher) and student networks.
    #     """
    #     # Student production, teacher comprehension
    #     network1 = Network.join(student.networks[1], self.networks[0])
    #     # Teacher production, student comprehension
    #     network2 = Network.join(self.networks[1], student.networks[0])
    #     return network1, network2

    def make_joint_TS_experiments(self, student):
        jointpatfunc = \
        self.make_patfunc(self.lexicon.jointpats,
                          lambda: [e[0] for e in self.lexicon.entries])
        # Student production, teacher comprehension
        network1 = Network.join(student.networks[1], self.networks[0])
        # Teacher production, student comprehension
        network2 = Network.join(self.networks[1], student.networks[0])
        exp1 = \
        Experiment("P{}->C{}".format(student.id, self.id),
                   network=network1, test_nearest=True,
                   conditions=[[jointpatfunc, jointpatfunc]])
        exp2 = \
        Experiment("P{}->C{}".format(self.id, student.id),
                   network=network2, test_nearest=True,
                   conditions=[[jointpatfunc, jointpatfunc]])
        return exp1, exp2

    def gen_PC_target(self, jointnet, lexicon, miss_index, incr=0.1):
        """
        A function that generates a form target for the output layer of
        the prodnet, which is the middle layer of the joint PC network.
        """
        # The form output by the network
        form = jointnet.layers[2].get_activations()
        # The form associated with the incorrect meaning
        miss_form = lexicon.entries[miss_index].get_form()
        # Differences between form dimensions
        diffs = form - miss_form
        nearest_index = np.argmin(np.abs(diffs))
        nearest_diff = diffs[nearest_index]
        target = np.copy(form)
        targ_incr = incr if nearest_diff > 0 else -incr
        target[nearest_index] += targ_incr
        return target

    def run_joint(self, exp, ntrials, lr=0.01):
        """
        Reconnect joint network, run it, and then reconnect prod network.
        """
        self.joint_network.reconnect()
        exp.run(ntrials, lr=lr)
        self.networks[1].reconnect()

    def test_joint(self, exp, record=2, update=False, verbose=0):
        """
        Reconnect joint network, test it, and then reconnect prod network.
        """
        self.joint_network.reconnect()
        run_err, miss_err, index_err, form_dict =\
        exp.test_all(record=record, verbose=verbose)
        self.networks[1].reconnect()
        if update:
            self.lexicon.update_forms(form_dict, index_err)
        return run_error, miss_err, index_err, form_dict

    def adjust_misses(self, exp, layer=2, verbose=0):
        """
        Test on the joint multi-Person experiment, for each miss,
        adjusting the form representation from the lower network's
        output away from the form representation associated with the
        incorrect meaning.
        """
        run_err, miss_err, index_err, form_dict = \
        exp.test_all(record=layer, verbose=verbose)
        if index_err:
            # There were missed meanings
            for pindex, err_index in index_err.items():
                orig_form = self.lexicon.entries[pindex].get_form()
                err_form = self.lexicon.entries[err_index].get_form()
#                print("** Error with {}->{}".format(pindex, err_index))
#                print("** Moving {} away from {}".format(orig_form, err_form))
                orig_form.separate(err_form)

#    @staticmethod
#    def run_teaching_joint(teacher, student, exp, ntrials, lr=0.01):
#        """
#        Run network, then reset LR for component networks.
#        """

    def teach(self, student, trials_per_lex=300, joint_trials_per_lex=200):
        """
        Train student on self's current lexicon.
        """
        print("{} TEACHING {}...".format(self, student))
        compexp, prodexp, jointexp = self.create_experiments(self, student)
        nlex = self.population.nmeanings
        flength = self.population.flength
        environment = self.population.environment
        nvalues = environment.mvalues
        # Train comprehension network
        print("\nTRAINING COMPREHENSION NETWORK")
        compexp.run(trials_per_lex * nlex)
        # Train production network
        print("\nTRAINING PRODUCTION NETWORK")
        student.networks[1].reconnect()
        prodexp.run(trials_per_lex * nlex)
        # Train joint network
        print("\nTRAINING JOINT NETWORK")
        student.run_joint(jointexp, joint_trials_per_lex * nlex, lr=0.01)
        # Save successful form representations from joint network in lexicon
        print("\nASSIGNING STUDENT LEXICON FORMS")
        run_err, miss_err, index_err, form_dict = jointexp.test_all(record=2)
        # For forms that failed, use teacher's forms
        student_lex_patterns = \
        [[m, Form(form_dict.get(i, gen_array(nvalues, flength)))] \
         for i, (m, f) in enumerate(self.lexicon.entries)]
        student.lexicon = \
        Lexicon(student.id, environment, flength, nlex=nlex,
                patterns=student_lex_patterns)
#        self.lexicon.update_forms(form_dict)
        print("\nCREATING TEACHER-STUDENT JOINT NETWORKS AND EXPERIMENTS")
        joint_ts_exp1, joint_ts_exp2 = self.make_joint_TS_experiments(student)
#        joint_ts1, joint_ts2 = self.make_joint_TS_networks(student)
#        joint_ts_exp1 = \
#        Experiment("PE{}->CE{}".format(student.id, self.id),
#                   network=joint_ts1, test_nearest=True,
#                   conditions=[[jointpatfunc, jointpatfunc]])
#        joint_ts_exp2 = \
#        Experiment("PE{}->CE{}".format(self.id, student.id),
#                   network=joint_ts2, test_nearest=True,
#                   conditions=[[jointpatfunc, jointpatfunc]])
        print("\nTraining Student->Teacher Network")
#        joint_ts_exp1.run(joint_trials_per_lex * nlex,
#                          lr={3: 0.005, 4: 0.005, 1: 0.01, 2: 0.01})
        print("\nTraining Teaching->Student Network")
#        joint_ts_exp2.run(joint_trials_per_lex * nlex,
#                          lr={1: 0.005, 2: 0.005, 3: 0.01, 4: 0.01})
        # Reset LR
#        self.networks[0].reset_lr()
#        self.networks[1].reset_lr()
        return joint_ts_exp1, joint_ts_exp2

    def communicate(self, other):
        '''
        Test comprehension and production between this and another Person.
        '''
        print("\n{} AND {} COMMUNICATING".format(self, other))
        joint_exp1, joint_exp2 = self.make_joint_TS_experiments(other)
        print("\n{} SPEAKING, {} LISTENING".format(other, self))
        joint_exp1.test_all(record=2)
        print("\n{} SPEAKING, {} LISTENING".format(self, other))
        joint_exp2.test_all(record=2)
