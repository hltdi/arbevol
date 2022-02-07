"""
arbevol: Population of Persons.
"""

from lex import *

class Population(list):

    def __init__(self, size, iconic=True, init_teachers=0.1,
                 mlength=6, flength=6, nhid=20, nmeanings=10, noise=0.1,
                 mvalues=4, clusters=None, sep_amount=0.01,
                 init_lex=None, init_meanings=None,
#                 compprob=0.0, prodprob=0.0,
                 dont_init=False):
        self.size = size
        self.mlength = mlength
        self.flength = flength
        self.mvalues = mvalues
        self.nhid = nhid
        self.iconic = iconic
        self.nmeanings = nmeanings
        self.noise = noise
        self.sep_amount = sep_amount
#        self.compprob = compprob
#        self.prodprob = prodprob
        self.teachers = set()
        # Whether all Persons have been taught a Lexicon
        self.taught = False
        self.runs = 0
        self.mutualities = []
        self.distances = []
        self.iconicities = []
        self.make_environment(clusters=clusters, init_meanings=init_meanings)
        print("Creating population of size {} with lexicon of size {}".format(size, nmeanings))
        for i in range(size):
            self.add(dont_init=dont_init, init_lex=init_lex)
        if not dont_init:
            self.teach()

    def add(self, dont_init=False, init_lex=None):
        self.append(Person(self, teacher=not self, dont_init=dont_init,
                           init_lex=init_lex))

    def extend_lexicons(self, meanings):
        """
        Add lexical entries for each of meanings. Re-init() master teacher
        and have them then teach() others.
        """
        self.nmeanings += len(meanings)
        self[0].init(meanings=meanings)

    def make_environment(self, clusters=None, init_meanings=None):
        """
        Create an environment of Meanings. ADD CLUSTERING PARAMETERS LATER.
        """
        e = \
        Environment(self.mvalues, self.mlength, nmeanings=self.nmeanings,
                    clusters=clusters, init_meanings=init_meanings)
        self.environment = e

#    def communicate(self, p1, p2, verbose=0):
#        p1.teach(p2, verbose=verbose)
#        p2.learn_from_misses(p1, verbose=verbose)

    def add_to_teachers(self, person):
        self.teachers.add(person)
        if len(self.teachers) == self.size:
            self.taught = True

    def get_untaught(self):
        return set(self) - self.teachers

    def pair(self):
        """Randomly select pair of Persons."""
        if self.taught:
            return [self[i] for i in random.sample(range(self.size), 2)]
        teacher = random.choice(list(self.teachers))
        student = random.choice(list(self.get_untaught()))
        return teacher, student

    def run(self, iterations, score=True):
        '''
        On each iteration, for each Person, select a random Person
        to communicate with.
        '''
        for i in range(iterations):
            print(">>>>>>>>ITERATION {}<<<<<<<<".format(self.runs))
            indices = list(range(self.size))
            random.shuffle(indices)
            for index in indices:
                student = self[index]
                teacher1 = self.select_other(index)
#                other_indices = list(range(self.size))
#                other_indices.remove(index)
#                teacher = self[random.choice(other_indices)]
#                t, s = self.pair()
#                student.learn_from_misses(teacher1, sep_amount=self.sep_amount)
                teacher2 = self.select_other(index)
                teacher2.teach(student)
#                self.communicate(teacher, student)
            self.runs += 1
        if score:
            self.mutualities.append((self.runs, self.mean_mutuality()))
            self.distances.append((self.runs, self.mean_distance()))
            self.iconicities.append((self.runs, self.mean_iconicity()))

    def select_other(self, index):
        """
        Given a Person index, randomly select another Person.
        """
        other_indices = list(range(self.size))
        other_indices.remove(index)
        return self[random.choice(other_indices)]

    def teach(self, one_teacher=True):
        """
        Have a Lexicon taught to the whole Population,
        by default by the single initial master teacher.
        """
        teacher = self[0]
        if teacher not in self.teachers:
            teacher.init(sep_amount=self.sep_amount)
        for student in self[1:]:
#            self.communicate(teacher, student)
            teacher.teach(student)
#            student.teach(teacher)
#            teacher.learn_from_misses(student)
#            student.learn_from_misses(teacher)
            if not one_teacher:
                teacher = student
        self.mutualities.append((0, self.mean_mutuality()))
        self.distances.append((0, self.mean_distance()))
        self.iconicities.append((0, self.mean_iconicity()))

    def mean_distance(self):
        """
        Mean distance between Lexicons of Persons.
        """
        distances = []
        for person1 in self:
            d = 0
            for person2 in self:
                if person2 != person1:
                    d += person1.lexicon.distance(person2.lexicon)
            d /= (self.size - 1)
            distances.append(d)
        return sum(distances) / self.size

    def mean_iconicity(self):
        """
        Mean iconicity of Lexicons of Persons.
        """
        return sum([p.lexicon.iconicity() for p in self]) / self.size

    def mutuality(self, p1, p2):
        """
        Measure of the extent to which Persons p1 and p2 understanding
        one another.
        """
        run_err1, miss_err1, err_index1 = p1.test_paired(p2)
        run_err2, miss_err2, err_index2 = p2.test_paired(p1)
        return (run_err1 + run_err2) / 2.0, (miss_err1 + miss_err2) / 2.0

    def mean_mutuality(self):
        m = [0.0, 0.0]
        n = 0
        for i, person1 in enumerate(self[:-1]):
            for person2 in self[i+1:]:
                m1, m2 = self.mutuality(person1, person2)
                m[0] += m1
                m[1] += m2
                n += 1
        m[0] /= n
        m[1] /= n
        return m

class Person:

    n=0

    def __init__(self, population, teacher=True, newarb=0.1, init_lex=None,
                 dont_init=False):
        # dont_init for debugging
        self.id = Person.n
        self.teacher = teacher
        self.population = population
        self.environment = population.environment
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
            self.lexicon = self.make_lexicon(iconic=population.iconic, init_lex=init_lex)
#                              compprob=population.compprob,
#                              prodprob=population.prodprob
            self.compexp, self.prodexp, self.jointexp = \
            self.create_experiments(self)
            if not dont_init:
                self.init()
#            self.compexp, self.prodexp = self.make_experiments()
#            self.jointexp = self.make_joint_experiment()
        else:
            self.compexp = self.prodexp = self.jointexp = self.lexicon = None
        # Dict of networks combining prod network of this Person with
        # comp network of Person identified by id
        self.paired_exps = {}

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
        return Network.join(self.networks[1], self.networks[0], Network.joint)

    def make_joint_experiment(self):
        mean2mean_patfunc = self.lexicon.make_joint_patfunc()
        return \
        Experiment("PCE{}".format(self.id),
                   network=self.joint_network,
                   test_nearest=True,
                   conditions=[[mean2mean_patfunc, mean2mean_patfunc]])

    def make_lexicon(self, iconic=True, init_lex=None):
        """
        Create initial lexicon for master teacher. init_lex is a file
        to read in a lexicon from.
        """
        patterns = None
        if init_lex:
            patterns = Lexicon.read(init_lex, self.population.mvalues, self.population.mvalues)
        return \
        Lexicon(self.id, self.environment, self.population.flength,
                nlex=self.population.nmeanings,
                noise=self.noise, iconic=iconic,
                patterns=patterns)

    def extend_lexicon(self, meanings, iconic=True):
        """
        For master teacher only, add new meanings to lexicon, creating new initial
        forms for them.
        """
        self.lexicon.add(meanings)

    # def make_experiment(self, name='lex_exp'):
    #     return Experiment(name, network=self.network,
    #                       conditions=self.lexicon.exp_conds,
    #                       test_nearest=True)

    # def make_experiments(self):
    #     comppatfunc = self.lexicon.make_comp_patfunc()
    #     prodpatfunc = self.lexicon.make_prod_patfunc()
    #     comp = Experiment("CE{}".format(self.id), network=self.networks[0],
    #                       test_nearest=True,
    #                       conditions=[[comppatfunc, comppatfunc]])
    #     prod = Experiment("PE{}".format(self.id), network=self.networks[1],
    #                       test_nearest=True,
    #                       conditions=[[prodpatfunc, prodpatfunc]])
    #     return comp, prod

    def create_experiments(self, student):
        # Teacher's lexicon
        lexicon = self.lexicon
        entries = lexicon.entries
        # Student's networks
        compnet = student.networks[0]
        prodnet = student.networks[1]
        jointnet = student.joint_network
        # Patterns and pattern generation functions
        comppats = lambda: lexicon.comppats #[l.make_comprehension_IT(simple=True) for l in entries]
        prodpats = lambda: lexicon.prodpats # [l.make_production_IT(simple=True) for l in entries]
#        mean2mean_pats = lambda: lexicon.mean2mean_pats # [[l.get_meaning(), l.get_meaning()] for l in entries]
        meaningtargfunc = lambda: [e[0] for e in entries]
        formtargfunc = lambda: [e[1] for e in entries]
        comppatfunc = self.make_patfunc(comppats, meaningtargfunc)
        prodpatfunc = self.make_patfunc(prodpats, formtargfunc)
        mean2mean_patfunc = self.make_mean2mean_func()
#        patfunc(mean2mean_pats, meaningtargfunc)
        comp = \
        Experiment("C{}".format(student.id), network=compnet,
                   test_nearest=True, conditions=[[comppatfunc, comppatfunc]])
        prod = \
        Experiment("P{}".format(student.id), network=prodnet,
                   test_nearest=True, conditions=[[prodpatfunc, prodpatfunc]])
        joint = \
        Experiment("PC{}".format(self.id), network=jointnet,
                   test_nearest=True, conditions=[[mean2mean_patfunc, mean2mean_patfunc]])
        return comp, prod, joint

    # def address(self, addressee):
    #     """
    #     Try all of self's Lex forms on addressee, comparing addressee's output
    #     meanings to input meanings to self.
    #     """

    def init(self, trials_per_lex=500, joint_trials_per_lex=300,
             sep_amount=0.01, new_meanings=None, verbose=0):
        """
        Train on the Person's own Lexicon.
        """
        nlex = self.population.nmeanings
        if new_meanings:
            print("MT {} RE-SELF-INITIALIZING WITH NEW MEANINGS...".format(self))
        else:
            print("MT {} SELF-INITIALIZING...".format(self))
        if verbose:
            print("** Initial lexicon")
            self.lexicon.show()
        # Train comprehension network
        print("Comprehension".format(self), end='; ')
        self.compexp.run(trials_per_lex * nlex, lr=0.1)
        # Train production network
        print("Production   ".format(self), end='; ')
        self.networks[1].reconnect()
        self.prodexp.run(trials_per_lex * nlex, lr=0.1)
        # Train joint network
        print("Prod->comp   ".format(self), end='; ')
        self.run_joint(self.jointexp, joint_trials_per_lex * nlex, lr=0.01)
        # Save successful form representations from joint network in lexicon
        print("Updating forms in {}'s lexicon".format(self))
        run_err, miss_err, index_err, form_dict = \
        self.jointexp.test_all(record=2)
#        if not index_err:
#            return
        self.lexicon.update_forms(form_dict, index_err, sep_amount=sep_amount,
                                  verbose=verbose)
        # At least one target category error, so retrain
        self.population.add_to_teachers(self)

    def make_patfunc(self, patterns_func, target_func, noisify=False):
        '''
        Make a pattern generation function for training or testing a network on
        comprehension or production or joint production-comprehension.
        '''
        nlex = self.population.nmeanings
        def patfunc(index=-1):
            patterns = patterns_func()
            if index < 0:
                input, target = random.choice(patterns)
            elif index >= nlex:
                return False, 0
            else:
                input, target = patterns[index]
            if noisify and self.noise:
                input = noisify(input, sd=self.noise)
            return [input, target], 0
        return PatGen(nlex, function=patfunc, targets=target_func)

    def make_mean2mean_func(self):
        return \
        self.make_patfunc(lambda: self.lexicon.mean2mean_pats,
                          lambda: [e.get_meaning() for e in self.lexicon.entries])

    # def make_paired_networks(self, student):
    #     """
    #     Make networks combining this (teacher) and student networks.
    #     """
    #     # Student production, teacher comprehension
    #     network1 = Network.join(student.networks[1], self.networks[0])
    #     # Teacher production, student comprehension
    #     network2 = Network.join(self.networks[1], student.networks[0])
    #     return network1, network2

    def make_paired_experiments(self, other):
        self_in_exp = other_in_exp = None
        if other.id in self.paired_exps:
            self_in_exp = self.paired_exps[other.id]
        if self.id in other.paired_exps:
            other_in_exp = other.paired_exps[self.id]
        if not self_in_exp or not other_in_exp:
            mean2mean_patfunc = self.make_mean2mean_func()
#            self.make_patfunc(lambda: self.lexicon.mean2mean_pats,
#                              lambda: [e[0] for e in self.lexicon.entries])
            if not other_in_exp:
                network1 = Network.join(other.networks[1], self.networks[0],
                                        Network.paired)
                other_in_exp = \
                Experiment("P{}->C{}".format(other.id, self.id),
                           network=network1, test_nearest=True,
                           conditions=[[mean2mean_patfunc, mean2mean_patfunc]])
                network1.disconnect()
                other.paired_exps[self.id] = other_in_exp
            if not self_in_exp:
                network2 = Network.join(self.networks[1], other.networks[0],
                                        Network.paired)
                self_in_exp = \
                Experiment("P{}->C{}".format(self.id, other.id),
                           network=network2, test_nearest=True,
                           conditions=[[mean2mean_patfunc, mean2mean_patfunc]])
                network2.disconnect()
                self.paired_exps[other.id] = self_in_exp
        return other_in_exp, self_in_exp

    def run_joint(self, exp, ntrials, lr=0.01):
        """
        Reconnect joint network, run it, and then reconnect prod network.
        """
        self.joint_network.reconnect()
        exp.run(ntrials, lr=lr)
        self.networks[1].reconnect()

    def test_joint(self, exp, record=2, update=False, sep_amount=0.01, verbose=0):
        """
        Reconnect joint network, test it, and then reconnect prod network.
        """
        self.joint_network.reconnect()
        run_err, miss_err, index_err, form_dict =\
        exp.test_all(record=record, verbose=verbose)
        self.joint_network.disconnect()
        if update:
            self.lexicon.update_forms(form_dict, index_err, sep_amount=sep_amount)
        return run_err, miss_err, index_err, form_dict

    def test_paired(self, other, record=None, verbose=0):
        """
        Test the paired network (self's production, other's comprehension).
        """
        exp = self.paired_exps.get(other.id)
        if not exp:
            other_in, exp = self.make_paired_experiments(other)
        exp.network.reconnect()
        run_err, miss_err, index_err, form_dict =\
        exp.test_all(record=record, verbose=verbose)
        exp.network.disconnect()
        return run_err, miss_err, index_err

    def learn_from_misses(self, other, exp=None, layer=2, sep_amount=0.01, verbose=0):
        print("- - - {} LEARNING FROM MISCOMMUNICATIONS WITH {} - - -".format(self, other))
        exp = exp or self.paired_exps.get(other.id)
        if not exp:
            other_in, exp = self.make_paired_experiments(other)
        n_misses = 100
        miss_gain = -1
        while True:
            exp.network.reconnect()
            run_err, miss_err, index_err, form_dict = exp.test_all(record=layer)
#            , verbose=verbose)
            exp.network.disconnect()
            n_new_misses = len(index_err)
            miss_gain = n_misses - n_new_misses
            n_misses = n_new_misses
            if n_misses and miss_gain > 0:
                clusters = self.environment.clusters
                meanings = self.environment.meanings
                if clusters:
                    in_cluster = 0
                    out_cluster = 0
                    for l1, l2 in index_err.items():
                        c1 = meanings[l1].cluster
                        c2 = meanings[l2].cluster
                        if c1 == c2:
                            in_cluster += 1
                        else:
                            out_cluster += 1
                    print("Updating missed forms: {} inside clusters, {} outside clusters".format(in_cluster, out_cluster))
                else:
                    print("Updating {} missed forms".format(len(index_err)))
                self.lexicon.update_forms(form_dict, index_err, sep_amount=sep_amount, verbose=verbose)
                print("Retraining production and comprehension networks")
                # Don't stop for misses and train past usual error threshold
                print("Production   ", end='; ')
                self.prodexp.run(10000, miss_thresh=-1, error_thresh=0.002)
                print("Comprehension", end='; ')
                self.compexp.run(10000, miss_thresh=-1, error_thresh=0.002)
            else:
                return

    def teach(self, student, trials_per_lex=200, joint_trials_per_lex=10000,
              lr=0.05, joint_lr=0.025, dont_train=False, verbose=0):
        """
        Train student on self's current lexicon.
        dont_train is there for debugging.
        """
        if student in self.population.teachers:
            # student is already a teacher, so do less training
            trials_per_lex //= 4
            joint_trials_per_lex //= 4
            lr //= 10
            joint_lr //= 10
        print("- - - {} TEACHING {} - - -".format(self, student))
        compexp, prodexp, jointexp = self.create_experiments(student)
        if dont_train:
            return compexp, prodexp, jointexp
        nlex = self.population.nmeanings
        flength = self.population.flength
        environment = self.environment
        nvalues = environment.mvalues
        print("Comprehension", end='; ')
        compexp.run(trials_per_lex * nlex, lr=lr)
        print("Production   ", end='; ')
        student.networks[1].reconnect()
        prodexp.run(trials_per_lex * nlex, lr=lr)
        print("Prod->Comp   ", end='; ')
        student.run_joint(jointexp, joint_trials_per_lex * nlex, lr=joint_lr)
        run_err, miss_err, index_err, form_dict = jointexp.test_all(record=2)
        student_lex_patterns = \
        [[m, Form(form_dict.get(i, gen_array(nvalues, flength)))] \
         for i, (m, f) in enumerate(self.lexicon.entries)]
        print("Assigning forms to {}'s lexicon".format(student))
        if verbose:
            print("** Student lex patterns")
            for p in student_lex_patterns:
                print(p)
        student.lexicon = \
        Lexicon(student.id, environment, flength, nlex=nlex,
                patterns=student_lex_patterns)
        # Create student-internal experiments, if they're not already created
        if not student.compexp:
            student.compexp, student.prodexp, student.jointexp = \
            student.create_experiments(student)
        if not student.id in self.paired_exps:
#            print("Creating joint networks and experiments")
            other_in_exp, self_in_exp = self.make_paired_experiments(student)
        # Add student to teacher list
        self.population.add_to_teachers(student)
#        return other_in_exp, self_in_exp
