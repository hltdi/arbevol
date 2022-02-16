"""
arbevol: Population of Persons.
"""

from lex import *
import time

def meta_experiment(npops=10, nruns=10, popsize=10, nlex=50, mvalues=5, clusters=None,
                    match=True, teach=True,
                    iconic=True, constant_flip=True, nearest_cluster=False,
                    file=None):
    string = "Training {} populations of size {} for {} runs, {} lexicons of size {}"
    print(string.format(npops, popsize, nruns, "iconic" if iconic else "arbitrary", nlex))
    results = []
    all_results = []
    if clusters:
        print("Meaning cluster properties: {}/{}".format(clusters, nearest_cluster))
    else:
        print("Random meanings")
    for i in range(npops):
        print("++++++++++++++++++++++++")
        print("POPULATION {}".format(i))
        print("++++++++++++++++++++++++")
        p = Population(popsize, nmeanings=nlex, mvalues=mvalues, clusters=clusters,
                       iconic=iconic, nearest_cluster=nearest_cluster, terse=True)
        if nruns:
            p.run(nruns, match=match, teach=teach, terse=True)
        all_results.append(p.stats)
        results.append((p.stats['m'][-1][1], p.stats['d'][-1][1], p.stats['i'][-1][1]))
    m = sum([x[0][0] for x in results])/npops, sum([x[0][1] for x in results])/npops
    d = sum([x[1] for x in results])/npops
    i = sum([x[2][0] for x in results])/npops, sum([x[2][1] for x in results])/npops
    print("m {}, d {}, i {}".format([np.round(mm, 3) for mm in m],
                                     np.round(d, 3),
                                    [np.round(ii, 3) for ii in i]))
    if file:
        pass
    return results, all_results

    def write_meta(filename):
        """
        Write the Lexicon's entries to a file.
        """
        path = os.path.join('results', filename)
        with open(path, 'w') as out:
            print(file=out)

#    def read_meta(filename, mvalues=5, fvalues=5):
#        path = os.path.join('data', filename)
#        entries = []
#        with open(path) as infile:
#            for line in infile:
#                meaning, form = line.split(" ‖ ")
#                meaning = Meaning.read(meaning, mvalues)
#                form = Form.read(form, fvalues)
#                entries.append([meaning, form])
#        return entries

class Population(list):

    def __init__(self, size, iconic=True, init_teachers=0.1,
                 mlength=5, flength=5, nhid=20, nmeanings=10, noise=0.1,
                 mvalues=5, clusters=None, sep_amount=0.02, nearest_cluster=False,
                 init_lex=None, init_meanings=None,
                 # iconicity parameters
                 constant_flip=True, deiconize=0.5,
                 dont_init=False, terse=False):
        self.size = size
        self.mlength = mlength
        self.flength = flength
        self.mvalues = mvalues
        self.nhid = nhid
        self.iconic = iconic
        self.nmeanings = nmeanings
        self.noise = noise
        self.sep_amount = sep_amount
        self.constant_flip = constant_flip
        self.deiconize = deiconize
        self.teachers = set()
        # Whether all Persons have been taught a Lexicon
        self.taught = False
        self.runs = 0
        self.commruns = 0
        self.stats = {'m': [], 'd': [], 'i': [], 'c': []}
#        self.mutualities = []
#        self.distances = []
#        self.iconicities = []
        # Whether nearest target counts as correct if in right cluster
        self.nearest_cluster = nearest_cluster
        print("Creating population of {}: lexicon size {}, form length {}, hidden layer size {}".format(size, nmeanings, flength, nhid))
        if iconic:
            print("  Iconicity: deiconization {}, constant flip feature {}, nearest cluster {}".format(deiconize, constant_flip, nearest_cluster))
        self.make_environment(clusters=clusters, init_meanings=init_meanings)
        for i in range(size):
            self.add(dont_init=dont_init, init_lex=init_lex, terse=terse)
        if not dont_init:
            self.teach(terse=terse)

    def add(self, dont_init=False, init_lex=None, terse=False):
        self.append(Person(self, teacher=not self, dont_init=dont_init,
                           init_lex=init_lex, terse=terse))

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

    @staticmethod
    def make_pairs(n):
        items1 = list(range(n))
        items2 = list(range(n))
        random.shuffle(items2)
        overlap = True
        while overlap:
            o = []
            for i in range(n):
                if items1[i] == items2[i]:
                    o.append(i)
            if not o:
                overlap = False
            else:
                positions = list(range(n))
                swap1 = o[0]
                positions.remove(swap1)
                swap2 = positions[0]
                items2[swap1], items2[swap2] = items2[swap2], items2[swap1]
        return list(zip(items1, items2))

    def communicate(self, p1, p2, verbose=0, terse=False):
#        p1.teach(p2, verbose=verbose)
        p2.match_meanings(p1, verbose=verbose, terse=terse)

    # def communicate(self, iterations=1, score=True):
    #     '''
    #     On each iteration, for each Person, select a random Person
    #     to communicate with.
    #     '''
    #     for i in range(iterations):
    #         print(">>>>>>>>COMMUNICATION ITERATION {}<<<<<<<<".format(self.commruns))
    #         indices = list(range(self.size))
    #         random.shuffle(indices)
    #         for index in indices:
    #             student = self[index]
    #             teacher = self.select_other(index)
    #             student.match_meanings(teacher, sep_amount=self.sep_amount)
    #         self.commruns += 1
    #     if score:
    #         self.update_stats()

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

    def run(self, iterations=1, match=True, teach=True, random_form=False, score=2,
            terse=False):
        '''
        On each iteration, for each Person, select a random Person
        to communicate with.
        '''
        if len(self.teachers) == 1:
            self.teach(random_form=random_form, terse=terse)
        for i in range(iterations):
            print(">>>>>>>>ITERATION {}<<<<<<<<".format(self.runs))
            pairs = Population.make_pairs(self.size)
            for studenti, teacheri in pairs:
                student = self[studenti]
                teacher = self[teacheri]
#            indices = list(range(self.size))
#            random.shuffle(indices)
#            for index in indices:
#                student = self[index]
#                teacher1 = self.select_other(index)
#                other_indices = list(range(self.size))
#                other_indices.remove(index)
#                teacher = self[random.choice(other_indices)]
#                t, s = self.pair()
#                student.match_meanings(teacher1, sep_amount=self.sep_amount)
#                teacher = self.select_other(index)
                match_out = None
                if match:
                    match_out = teacher.match_meanings(student, sep_amount=self.sep_amount, terse=terse)
                if teach and (match_out or not match):
                    teacher.teach(student, random_form=random_form, terse=terse)
            self.runs += 1
            if score and self.runs % score == 0:
                self.update_stats()

        if score:
            self.update_stats()

    def update_stats(self):
        last_update = self.stats['i'][-1][0] if self.stats['i'] else -1
        if self.runs > last_update:
            print("--Updating stats--")
            self.stats['m'].append((self.runs, self.mean_mutuality()))
            self.stats['d'].append((self.runs, self.mean_distance()))
            self.stats['i'].append((self.runs, (self.mean_iconicity(), self.mean_m_iconicity())))
            self.stats['c'].append((self.runs, self.mean_dc()))
#            self.stats['c'].append((self.runs, [np.rond()]))

    def select_other(self, index):
        """
        Given a Person index, randomly select another Person.
        """
        other_indices = list(range(self.size))
        other_indices.remove(index)
        return self[random.choice(other_indices)]

    def teach(self, one_teacher=False, random_form=False, terse=False):
        """
        Have a Lexicon taught to the whole Population,
        by default by the single initial master teacher.
        """
        teacher = self[0]
        if teacher not in self.teachers:
            teacher.init(sep_amount=self.sep_amount, terse=terse)
        for student in self[1:]:
#            self.communicate(teacher, student)
            teacher.teach(student, random_form=random_form, terse=terse)
#            student.teach(teacher)
#            teacher.match_meanings(student)
#            student.match_meanings(teacher)
            if not one_teacher:
                teacher = random.choice(list(self.teachers))
#                teacher = student
        self.update_stats()

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
        return np.round(sum(distances) / self.size, 3)

    def mean_dc(self):
        """
        Mean distance correlations within Lexicons of Persons.
        """
        corrs = np.array([0.0, 0.0, 0.0])
        for person in self:
            corrs += person.lexicon.dc()
        return tuple(np.round(corrs / self.size, 3))

    def mean_iconicity(self):
        """
        Mean iconicity of Lexicons of Persons.
        """
        return np.round(sum([p.lexicon.iconicity() for p in self]) / self.size, 3)

    def mean_m_iconicity(self):
        """
        Mean iconicity of Lexicons of Persons.
        """
        return np.round(sum([p.lexicon.m_iconicity() for p in self]) / self.size, 3)

    def mutuality(self, p1, p2):
        """
        Measure of the extent to which Persons p1 and p2 understand
        one another.
        """
        run_err1, miss_err1, err_index1, form_dict1 = p1.test_paired(p2)
        run_err2, miss_err2, err_index2, form_dict2 = p2.test_paired(p1)
        return round((run_err1 + run_err2) / 2.0, 3), round((miss_err1 + miss_err2) / 2.0, 3)

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
        return round(m[0], 3), round(m[1], 3)

class Person:

    n=0

    def __init__(self, population, teacher=True, newarb=0.1, init_lex=None,
                 dont_init=False, terse=False):
        # dont_init for debugging
        self.id = Person.n
        self.teacher = teacher
        self.population = population
        self.environment = population.environment
        self.noise = population.noise
        self.newarb = newarb
        self.meanings = self.environment.meanings
        self.clusters = self.environment.clus_indices
        self.noisify_errors = False
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
            self.compexp, self.prodexp, self.jointexp = self.make_experiments(self)
            if not dont_init:
                self.init(terse=terse)
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

    def make_joint_network(self):
        return Network.join(self.networks[1], self.networks[0], Network.joint)

    def make_lexicon(self, iconic=True, init_lex=None):
        """
        Create initial lexicon for master teacher. init_lex is a file
        to read in a lexicon from.
        """
        patterns = None
        if init_lex:
            patterns = \
            Lexicon.read(init_lex, self.population.mvalues, self.population.mvalues)
#                         constant_flip=self.population.constant_flip,
#                         deiconize=self.population.deiconize)
        return \
        Lexicon(self.id, self.environment, self.population.flength,
                nlex=self.population.nmeanings,
                constant_flip=self.population.constant_flip,
                deiconize=self.population.deiconize,
                noise=self.noise, iconic=iconic,
                init=True,
                patterns=patterns)

    def extend_lexicon(self, meanings, iconic=True):
        """
        For master teacher only, add new meanings to lexicon, creating new initial
        forms for them.
        """
        self.lexicon.add(meanings)

    def make_experiments(self, student):
        """
        Make experiments for training student on this Person's (master teacher's)
        Lexicon. If student is self, these are for the master teacher's
        initialization self-teaching.
        """
        # Teacher's lexicon
        lexicon = self.lexicon
        entries = lexicon.entries
        # Student's networks
        compnet = student.networks[0]
        prodnet = student.networks[1]
        jointnet = student.joint_network
        teacher_prodnet = self.networks[1]
        # Patterns and pattern generation functions
        comppats = lambda: lexicon.comppats #[l.make_comprehension_IT(simple=True) for l in entries]
        prodpats = lambda: lexicon.prodpats # [l.make_production_IT(simple=True) for l in entries]
#        mean2mean_pats = lambda: lexicon.mean2mean_pats # [[l.get_meaning(), l.get_meaning()] for l in entries]
        meaningtargfunc = lambda: [e[0] for e in entries]
        if self == student:
            formtargfunc = lambda: [e[1] for e in entries]
        else:
            formtargfunc = lambda: [self.get_form_target(teacher_prodnet, e[0], e[1]) for e in entries]
        comppatfunc = self.make_patfunc(comppats, meaningtargfunc, noisify_input=student!=self)
        prodpatfunc = self.make_patfunc(prodpats, formtargfunc, noisify_target=student!=self)
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
                   test_nearest=True, nearest_cluster=self.population.nearest_cluster,
                   conditions=[[mean2mean_patfunc, mean2mean_patfunc]])
        return comp, prod, joint

    def get_form_target(self, teacher_prod_net, meaning, lexform, verbose=0):
        teacher_prod_net.reconnect()
#        print("** Getting form from target {} with input {}".format(teacher_prod_net, meaning))
        teacher_prod_net.step([meaning, None], train=False, verbose=verbose)
        teacher_out = teacher_prod_net.layers[-1].activations
        if Form.roundQ:
            Form.round(teacher_out, nvalues=self.population.mvalues)
#        print("** Got target {}".format(teacher_out))
#        print("** (Expected target {})".format(lexform))
        return teacher_out

    # def make_paired_experiments(self, other):
    #     self_in_exp = other_in_exp = None
    #     if other.id in self.paired_exps:
    #         self_in_exp = self.paired_exps[other.id]
    #     if self.id in other.paired_exps:
    #         other_in_exp = other.paired_exps[self.id]
    #     if not self_in_exp or not other_in_exp:
    #         mean2mean_patfunc = self.make_mean2mean_func()
    #         if not other_in_exp:
    #             network1 = Network.join(other.networks[1], self.networks[0],
    #                                     Network.paired)
    #             other_in_exp = \
    #             Experiment("P{}->C{}".format(other.id, self.id),
    #                        network=network1, test_nearest=True,
    #                        nearest_cluster=self.population.nearest_cluster,
    #                        conditions=[[mean2mean_patfunc, mean2mean_patfunc]])
    #             network1.disconnect()
    #             other.paired_exps[self.id] = other_in_exp
    #         if not self_in_exp:
    #             network2 = Network.join(self.networks[1], other.networks[0],
    #                                     Network.paired)
    #             self_in_exp = \
    #             Experiment("P{}->C{}".format(self.id, other.id),
    #                        network=network2, test_nearest=True,
    #                        nearest_cluster=self.population.nearest_cluster,
    #                        conditions=[[mean2mean_patfunc, mean2mean_patfunc]])
    #             network2.disconnect()
    #             self.paired_exps[other.id] = self_in_exp
    #     return other_in_exp, self_in_exp

    def make_paired_experiments(self, other):
        other_out_exp = other_in_exp = None
        if other.id in self.paired_exps:
            other_out_exp = self.paired_exps[other.id]
        if self.id in other.paired_exps:
            other_in_exp = other.paired_exps[self.id]
        if not other_out_exp or not other_in_exp:
            if not other_in_exp:
                # other production in, self comprehension out
                otherinfunc = self.make_paired_patfunc(other)
                selfcomp = self.networks[0]
                other_in_exp = \
                Experiment("P{}->C{}".format(other.id, self.id),
                           network=selfcomp, test_nearest=True,
                           nearest_cluster=self.population.nearest_cluster,
                           conditions=[[otherinfunc, otherinfunc]])
                other.paired_exps[self.id] = other_in_exp
            if not other_out_exp:
                # other comprehension out, self production in
                otheroutfunc = other.make_paired_patfunc(self)
                othercomp = other.networks[0]
                other_out_exp = \
                Experiment("P{}->C{}".format(self.id, other.id),
                           network=othercomp, test_nearest=True,
                           nearest_cluster=self.population.nearest_cluster,
                           conditions=[[otheroutfunc, otheroutfunc]])
                self.paired_exps[other.id] = other_out_exp
        return other_in_exp, other_out_exp

    def init(self, trials_per_lex=100, joint_trials_per_lex=50,
             sep_amount=0.01, new_meanings=None, verbose=0, terse=False):
        """
        Train on the Person's own Lexicon.
        """
        nlex = self.population.nmeanings
        print(">>>>>>>>INITIALIZATION<<<<<<<<")
        if new_meanings:
            print("- - - {} RE-SELF-INITIALIZING WITH NEW MEANINGS - - -".format(self))
        else:
            print("- - - {} SELF-INITIALIZING - - -".format(self))
        if verbose:
            print("** Initial lexicon")
            self.lexicon.show()
        icon1 = round(self.lexicon.iconicity(), 3)
        micon1 = round(self.lexicon.m_iconicity(), 3)
        if not terse:
            print("Initial iconicity: {}".format(icon1))
        self.population.stats['i'].append((-1, [icon1, micon1]))
        # Train comprehension network
        if not terse:
            print("Comprehension".format(self), end='; ')
        self.compexp.run(trials_per_lex * nlex, lr=0.1, terse=terse)
        # Train production network
        if not terse:
            print("Production   ".format(self), end='; ')
        self.networks[1].reconnect()
        self.prodexp.run(trials_per_lex * nlex, lr=0.1, terse=terse)
        # Train joint network
        if not terse:
            print("Prod->comp   ".format(self), end='; ')
        miss_thresh = -0.1 if self.population.nearest_cluster else 0.0
        self.run_joint(self.jointexp, joint_trials_per_lex * nlex, lr=0.05,
                       miss_thresh=miss_thresh, terse=terse)
        # Save successful form representations from joint network in lexicon
#        if not terse:
#            print("Updating forms in {}'s lexicon".format(self))
        run_err, miss_err, index_err, form_dict = self.test_joint(self.jointexp)
        if index_err and not terse:
            print("Updating {}".format(index_err))
        self.lexicon.update_forms(form_dict, index_err, sep_amount=sep_amount,
                                  verbose=verbose)
        # At least one target category error, so retrain
        self.population.add_to_teachers(self)

    # def make_paired_networks(self, student):
    #     """
    #     Make networks combining this (teacher) and student networks.
    #     """
    #     # Student production, teacher comprehension
    #     network1 = Network.join(student.networks[1], self.networks[0])
    #     # Teacher production, student comprehension
    #     network2 = Network.join(self.networks[1], student.networks[0])
    #     return network1, network2

    def teach(self, student, trials_per_lex=100, joint_trials_per_lex=50,
              lr=0.1, joint_lr=0.05, dont_train=False,
              random_form=False, verbose=0, terse=False):
        """
        Train student on self's current lexicon.
        dont_train is there for debugging.
        """
#        if student in self.population.teachers:
            # student is already a teacher, so do less training
#            trials_per_lex //= 2
#            joint_trials_per_lex //= 2
#            lr //= 2
#            joint_lr //= 2
        print("- - - {} TEACHING {} - - -".format(self, student))
#        print("** trials per lex {}".format(trials_per_lex))
        compexp, prodexp, jointexp = self.make_experiments(student)
        if dont_train:
            return compexp, prodexp, jointexp
        nlex = self.population.nmeanings
        flength = self.population.flength
        environment = self.environment
        nvalues = environment.mvalues
        if verbose:
            print("** New lexicon")
            self.lexicon.show()
        if not terse:
            print("Comprehension", end='; ')
        compexp.run(trials_per_lex * nlex, lr=lr, miss_thresh=-.1, terse=terse)
        if not terse:
            print("Production   ", end='; ')
        student.networks[1].reconnect()
        prodexp.run(trials_per_lex * nlex, lr=lr, miss_thresh=-.1, terse=terse)
        if not terse:
            print("Prod->Comp   ", end='; ')
        miss_thresh = -0.1 if self.population.nearest_cluster else 0.0
        student.run_joint(jointexp, joint_trials_per_lex * nlex, lr=joint_lr,
                          miss_thresh=miss_thresh, terse=terse)
        run_err, miss_err, index_err, form_dict = self.test_joint(jointexp)
#        jointexp.test_all(record=2)
#        if self.clusters:
#            out_err = self.cluster_error(index_err, miss_err)
        if index_err and not terse:
            print("Updating {}".format(index_err))
        student_lex_patterns = \
        self.make_student_lex_patterns(form_dict, index_err, nvalues, flength,
                                       random_form=random_form)
#        [[m, Form(form_dict.get(i, gen_array(nvalues, flength)))] \
#         for i, (m, f) in enumerate(self.lexicon.entries)]
#        print("Assigning forms to {}'s lexicon".format(student))
        if verbose:
            print("** Student lex patterns")
            for p in student_lex_patterns:
                print(p)
        student.lexicon = \
        Lexicon(student.id, environment, flength, nlex=nlex,
                patterns=student_lex_patterns,
                constant_flip=self.population.constant_flip,
                deiconize=self.population.deiconize)
        # Create student-internal experiments, if they're not already created
        if not student.compexp:
            student.compexp, student.prodexp, student.jointexp = \
            student.make_experiments(student)
        if not student.id in self.paired_exps:
#            print("Creating joint networks and experiments")
            other_in_exp, self_in_exp = self.make_paired_experiments(student)
        # Add student to teacher list
        self.population.add_to_teachers(student)
#        return other_in_exp, self_in_exp

    def make_student_lex_patterns(self, form_dict, index_error, nvalues, flength,
                                  random_form=False, verbose=0):
        '''
        Create lexical patterns for a student network based on the saved
        values from the joint network in form_dict.
        For failed forms,a new form is generated by flipping the old form.
        '''
        patterns = []
        old_entries = [entry.copy() for entry in self.lexicon.entries]
        for i, (m, f) in enumerate(self.lexicon.entries):
            if i not in index_error:
                form = form_dict.get(i)
                form = Form(form, nvalues=nvalues, init=False)
            else:
                if random_form:
                    form = gen_array(nvalues, flength)
                elif self.noisify_errors:
                    if Form.roundQ:
                        form = f.copy()
                        Form.round(form, nvalues)
                        if verbose:
                            print("  ** Updating {} ->".format(form), end=' ')
                        # Then flip one value
                        flip_values(form, 2, nvalues, length=flength,
                                    opts=Form.value_opts[nvalues])
                        if verbose:
                            print("  {}".format(form))
                    else:
                        form = noisify(f, sd=0.05)
                else:
                    # Separate form from error form
                    form = Form(form_dict.get(i), nvalues=nvalues)
                    error_index = index_error[i]
                    error_form = old_entries[error_index][1]
#                    print("** form {}, error form {}".format(form, error_form))
                    form.separate(error_form, amount=self.population.sep_amount,
                                  verbose=verbose)
#                    print("** separated {}".format(form))
#                    Form.round(form, nvalues)
#                print("** Generating novel form for {}::{}; {} -> {}".format(i, m, f, form))
                form = Form(form, nvalues=nvalues, init=True)
            patterns.append([m, form])
        return patterns

    def match_meanings(self, other, exp=None, layer=2,
                       trials_per_lex=50, joint_trials_per_lex=25, nreps=3,
                       sep_amount=0.01, verbose=0, terse=False):
        print("- - - {} MATCHING MEANINGS WITH {} - - -".format(self, other, sep_amount))
        exp = exp or self.paired_exps.get(other.id)
        nlex = self.population.nmeanings
        environment = self.environment
        nvalues = environment.mvalues
        flength = self.population.flength
        if not exp:
            other_in, exp = self.make_paired_experiments(other)
        n_misses = 100
        miss_gain = -1
        miss_thresh = -0.1 if self.population.nearest_cluster else 0.0
        reps = 0
        miss_err = 0.0
        found_misses = False
        while reps < nreps:
#            exp.network.reconnect()
            run_err, miss_err, index_err, form_dict = exp.test_all(record=True)
#            , verbose=verbose)
#            exp.network.disconnect()
            n_new_misses = len(index_err)
            if not found_misses and n_new_misses:
                found_misses = True
            miss_gain = n_misses - n_new_misses
            n_misses = n_new_misses
            if n_misses and miss_gain > 0:
                if self.clusters:
                    in_cluster = 0
                    out_cluster = 0
                    for l1, l2 in index_err.items():
                        c1 = self.meanings[l1].cluster
                        c2 = self.meanings[l2].cluster
                        if c1 == c2:
                            in_cluster += 1
                        else:
                            out_cluster += 1
                    if not terse:
                        print("Updating missed forms: {} inside clusters, {} outside clusters".format(in_cluster, out_cluster))
                else:
                    if not terse:
                        print("Updating {} missed forms".format(len(index_err)))
                self.lexicon.update_forms(form_dict, index_err,
                                          sep_amount=sep_amount, verbose=verbose)
                if not terse:
#                    print("Retraining production and comprehension networks")
                # Don't stop for misses and train past usual error threshold
                    print("Production   ", end='; ')
                self.prodexp.run(trials_per_lex * nlex, miss_thresh=-1,
                                 error_thresh=0.002, terse=terse)
                if not terse:
                    print("Comprehension", end='; ')
                self.compexp.run(trials_per_lex * nlex, miss_thresh=-1,
                                 error_thresh=0.002, terse=terse)
                if not terse:
                    print("Prod->Comp   ", end='; ')
                self.run_joint(self.jointexp, joint_trials_per_lex * nlex,
                               miss_thresh=miss_thresh, terse=terse)
                run_err, miss_err, index_err, form_dict = self.test_joint(self.jointexp)
                if index_err and not terse:
                    print("Updating forms, including internal errors {}".format(index_err))
                self.lexicon.update_forms(form_dict, index_err, sep_amount=sep_amount, verbose=verbose)
#                student_lex_patterns = self.make_student_lex_patterns(form_dict, nvalues, flength)
#                self.lexicon = \
#                Lexicon(self.id, environment, flength, nlex=nlex,
#                        patterns=student_lex_patterns,
#                        constant_flip=self.population.constant_flip,
#                        deiconize=self.population.deiconize)
                reps += 1
            else:
                return found_misses
        return found_misses

    def run_joint(self, exp, ntrials, lr=0.01, miss_thresh=0.0, terse=False):
        """
        Reconnect joint network, run it, and then reconnect prod network.
        """
        self.joint_network.reconnect()
        exp.run(ntrials, lr=lr, miss_thresh=miss_thresh, terse=terse)
        self.networks[1].reconnect()

    def test_joint(self, exp, record=2, update=False, sep_amount=0.01,
                   verbose=0, terse=False):
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

    def cluster_error(self, index_err, miss_err):
        if index_err:
            # There were misses
            out_index_err = {}
            in_cluster = 0
            out_cluster = 0
            for l1, l2 in index_err.items():
                c1 = self.meanings[l1].cluster
                c2 = self.meanings[l2].cluster
                if c1 == c2:
                    in_cluster += 1
                else:
                    out_cluster += 1
                    out_index_err[l1] = l2
            print("** in errors {}, out errors {}".format(in_cluster, out_cluster))
            out_err = out_cluster / len(self.meanings)
            print("** miss err {}, out miss err {}".format(miss_err, out_err))
            return out_err

    def test_paired(self, other, record=None, ignore_clusters=False, verbose=0):
        """
        Test the paired network (self's production, other's comprehension).
        """
        exp = self.paired_exps.get(other.id)
        if not exp:
            other_in, exp = self.make_paired_experiments(other)
#        exp.network.reconnect()
        run_err, miss_err, index_err, form_dict =\
        exp.test_all(record=record, ignore_clusters=ignore_clusters, verbose=verbose)
#        exp.network.disconnect()
#        if self.clusters:
#            out_err = self.cluster_error(index_err, miss_err)
        return run_err, miss_err, index_err, form_dict

    def make_patfunc(self, patterns_func, target_func,
                     cluster_func=None, noisify_input=False, noisify_target=False):
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
                return False, 0, None
            else:
                input, target = patterns[index]
            if noisify_input and self.noise:
                input = noisify(input, sd=self.noise)
#                print("** input:     {}".format(input))
            if noisify_target and self.noise:
#                print("** noisifying target: {}".format(target) end=', ')
                target = noisify(target, sd=self.noise)
#                print("{}".format(target))
            return [input, target], 0, None
        return PatGen(nlex, function=patfunc, targets=target_func, clusters=cluster_func)

    def make_paired_patfunc(self, producer):
        nlex = self.population.nmeanings
        meanings = [l.get_meaning() for l in self.lexicon.entries]
        prodnet = producer.networks[1]
        target_func = lambda: [e.get_meaning() for e in self.lexicon.entries]
        def patfunc(index=-1):
            if index < 0:
                meaning = random.choice(meanings)
            elif index >= nlex:
                return False, 0, None
            else:
                meaning = meanings[index]
            # Run the producer network on the meaning and get its output
            prodnet.step([meaning, None], train=False, show_act=False)
            input = Form(prodnet.layers[-1].activations, nvalues=self.population.mvalues)
            to_record = input
            if self.noise:
#                print("** Noisifying {} -> ".format(input), end=', ')
                input = noisify(input, sd=self.noise)
#                print("{}".format(input))
            return [input, meaning], 0, to_record
        return PatGen(nlex, function=patfunc, targets=target_func)

    def make_mean2mean_func(self):
        return \
        self.make_patfunc(lambda: self.lexicon.mean2mean_pats,
                          lambda: [e.get_meaning() for e in self.lexicon.entries],
                          lambda: self.clusters,
                          noisify_input=False, noisify_target=False)
