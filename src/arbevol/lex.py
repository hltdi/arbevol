"""
arbevol lexicon: networks, training pattern generators, experiments.
"""

from experiment import *
from environment import *

class Lexicon:

    def __init__(self, id, environment, flength, iconic=True,
                 nlex=10,
                 enforce_distance=False, noise=0.1,
                 compprob=0.0, prodprob=0.0, patterns=None):
        self.id = id
        self.environment = environment
        # This may be less than the number of meanings in the environment
        self.flength = flength
        self.mlength = environment.mlength
        self.patlength = self.flength + self.mlength
        self.iconic = iconic
        self.nvalues = environment.mvalues
        self.enforce_distance = enforce_distance
        self.noise = noise
        self.compprob = compprob
        self.prodprob = prodprob
        self.entries = []
        self.nlex = nlex
        # Later a subset of the environment's meanings?
        self.meanings = environment.meanings[:nlex]
        if not iconic:
            # Create form space for arbitrary Lex entries (must precede make())
            self.form_space = self.gen_form_space()
        else:
            self.form_space = None
        self.make(self.meanings, patterns=patterns)
        self.make_patterns2()
        # self.make_patterns()
        # self.make_exp_conditions()
        # After make()
        # Comprehension and production training patterns

    def __repr__(self):
        return "L{}".format(self.id)

#    def make_network(self, nhidden=10, name='lex'):
#        return \
#        Network(name,
#                layers = [Layer('in', self.flength+self.nmeaning),
#                          Layer('hid', nhidden),
#                          Layer('out', self.flength+self.nmeaning)])

#    def adjust_forms(self):

    def update_forms(self, form_dict):
        '''
        Adjust the form values in the lexicon based on those in the form
        dict (entry_index: array)
        '''
        for index, form in form_dict.items():
            entry = self.entries[index]
            entry[1] = Form(form)

    @staticmethod
    def from_network(experiment, meanings, person):
        '''
        Create a Lexicon from the output of a production network presented
        with Meanings as input.
        '''
        flength = person.population.flength
        nvalues = person.population.mvalues
        newarb = person.newarb
        paterror, nearest_error, errors, form_out = \
        experiment.test_all('prod', reps=1, record=True)
        patterns = []
        for i, meaning in enumerate(meanings):
            form = form_out.get(i)
            if type(form) == np.ndarray:
                form = Form(form)
            patterns.append([meaning, form])
        for m, f in patterns:
            print("** {} | {}".format(m, f))
        for index, (m, f) in enumerate(patterns):
            if type(f) != Form:
                # no form for this entry
                print("Setting form for {}".format((m, f)))
                forms = [p[1] for p in patterns]
                if np.random.rand() < newarb:
                    # Generate a random form
                    patterns[index][1] =  Form.make_random_form(nvalues, flength, forms)
                else:
                    # Generate an iconic form
                    patterns[index][1] = Form.make_iconic_form(m, nvalues, flength, forms)

        return Lexicon(person.id, person.population.environment,
                       flength, patterns=patterns, nlex=len(meanings))

    def make(self, meanings, patterns=None):
        """
        Create nlex Lex instances. patterns are provided for a lexicon
        generated from network output.
        """
        if patterns:
            for meaning, form in patterns:
                self.entries.append(Lex(self, meaning, form=form))
        elif self.iconic:
            for m in meanings:
                self.entries.append(Lex(self, m))
        else:
            for index, m in enumerate(meanings):
                self.entries.append(Lex(self, m, form_space=self.form_space, index=index))

    def add(self, meanings, patterns=None):
        """
        Create new Lex instances for each of meanings, and add them to
        entries.
        """
        if patterns:
            for meaning, form in patterns:
                self.entries.append(Lex(self, meaning, form=form))
        elif self.iconic:
            for m in meanings:
                self.entries.append(Lex(self, m))
        else:
            for index, m in enumerate(meanings):
                self.entries.append(Lex(self, m, form_space=form_space,
                                        index=self.nlex + index))
        self.nlex += len(meanings)

    def gen_form_space(self):
        """
        Generate a list of form arrays to select arbitrary forms from.
        Only done for arbitrary Lexicons.
        """
        # Generate corners of flength space
        corners = cornersA(self.flength)
        # 2 ^ flength
        ncorners = len(corners)
        if self.nlex <= ncorners:
            random.shuffle(corners)
            return corners[:self.nlex]
        # Add new random flength array to corners
        # LATER CONSTRAIN DISTANCE FROM OTHER ARRAYS
        for i in range(ncorners, self.nlex):
            a = gen_novel_array(self.nvalues, self.flength, spec=None, existing=corners)
            corners.append(a)
        random.shuffle(corners)
        return corners

    def get_forms(self):
        """
        All of the current form arrays.
        """
        return [l.get_form() for l in self.entries]

    def comppat_from_network(self, network, lex, showmeaning=0.2):
        """
        Given a lex meaning as input, return the form output of the network,
        and the meaning as target, for training and testing another network
        on comprehension.
        """
        meaning = lex.get_meaning()
        target = lex.make_pattern(meaning=True, form=False, target=True)
        form_input = np.zeros(self.flength)
        input = np.concatenate((meaning, form_input))
        network.step([input, None], train=False)
        output = np.copy(network.layers[-1].activations)
        # Set output meaning to 0.0 except in showmeaning positions
        nzero = round((1.0 - showmeaning) * self.mlength)
        positions = list(range(self.mlength))
        random.shuffle(positions)
        zeropos = positions[:nzero]
        for pos in zeropos:
            output[pos] = 0.0
        return output, target

    def comppats_from_network(self, network, showmeaning=0.2):
        return [self.comppat_from_network(network, lex, showmeaning=showmeaning) \
                for lex in self.entries]

    # def make_patterns(self):
    #     self.patterns = [l.make_input_target() for l in self.entries]
    #     self.comp_test_patterns = [l.make_comprehension_IT(test=True) for l in self.entries]
    #     self.prod_test_patterns = [l.make_production_IT(test=True) for l in self.entries]
    #     self.comp_train_patterns = [l.make_comprehension_IT(test=False) for l in self.entries]
    #     self.prod_train_patterns = [l.make_production_IT(test=False) for l in self.entries]

    def make_patterns2(self):
        self.comppats = [l.make_comprehension_IT(simple=True) for l in self.entries]
        self.prodpats = [l.make_production_IT(simple=True) for l in self.entries]

    def get_meaning_targets(self):
        return [e[0] for e in self.entries]

    def get_form_targets(self):
        return [e[1] for e in self.entries]

    def make_comp_patfunc(self, noise=False):
        noise = noise and self.noise
        def patfunc(index=-1):
            if index < 0:
                input, target = random.choice(self.comppats)
            elif index >= self.nlex:
                return False, 0
            else:
                input, target = self.comppats[index]
            if noise:
                input = noisify(input, sd=noise)
            return [input, target], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_meaning_targets)

    def make_prod_patfunc(self, noise=False):
        noise = noise and self.noise
        def patfunc(index=-1):
            if index < 0:
                input, target = random.choice(self.prodpats)
            elif index >= self.nlex:
                return False, 0
            else:
                input, target = self.prodpats[index]
            if noise:
                input = noisify(input, sd=noise)
            return [input, target], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_form_targets)

    def make_joint_patfunc(self, noise=False):
        noise = noise and self.noise
        def patfunc(index=-1):
            if index < 0:
                input = random.choice(self.meanings)
            elif index >= self.nlex:
                return False, 0
            else:
                input = self.meanings[index]
            if noise:
                input = noisify(input, sd=noise)
            return [input, input], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_meaning_targets)

    def perf_pattern(self, pattern, comp=True):
        """
        Convert a full input pattern to a comprehension or production pattern.
        """
        p = np.copy(pattern)
        if comp:
            for i in range(self.mlength, self.patlength):
                p[i] = 0.0
            return p
        else:
            for i in range(self.mlength):
                p[i] = 0.0
            return p

    def make_patfunc(self, compprob=1.0, prodprob=0.0, showmeaning=0.2,
                     noise=True):
        """
        perfprob is the probability of training on either a comprehension
        or a production input pattern.
        showmeaning is the probability of revealing one of the meaning units
        in a comprehension input pattern.
        """
        noise = noise and self.noise
        perfprob = compprob + prodprob
        if perfprob > 1.0:
            print("Comp prob {} and prod prob {} sum > 1".format(compprob, prodprob))
        def patfunc(index=-1):
            pindex = np.random.randint(0, len(self.entries))
            rand = np.random.rand()
#            print("** compprob: {}, prodprob: {}, r: {}".format(compprob, prodprob, rand))
            if perfprob and rand < perfprob:
                # Performance training instance
                if compprob and rand < compprob:
                    # Comprehension
                    pattern = self.comp_train_patterns[pindex]
                    if noise:
                        input = noisify(pattern[0], sd=noise, indices=(self.mlength, self.patlength))
                    else:
                        input = np.copy(pattern[0])
                    # Set input meaning to 0.0 except in showmeaning positions
                    nzero = round((1.0 - showmeaning) * self.mlength)
                    positions = list(range(self.mlength))
                    random.shuffle(positions)
                    zeropos = positions[:nzero]
                    for pos in zeropos:
                        input[pos] = 0.0
                else:
                    # Production
                    pattern = self.prod_train_patterns[pindex]
                    if noise:
                        input = noisify(pattern[0], sd=noise, indices=(0, self.mlength))
                    else:
                        input = pattern[0]
            else:
                pattern = self.patterns[pindex]
                input = noisify(pattern[0], sd=noise) if noise else pattern[0]
            target = pattern[1]
            return [input, target], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_targets)

    def make_test_patfunc(self, noise=True):
        def patfunc(index=-1):
            if index < 0:
                pattern = random.choice(self.patterns)
            elif index >= self.nlex:
                return False, 0
            else:
                pattern = self.patterns[index]
            input = pattern[0] # noisify(pattern[0], sd=self.noise) if (noise and self.noise) else pattern[0]
            target = pattern[1]
            return [input, target], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_targets)

    def make_comptest_patfunc(self, noise=True, showmeaning=0.2):
        def patfunc(index=-1):
            if index < 0:
                pattern = random.choice(self.comp_test_patterns)
            elif index >= self.nlex:
                return False, 0
            else:
                pattern = self.comp_test_patterns[index]
            input = np.copy(pattern[0]) # noisify(pattern[0], sd=self.noise, indices=(self.mlength, self.patlength)) if (noise and self.noise) else pattern[0]
            # Set input meaning to 0.0 except in showmeaning positions
            nzero = round((1.0 - showmeaning) * self.mlength)
            positions = list(range(self.mlength))
            random.shuffle(positions)
            zeropos = positions[:nzero]
            for pos in zeropos:
                input[pos] = 0.0
            target = pattern[1]
            return [input, target], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_comp_targets)

    def make_prodtest_patfunc(self, noise=True):
        def patfunc(index=-1):
            if index < 0:
                pattern = random.choice(self.prod_test_patterns)
            elif index >= self.nlex:
                return False, 0
            else:
                pattern = self.prod_test_patterns[index]
            input = pattern[0] # noisify(pattern[0], sd=self.noise, indices=(0, self.mlength)) if (noise and self.noise) else pattern[0]
            target = pattern[1]
            return [input, target], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_prod_targets)

#    def make_testpatfunc(self):
#        def patfunc():
#            pattern = random.choice(self.patterns)

    def get_targets(self):
        """
        Returns targets for all patterns.
        """
        return [p[1] for p in self.patterns]

    def get_comp_targets(self):
        """
        Returns targets for comprehension test patterns.
        """
        return [p[1] for p in self.comp_test_patterns]

    def get_prod_targets(self):
        """
        Returns targets for production test patterns.
        """
        return [p[1] for p in self.prod_test_patterns]

    def spec_exp_conditions(self, comptrain=0.33, prodtrain=0.33,
                            noise=True):
        """
        Create specific pattern generation functions for an experiment.
        """
        patfunc = self.make_patfunc(compprob=comptrain, prodprob=prodtrain,
                                    noise=noise)
        if comptrain == 1.0:
            return [ [patfunc, self.make_comptest_patfunc()] ]
        elif prodtrain == 1.0:
            return [ [patfunc, self.make_prodtest_patfunc()] ]
#        elif comptrain + prodtrain == 1.0:
#            return [ [patfunc, self.make_test_patfunc()] ]
        else:
            return \
            [ [patfunc,
               {'full': self.make_test_patfunc(),
                'comp': self.make_comptest_patfunc(),
                'prod': self.make_prodtest_patfunc()}] ]

    # def make_exp_conditions(self, compprob=0.33, prodprob=0.33):
    #     patfunc = self.make_patfunc(compprob=compprob, prodprob=prodprob)
    #     testpatfunc = self.make_test_patfunc()
    #     comppatfunc = self.make_comptest_patfunc()
    #     prodpatfunc = self.make_prodtest_patfunc()
    #     self.exp_conds = \
    #     [ [patfunc,
    #        {'comp': comppatfunc, 'prod': prodpatfunc, 'full': testpatfunc}] ]

    def calc_iconicity(self):
        """
        Correlations between form and meaning dimensions.
        """
        result = []
        for fdim in range(self.flength):
            maxcorr = -1.0
            maxmdim = -1
            for mdim in range(self.mlength):
                corr = self.calc_iconicity_dim(fdim, mdim)
                if corr > maxcorr:
                    maxcorr = corr
                    maxmdim = mdim
            result.append((maxcorr, maxmdim))
        return result

    def calc_iconicity_dim(self, fdim, mdim):
        """
        Correlation between form dimension findex and meaning dimension mindex.
        """
        f = [l.get_form()[fdim] for l in self.entries]
        m = [l.get_meaning()[mdim] for l in self.entries]
        cc = np.corrcoef([f, m])
        return cc[0][1]

    # def make_patgen(self, npats):
    #     return PatGen(npats, patterns=self.make_patterns(npats))

class Lex(list):

    mpre = "M"
    fpre = "F"

    def __init__(self, lexicon, meaning, form=None, form_space=None, index=0,
                 arbitrary=False):
        self.iconic = lexicon.iconic
        self.lexicon = lexicon
        self.mlength = lexicon.mlength
        self.flength = lexicon.flength
#        self.deiconize = 0.5
        self.nvalues = lexicon.nvalues
        # Make Lex arbitrary even if Lexicon is supposed to be iconic
        self.arbitrary = arbitrary
        if not isinstance(form, Form):
            form = self.make_form(meaning, form_space, index)
        list.__init__(self, [meaning, form])

    def __repr__(self):
        return \
         "{} ‖ {}".format(self[0].__repr__(), self[1].__repr__())

    def get_meaning(self):
        return self[0]

    def get_form(self):
        return self[1]

    def make_arbitrary_form(self, form_space, index):
        f = form_space[index]
#        f = gen_array(self.nvalues, self.flength)
        return Form(f)

    def make_form(self, meaning, form_space=None, index=0):
        if self.arbitrary or not self.iconic:
            return self.make_arbitrary_form(form_space, index)
        else:
            return Form.make_iconic_form(meaning, self.nvalues, self.flength, self.lexicon.get_forms())
#            self.make_iconic_form(meaning)

    def make_empty_pattern(self, length, target=False):
        """
        Constant form or meaning pattern, 0.0 if input, DONT_CARE if target.
        """
        return np.full((length,), DONT_CARE if target else 0.0)

    def make_pattern(self, meaning=True, form=True, target=False):
        """
        Create a network input pattern or target from form and meaning.
        """
        if meaning:
            # meaning is either True or a float specifying the proportion
            # of non-zero values to set
            # if isinstance(meaning, float):
            #     m_full = self.get_meaning()
            #     m = self.make_empty_pattern(self.mlength, target=target)
            #     n_show_positions = round(self.mlength * meaning)
            #     positions = list(range(self.mlength))
            #     random.shuffle(positions)
            #     show_positions = positions[:n_show_positions]
            #     for p in show_positions:
            #         m[p] = m_full[p]
            # else:
            m = self.get_meaning()
        else:
            m = self.make_empty_pattern(self.mlength, target=target)
        if form:
            f = self.get_form()
#            for index in range(len(f)):
#                if f[index] == 0.0:
#                    f[index] = 0.05
        else:
            f = self.make_empty_pattern(self.flength, target=target)
#        m = self.get_meaning() if meaning else np.zeros(self.lexicon.mlength)
#        f = self.get_form() if form else np.zeros(self.lexicon.flength)
        return np.concatenate((m, f))

    def make_input_target(self, meaning=(True, True), form=(True, True)):
        input = self.make_pattern(meaning=meaning[0], form=form[0], target=False)
        target = self.make_pattern(meaning=meaning[1], form=form[1], target=True)
        return [input, target]

    def make_comprehension_IT(self, test=False, copy=False, simple=False):
        """
        Input-target pair for comprehension. If copy is True, form pattern
        is included in training target.
        """
        if simple:
            return [self.get_form(), self.get_meaning()]
        return \
        self.make_input_target(meaning=(True, True), form=(True, copy and not test))

    def make_production_IT(self, test=False, copy=False, simple=False):
        """
        Input-target for production. If copy is True, meaning pattern is included
        in training target.
        """
        if simple:
            return [self.get_meaning(), self.get_form()]
        return \
        self.make_input_target(meaning=(True, copy and not test), form=(False, True))

class Form(np.ndarray):
     """
     An array representing a linguistic form pattern or category,
     each element one dimension with nvalues possible values.
     """

     deiconize = 0.5
     pre = "F"

     def __new__(cls, input_array, nvalues=3, spec=None):
         a = np.asarray(input_array).view(cls)
         a.nvalues = nvalues
         return a

     def __array_finalize__(self, obj):
         if obj is None: return
         self.nvalues = getattr(obj, 'nvalues', None)

     def __repr__(self):
         if not self.shape:
             # The meaning somehow ended up scalar
             return "{}".format(float(self))
         s = Form.pre
#         mult = self.nvalues - 1
         for value in self:
#             if value < 0.1:
#                 value = 0.0
#             value = int(mult * value)
             value = round(100 * value)
             s += "{:> 4}".format(value)
         return s

     def __str__(self):
         s = Form.pre
#         mult = self.nvalues - 1
         for value in self:
#             if value < 0.1:
#                 value = 0.0
#             value = int(mult * value)
             value = round(100 * value)
             s += "{:> 4}".format(value)
         return s

     @staticmethod
     def make_random_form(nvalues, length, forms, threshold=0.4):
         candidate = None
         while True:
             candidate = gen_array(nvalues, length)
             found = True
             for f in forms:
                 if type(f) != np.ndarray:
                     continue
                 if array_distance(candidate, f) < threshold:
                     found = False
                     break
             if not found:
                 continue
             return Form(candidate)

     @staticmethod
     def make_iconic_form(meaning, nvalues, length, forms, threshold=0.4):
#         print("Make iconic form {}".format(forms))
         candidate = None
         while True:
             candidate = Form.iconic_form_candidate(meaning, nvalues, length)
             found = True
             for f in forms:
                 if type(f) != Form:
                     continue
                 if array_distance(candidate, f) < threshold:
#                     print("Distance {}".format(array_distance(candidate, f)))
#                 (candidate == f).all():
#                     print("{} already in forms".format(candidate))
                     found = False
                     break
             if not found:
                 continue
             return Form(candidate)

     @staticmethod
     def iconic_form_candidate(meaning, nvalues, length):
         form = np.copy(meaning)[:length]
         if not Form.deiconize:
             return form
         nflip = int(Form.deiconize * (1.0 - 1.0 / nvalues) * length)
         flip_positions = list(range(length))
         random.shuffle(flip_positions)
         flip_positions = flip_positions[:nflip]
         values = gen_value_opts(nvalues)
         for pos in flip_positions:
             form[pos] = random.choice(values)
         return form
