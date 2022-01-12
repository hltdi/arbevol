"""
arbevol lexicon: networks, training pattern generators, experiments.
"""

from experiment import *
from environment import *

class Lexicon:

    def __init__(self, id, environment, flength, iconic=True,
                 enforce_distance=False, noise=0.1,
                 compprob=0.0, prodprob=0.0):
        self.id = id
        self.environment = environment
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
        self.nlex = environment.nmeanings
#        if not iconic:
#            # Create form space for arbitrary Lex entries (must precede make())
#            self.form_space = self.gen_form_space()
#        else:
#            self.form_space = None
        self.make(environment.meanings)
        self.make_patterns()
        self.make_exp_conditions()
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

    def make(self, meanings):
        """
        Create nlex Lex instances.
        """
        if self.iconic:
            for m in meanings:
                self.entries.append(Lex(self, m))
        else:
            form_space = self.gen_form_space()
            for index, m in enumerate(meanings):
                self.entries.append(Lex(self, m, form_space, index))

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

    # def make_lex(self, meaning):
    #     """
    #     Create a single Lex entry.
    #     """
    #     self.entries.append(Lex(self, meaning))

    def make_patterns(self):
        self.patterns = [l.make_input_target() for l in self.entries]
        self.comp_test_patterns = [l.make_comprehension_IT(test=True) for l in self.entries]
        self.prod_test_patterns = [l.make_production_IT(test=True) for l in self.entries]
        self.comp_train_patterns = [l.make_comprehension_IT(test=False) for l in self.entries]
        self.prod_train_patterns = [l.make_production_IT(test=False) for l in self.entries]

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

    def make_patfunc(self, compprob=1.0, prodprob=0.0):
        """
        perfprob is the probability of training on either a comprehension
        or a production input pattern.
        """
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
                    if self.noise:
                        input = noisify(pattern[0], sd=self.noise, indices=(self.mlength, self.patlength))
                    else:
                        input = pattern[0]
                else:
                    # Production
                    pattern = self.prod_train_patterns[pindex]
                    if self.noise:
                        input = noisify(pattern[0], sd=self.noise, indices=(0, self.mlength))
                    else:
                        input = pattern[0]
            else:
                pattern = self.patterns[pindex]
                input = noisify(pattern[0], sd=self.noise) if self.noise else pattern[0]
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
            input = noisify(pattern[0], sd=self.noise) if (noise and self.noise) else pattern[0]
            target = pattern[1]
            return [input, target], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_targets)

    def make_comptest_patfunc(self, noise=True):
        def patfunc(index=-1):
            if index < 0:
                pattern = random.choice(self.comp_test_patterns)
            elif index >= self.nlex:
                return False, 0
            else:
                pattern = self.comp_test_patterns[index]
            input = noisify(pattern[0], sd=self.noise, indices=(self.mlength, self.patlength)) if (noise and self.noise) else pattern[0]
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
            input = noisify(pattern[0], sd=self.noise, indices=(0, self.mlength)) if (noise and self.noise) else pattern[0]
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

    def spec_exp_conditions(self, comptrain=0.33, prodtrain=0.33):
        """
        Create specific pattern generation functions for an experiment.
        """
        patfunc = self.make_patfunc(compprob=comptrain, prodprob=prodtrain)
        if comptrain == 1.0:
            return [ [patfunc, self.make_comptest_patfunc()] ]
        elif prodtrain == 1.0:
            return [ [patfunc, self.make_prodtest_patfunc()] ]
        elif comptrain + prodtrain == 1.0:
            return [ [patfunc, self.make_test_patfunc()] ]
        else:
            return \
            [ [patfunc,
               {'full': self.make_test_patfunc(),
                'comp': self.make_comptest_patfunc(),
                'prod': self.make_prodtest_patfunc()}] ]

    def make_exp_conditions(self, compprob=0.33, prodprob=0.33):
        patfunc = self.make_patfunc(compprob=compprob, prodprob=prodprob)
        testpatfunc = self.make_test_patfunc()
        comppatfunc = self.make_comptest_patfunc()
        prodpatfunc = self.make_prodtest_patfunc()
        self.exp_conds = \
        [ [patfunc,
           {'comp': comppatfunc, 'prod': prodpatfunc, 'full': testpatfunc}] ]

    # def make_patgen(self, npats):
    #     return PatGen(npats, patterns=self.make_patterns(npats))

class Lex(list):

    mpre = "$"
    fpre = "&"

    def __init__(self, lexicon, meaning, form_space=None, index=0):
        self.iconic = lexicon.iconic
        self.lexicon = lexicon
        self.mlength = lexicon.mlength
        self.flength = lexicon.flength
        self.deiconize = 0.5
        self.nvalues = lexicon.nvalues
        list.__init__(self, [meaning, self.make_form(meaning, form_space, index)])

    def show_form(self):
        '''
        Display meaning or form.
        '''
        form = self[1]
        s = Lex.fpre
        mult = self.lexicon.nvalues - 1
        for value in form:
            value = int(mult * value)
            s += str(value)
        return s

    def __repr__(self):
        return \
         "{}{}".format(self[0].__repr__(), self.show_form())

    def get_meaning(self):
        return self[0]

    def get_form(self):
        return self[1]

    def make_iconic_form(self, meaning):
        form = np.copy(meaning)
        nflip = int(self.deiconize * (1.0 - 1.0 / self.nvalues) * self.flength)
        flip_positions = list(range(self.flength))
        random.shuffle(flip_positions)
        flip_positions = flip_positions[:nflip]
#        print("flip: {}".format(flip_positions))
        values = gen_value_opts(self.nvalues)
#        value_dict = dict((v, [vv for vv in values if vv != v]) for v in values)
#        print("Flip positions: {}".format(flip_positions))
#        print("Value dict: {}".format(value_dict))
        for pos in flip_positions:
#            iconvalue = form[pos]
#            options = value_dict[iconvalue]
#            choice = random.choice(options)
            form[pos] = random.choice(values)
        return form

    def make_arbitrary_form(self, form_space, index):
        f = form_space[index]
#        for index in range(len(f)):
#            if f[index] == 0.0:
#                f[index] = 0.05
        return f

    def make_form(self, meaning, form_space=None, index=0):
        if self.iconic:
            return self.make_iconic_form(meaning)
        else:
            return self.make_arbitrary_form(form_space, index)

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
            m = self.get_meaning()
#            for index in range(len(m)):
#                if m[index] == 0.0:
#                    m[index] = 0.05
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

    def make_comprehension_IT(self, test=False, copy=False):
        """
        Input-target pair for comprehension. If copy is True, form pattern
        is included in training target.
        """
        return \
        self.make_input_target(meaning=(False, True), form=(True, copy and not test))

    def make_production_IT(self, test=False, copy=False):
        """
        Input-target for production. If copy is True, meaning pattern is included
        in training target.
        """
        return \
        self.make_input_target(meaning=(True, copy and not test), form=(False, True))
