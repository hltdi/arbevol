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
        # comppats and prodpats
        self.make_patterns()
        self.mean2mean_pats = [[l.get_meaning(), l.get_meaning()] for l in self.entries]
        # constant empty Form array
        self.empty_form = Form(np.full(self.flength, -100.0))
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

    def show(self):
        for i, e in enumerate(self.entries):
            print("{:>3} {}".format(i, e))

    def update_forms(self, form_dict, index_errors,
                     sep_amount=0.01, verbose=0):
        '''
        Adjust the form values in the lexicon based on those in the form
        dict (entry_index: array).
        index_errors is a dict with indices of patterns for which there were
        target category errors and the indices of the errors.
        Patterns must be regenerated after this.
        '''
        old_entries = [entry.copy() for entry in self.entries]
        for index, entry in enumerate(self.entries):
            form = form_dict.get(index, self.empty_form)
            if form is self.empty_form:
                if verbose:
                    print("Updating {}".format(index))
                # No new form for this pattern because of an error
                error_index = index_errors[index]
#                print("** Error indices: {} {}".format(index, error_index))
                old_form = entry.get_form()
                error_form = old_entries[error_index][1]
#                print("** Forms {} {}".format(old_form, error_form))
                old_form.separate(error_form, amount=sep_amount, verbose=verbose)
            else:
                if verbose:
                    print("Updating {} from network: {}".format(index, form))
                entry[1] = Form(form)
#        for index, form in form_dict.items():
#            entry = self.entries[index]
#            entry[1] = Form(form)
        # Make patterns using new forms.
        self.make_patterns()

    def make(self, meanings, patterns=None):
        """
        Create nlex Lex instances. patterns are provided for a lexicon
        generated from network output.
        """
        if patterns:
            for meaning, form in patterns:
                self.entries.append(Lex(self, meaning, form=form))
#        elif self.iconic:
        else:
            for m in meanings:
                self.entries.append(Lex(self, m))
#        else:
#            for index, m in enumerate(meanings):
#                self.entries.append(Lex(self, m, form_space=self.form_space, index=index))

    def add(self, meanings, patterns=None):
        """
        Create new Lex instances for each of meanings, and add them to
        entries. Or if patterns is given, make an entry for each of these.
        """
        if patterns:
            for meaning, form in patterns:
                self.entries.append(Lex(self, meaning, form=form))
#        elif self.iconic:
        else:
            for m in meanings:
                self.entries.append(Lex(self, m))
#        else:
#            for index, m in enumerate(meanings):
#                self.entries.append(Lex(self, m, form_space=form_space,
#                                        index=self.nlex + index))
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

    def make_patterns(self):
        self.comppats = [l.make_comprehension_IT(simple=True) for l in self.entries]
        self.prodpats = [l.make_production_IT(simple=True) for l in self.entries]

    def get_meaning_targets(self):
        return [e[0] for e in self.entries]

    def get_form_targets(self):
        return [e[1] for e in self.entries]

    # def perf_pattern(self, pattern, comp=True):
    #     """
    #     Convert a full input pattern to a comprehension or production pattern.
    #     """
    #     p = np.copy(pattern)
    #     if comp:
    #         for i in range(self.mlength, self.patlength):
    #             p[i] = 0.0
    #         return p
    #     else:
    #         for i in range(self.mlength):
    #             p[i] = 0.0
    #         return p

    # def make_patfunc(self, compprob=1.0, prodprob=0.0, showmeaning=0.2,
    #                  noise=True):
    #     """
    #     perfprob is the probability of training on either a comprehension
    #     or a production input pattern.
    #     showmeaning is the probability of revealing one of the meaning units
    #     in a comprehension input pattern.
    #     """
    #     noise = noise and self.noise
    #     perfprob = compprob + prodprob
    #     if perfprob > 1.0:
    #         print("Comp prob {} and prod prob {} sum > 1".format(compprob, prodprob))
    #     def patfunc(index=-1):
    #         pindex = np.random.randint(0, len(self.entries))
    #         rand = np.random.rand()
    #         if perfprob and rand < perfprob:
    #             # Performance training instance
    #             if compprob and rand < compprob:
    #                 # Comprehension
    #                 pattern = self.comp_train_patterns[pindex]
    #                 if noise:
    #                     input = noisify(pattern[0], sd=noise, indices=(self.mlength, self.patlength))
    #                 else:
    #                     input = np.copy(pattern[0])
    #                 # Set input meaning to 0.0 except in showmeaning positions
    #                 nzero = round((1.0 - showmeaning) * self.mlength)
    #                 positions = list(range(self.mlength))
    #                 random.shuffle(positions)
    #                 zeropos = positions[:nzero]
    #                 for pos in zeropos:
    #                     input[pos] = 0.0
    #             else:
    #                 # Production
    #                 pattern = self.prod_train_patterns[pindex]
    #                 if noise:
    #                     input = noisify(pattern[0], sd=noise, indices=(0, self.mlength))
    #                 else:
    #                     input = pattern[0]
    #         else:
    #             pattern = self.patterns[pindex]
    #             input = noisify(pattern[0], sd=noise) if noise else pattern[0]
    #         target = pattern[1]
    #         return [input, target], 0
    #     return PatGen(self.nlex, function=patfunc, targets=self.get_targets)
    #
    # def make_test_patfunc(self, noise=True):
    #     def patfunc(index=-1):
    #         if index < 0:
    #             pattern = random.choice(self.patterns)
    #         elif index >= self.nlex:
    #             return False, 0
    #         else:
    #             pattern = self.patterns[index]
    #         input = pattern[0] # noisify(pattern[0], sd=self.noise) if (noise and self.noise) else pattern[0]
    #         target = pattern[1]
    #         return [input, target], 0
    #     return PatGen(self.nlex, function=patfunc, targets=self.get_targets)
    #
    # def make_comptest_patfunc(self, noise=True, showmeaning=0.2):
    #     def patfunc(index=-1):
    #         if index < 0:
    #             pattern = random.choice(self.comp_test_patterns)
    #         elif index >= self.nlex:
    #             return False, 0
    #         else:
    #             pattern = self.comp_test_patterns[index]
    #         input = np.copy(pattern[0]) # noisify(pattern[0], sd=self.noise, indices=(self.mlength, self.patlength)) if (noise and self.noise) else pattern[0]
    #         # Set input meaning to 0.0 except in showmeaning positions
    #         nzero = round((1.0 - showmeaning) * self.mlength)
    #         positions = list(range(self.mlength))
    #         random.shuffle(positions)
    #         zeropos = positions[:nzero]
    #         for pos in zeropos:
    #             input[pos] = 0.0
    #         target = pattern[1]
    #         return [input, target], 0
    #     return PatGen(self.nlex, function=patfunc, targets=self.get_comp_targets)
    #
    # def make_prodtest_patfunc(self, noise=True):
    #     def patfunc(index=-1):
    #         if index < 0:
    #             pattern = random.choice(self.prod_test_patterns)
    #         elif index >= self.nlex:
    #             return False, 0
    #         else:
    #             pattern = self.prod_test_patterns[index]
    #         input = pattern[0] # noisify(pattern[0], sd=self.noise, indices=(0, self.mlength)) if (noise and self.noise) else pattern[0]
    #         target = pattern[1]
    #         return [input, target], 0
    #     return PatGen(self.nlex, function=patfunc, targets=self.get_prod_targets)

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

    def iconicity(self):
        """
        Correlations between form and meaning dimensions.
        """
        result = []
        for fdim in range(self.flength):
            maxcorr = -1.0
            maxmdim = -1
            for mdim in range(self.mlength):
                corr = self.iconicity_dim(fdim, mdim)
                if corr > maxcorr:
                    maxcorr = corr
                    maxmdim = mdim
            result.append((maxcorr, maxmdim))
        result = sum([x[0] for x in result]) / self.flength
        return result

    def iconicity_dim(self, fdim, mdim):
        """
        Correlation between form dimension findex and meaning dimension mindex.
        """
        f = [l.get_form()[fdim] for l in self.entries]
        m = [l.get_meaning()[mdim] for l in self.entries]
        cc = np.corrcoef([f, m])
        return cc[0][1]

    def distance(self, lexicon):
        '''
        Average distance between forms of this and another lexicon.
        '''
        dist = 0.0
        for e1, e2 in zip(self.entries, lexicon.entries):
            dist += array_distance(e1.get_form(), e2.get_form())
        dist /= self.nlex
        return dist

    def write(self, filename):
        """
        Write the Lexicon's entries to a file.
        """
        path = os.path.join('data', filename)
        with open(path, 'w') as out:
            for entry in self.entries:
                print(entry, file=out)

    @staticmethod
    def read(filename, mvalues=5, fvalues=5):
        path = os.path.join('data', filename)
        entries = []
        with open(path) as infile:
            for line in infile:
                meaning, form = line.split(" ‖ ")
                meaning = Meaning.read(meaning, mvalues)
                form = Form.read(form, fvalues)
                entries.append([meaning, form])
        return entries

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
        if not isinstance(form, (Form, np.ndarray)):
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
#        f = form_space[index]
        f = gen_array(self.nvalues, self.flength)
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

     def separate(self, form2, amount=0.02, onedim=False, verbose=1):
         '''
         Move this Form away from form2.
         '''
#         if verbose:
         print("** Separating {} from {}".format(self, form2), end=' ')
         diffs = self - form2
         if onedim:
             # Find the dimension where the 2 forms are closest
             nearest_dim = np.argmin(np.abs(diffs))
             diff = diffs[nearest_dim]
             incr = amount if diff > 0 else -amount
             self[nearest_dim] += incr
         else:
             # Adjust values on all dimensions, in proportion to 1-abs(diff)
             for dim in range(len(self)):
                 diff = diffs[dim]
                 mult = (1.0 - abs(diff)) * amount
                 incr = mult if diff > 0 else -mult
                 self[dim] += incr
#         if verbose:
         print("-> {}".format(self))

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
     def make_iconic_form(meaning, nvalues, length, forms,
                          threshold=0.2, constant_flip=True):
#         print("Make iconic form {}".format(forms))
         candidate = None
         while True:
             candidate = Form.iconic_form_candidate(meaning, nvalues, length, constant_flip=constant_flip)
             if Form.deiconize < 0.1:
                 return Form(candidate)
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
     def iconic_form_candidate(meaning, nvalues, length, constant_flip=True):
         form = np.copy(meaning)[:length]
         if Form.deiconize < 0.1:
             return form
         nflip = int(Form.deiconize * (1.0 - 1.0 / nvalues) * length)
         if constant_flip:
             flip_positions = range(length-1, length-1-nflip, -1)
         else:
             flip_positions = list(range(length))
             random.shuffle(flip_positions)
             flip_positions = flip_positions[:nflip]
         values = gen_value_opts(nvalues)
         for pos in flip_positions:
             form[pos] = random.choice(values)
         return form

     @staticmethod
     def read(string, fvalues):
         elements = string.split()
         label = elements[0]
         values = [int(v)/100.0 for v in elements[1:]]
         array = np.array(values)
         return Form(array, nvalues=fvalues)
