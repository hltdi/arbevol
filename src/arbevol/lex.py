"""
arbevol lexicon: networks, training pattern generators, experiments.
"""

from experiment import *
from environment import *
from itertools import combinations

class Lexicon:

    # SD for noisify when noisifying form resulting in missed meanings
    form_noise = 0.05

    def __init__(self, id, environment, flength, iconic=True,
                 nlex=10,
                 enforce_distance=False, noise=0.1, noisify_errors=False,
                 # Iconicity params
                 constant_flip=True, deiconize=0.5,
                 # True only for initial lexicon
                 init=False,
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
        self.noisify_errors = noisify_errors
        self.constant_flip = constant_flip
        self.deiconize = deiconize
        # Later a subset of the environment's meanings?
        self.meanings = environment.meanings[:nlex]
        self.init = init
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
        self.empty_form = Form(np.full(self.flength, -100.0), nvalues=self.nvalues)
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
                     sep_amount=0.01, verbose=1):
        '''
        Adjust the form values in the lexicon based on those in the form
        dict (entry_index: array).
        index_errors is a dict with indices of patterns for which there were
        target category errors and the indices of the errors.
        Patterns must be regenerated after this.
        '''
        old_entries = [entry.copy() for entry in self.entries]
        for index, entry in enumerate(self.entries):
            form = form_dict.get(index) #, self.empty_form)
            if index in index_errors:
                if verbose:
                    print("** Updating missed form {}: {}".format(index, old_form))
                if self.noisify_errors:
                    if Form.roundQ:
                        form = form.copy()
                        # First round form
                        Form.round(form, self.nvalues)
                        if verbose:
                            print("  ** Updating {} ->".format(form), end=' ')
                        # Then flip one value
                        flip_values(form, 1, self.nvalues, length=self.flength,
                                    opts=Form.value_opts[self.nvalues])
                        if verbose:
                            print("  {}".format(form))
                    else:
                        if verbose:
                            print("  ** Updating {} ->".format(form), end=' ')
                        form = Form(noisify(form, sd=Lexicon.form_noise),
                                    init=True, nvalues=self.nvalues)
                        if verbose:
                            print("{}".format(form))
#                    print("** Noisifying {}: {}".format(old_form, form))
                    entry[1] = form
                else:
                    # No new form for this pattern because of an error
                    form = Form(form, self.nvalues)
                    error_index = index_errors[index]
                    error_form = old_entries[error_index][1]
                    form.separate(error_form, amount=sep_amount, verbose=verbose)
                    entry[1] = form
            else:
                if verbose:
                    print("Updating {} from network: {}".format(index, form))
                entry[1] = Form(form, self.nvalues)
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

    def m_iconicity(self):
        result = []
        for mdim in range(self.mlength):
            maxcorr = -1.0
            maxfdim= -1
            for fdim in range(self.flength):
                corr = self.iconicity_dim(fdim, mdim)
                if corr > maxcorr:
                    maxcorr = corr
                    maxfdim = fdim
            result.append((maxcorr, maxfdim))
        result = sum([x[0] for x in result]) / self.mlength
        return result

    def iconicity(self, rows=None, verbose=0):
        """
        Correlations between form and meaning dimensions.
        """
        result = []
        for fdim in range(self.flength):
            maxcorr = -1.0
            maxmdim = -1
            for mdim in range(self.mlength):
                # Use abs because negative and positive correlations are
                # equally useful
                corr = abs(self.iconicity_dim(fdim, mdim, rows=rows))
                if corr > maxcorr:
                    maxcorr = corr
                    maxmdim = mdim
            result.append((maxcorr, maxmdim))
        if verbose:
            print("** {}".format(result))
        result = sum([x[0] for x in result]) / self.flength
        return result

    def dim_iconicity(self, rows=None):
        results = []
        for fdim in range(self.flength):
            results1 = []
            for mi, mdim in enumerate(range(self.mlength)):
                corr = self.iconicity_dim(fdim, mdim, rows=rows)
                results1.append(np.round(corr, 3))
            results.append(results1)
        return results

#    def c_iconicity(self):
#        clusters = self.environment.clus_indices
#        for c, rows in clusters.items():
#            entries = [self.entries[row] for row in rows]

    def matched_iconicity(self):
        result = []
        for fdim in range(self.flength):
            corr = self.iconicity_dim(fdim, fdim)
            result.append(corr)
        result = sum(result) / self.flength
        return result

    def distcorr(self, indices=None, between_clusters=False, within_clusters=False, verbose=0):
        mdists = []
        fdists = []
        if between_clusters:
            pairs = self.environment.get_between_cluster_pairs()
        elif within_clusters:
            pairs = self.environment.get_within_cluster_pairs()
        else:
            if not indices:
                indices = range(len(self.entries))
            pairs = combinations(indices, 2)

        for i1, i2 in pairs:
            dm, df = self.entry_distances(i1, i2)
            mdists.append(dm)
            fdists.append(df)
#        if verbose:
#            print("{}".format([np.round(x) for x in mdists])
#            print("{}".format([np.round(x) for x in fdists])
        cc = np.corrcoef([mdists, fdists])
        return cc[0][1]

    def dc(self):
        return self.distcorr(), self.distcorr(between_clusters=True), self.distcorr(within_clusters=True)

    def entry_distances(self, i1, i2):
        e1 = self.entries[i1]
        e2 = self.entries[i2]
        dm = adistance(e1[0], e2[0])
        df = adistance(e1[1], e2[1])
        return dm, df

    def iconicity_dim(self, fdim, mdim, rows=None):
        """
        Correlation between form dimension findex and meaning dimension mindex.
        """
        if rows:
            entries = [self.entries[row] for row in rows]
        else:
            entries = self.entries
        f = [l.get_form()[fdim] for l in entries]
        m = [l.get_meaning()[mdim] for l in entries]
        cc = np.corrcoef([f, m])
        return cc[0][1]

    def distance(self, lexicon):
        '''
        Average distance between forms of this and another lexicon.
        '''
        dist = 0.0
        for e1, e2 in zip(self.entries, lexicon.entries):
            dist += adistance(e1.get_form(), e2.get_form())
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
        self.deiconize = lexicon.deiconize
        self.constant_flip = lexicon.constant_flip
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
        return Form(f, nvalues=self.nvalues)

    def make_iconic_form(self, meaning):
        f = \
        Form.make_iconic_form(meaning, self.nvalues, self.flength,
                              self.lexicon.get_forms(),
                              constant_flip=self.constant_flip,
                              deiconize=self.deiconize)
        return f

    def make_form(self, meaning, form_space=None, index=0):
        if self.arbitrary or not self.iconic:
            return self.make_arbitrary_form(form_space, index)
        else:
            return self.make_iconic_form(meaning)

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

#     deiconize = 0.5
     pre = "F"
     roundQ = False

     value_opts = {3: gen_value_opts(3), 4: gen_value_opts(4), 5: gen_value_opts(5),
                   6: gen_value_opts(6), 7: gen_value_opts(7), 8: gen_value_opts(8),
                   9: gen_value_opts(9), 10: gen_value_opts(10), 11: gen_value_opts(11)}

     def __new__(cls, input_array, nvalues=5, init=False, spec=None):
         if not init and Form.roundQ:
             Form.round(input_array, nvalues)
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

     def copy(self):
         array = np.copy(self)
         return Form(array, nvalues=self.nvalues, init=True)

     def separate(self, form2, amount=0.02, onedim=False, verbose=1):
         '''
         Move this Form away from form2.
         '''
         if verbose:
             print("** Separating {} from {} ({})".format(self, form2, amount), end=' ')
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
         if verbose:
             print("-> {}".format(self))

     @staticmethod
     def round(array, nvalues):
         opts = Form.value_opts[nvalues]
         for i, value in enumerate(array):
             array[i] = round_nvalues(value, nvalues, opts=opts)

     @staticmethod
     def make_random_form(nvalues, length, forms, threshold=0.1):
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
             return Form(candidate, nvalues=nvalues, init=True)

     @staticmethod
     def make_iconic_form(meaning, nvalues, length, forms,
                          threshold=0.05, constant_flip=False,
                          deiconize=0.5):
#         print("Make iconic form {}".format(forms))
         candidate = None
         while True:
             candidate = \
             Form.iconic_form_candidate(meaning, nvalues, length,
                                        constant_flip=constant_flip,
                                        deiconize=deiconize)
             if deiconize < 0.1:
                 return Form(candidate, nvalues=nvalues)
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
             return Form(candidate, nvalues=nvalues, init=True)

     @staticmethod
     def iconic_form_candidate(meaning, nvalues, length,
                               constant_flip=True, deiconize=0.5):
         form = np.copy(meaning)[:length]
         if deiconize < 0.1:
             return form
         nflip = int(deiconize * (1.0 - 1.0 / nvalues) * length)
         if constant_flip:
             flip_positions = range(length-1, length-1-nflip, -1)
         else:
             flip_positions = None
         flip_values(form, nflip, nvalues, flip_positions=flip_positions,
                     length=length, ran=True)
#         else:
#             flip_positions = list(range(length))
#             random.shuffle(flip_positions)
#             flip_positions = flip_positions[:nflip]
#         values = gen_value_opts(nvalues)
#         for pos in flip_positions:
#             form[pos] = random.choice(values)
         return form

     @staticmethod
     def read(string, fvalues):
         elements = string.split()
         label = elements[0]
         values = [int(v)/100.0 for v in elements[1:]]
         array = np.array(values)
         return Form(array, nvalues=fvalues)
