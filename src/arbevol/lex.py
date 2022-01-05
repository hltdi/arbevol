"""
arbevol lexicon: networks, training pattern generators, experiments.
"""

from experiment import *

def noisify(array, sd=0.05):
    func = np.vectorize(lambda x: noisify1(x, sd=sd))
    return func(array)

def noisify1(value, sd=0.05):
    return value + np.random.normal(0.0, sd)

def gen_value(nvalues):
    ran = np.random.rand()
    minval = 1.0 / nvalues
    interval = 1.0 / (nvalues-1.0)
    values = [(i * interval, (i+1) * minval) for i in range(nvalues)[:-1]]
    for v1, v2 in values:
#            print("{}, {}, {}".format(v1, v2, ran))
        if ran < v2:
            return v1
    return 1.0

# def gen_value(nvalues):
#     ran = np.random.rand()
#     minval = 1.0 / nvalues
#     interval = 1.0 # / (nvalues-1.0)
#     values = [(i * interval, (i+1) * minval) for i in range(nvalues)[:-1]]
#     for v1, v2 in values:
# #        print("{}, {}, {}".format(v1, v2, ran))
#         if ran < v2:
#             return v1
#     return (nvalues-1) * interval

def gen_array(nvalues, length):
    iter = (gen_value(nvalues) for i in range(length))
    return np.fromiter(iter, float)

class Lexicon:

    def __init__(self, nform, nmeaning, iconic=True, nvalues=3,
                 nlex=100, enforce_distance=False):
        self.nform = nform
        self.nmeaning = nmeaning
        self.iconic = iconic
        self.nvalues = nvalues
        self.enforce_distance = enforce_distance
        self.entries = []
        self.patterns = []
        self.nlex = nlex
        self.make(nlex)

    def make_experiment(self, nhidden, name='lex_exp'):
        network = self.make_network(nhidden=nhidden)
#        patgen = self.make_patgen(npatterns)
        patfunc = self.make_patfunc()
        return Experiment(name, network=network, conditions=[[patfunc]],
#        [[patgen]],
                          test_nearest=True)

    def make_network(self, nhidden=10, name='lex'):
        return \
        Network(name,
                layers = [Layer('in', self.nform+self.nmeaning),
                          Layer('hid', nhidden),
                          Layer('out', self.nform+self.nmeaning)])

    def make(self, n):
        """
        Create n Lex instances.
        """
        for i in range(n):
            self.make_lex()

    def make_lex(self):
        """
        Create a single Lex entry.
        """
        self.entries.append(Lex(self))

    def make_patterns(self):
        self.patterns = [l.make_input_target() for l in self.entries]

    # def make_patterns(self, n):
    #     """
    #     Create n unique patterns. (Should check for an infinite loop.)
    #     """
    #     patterns = []
    #     inputs = []
    #     while len(patterns) < n:
    #         input = self.make_pattern()
    #         for i in inputs:
    #             if (input == i).all():
    #                 continue
    #         inputs.append(input)
    #         target = self.make_target(input)
    #         p = [input, target]
    #         patterns.append(p)
    #     return patterns
    #
    # def make_meaning_pattern(self):
    #     return gen_array(self.nvalues, self.nmeaning)
    #
    # def make_form_pattern(self, meaning):
    #     if self.iconic:
    #         return np.copy(meaning)
    #     else:
    #         return gen_array(self.nvalues, self.nform)

    # def make_pattern(self):
    #     """
    #     Make form and meaning arrays and concatenate them.
    #     """
    #     meaning = self.make_meaning_pattern()
    #     form = self.make_form_pattern(meaning)
    #     return np.concatenate((meaning, form))
    #
    # def make_target(self, input):
    #     """
    #     For now just copy the input as target.
    #     """
    #     return np.copy(input)

    def make_patfunc(self):
        self.make_patterns()
        def patfunc():
            pattern = random.choice(self.patterns)
            return [noisify(pattern[0]), pattern[1]], 0
        return PatGen(self.nlex, function=patfunc, targets=self.get_targets)

#    def make_testpatfunc(self):
#        def patfunc():
#            pattern = random.choice(self.patterns)

    def get_targets(self):
        """
        Returns targets for all patterns.
        """
        return [p[1] for p in self.patterns]

    # def make_patgen(self, npats):
    #     return PatGen(npats, patterns=self.make_patterns(npats))

class Lex(list):

    mpre = "$"
    fpre = "&"

    def __init__(self, lexicon):
        self.iconic = lexicon.iconic
        self.lexicon = lexicon
        self.init()

    def init(self):
        while True:
            meaning = self.make_meaning()
            found = True
            for l in self.lexicon.entries:
                m = l.get_meaning()
                if (meaning == m).all():
                    found = False
                    break
            if not found:
                continue
            self.append(meaning)
            self.append(self.make_form(meaning))
            break

    def show_part(self, index):
        '''
        Display meaning or form.
        '''
        array = self[index]
        s = Lex.mpre if index == 0 else Lex.fpre
        mult = self.lexicon.nvalues - 1
        for value in array:
            value = int(mult * value)
            s += str(value)
        return s

    def __repr__(self):
        return \
         "{}{}".format(self.show_part(0), self.show_part(1))

    def get_meaning(self):
        return self[0]

    def get_form(self):
        return self[1]

    def make_meaning(self):
        return gen_array(self.lexicon.nvalues, self.lexicon.nmeaning)

    def make_form(self, meaning):
        if self.iconic:
            return np.copy(meaning)
        return gen_array(self.lexicon.nvalues, self.lexicon.nform)

    def make_pattern(self, meaning=True, form=True):
        """
        Create a network input pattern or target from form and meaning.
        """
        m = self.get_meaning() if meaning else np.zeros(self.lexicon.nmeaning)
        f = self.get_form() if form else np.zeros(self.lexicon.nform)
        return np.concatenate((m, f))

    def make_input_target(self, meaning=(True, True), form=(True, True)):
        input = self.make_pattern(meaning=meaning[0], form=form[0])
        target = self.make_pattern(meaning=meaning[1], form=form[1])
        return [input, target]
