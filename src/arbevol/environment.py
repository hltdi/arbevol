"""
arbevol
Environment: meanings and time
"""

import numpy as np
from utils import *

class Environment:

    time = 0

    def __init__(self, mvalues=3, mlength=6, nmeanings=500,
                 mspec=None):
        self.mvalues = mvalues
        self.mlength = mlength
        self.mspec = mspec
        maxn = self.max_nmeanings()
        print("Creating environment: mlength {}, mvalues {}, max {} meanings".format(mlength, mvalues, maxn))
        self.nmeanings = round(maxn / 2)
        self.meanings = []
        # Meanings grouped by cluster
        self.clusters = []
        self.init_meanings(self.nmeanings)

    def init_meanings(self, n):
        for i in range(n):
            self.make_meaning(self.mspec)

    def max_nmeanings(self):
        return np.power(self.mvalues, self.mlength)

    def make_meaning(self, spec=None):
        """
        Create a unique meaning.
        """
        while True:
            meaning = gen_array(self.mvalues, self.mlength, spec=spec)
#            for i in range(len(meaning)):
#                if meaning[i] == 0.0:
#                    meaning[1] = 0.05
            found = True
            for m in self.meanings:
                if (meaning == m).all():
                    found = False
                    break
            if not found:
                continue
            meaning = Meaning(meaning, nvalues=self.mvalues)
            self.meanings.append(meaning)
            return meaning

    def make_meaning_cluster(self, mdims, n):
        """
        Create n meanings with value 1.0 on all dimensions in mdims,
        random values in other dimensions.
        """
        spec = dict([(d, 1.0) for d in mdims])
        for i in range(n):
            self.make_meaning(spec=spec)

class Meaning(np.ndarray):
     """
     An array representing a category in the environment,
     each element one dimension with nvalues possible values.
     """

     pre = "M"

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
         s = Meaning.pre
#         mult = self.nvalues - 1
         for value in self:
#             if value < 0.1:
#                 value = 0.0
#             value = int(mult * value)
             value = round(100 * value)
             s += "{:> 4}".format(value)
         return s

     def __str__(self):
         s = Meaning.pre
#         mult = self.nvalues - 1
         for value in self:
#             if value < 0.1:
#                 value = 0.0
#             value = int(mult * value)
             value = round(100 * value)
             s += "{:> 4}".format(value)
         return s
