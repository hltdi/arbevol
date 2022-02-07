"""
arbevol
Environment: meanings and time
"""

import numpy as np
from utils import *
from itertools import permutations

class Environment:

    time = 0

    def __init__(self, mvalues=3, mlength=6, nmeanings=500, clusters=None,
                 init_meanings=None):
        self.mvalues = mvalues
        self.mlength = mlength
        maxn = self.max_nmeanings()
        self.nmeanings = nmeanings
#        round(maxn / 2)
        self.meanings = []
        # clusters is either None (no clusters) or a triple consisting of the proportion of fixed
        # dimensions in each cluster, the number of clusters, and the number of meanings
        # in each cluster
        if clusters:
            self.cluster_spec = self.make_meaning_clusters(clusters[0], clusters[1])
            self.n_per_cluster = clusters[2]
        else:
            self.cluster_spec = None
            self.n_per_cluster = 0
        # Meanings grouped by cluster
        self.clusters = {}
        if init_meanings:
            self.read_meanings(init_meanings)
        else:
            self.init_meanings(self.nmeanings, self.cluster_spec, self.n_per_cluster)
        print("Creating environment: mlength {}, mvalues {}, {} meanings".format(mlength, mvalues, self.nmeanings), end='')
        if clusters:
            print(", {} clusters with {} of dimensions fixed".format(clusters[1], clusters[0]))
        else:
            print()

    def init_meanings(self, n, cluster_spec=None, n_per_cluster=0):
#        print("** init_meaning: {}, {}, {}".format(n, cluster_spec, n_per_cluster))
        if cluster_spec:
            for ci, spec in enumerate(cluster_spec):
                for i in range(n_per_cluster):
                    self.make_meaning(spec=spec, cluster=ci)
        nmeanings = len(self.meanings)
        # Make additional meanings not belonging to clusters
        for i in range(nmeanings, n):
            self.make_meaning()

    def max_nmeanings(self):
        return np.power(self.mvalues, self.mlength)

    def value2act(self, value):
        """
        Convert a 'value' (int between 0 and self.mvalues-1) to an activation
        between 0 and 1.
        """
        return value * (1.0 / (self.mvalues-1.0))

    def make_meaning(self, spec=None, cluster=-1):
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
                if (cluster < 0 and distance(meaning, m) < 0.5) or (cluster >= 0 and (meaning==m).all()):
#                if (meaning == m).all():
                    found = False
                    break
            if not found:
                continue
            meaning = Meaning(meaning, nvalues=self.mvalues, cluster=cluster)
            self.meanings.append(meaning)
            if cluster >=0:
                if cluster not in self.clusters:
                    self.clusters[cluster] = []
                self.clusters[cluster].append(meaning)
            return meaning

    # def make_meaning_cluster(self, mdims, n):
    #     """
    #     Create n meanings with value 1.0 on all dimensions in mdims,
    #     random values in other dimensions.
    #     """
    #     spec = dict([(d, 1.0) for d in mdims])
    #     for i in range(n):
    #         self.make_meaning(spec=spec)

    def make_meaning_clusters(self, fixed_frac, n):
        """
        Create n meaning clusters with fixed_frac of dimensions fixed
        for each cluster.
        """
        n_fixed_dims = round(fixed_frac * self.mlength)
        fixed_dims = range(0, n_fixed_dims)
        clusters = []
        # First create cluster with constant values
        if self.mvalues > n:
            # There are more meaning feature values than clusters, so
            # distribute the values evenly from 0 to 1
            frac = 1.0 / (n-1)
            values = [round((i+1) * frac * self.mvalues) for i in range(n-2)]
            d = {}
            for dim in fixed_dims:
                d[dim] = 0.0
            clusters.append(d)
            for v in values:
                d = {}
                value = self.value2act(v)
                for dim in fixed_dims:
                    d[dim] = value
                clusters.append(d)
            d = {}
            for dim in fixed_dims:
                d[dim] = 1.0
            clusters.append(d)
        else:
            for value in range(self.mvalues):
                d = {}
                for dim in fixed_dims:
                    d[dim] = self.value2act(value)
                clusters.append(d)

        if len(clusters) >= n:
            return clusters[:n]
#        l = len(clusters)
        # Create clusters with different values on each dimension
        perms = list(permutations(range(self.mvalues)))
        random.shuffle(perms)
        for p in perms:
            p = list(p)
            if len(fixed_dims) < self.mvalues:
                p = p + p
            d = {}
            for val, dim in zip(p, fixed_dims):
                d[dim] = self.value2act(val)
            clusters.append(d)
            if len(clusters) == n:
                return clusters
#        for i in range(l, n):
#            values1 = list(range(self.mvalues))
#            random.shuffle(values1)
#            if len(fixed_dims) < self.mvalues:
#                values2 = list(range(self.mvalues))
#                random.shuffle(values2)
#                values1 = values1 + values2
#            d = {}
#            for val, dim in zip(values1, fixed_dims):
#                d[dim] = self.value2act(val)
#            clusters.append(d)
#        return clusters

    def write_meanings(self, filename):
        """
        Write the environment's meanings to a file.
        """
        path = os.path.join('data', filename)
        with open(path, 'w') as out:
            for meaning in self.meanings:
                print(meaning, file=out)

    def read_meanings(self, filename):
        path = os.path.join('data', filename)
        with open(path) as infile:
            for line in infile:
                meaning = Meaning.read(line.strip(), self.mvalues)
                cluster = meaning.cluster
                if cluster >= 0:
                    if cluster not in self.clusters:
                        self.clusters[cluster] = []
                    self.clusters[cluster].append(meaning)
                self.meanings.append(meaning)

class Meaning(np.ndarray):
     """
     An array representing a category in the environment,
     each element one dimension with nvalues possible values.
     """

     pre = "M"

     def __new__(cls, input_array, nvalues=3, cluster=-1, spec=None):
         a = np.asarray(input_array).view(cls)
         a.nvalues = nvalues
         a.cluster = cluster
         return a

     def __array_finalize__(self, obj):
         if obj is None: return
         self.nvalues = getattr(obj, 'nvalues', None)
         self.cluster = getattr(obj, 'cluster', None)

     def __repr__(self):
         if not self.shape:
             # The meaning somehow ended up scalar
             return "{}".format(float(self))
         s = Meaning.pre
         if self.cluster >= 0:
             s += "({})".format(self.cluster)
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
         if self.cluster >= 0:
             s += "({})".format(self.cluster)
         for value in self:
#             if value < 0.1:
#                 value = 0.0
#             value = int(mult * value)
             value = round(100 * value)
             s += "{:> 4}".format(value)
         return s

     @staticmethod
     def read(string, mvalues):
         elements = string.split()
         label = elements[0]
         cluster=-1
         if '(' in label:
             # Meaning is in a cluster
             cluster = int(label.split('(')[-1][:-1])
         values = [int(v)/100.0 for v in elements[1:]]
         array = np.array(values)
         return Meaning(array, mvalues, cluster=cluster)
