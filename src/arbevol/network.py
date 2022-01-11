'''
arbevol.
Neural network with some number of layers.
'''

from utils import *

NO_TARGET = -100.0

## RANDOM WEIGHT AND ACTIVATION GENERATION

def random_act(min_act, max_act):
    '''
    Generate a random activation between min_act and max_act.
    '''
    return min_act + np.random.random() * (max_act - min_act)

def random_weight(wt_range, neg=True):
    '''
    Generate a random weight in range wt_range. If neg is True, value may
    be negative.
    '''
    val = np.random.random() * wt_range
    if neg:
        val -= wt_range / 2
    return val

class Network:

    LR = 0.1

    def __init__(self, name, layers, array=True, supervised=True, verbose=0):
        '''
        Initialize, creating input, hidden, and output Layers and weights.
        '''
        self.name = name
        # A list of layers in input - hidden - output order
        self.layers = layers
        # By default this is a supervised pattern associator
        self.supervised = supervised
        self.has_bias = True
        self.competitive = False
        self.settling = False
        self.target = []
        self.array = array
        self.verbose = verbose
        self.initialize()

    def __repr__(self):
        return "<<" + self.name + ">>"

    def initialize(self):
        '''
        Initialize the Layers, including weights to non-input Layers.
        '''
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            last_layer = self.layers[i-1]
            # Default connectivity: each layer feeds into the next
            layer.connect(last_layer)
        for l in self.layers:
            l.initialize()

    def reinit(self):
        '''
        Reinitialize; only initialize the weights into layers other than input.
        '''
        for l in self.layers[1:]:
            l.reinit()

    def get_n_weights(self):
        '''
        Number of weights and biases in the network.
        '''
        return sum([l.get_n_weights() for l in self.layers])

    def assign_weights(self, weights):
        '''
        Assign a list of weights (as long as it lasts) to the connections in
        the network.
        '''
        index = 0
        for l in self.layers[1:]:
            if l.initialized:
                for u in range(l.size):
                    for w in range(len(l.weights[u])):
                        l.weights[u][w] = weights[index]
                        index += 1

    def propagate_forward(self, verbose=0):
        '''Propagate activation forward through the network.'''
        for l in self.layers[1:]:
            l.update(verbose=verbose)

    def propagate_backward(self, verbose=0):
        '''Propagate error backward through the network.'''
        for l in reversed(self.layers[1:-1]):
            if l.recurrent:
                l.delayed_deltas.clear()
            l.update_error(delay=0, verbose=verbose)

    def update_weights(self, lr=None, verbose=0):
        '''Update the weights into each layer other than the first.'''
        for l in reversed(self.layers[1:]):
            l.update_weights(lr=lr, verbose=verbose)

    def step(self, pattern, train=True, show_act=False, lr=None, seqfirst=False, verbose=0):
        '''
        Run the network on one pattern, returning the error and []
        (winner for CompetNetwork).
        '''
        error = 0.0
        self.pre_stuff(self.target if self.supervised else [], seqfirst=seqfirst)
        # Clamp the input Layer to the list within the input part of the pattern.
        self.layers[0].clamp(pattern[0] if self.supervised else pattern)
        # Update the other layers in sequence
        self.propagate_forward(verbose=verbose)
        # Figure the error into each output unit, given a target sublist
        if self.supervised and pattern[1] is not None:
            # If this is the recurrent layer, also set recurrent deltas
            error += self.layers[-1].do_errors(pattern[1])
            self.target = pattern[1]
        if train:
            # If we're training, propagate error back
            self.propagate_backward(verbose=verbose)
            # Update weights
            self.update_weights(lr=lr, verbose=verbose)
        if show_act:
            self.show()
        self.post_stuff()
        return error, []

    def pre_stuff(self, target, seqfirst=False):
        pass

    def post_stuff(self, target=None):
        pass

    def get_output(self):
        """
        Current activation of the output layer.
        """
        return self.layers[-1].activations

    def show(self, hidden=True):
        '''Print input pattern and output activation, possibly also hidden.'''
        self.layers[0].show()
        if hidden:
            for layer in self.layers[1:-1]:
                layer.show()
        self.layers[-1].show()

    def show_weights(self):
        '''Print the weights and biases in the newtork.'''
        for l in self.layers[1:]:
            l.show_weights()

class Layer:
    '''A group of units and the weights into them. May be a recurrent layer.'''

    # Parameter for the sigmoid activation function
    threshold = 0.0

    # Parameters for ADAM
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.0000001

    momentum = 0.9

    # Leaky RELU slope
    leaky_slope = 0.01

    def __init__(self, name,
                 size=10,             # number of units
                 weight_range=.16,    # range of initial weights
                 has_bias=True,       # is there a bias into each unit?
                 neg_weights=True,    # can initial weights be negative?
                 spec_weights = None, # initial weights to use instead of random ones
                 gain=1.0,            # gain for the activation function (if non-linear)
                 lr=None,             # learning rate
                 momentum=0,          # momentum if is to be used
                 grad_clip=2.0,       # clip gradient norm here
                 error_func='xent',   # error function: quadratic or cross-entropy
                 linear=True,        # is the activation function linear?
                 rectified=True,      # is the activation rectified linear (if linear)?
                 leaky=True,          # is the activation rectified leaky linear?
                 bipolar=True,        # are activations negative and positive?
                 array=True):         # whether to create arrays instead of lists
        '''Initialize variables, but not the list of activations or weights.'''
        self.size = size
        # These are only needed for layers with recurrent units
        self.real_size = self.size
        self.recurrent_indices = []
        self.recur_layer = None
        # default is not recurrent
        self.recurrent = False
        self.rectype = None
        # Layer feeding this layer
        self.input_layer = []
        # Layer fed from this layer
        self.output_layer = []
        self.spec_weights = spec_weights
        self.weights = []
        self.gradients = []
        self.grad_clip = grad_clip
        # Whether to make a bias unit
        self.has_bias = has_bias
        self.errors = []
        self.deltas = []
        self.gain = gain
        self.lr = lr or 0.1
        # Only needed for leaky RELU
        self.act_arg = None
        self.array = array
        # Activation function
        if linear:
            if rectified:
                if leaky:
                    self.act_function = leaky_relu
                    self.act_slope = leaky_relu_slope
                    self.act_arg = Layer.leaky_slope
                else:
                    self.act_function = relu
                    self.act_slope = relu_slope
            else:
                self.act_function = linear
                self.act_slope = linear_slope
        elif bipolar:
            self.act_function = tanh
            self.act_slope = tanh_slope
        else:
            self.act_function = sigmoid
            self.act_slope = sigmoid_slope
        if self.act_function == sigmoid and error_func != 'quad':
            # Cross entropy is the default for sigmoid
            self.error_function = cross_entropy
            self.error_deriv = crossent_slope
        else:
            self.error_function = quadratic
            self.error_deriv = quadratic_slope
        self.name = name
        self.min_activation = -1.0 if bipolar else 0.0
        self.max_activation = 1.0
        self.weight_range = weight_range
        # Whether to create initial negative weights; don't for relu actfunc
        self.neg_weights = False if self.act_function in (relu, leaky_relu) else True
        # Needed for momentum
        self.momentum = momentum
        self.last_wt_updates = None
        # Past activations for recurrent network
        self.delayed_activations = []
        self.index = 0
        # whether weights and errors have been created
        self.initialized = False

    def __repr__(self):
        return "<" + self.name + ">"

    def connect(self, input_layer):
        '''
        Make input_layer the input layer for this layer, and this an
        output_layer for input_layer.
        '''
        self.input_layer = input_layer
        input_layer.output_layer = self

    def initialize(self):
        '''Create the activations and weights lists.'''
        self.set_activations()
        if self.input_layer:
            # This has an input_layer, so it has errors
            self.initialize_weights()
            self.set_errors()
        self.initialized = True

    def set_activations(self):
         midpoint = (self.max_activation + self.min_activation) / 2.0
         acts = [midpoint for u in range(self.size)]
         if self.array:
             acts = l2a(acts)
         self.activations = acts

    def set_errors(self):
        '''Set the errors for the units.'''
        e = [0.0 for u in range(self.size)]
        d = [0.0 for u in range(self.size)]
        if self.array:
            e = l2a(e)
            d = l2a(d)
        self.errors = e
        self.deltas = d

    def initialize_weights(self):
        '''
        Create the weights list of lists and a bias for each unit
        (if self.has_bias is True).

        Indices: [dest-index][input-unit-index]
        '''
        if self.spec_weights:
            # Not checking whether the number of weights is right
            self.weights = self.spec_weights
        elif self.array:
            self.weights = self.make_weight_array(True)
        else:
            w = \
            [ [self.genweight() for i in range(self.input_layer.size)] \
                             # Bias weight
              + ([self.genweight()] if self.has_bias else []) \
              for u in range(self.size) ]
#            if self.array:
#                w = l2a(w)
            self.weights = w
        if self.momentum:
            if self.array:
                self.last_wt_updates = self.make_weight_array(False)
            else:
                u = \
                [ [0.0 for i in range(self.input_layer.size)] + ([0.0] if self.has_bias else []) \
                                        for u in range(self.size) ]
#                if self.array:
#                    u = l2a(u)
                self.last_wt_updates = u
        # gradients
        if self.array:
            self.gradients = self.make_weight_array(False)
        else:
            g = [ [0.0 for i in range(self.input_layer.size)] + ([0.0] if self.has_bias else []) for u in range(self.size) ]
#            if self.array:
#                g = l2a(g)
            self.gradients = g

    def genweight(self, random=True):
        '''Generate a random weight using parameters for this layer.'''
        if not random:
            return 0.0
        return random_weight(self.weight_range, self.neg_weights)

    def genweightA(self, random=True):
        return np.vectorize(self.genweight)(random=random)

    def make_weight_array(self, random=True):
        '''
        Create an array to used for weights, last weight updates, and gradients.
        '''
        d1 = self.input_layer.size
        if self.has_bias:
            d1 += 1
        d2 = self.size
        return np.array([ [self.genweightA(random=random) for i in range(d1)] for u in range(d2) ])

    def add_recurrent_units(self, recur_layer):
        '''Add recurrent units to the layer, representing copies of the activations on the previous time step
        of units in a higher layer.'''
        self.size += recur_layer.size
        self.recurrent_indices = range(self.real_size, self.size)
        self.recur_layer = recur_layer

    def is_recurrent(self, index):
        '''Is the unit with index index a recurrent unit?'''
        return self.recur_layer and index in self.recurrent_indices

    def get_recur_source(self, index):
        '''Get the index of the unit in the recurrent layer that is copied to this index.'''
        if self.is_recurrent(index):
            return index - self.real_size
        return -1

    def get_n_weights(self):
        '''Number of weights and biases into the layer.'''
        if self.weights:
            return sum([len(self.weights[u]) for u in range(self.size)])
        else:
            return 0

    def reinit(self):
        '''Reinitialize the Layer by assigning new weights and biases.'''
        for u in range(self.size):
            for j in range(len(self.weights[u])):
                self.weights[u][j] = random_weight(self.weight_range, self.neg_weights)

    def gen_random_acts(self):
        '''Generate random activations.'''
        a = [random_act(self.min_activation, self.max_activation) for u in range(self.size)]
        if self.array:
            a = l2a(a)
        return a

    def get_weights(self, dest_i, forward=True, limit=0):
        '''Get weights into (if forward) or out of unit dest_i.'''
        weights = self.weights[dest_i] if forward else self.get_reverse_weights(dest_i)
        if limit:
            # Only weights from limit inputs (or outputs if forward=False)
            weights = weights[:limit]
        return weights

    def get_reverse_weights(self, dest_i):
        '''Weights out of dest_i into output_layer.
        LATER USE transpose().
        '''
        w = [out[dest_i] for out in self.output_layer.weights] if self.output_layer else []
        if self.array:
            w = l2a(w)
        return w

    def clamp(self, v):
        '''Clamp pattern vector v on this Layer.'''
        # %% OR SHOULD THIS BE A COPY
        self.activations = v
#        for i in range(min([self.size, len(v)])):
#            self.activations[i] = v[i]

    def get_unit_output_error(self, targ, act):
        """
        Error for a single output unit.
        """
        if targ == NO_TARGET:
            return 0.0
        else:
            return targ - act

    def get_unit_output_error_array(self, targ, act):
        return np.vectorize(self.get_unit_output_error)(targ, act)

    def do_errors(self, target):
        '''
        Figure the errors for each (output) unit, given the target pattern
        (list), returning RMS error.
        '''
        if self.array:
            self.errors = self.get_unit_output_error_array(target, self.activations)
            if self.error_function == cross_entropy:
                np.copyto(self.deltas, self.errors)
            else:
                self.deltas = self.errors * self.act_slope(self.activations, var=self.act_arg)
            return math.sqrt(np.sum(self.errors * self.errors) / self.size)

        error = 0.0
        for i in range(self.size):
            targ = target[i]
            act = self.activations[i]
            e = 0.0 if (targ == NO_TARGET) else (targ - act)
            self.errors[i] = e
            if self.error_function == cross_entropy:
                # Assumes the sigmoid activation function
                self.deltas[i] = e
            else:
#            error_deriv = self.error_deriv(act, targ)
                self.deltas[i] = e * self.act_slope(act, var=self.act_arg)
            error += e * e
        return math.sqrt(error / self.size)

    def gradient_norm(self):
        if self.array:
            sumofsquares = np.sum(np.square(self.gradients))
            norm = math.sqrt(sumofsquares)
            return norm
        sumofsqrs = 0.0
        for u in range(len(self.gradients)):
            # for each weight vector calculate the dot product with itself
            grads = self.gradients[u]
            sumofsqrs += dot_product(grads, grads)
        norm = math.sqrt(sumofsqrs)
        print("** old norm {}".format(norm))
        return norm

    def clip_gradients(self, norm):
        scale = self.grad_clip / norm
#        print("Clip scale: {}".format(scale))
        for u in range(len(self.gradients)):
            for g in range(len(self.gradients[u])):
                self.gradients[u][g] *= scale

    def get_recurrent_weights_in(self, unit):
        indices = self.input_layer.recurrent_indices
#        print("indices: {}".format(indices))
        return self.weights[unit][indices[0]:indices[-1]+1]

    def calc_gradients(self, delay=0, verbose=0):
#        print("{} calculating gradients with delay {}".format(self, delay))
        deltas = self.get_deltas(delay)
        if self.array:
            if delay > 0:
                input_acts = self.delayed_activations[delay-1]
                self.gradients = np.outer(deltas, input_acts)
            else:
                input_acts = self.input_vector()
                self.gradients = np.outer(deltas, input_acts)
        else:
            for u in range(self.size):
                delta = deltas[u]
                if verbose:
                    print("Delta for {}|{}: {}".format(self, u, delta))
                if delay > 0:
                    for i in range(self.size):
                        src_act = self.get_activation(i, delay=delay+1)
                        gradient = src_act * delta
                        self.gradients[u][i] = gradient
                else:
                     for i in range(self.input_layer.size):
                        src_act = self.input_layer.activations[i]
                        gradient = src_act * delta
                        self.gradients[u][i] = gradient
                     # Bias
                     self.gradients[u][self.input_layer.size] = delta
        return self.gradient_norm()

    def update_weights(self, delay=0, lr=None, verbose=0):
        '''Update the weights, first calculating the gradient norm.'''
        grad_norm = self.calc_gradients(delay=delay)
        # Clip gradients if this is not an output layer
#        if self.output_layer:
#            if grad_norm > self.grad_clip:
#                self.clip_gradients(grad_norm)
        lr = lr or self.lr
#        if delay > 0:
#            print("Updating recurrent weights, delay: {}".format(delay))
        if self.array:
            # %% FIGURE OUT HOW TO DO DELAYS
            wt_incrs = lr * self.gradients
            if self.momentum:
                wt_incrs *= self.last_wt_updates
                self.last_wt_updates = np.copy(wt_incrs)
            self.weights += wt_incrs
#        print("** new weight incrs\n{}".format(wt_incrs_new))
        else:
            for u in range(self.size):
                if delay > 0:
                    # Update recurrent weights only
                    recurrent_indices = self.input_layer.recurrent_indices
                    weight_offset = recurrent_indices[0]
                    for i in range(len(recurrent_indices)):
                        weight_index = i + weight_offset
                        gradient = self.gradients[u][i]
                        incr = gradient * lr
                        self.weights[u][weight_index] += incr
                    continue
                for i in range(self.input_layer.size):
                    gradient = self.gradients[u][i]
                    incr = gradient * lr
                    if self.momentum:
                        last_upd = self.last_wt_updates[u][i]
                        incr += self.momentum * last_upd
                    if verbose:
                        print("  Weight increment for {},{}: {}".format(u, i, incr))
                    wt_incrs[u][i] = incr
                    self.weights[u][i] += incr
                    if self.momentum:
                        self.last_wt_updates[u][i] = incr
                # Bias weight
                gradient = self.gradients[u][self.input_layer.size]
                incr = gradient * lr
                if self.momentum:
                    last_upd = self.last_wt_updates[u][self.input_layer.size]
                    incr += self.momentum * last_upd
                if verbose:
                    print(" Bias increment for {}: {}".format(u, incr))
                wt_incrs[u][self.input_layer.size] = incr
                self.incr_bias(u, incr)
        if self.recurrent and self.rectype == 'elman' and delay < 3 and len(self.delayed_activations) > delay+1:
            self.update_weights(delay=delay+1, lr=lr, verbose=verbose)

    def get_input_deltas(self, delay=0):
        if not delay:
            return self.output_layer.deltas
        else:
            return self.delayed_deltas[delay-1]

    def get_deltas(self, delay=0):
        if not delay:
            return self.deltas
        else:
            return self.delayed_deltas[delay-1]

#    def get_act_array(self, delay=0, bias=False):
#        if delay == 0:
#            act = self.activations
#        else:
#            act = self.delayed_activations[delay-1]
#        if bias:
#            act = np.concat

    def get_activations(self, delay=0):
        if not delay:
            return self.activations
        else:
            return self.delayed_activations[delay-1]

    def get_activation(self, unit, delay=0):
        activations = self.get_activations(delay=delay)
        return activations[unit]

    def input_vector(self, verbose=0):
        '''Make the input vector: activations + 1.0 for bias unit.'''
        ia = self.input_layer.activations
        if self.array:
            return np.concatenate((ia, [1.0]))
        else:
            return ia + [1.0]

    def get_input(self, verbose=0):
        """
        Calculate the vector of inputs to this layer.
        """
        activations = self.input_vector(verbose=verbose)
        if verbose > 1:
            print("   Activations {}".format(activations))
        return activations.dot(self.weights.transpose())

    def get_unit_input(self, dest_i, verbose=0):
        '''Get input into unit dest_i, including [1.0] for bias.'''
        activations = self.input_vector(verbose=verbose)
#        self.input_layer.activations + [1.0]
        weights = self.weights[dest_i]
        if verbose > 1:
            print("   Activations: {}".format(activations))
            print("   Weights: {}".format(weights))
        if self.array:
            return activations.dot(weights)
        return dot_product(activations, weights)

    def get_unit_error_input(self, dest_i, delay=0):
        '''Get the error input from the next layer into unit dest_i.'''
        deltas = self.get_input_deltas(delay=delay)
        if self.array:
            return deltas.dot(self.get_reverse_weights(dest_i))
        return dot_product(deltas, self.get_reverse_weights(dest_i))

    def update_unit_error(self, unit, delay=0, new_deltas=None, verbose=0):
#        if self.recurrent and delay > 0:
#            print("Updating error for recurrent unit {} with new_deltas {}".format(unit, new_deltas))
        # This is constant for all time steps, so only set it at delay=0
        error_in = self.get_unit_error_input(unit, delay=delay)
        act = self.get_activation(unit, delay=delay)
#        print(" Layer {} updating error for {}; delay {}; slope act {}".format(self, unit, delay, act))
        self.errors[unit] = error_in
        new_deltas[unit] = error_in * self.act_slope(act, var=self.act_arg)

    def get_back_weights(self, drop_bias=True):
        '''
        Weights connecting this layer to the next layer, needed during backward
        propagation.
        '''
        if self.output_layer:
            weights = self.output_layer.weights
            if drop_bias:
                weights = weights[::,:-1]
            return weights
        return []

    def update_error(self, delay=0, verbose=0):
        '''Update unit errors.'''
#        print("** {} updating errors with delay {}".format(self, delay))
        if self.array:
            deltas = self.get_input_deltas(delay=delay)
            # leave off bias weights
            weights = self.get_back_weights(drop_bias=True)
            # leave off last element (bias error not needed for backprop)
            self.errors = deltas.dot(weights) #[:-1]
            new_deltas = self.errors * self.act_slope(self.activations, var=self.act_arg)
            if delay == 0:
                self.deltas = new_deltas
        else:
            new_deltas = [0.0 for x in range(self.size)] if delay > 0 else self.deltas
            for unit in range(self.size):
                self.update_unit_error(unit, delay=delay, new_deltas=new_deltas, verbose=verbose)
        if self.recurrent:
            deltas = new_deltas[:]
#            print("Saving recurrent deltas")
            self.delayed_deltas.append(deltas)
            if delay < 3 and len(self.delayed_activations) > delay+1:
                self.update_error(delay=delay+1, verbose=verbose)

    def update(self, verbose=0):
        '''Update unit activations.'''
        if self.array:
            inp = self.get_input(verbose=verbose)
            if verbose:
                print(" Input: {}".format(inp))
            activations = self.act_function(inp, self.threshold, self.gain, self.act_arg)
            self.activations = activations
            if verbose:
                print(" New activations: {}".format(self.activations))
        else:
            for index in range(self.size):
                if verbose:
                    print("Updating {}|{}".format(self, index))
                inp = self.get_unit_input(index, verbose=verbose)
                if verbose:
                    print(" Input: {}".format(inp))
                # Apply activation function
                activation = self.act_function(inp, self.threshold, self.gain, self.act_arg)
                self.activations[index] = activation
                if verbose:
                    print(" New activation: {}".format(self.activations[index]))

    def get_bias(self, unit):
        '''The bias into unit.'''
        return self.weights[unit][self.input_layer.size]

    def set_bias(self, unit, bias):
        '''Set the bias into unit.'''
        self.weights[unit][self.input_layer.size] = bias

    def incr_bias(self, unit, incr):
        '''Increment the bias into unit.'''
        bias_index = self.input_layer.size
        self.weights[unit][bias_index] += incr
        if self.momentum:
            self.last_wt_updates[unit][bias_index] = incr

    def show(self):
        '''Print activations.'''
        print(self.name.ljust(12), end=' ')
        for a in self.activations:
            print("{: .2f}".format(a), end=' ')
#            print('%.3f' % a, end=' ')
        print()

    def show_weights(self):
        '''Print the weights and biases in the layer.'''
        print(self.name)
        for u in range(self.size):
            print(str(u).ljust(5), end=' ')
            for w in range(len(self.weights[u])):
                print("{: .2f}".format(self.weights[u][w]), end=' ')
#                        %.3f' % self.weights[u][w], end=' ')
            print()

    def adam_update(self, gradient):
        '''Update the m and v values for all weights.'''
        pass
