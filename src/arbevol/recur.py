'''
arbevol.
Recurrent networks.
'''

from network import *
from collections import deque

MU = .5
TDMAX = 3

class RecurrentNetwork(Network):

    def __init__(self, name, layers,
                 # Source of the recurrent input layer, either 'o(utput)' or 'h(idden)'
                 source = 'h',
                 mu = MU,
                 reinit_seq = True, supervised=True,
                 teacher_forcing = False):
        self.source = 'h' if source[0] == 'h' else 'o'
        self.mu = mu
        self.teacher_forcing = teacher_forcing
        self.reinit_seq = reinit_seq
        Network.__init__(self, name, layers, supervised=supervised)

    def initialize(self):
        '''Increase input Layer size and assign weights.'''
        # The source of the recurrent connections
        self.recurrent_layer = self.layers[-1] if self.source == 'o' else self.layers[1]
        # Add this many units to the network input layer
        self.layers[0].add_recurrent_units(self.recurrent_layer)
        # Make the hidden layer recurrent (this won't be the recurrent source if it's a jordan net).
        self.recurrent_target = self.layers[1]
        self.recurrent_target.recurrent = True
        if self.source == 'o':
            self.recurrent_target.rectype = 'jordan'
        else:
            self.recurrent_target.rectype = 'elman'
        self.recurrent_target.delayed_activations = deque([], TDMAX)
        self.recurrent_target.delayed_deltas = deque([], TDMAX)
        Network.initialize(self)

    def pre_stuff(self, target = [], seqfirst=False):
        '''Copy the recurrent activations to the end of the input layer.'''
        if self.source == 'o':
            recur_source = target if self.teacher_forcing else self.recurrent_layer.activations
            for u in range(self.recurrent_layer.size):
                if self.reinit_seq and seqfirst:
                    self.layers[0].activations[-(u + 1)] = 0.0
                else:
                    # Decay the context unit
                    self.layers[0].activations[-(u + 1)] *= self.mu
                    self.layers[0].activations[-(u + 1)] += (1 - self.mu) * recur_source[-(u + 1)]
        else:
            for u in range(self.recurrent_layer.size):
                if self.reinit_seq and seqfirst:
                    self.layers[0].activations[-(u + 1)] = 0.0
                else:
                    self.layers[0].activations[-(u + 1)] = self.recurrent_layer.activations[-(u + 1)]

    def post_stuff(self):
        '''Copy the time-delayed activations.'''
        rl = self.recurrent_target
        # Copy of the activations
        activations = rl.activations[:]
        # Add the activations to the right end of the queue
        rl.delayed_activations.appendleft(activations)

class GRULayer(Layer):
    '''A Layer of gated recurrent units, each consisting of three real units and weights into
    them. Three actual Layers are created.'''

    def __init__(self, name, size=10, weight_range=.5, gain=1.0,
                 lr=None, bipolar=True, unittype=0, momentum=0.9,
                 spec_weights=None, update_spec_weights=None, reset_spec_weights=None):
        # Create layers for the update and reset gates, but don't add them to the network list of layers
        self.update_layer = Layer(name + '_upd', size=size, rectified=False, bipolar=False, momentum=momentum,
                                  spec_weights=update_spec_weights)
        self.reset_layer = Layer(name + '_res', size=size, rectified=False, bipolar=False, momentum=momentum,
                                 spec_weights=reset_spec_weights)
        # Set during initialization
        Layer.__init__(self, name, size=size,
                       weight_range=weight_range, has_bias=True, neg_weights=True, spec_weights=spec_weights,
                       gain=gain, lr=lr, bipolar=bipolar, linear=False, momentum=momentum)

    def set_activations(self):
        # Output activations for this layer
        Layer.set_activations(self)
        self.candidate_activations = self.gen_random_acts()
        self.update_layer.activations = [1.0 for u in range(self.size)]
        self.reset_layer.activations = [1.0 for u in range(self.size)]

    def connect(self, input_layer):
        '''Override Layer.connect() so that reset and update layers also connect to input_layer.'''
        Layer.connect(self, input_layer)
        input_size = input_layer.real_size
        self.input_indices = range(input_size)
        self.recurrent_indices = range(input_size, input_size + self.size)
        self.bias_index = input_size + self.size
        self.update_layer.input_layer = input_layer
        self.reset_layer.input_layer = input_layer

    def set_errors(self):
        # This create the usual error (delta) terms, which are actually those for the candidate units.
        Layer.set_errors(self)
        self.reset_layer.errors = [0.0 for u in range(self.size)]
        self.reset_layer.deltas = [0.0 for u in range(self.size)]
        self.update_layer.errors = [0.0 for u in range(self.size)]
        self.update_layer.deltas = [0.0 for u in range(self.size)]

    def initialize_weights(self):
        '''Override Layer.initialize_weights to also initialize the weights in the reset and update layers.'''
        Layer.initialize_weights(self)
        self.update_layer.initialize_weights()
        self.reset_layer.initialize_weights()

    def reinit(self):
        '''Need to reinitialize all the weights.'''
        Layer.reinit(self)
        weights = self.update_layer.weights
        wt_range = self.update_layer.weight_range
        for u in range(self.size):
            for j in range(len(weights[u])):
                weights[u][j] = random_weight(wt_range, True)
        weights = self.reset_layer.weights
        wt_range = self.reset_layer.weight_range
        for u in range(self.size):
            for j in range(len(weights[u])):
                weights[u][j] = random_weight(wt_range, True)

    def get_bias(self, unit):
        return self.weights[unit][self.bias_index]

    def get_input_weights(self, unit):
        weights = self.weights[unit]
        return [weights[i] for i in self.input_indices]

    def get_input_activations(self):
        activations = self.input_layer.activations
        return [activations[i] for i in self.input_indices]

    def get_recurrent_weights(self, unit):
        weights = self.weights[unit]
        return [weights[i] for i in self.recurrent_indices]

    def get_recurrent_activations(self):
        activations = self.input_layer.activations
        return [activations[i] for i in self.recurrent_indices]

    def get_recurrent_activation(self, unit):
        '''The activation of this unit on the previous time step.'''
        return self.get_recurrent_activations()[unit]

    def get_reset_activation(self, unit):
        '''The activation of the reset unit for unit.'''
        return self.reset_layer.activations[unit]

    def get_update_activation(self, unit):
        '''The activation of the update unit for unit.'''
        return self.update_layer.activations[unit]

    def update(self, verbose=0):
        # Activate all reset and update units
        self.reset_layer.update()
        self.update_layer.update()
        # Activate the candidate units
        for u in range(self.size):
            self.update_unit(u, verbose=verbose)

    def update_unit(self, unit, verbose=0):
        '''Update the activation of the candidate unit and then the unit itself.'''
        if verbose:
            print("Updating GRU {}".format(unit))
        recurrents = self.get_recurrent_activations()
        inputs = self.get_input_activations()
        recurrent = recurrents[unit]
        input_input = dot_product(inputs + [1.0],
                                  self.get_input_weights(unit) + [self.get_bias(unit)])
        if verbose:
            print(" Input input: {}".format(input_input))
        reset_input_vec = pairwise_product(recurrents,
                                           self.reset_layer.activations)
        recurrent_input = dot_product(recurrents,
#            reset_input_vec,
                                      self.get_recurrent_weights(unit))
        if verbose:
            print(" Recurrent input: {}".format(recurrent_input))
        cand_act = self.act_function(input_input + recurrent_input, self.threshold, self.gain)
        self.candidate_activations[unit] = cand_act
        if verbose:
            print(" Candidate activation: {}".format(cand_act))
        updu_act = self.update_layer.activations[unit]
        self.activations[unit] = (1.0 - updu_act) * recurrent + updu_act * cand_act
        if verbose:
            print(" New activation: {}".format(self.activations[unit]))

    def update_error(self, delay=0, verbose=1):
        '''Update unit errors.'''
#        print("{} updating errors with delay {}".format(self, delay))
        if delay > 0:
            new_deltas = [0.0 for x in range(self.size)]
            new_update_deltas = [0.0 for x in range(self.size)]
            new_reset_deltas = [0.0 for x in range(self.size)]
        else:
            new_deltas = self.deltas
            new_update_deltas = self.update_layer.deltas
            new_reset_deltas = self.reset_layer.deltas
        for unit in range(self.size):
            self.update_unit_error(unit, delay=delay,
                                   new_deltas=(new_deltas, new_update_deltas, new_reset_deltas),
                                   verbose=verbose)
        if self.recurrent:
            deltas = new_deltas[:]
#            print("Saving recurrent deltas")
            self.delayed_deltas.append(deltas)
            if delay < 3 and len(self.delayed_activations) > delay+1:
                self.update_error(delay=delay+1, verbose=verbose)

    def update_unit_error(self, unit, delay=0, new_deltas=None, verbose=0):
        deltas, update_deltas, reset_deltas = new_deltas
        # weighted error back-propagated from next layer's error terms
        if verbose:
            print("Updating error for GRU {}".format(unit))
        # Current activations of unit components
        cand_act = self.candidate_activations[unit]
        update_act = self.get_update_activation(unit)
        reset_act = self.get_reset_activation(unit)
        recurrent_act = self.get_recurrent_activation(unit)
        # Error terms
        error_in = self.get_error_input(unit, delay=delay)
        if verbose:
            print(" Error in: {}".format(error_in))
        # Calculate the candidate error
        cand_delta = error_in * self.act_slope(cand_act) * update_act
        deltas[unit] = cand_delta
#        self.errors[unit] = cand_error
        if verbose:
            print(" Candidate delta: {}".format(cand_delta))
        # Calculate the error for the update unit
        slope_func = self.update_layer.act_slope
        update_delta = error_in * slope_func(update_act) * (cand_act - recurrent_act)
        update_deltas[unit] = update_delta
#        self.update_layer.errors[unit] = update_error
        if verbose:
            print(" Update delta: {}".format(update_delta))
        # Calculate the error for the reset unit
        slope_func = self.reset_layer.act_slope
        recurrent_weight = self.get_recurrent_weights(unit)[unit]
        reset_delta = cand_delta * slope_func(reset_act) * recurrent_act * recurrent_weight
        reset_deltas[unit] = reset_delta
#        self.reset_layer.errors[unit] = reset_error
        if verbose:
            print(" Reset delta: {}".format(reset_delta))

    def get_update_deltas(self, delay=0):
        if not delay:
            return self.update_layer.deltas
        else:
            return self.update_layer.delayed_deltas[delay-1]

    def get_reset_deltas(self, delay=0):
        if not delay:
            return self.reset_layer.deltas
        else:
            return self.reset_layer.delayed_deltas[delay-1]

    def calc_gradients(self, delay=0, verbose=0):
        deltas = self.get_deltas(delay)
        update_deltas = self.get_update_deltas(delay)
        reset_deltas = self.get_reset_deltas(delay)
        for u in range(self.size):
            cand_delta = deltas[u]
            update_delta = update_deltas[u]
            reset_delta = reset_deltas[u]
#            cand_error = self.errors[u]
#            update_error = self.update_layer.errors[u]
#            reset_error = self.reset_layer.errors[u]
            if delay > 0:
                for i in range(self.size):
                    src_act = self.get_activation(i, delay=delay+1)
                    gradient = src_act * cand_delta
                    self.gradients[u][i] = gradient
                    update_grad = src_act * update_delta
                    self.update_layer.gradients[u][i] = update_grad
                    reset_grad = src_act * reset_delta
                    self.reset_layer.gradients[u][i] = reset_grad
            else:
                for i in range(self.input_layer.size):
                    src_act = self.input_layer.activations[i]
                    cand_grad = src_act * cand_delta
                    update_grad = src_act * update_delta
                    reset_grad = src_act * reset_delta
                    self.gradients[u][i] = cand_grad
                    self.update_layer.gradients[u][i] = update_grad
                    self.reset_layer.gradients[u][i] = reset_grad
                # Bias gradients
                self.gradients[u][self.input_layer.size] = cand_delta
                self.update_layer.gradients[u][self.input_layer.size] = update_delta
                self.reset_layer.gradients[u][self.input_layer.size] = reset_delta
        return self.gradient_norm(), self.update_layer.gradient_norm(), self.reset_layer.gradient_norm()

    def learn(self, delay=0, lr=None, verbose=0):
        '''Update the weights.'''
        gnorm, update_gnorm, reset_gnorm = self.calc_gradients(verbose=verbose)
        if verbose:
            print("Grad norms: {}, update_gnorm {}, reset_gnorm {}".format(gnorm, update_gnorm, reset_gnorm))
        lr = lr or self.lr
        for u in range(self.size):
            if verbose:
                print("Adapting weights into {}|{} with delay {}".format(self, u, delay))
            if delay > 0:
                # Update recurrent weights only
                weight_offset = self.input_layer.recurrent_indices[0]
                for i in range(self.size):
                    weight_index = i + weight_offset
                    cand_grad = self.gradients[u][i]
                    update_grad = self.update_layer.gradients[u][i]
                    reset_grad = self.reset_layer.gradients[u][i]
                    cand_incr = cand_grad * lr
                    update_incr = update_grad * lr
                    reset_incr = reset_grad * lr
#                    print(" Weight increment for {}|{} at delay {}: {}".format(u, weight_index, delay, incr))
                    self.weights[u][weight_index] += cand_incr
                    self.update_layer.weights[u][weight_index] += update_incr
                    self.reset_layer.weights[u][weight_index] += reset_incr
                continue
            # Update the weights into the update unit and reset unit
            for i in range(self.input_layer.size):
                # Weights from input and recurrent units are treated the same for update and reset units
                update_grad = self.update_layer.gradients[u][i]
                update_incr = lr * update_grad
                reset_grad = self.reset_layer.gradients[u][i]
                reset_incr = lr * reset_grad
                cand_grad = self.gradients[u][i]
                cand_incr = lr * cand_grad
                if self.momentum:
                    last_wt = self.last_wt_updates[u][i]
                    cand_incr += self.momentum * last_wt
                    last_update_wt = self.update_layer.last_wt_updates[u][i]
                    update_incr += self.momentum * last_update_wt
                    last_reset_wt = self.reset_layer.last_wt_updates[u][i]
                    reset_incr += self.momentum * last_reset_wt
                if i in self.input_layer.recurrent_indices:
                    # For update weights into candidate from recurrent units, we have to multiply
                    # by the current reset unit activation
                    cand_incr *= self.get_reset_activation(u)
                if verbose:
                    print("  Update weight increment for {},{}: {}".format(u, i, update_incr))
                    print("  Reset weight increment for {},{}: {}".format(u, i, reset_incr))
                    print("  Candidate weight increment for {},{}: {}".format(u, i, cand_incr))
#                print("Updating update weight {}{}: {}".format(u, i, update_incr))
                self.update_layer.weights[u][i] += update_incr
                self.reset_layer.weights[u][i] += reset_incr
                self.weights[u][i] += cand_incr
                if self.momentum:
                    self.last_wt_updates[u][i] = cand_incr
                    self.update_layer.last_wt_updates[u][i] = update_incr
                    self.reset_layer.last_wt_updates[u][i] = reset_incr
            # Bias weight
            update_bias_grad = self.update_layer.gradients[u][self.input_layer.size]
            reset_bias_grad = self.reset_layer.gradients[u][self.input_layer.size]
            cand_bias_grad = self.gradients[u][self.input_layer.size]
            update_bias_incr = lr * update_bias_grad
            reset_bias_incr = lr * reset_bias_grad
            cand_bias_incr = lr * cand_bias_grad
            if verbose:
                print(" Update bias increment for {}: {}".format(u, update_bias_incr))
#                print(" Reset bias increment for {}: {}".format(u, reset_bias_incr))
            self.incr_bias(u, cand_bias_incr)
            self.update_layer.incr_bias(u, update_bias_incr)
            self.reset_layer.incr_bias(u, reset_bias_incr)

    def show(self):
        """Print activations of all three unit types."""
        print(self.name.ljust(12), end=' ')
        for a in self.activations:
            print('%.3f' % a, end=' ')
        print()
        print("  upd".ljust(12), end=' ')
        for a in self.update_layer.activations:
            print('%.3f' % a, end=' ')
        print()
        print("  res".ljust(12), end=' ')
        for a in self.reset_layer.activations:
            print('%.3f' %a, end= ' ')
        print()

# A really simple network

def gru_test():
    gru1 = RecurrentNetwork('gru1', supervised=False,
                            layers=[Layer('in', 3),
                                    GRULayer('hid', 2,
                                             spec_weights=[[2.0, 0.0, -1.0,   2.0, 0.0,   0.0],
                                                           [0.0, 2.0, 0.0,    0.0, 0.0,   -1.0]],
                                             update_spec_weights=[[2.0, 0.0, 2.0,   0.0, 0.0,   0.0],
                                                                  [0.0, 4.0, 0.0,   0.0, 0.0,   0.0]]),
                                    Layer('out', 1,
                                          spec_weights=[[1.0, 1.0, -1.0]])])
    feat = [1.0, 0.0, 0.0]
    wait = [-1.0, 0.0, 0.0]
    remember = [-1.0, 1.0, 0.0]
    reset = [-1.0, 0.0, 1.0]
    gru1.feat = lambda: gru1.step(feat, train=False, show_act=True, verbose=1)
    gru1.wait = lambda: gru1.step(wait, train=False, show_act=True, verbose=1)
    gru1.reset = lambda: gru1.step(reset, train=False, show_act=True, verbose=1)
    gru1.remember = lambda: gru1.step(remember, train=False, show_act=True, verbose=1)

    return gru1
