'''
arbevol.
Visualization of networks.
'''

from tkinter import *
import tkinter.font
import recur

# Dimensions of the window for NN experiments
#WIDTH = 750
#HEIGHT = 600
WIDTH = 500
HEIGHT = 400
#MAX_CELL 50
MAX_CELL = 40
# Number of times the Experiment is stepped when the 'Run' button is clicked.
N_STEPS = 20000
# Number of microseconds between steps during run.
STEP_DELAY = 0

def strength_color(strength, rgb):
    '''Convert a list of ints representing a color and a strength between 0 and 1 to a color.'''
    return "#%02x%02x%02x" % (int(strength * rgb[0]), int(strength * rgb[1]), int(strength * rgb[2]))

def get_node_color(activation):
    '''Get a color for a node with a particular activation: shades of blue for +, red for -.'''
    if activation < 0.0:
        base = (255, 0, 0)
        activation = abs(activation)
    else:
        base = (0, 0, 255)
    if activation > 1.0:
        activation = 1.0
    return strength_color(activation, base)

#def get_weight_color_width(value):
#    if value < 0.0:
#        base = (255, 0, 0)
#        value = abs(value)
#    else:
#        base = (0, 0, 255)

def get_error_color(error):
    '''Get a color for an error value, shades of yellow.'''
    return strength_color(min([error, 1.0]), (255, 255, 0))

## class NNTop(Toplevel):

##     def __init__(self, width=INDIV_WIDTH, height=INDIV_HEIGHT,
##                  critter = None, network = None, title = None, menus=True):
##         Toplevel.__init__(self)
##         self.critter = critter
##         self.frame = NNFrame(self, critter = critter, network = network,
##                              width = width, height = height,
##                              buttons = False, menus = menus,
##                              title = title)
##         self.canvas = self.frame.canvas

##     def destroy(self):
##         '''When we destroy the network window, we need to tell its critter.'''
##         self.critter.net_view = None
##         Toplevel.destroy(self)

##     def paint(self):
##         '''Paint all nodes in the canvas.'''
##         self.canvas.paint_nodes()

class NNFrame(Frame):
    '''A window for the Canvas showing the Network.'''

    def __init__(self, parent, experiment = None, network = None,
                 width = WIDTH, height = HEIGHT,
                 critter = None,
                 buttons = True, menus = False, title = None):
        '''Create the buttons and the Canvas.  Give it (and the Canvas) an Experiment or a Critter.'''
        Frame.__init__(self, parent, width=width, height=height)
        self.grid(row = 0)
        parent.title(title if title else experiment.name)
        self.always_paint = False
#        self.show_input_space = False
        self.canvas = NNCanvas(self, experiment, width = width, height = height,
                               network = network, critter = critter)
        if buttons:
            self.__button0 = Button(self, text = "Reinit")
            self.__button0.grid(row = 2, column = 0)
            self.__button0.bind("<Button-1>", self.canvas.reinit)
            self.__button1 = Button(self, text = "Step")
            self.__button1.grid(row = 2, column = 1)
            self.__button1.bind("<Button-1>", self.canvas.step)
            self.__button2 = Button(self, text = "Run")
            self.__button2.grid(row = 2, column = 2)
            self.__button2.bind("<Button-1>", self.canvas.run)
            self.__button3 = Button(self, text = "Show")
            self.__button3.grid(row = 2, column = 3)
            self.__button3.bind("<Button-1>", self.canvas.show)
#            self.__button3 = Button(self, text = "Next cond")
#            self.__button3.grid(row = 2, column = 3)
#            self.__button3.bind("<Button-1>", self.canvas.next_cond)

        if menus:
            self.viewmenu = Menubutton(self, text = "View")
            self.viewmenu.grid(row = 0, columnspan = 4)
            self.viewmenu.menu = Menu(self.viewmenu)
            self.viewmenu["menu"] = self.viewmenu.menu
#            if self.canvas.is_nodes:
#                self.viewmenu.menu.add_command(label = "Show hidden units in input space",
#                                               command = self.__toggle_show_input_space)

#    def __toggle_show_input_space(self):
#        '''Display hidden nodes in input space (only when there are two dimensions).'''
#        if self.show_input_space:
#            self.viewmenu.menu.entryconfigure(1, label = "Show hidden units in input space")
#            self.show_input_space = False
#            self.canvas.unshow_input_space()
#        else:
#            self.viewmenu.menu.entryconfigure(1, label = "Show hidden units normally")
#            self.show_input_space = True
#            self.canvas.show_input_space()

class NNCanvas(Canvas):
    '''The Canvas where the Network is shown.'''

    node_radius = 1

    def __init__(self, parent, experiment = None, critter = None,
                 network = None, width = WIDTH, height = HEIGHT):
        Canvas.__init__(self, parent, width=width, height=height, bg="black")
        self.__width = width
        self.__height = height
        self.__parent = parent
        self.__experiment = experiment
        if experiment:
            self.__network = experiment.network
            self.__experiment.canvas = self
        else:
            self.__critter = critter
            self.__network = network
        self.__compet = self.__network.competitive
        self.__is_sup = self.__network.supervised
        self.__settling = self.__network.settling
        self.__bias = self.__network.has_bias
        self.grid(row = 1, columnspan = 4)
        self.__label_font = tkinter.font.Font(family='Helvetica', size = 9, weight = 'normal')
        self.__initialize()

    def __initialize(self):
        '''Create all of the objects for the nodes, weights, and targets, plus various lines.'''
        # Width and height of region where units are displayed
        width = 4 * self.__width / 5
        height = 7 * self.__height / 8
        y0 = self.__height / 16
        y1 = self.__height - y0
        # Lists and dict to store the objects
        self.__layers = self.__network.layers
#        self.__dim_layers = len([l for l in self.__layers if l.dimensions != None])
        # Number of layers
        self.__n_layers = len(self.__layers)
        # Number of layer regions
        self.__n_regions = self.__n_layers
#         = 2 * self.__dim_layers + (self.__n_layers - self.__dim_layers)
        # Height of layer region; depends on number of layers (+ target if supervised)
        layer_height = height / (self.__n_regions + (.5 if self.__is_sup else 0))
        max_cell = min(3 * layer_height / 4, MAX_CELL)
        layer_gap = layer_height / 8
        self.__units = []
        self.__unit_coords = []
        self.__update_units = []
        self.__reset_units = []
#        self.__labels = []
        self.__weights = []
        self.__update_weights = []
        self.__reset_weights = []
        self.__targets = []
        self.__weight_dests = {}
        self.__weight_lines = {}
        self.__weight_coords = []
        # Number of units
        self.__n_units = [l.size for l in self.__network.layers]

        # Error object: put in the upper right corner
        cell = self.__width / 30
        gap = cell / 2
        self.error_id = self.__add_graphic(self.__width - cell - gap, cell + gap, cell,
                                           outline='blue', fill='', rect=False)

        # Bias object: put in the lower left corner
        if self.__bias:
            cell = self.__width / 30
            gap = cell / 2
            self.bias_id = self.__add_graphic(cell + gap, self.__height - cell - gap, cell,
                                              outline='', fill='')

        for l, layer in enumerate(self.__layers):
            is_out = (l == self.__n_layers - 1)
            is_in = (l == 0)
            is_gru = isinstance(layer, recur.GRULayer)
            layer_width = layer.size
            x_gap = width / layer_width
            y_gap = 0
            cell = min(x_gap / 4, max_cell / 2)
            x0 = (self.__width - (x_gap * layer_width)) / 2
            x = x0 + x_gap / 2
            y0 = (y1 - cell - layer_gap)
            y = y0
            if not is_out:
                weights_y = y - 2 * cell - layer_gap
            elif self.__is_sup:
                target_y = y - 2 * cell - 2 * layer_gap
            units = []
            unit_coords = []
            update_units = []
            reset_units = []
            # List of weight icons OUT of this layer
            weights = []
            # List of source and destination coords for weights into this layer
            weight_coords = []
            update_weights = []
            reset_weights = []
            targets = []
            def __weights_hider(event, layer_index = l-1):
                return self.__unpaint_weights(event, layer_index)

            self.is_nodes = []

            for n in range(layer.size):
                rect = not layer.is_recurrent(n)
                unit_id = self.__add_graphic(x, y, cell, rect=rect)
                if is_gru:
                    update_id = self.__add_graphic(x + 4 * cell/3, y - 4*cell/3, cell/3, rect=True)
                    update_units.append(update_id)
                    reset_id = self.__add_graphic(x - 4*cell/3, y + 4*cell/3, cell/3, rect=True)
                    reset_units.append(reset_id)
                units.append(unit_id)
                unit_coords.append((x, y))
                if not is_out:
                    weight_id = self.__add_graphic(x, y, cell, outline='', fill='', rect=rect)
                    weights.append(weight_id)
                    if is_gru:
                        update_weight_id = \
                          self.__add_graphic(x + 4*cell/3, y - 4*cell/3, cell/3, outline='', fill='', rect=rect)
                        update_weights.append(update_weight_id)
                        reset_weight_id = \
                          self.__add_graphic(x - 4*cell/3, y + 4*cell/3, cell/3, outline='', fill='', rect=rect)
                        reset_weights.append(update_weight_id)
                elif self.__is_sup:
                    target_id = self.__add_graphic(x, target_y, cell, '', '')
                    targets.append(target_id)
                if not is_in:
                    # Handle displaying weights INTO this unit
                    weight_lines = []
                    for c in self.__unit_coords[-1]:
                        weight_lines.append(self.__add_weight_line(c[0], c[1], x, y))
#                        weight_coords.append(c)
                    self.__weight_lines[unit_id] = weight_lines
                    def __weights_shower(event, unit_id = unit_id):
                        return self.__paint_weights(event, unit_id)
                    def __weight_lines_hider(event, unit_id=unit_id):
                        return self.__unpaint_weight_lines(event, unit_id)
                    # Save this layer, to get the weights, and the index of the source layer
                    self.__weight_dests[unit_id] = (layer, l-1, n)
                    # Object that will get clicked to display the weights into the unit
                    button_unit_id = unit_id if is_out else weight_id
                    self.tag_bind(button_unit_id, "<Button-1>", __weights_shower)
                    self.tag_bind(button_unit_id, "<ButtonRelease-1>",
#                                  __weight_lines_hider)
                                   __weights_hider)
                    if is_gru:
                        def __weights_shower(event, unit_id = unit_id):
                            return self.__paint_weights(event, unit_id, gru=1)
                        button_unit_id = update_unit_id if is_out else update_weight_id
                        self.tag_bind(button_unit_id, "<Button-1>", __weights_shower)
                        self.tag_bind(button_unit_id, "<ButtonRelease-1>", __weights_hider)
                        def __weights_shower(event, unit_id = unit_id):
                            return self.__paint_weights(event, unit_id, gru=2)
                        button_unit_id = reset_unit_id if is_out else reset_weight_id
                        self.tag_bind(button_unit_id, "<Button-1>", __weights_shower)
                        self.tag_bind(button_unit_id, "<ButtonRelease-1>", __weights_hider)
                x += x_gap

            self.__units.append(units)
            self.__unit_coords.append(unit_coords)
            self.__weight_coords.append(weight_coords)
            if is_gru:
                self.__update_units.append(update_units)
                self.__reset_units.append(reset_units)
            if not is_out:
                self.__weights.append(weights)
            elif self.__is_sup:
                self.create_line(x0, y - cell - layer_gap, x0 + x_gap * layer_width, y - cell - layer_gap, fill='white')
                self.__targets = targets
            y1 -= layer_height
#        for w in self.__weight_coords:
#            print(w)

    def next_cond(self, event):
        '''Switch the Experiment to its next condition.'''
        self.__experiment.next_condition()

    def __add_graphic(self, x, y, w, outline = '#333', fill = 'blue', rect = True):
        '''Add an object in the given position with outline and fill color, rectangle or circle.'''
        return \
               (getattr(self, 'create_rectangle') \
                if rect else getattr(self, 'create_oval'))(x - w, y - w, x + w, y + w,
                                                           width = 1, outline = outline, fill = fill)

    def __add_weight_line(self, src_x, src_y, dest_x, dest_y, color='yellow'):
        return self.create_line(src_x, src_y, dest_x, dest_y, fill='')

    def __paint_weight_line(self, id, value, color='yellow'):
        self.itemconfigure(id, fill = get_node_color(value))

    def __unpaint_weight_line(self, id):
        self.itemconfigure(id, fill='')

    def __add_label(self, string, x, y, color = 'blue'):
        '''Add a text object for a unit.'''
        return self.create_text(x, y, text = string, fill = color, font = self.__label_font)

    def paint_nodes(self):
        '''Paint all of the nodes.'''
        grus = 0
        for l, layer in enumerate(self.__layers):
            is_gru = isinstance(layer, recur.GRULayer)
#            if self.__parent.show_input_space and self.is_nodes and \
#               l > 0 and self.is_displayable(layer, self.__layers[l - 1]):
#                self.update_is()
#            else:
            for u in range(layer.size):
                self.__paint_node(self.__units[l][u], layer.activations[u])
                if is_gru:
                    self.__paint_node(self.__update_units[grus][u], layer.update_layer.activations[u])
                    self.__paint_node(self.__reset_units[grus][u], layer.reset_layer.activations[u])
            if is_gru:
                grus += 1

    def __paint_node(self, nodeid, activation, color=None):
        '''Paint a node object.'''
        self.itemconfigure(nodeid, fill = get_node_color(activation))

    def hide_nodes(self):
        pass
#        for l, layer in enumerate(self.__layers[1:]):
#            if layer.dimensions:
#                for u in range(layer.size):
#                    self.__hide_node(self.__units[l + 1][u])

    def __hide_node(self, nodeid):
        self.itemconfigure(nodeid, fill = '', width = 0)

    def unhide_nodes(self):
        pass
#        for l, layer in enumerate(self.__layers[1:]):
#            if layer.dimensions:
#                for u in range(layer.size):
#                    self.__unhide_node(self.__units[l + 1][u], layer.activations[u])

    def __unhide_node(self, nodeid, activation):
        self.itemconfigure(nodeid, fill = get_node_color(activation), width = 1)

    def __paint_weights(self, event, unit_id, gru=0):
        '''Show weights into unit represented by object with unit_id.
        If gru is 0, these are the weights into the unit or candidate unit for
        a GRU unit. If gru is 1, paint the update weights, if 2, paint the
        reset weights.
        '''
        dest_unit = self.__weight_dests[unit_id]
        layer = dest_unit[0]
        src_index = dest_unit[1]
        index = dest_unit[2]
        # Weights into dest unit, ending with bias
        if gru == 1:
            weights = layer.update_layer.get_weights(index)
        elif gru == 2:
            weights = layer.reset_layer.get_weights(index)
        else:
            weights = layer.get_weights(index)
#        weight_lines = self.__weight_lines[unit_id]
#        for l, w in zipe(weight_lines, weights if not self.__bias else weights[:-1])
#            self.__paint_weight_line(l, w)
        for w_i, w in enumerate(weights if not self.__bias else weights[:-1]):
            self.__paint_weight(src_index, w_i, w)
        # Bias
        if self.__bias:
            self.__paint_bias(weights[-1])
        print(weights)

    def __unpaint_weights(self, event, layer_index):
        '''Hide the weights for the source layer.'''
        for i in self.__weights[layer_index]:
            self.__unpaint_weight(i)
        if self.__bias:
            self.__unpaint_bias()

    def __unpaint_weight_lines(self, event, dest_unit):
        for l in self.__weight_lines[dest_unit]:
            self.__unpaint_weight_line(l)

    def __paint_weight(self, layer_index, index, weight):
        '''Paint a weight object.'''
        self.itemconfigure(self.__weights[layer_index][index], fill = get_node_color(weight), outline = 'white')

    def __unpaint_weight(self, index):
        '''Unpaint (hide) the weight at index.'''
        self.itemconfigure(index, fill = '', outline = '')

    def __paint_bias(self, weight):
        '''Paint the bias into a unit.'''
        self.itemconfigure(self.bias_id, fill = get_node_color(weight), outline = 'white')

    def __unpaint_bias(self):
        self.itemconfigure(self.bias_id, fill = '', outline = '')

    def __paint_targets(self, target):
        '''Paint the targets (for a supervised Network).'''
        index = 0
        for u in range(len(target)):
            self.__paint_target(index, target[u])
            index += 1

    def __paint_target(self, index, target):
        '''Paint a target object.'''
        self.itemconfigure(self.__targets[index],
                           fill = get_node_color(target),
#                           0.0 if target == self.__network.NO_TARGET else target),
                           outline = 'white')

    def __paint_error(self, error):
        self.itemconfigure(self.error_id, fill = get_error_color(error), outline = 'blue')

    def reinit(self, arg):
        '''Reinitialize the Network.'''
        self.__network.reinit()

    def step0(self, show_error=True, paint=True, train=True, target=[]):
        '''Run the Experiment on one pattern.'''
        if self.__experiment:
            pat_err_win = self.__experiment.step(train=train, show_error=show_error)
            if self.__is_sup:
                target = pat_err_win[0][1]
            error = pat_err_win[1]
            if paint:
                self.__paint(target, error)
            return error, pat_err_win[2]
        elif paint:
            self.__paint(target)

    def __paint(self, target = [], error = None):
        self.paint_nodes()
        if error != None:
            self.__paint_error(error)
        if target:
#            print("Painting target: {}".format(target))
            self.__paint_targets(target)

    def show(self, arg):
        '''Show the network activations in the console.'''
        if self.__experiment:
            self.__experiment.show()

    def run(self, arg):
        '''Run the Experiment N_STEPS times and print out statistics.'''
        trial0 = self.__experiment.trials
        error = 0.0
        compet = False
        paint = self.__parent.always_paint
#        if not self.__network.supervised:
#            winners = set([])
#            compet = True
        for i in range(N_STEPS):
            if paint:
                self.after(STEP_DELAY)
            err_win = self.step0(show_error = False, paint = paint)
            error += err_win[0]
#            if compet:
#                winners = winners | err_win[1]
            if paint:
                self.update_idletasks()
        error /= N_STEPS
        print(self.__experiment.trials, 'trials')
        print('error:', error)
#        if self.__parent.show_input_space and self.is_nodes:
#            self.update_is()
        self.__paint_error(error)
#        if compet:
#            print('# winners:', len(winners))

    def step(self, arg):
        '''Run the Experiment or Network n one pattern and show error.'''
        self.step0(show_error = True)
