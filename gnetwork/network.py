from gnetwork.activator import *
from gnetwork.coder import Coder
from random import random
from statistics import mean, stdev
from time import perf_counter
from collections import OrderedDict

import signal

from gnetwork.parallel_tools import engine

CMP_THRESH = 0.95

class State:
    def __init__(self, layer, sym=None, vec=None, ctx_state=None):
        self.layer = layer
        self.sym = sym
        self.ctx_state = ctx_state

        if ctx_state is not None and vec is not None:
            self.vec = vec * (self.ctx_state.get_vector() > 0)
        else:
            self.vec = vec

        # If vector is provided and symbol is not, accuracy is unknown
        if sym is None and vec is not None:
            self.acc = None
        else:
            self.acc = 1.

    def __str__(self):
        return str((self.layer.name, self.get_symbol(),
            0 if self.vec is None else self.vec.size,
            self.ctx_state))

    def is_none(self):
        return self.sym is None and self.vec is None

    def get_acc(self, target_sym=None):
        target_sym = target_sym if target_sym else self.get_symbol()
        if self.acc is None:
            return self.compare(State(self.layer,
                sym=target_sym, ctx_state=self.ctx_state))
        else:
            return self.acc

    def tuple(self):
        if self.ctx_state is not None:
            ctx_tup = (self.ctx_state.layer,
                self.ctx_state.get_symbol())
        else:
            ctx_tup = (None, None)

        return (self.layer, self.get_symbol()) + ctx_tup

    def get_symbol(self):
        if self.vec is not None and self.sym is None:
            # Decode and set accuracy
            # TODO: verify that decode computes accuracy in the same way
            self.sym,self.acc = self.layer.decode(self.vec)

            # Adjust accuracy based on context density
            if self.ctx_state is not None:
                nonzero = np.sum(self.ctx_state.get_vector() > 0)
                self.acc = self.acc / (nonzero / self.layer.size)
        return self.sym

    def get_vector(self):
        if self.vec is None:
            if self.sym is None:
                self.vec = np.zeros((self.layer.size, 1), dtype=np.float32)
            else:
                self.vec = self.layer.encode(self.sym)

                if self.ctx_state:
                    self.vec = self.vec * (self.ctx_state.get_vector() > 0)

        return self.vec

    def mask(self, ctx_state):
        if ctx_state:
            return State(self.layer,
                sym=self.sym,
                vec=self.vec,
                ctx_state= (ctx_state
                    if self.ctx_state is None
                    else self.ctx_state.mult(ctx_state)))
        else:
            return self

    def add(self, delta):
        return State(self.layer,
            vec = self.get_vector() + delta,
            ctx_state = self.ctx_state)

    def sub(self, delta):
        return State(self.layer,
            vec = self.get_vector() - delta,
            ctx_state = self.ctx_state)

    def mult(self, coeff):
        if self.vec is None and coeff.vec is None \
                and self.get_symbol() == coeff.get_symbol():
            return self
        else:
            return State(self.layer,
                sym = self.sym,
                vec = np.multiply(coeff.get_vector(), self.get_vector()),
                ctx_state = self.ctx_state)

    def saturate(self):
        if self.vec is None:
            #out_vec = None
            return self
        else:
            out_vec = vec=self.layer.act.saturate(self.vec)
        return State(layer=self.layer,
            sym=self.sym,
            vec=out_vec,
            ctx_state=self.ctx_state)

    def g(self):
        if self.vec is None:
            return self
        else:
            out_vec = self.layer.act.g(
                self.layer.act.saturate(self.vec))
            return State(layer=self.layer,
                sym=self.sym,
                vec=out_vec,
                ctx_state=self.ctx_state)

    def f(self):
        if self.vec is None:
            return self
        else:
            out_vec = self.layer.act.f(self.vec)
            return State(layer=self.layer,
                sym=self.sym,
                vec=out_vec,
                ctx_state=self.ctx_state)

    def compare(self, other):
        if self.sym and other.sym:
            return 1. if self.sym == other.sym else 0.
        elif self.vec is not None or other.vec is not None:
            self_vec, other_vec = self.get_vector(), other.get_vector()

            # Compute similarity differently for heaviside and sign/tanh
            if self.layer.act.label == "heaviside":
                sim = np.sum(np.equal(self_vec, other_vec))
            else:
                sim = np.sum(np.multiply(self_vec, other_vec))

            # Adjust based on context patterns (if identical)
            if self.ctx_state and other.ctx_state \
                    and (self.ctx_state.compare(other.ctx_state) == 1.):
                N = np.sum(self.ctx_state.get_vector())
            else:
                N = self.layer.size
            return sim / N
        else:
            return 1.

class Mappings:
    def __init__(self, conn):
        self.conn = conn
        self.learned = []
        self.mappings = OrderedDict()

    def add(self, x, y):
        self.learned.append((x, y))
        k = (x.tuple(), y.ctx_state.tuple() if y.ctx_state else None)
        self.mappings[k] = y

    def lookup(self, state, to_ctx=None):
        to_ctx = to_ctx.tuple() if to_ctx else None
        ftup = state.tuple()
        try:
            # Try to look up mapping
            inp = self.mappings[ftup, to_ctx]
        except KeyError:
            try:
                # If failure, check for non-contextualized mapping
                inp = self.mappings[ftup[:2] + (None, None), None]
            except KeyError:
                # This means we tried to execute a non-existent transition
                # This happens deliberately, and is checked for using cmp
                inp = State(self.conn.to_layer)
        return inp

    def get_most_recent(self):
        if len(self.learned) == 0:
            raise RuntimeError("Compare activate before compare learn")
        return self.learned[-1]

    def sizes(self):
        return (len(self.mappings), len(self.learned))

    def empty(self):
        return len(self.learned) == 0

    def split_dict(self):
        to_syms, from_syms, to_ctx_syms, from_ctx_syms = [],[],[],[]
        to_ctx_layer, from_ctx_layer = None, None

        for ((from_name,from_sym,from_ctx_layer,from_ctx_sym),
                to_ctx_tuple),to_state in self.mappings.items():
            to_name, to_sym, to_ctx_layer, to_ctx_sym = to_state.tuple()

            to_syms.append(to_sym)
            from_syms.append(from_sym)
            from_ctx_syms.append(from_ctx_sym)
            from_ctx_layer = from_ctx_layer
            to_ctx_syms.append(to_ctx_sym)
            to_ctx_layer = to_ctx_layer

        return (to_syms, from_syms,
            to_ctx_syms, from_ctx_syms,
            to_ctx_layer, from_ctx_layer)

    def split_list(self):
        to_syms, from_syms, to_ctx_syms, from_ctx_syms = [],[],[],[]
        to_ctx_layer, from_ctx_layer = None, None

        for x,y in self.learned:
            from_name,from_sym,from_ctx_layer,from_ctx_sym = x.tuple()
            to_name,to_sym,to_ctx_layer,to_ctx_sym = y.tuple()

            to_syms.append(to_sym)
            from_syms.append(from_sym)
            from_ctx_syms.append(from_ctx_sym)
            from_ctx_layer = from_ctx_layer
            to_ctx_syms.append(to_ctx_sym)
            to_ctx_layer = to_ctx_layer

        return (to_syms, from_syms,
            to_ctx_syms, from_ctx_syms,
            to_ctx_layer, from_ctx_layer)

def make_flash_matrix(layer, symbols, ctx=False, g=False):
    # ctx flag creates integer bit mask
    if ctx:
        pats = tuple((layer.encode(sym) > 0).T for sym in symbols)
        dtype = np.int32
    else:
        pats = tuple(layer.encode(sym).T for sym in symbols)
        dtype = np.float32

    # g flag runs patterns through inverse activation function
    if g: pats = tuple(layer.act.g(pat) for pat in pats)

    shape = (len(symbols), layer.size)
    arr = engine.managed(shape=shape, dtype=dtype)
    return np.concatenate(pats, axis=0, out=arr)

class Connection:
    def __init__(self, to_layer, from_layer, name, diag=True, decay=1., compare=False):

        self.to_layer = to_layer
        self.from_layer = from_layer
        self.name = (to_layer.name, from_layer.name, name)
        self.emulate = self.to_layer.emulate
        self.compare = compare

        self.norm = (from_layer.act.rho**2)

        self.diag = diag
        self.decay = decay

        self.flashed_mappings = Mappings(self)
        self.online_mappings = Mappings(self)
        self.all_mappings = Mappings(self)

        self.activ_count = 0
        #self.activ_history = []

        self.device = engine.get_device()
        #self.stream = None

        if not self.emulate:
            # Connection weight matrix
            self.weights = engine.managed(
                shape=(to_layer.size, from_layer.size),
                dtype=engine.get_weight_dtype())

            # x : source output (x)
            # y : target input (y)
            # Generally, learning and activation is such that
            #    y = dot(w, x)
            self.x = engine.managed(self.from_layer.outputs.get_vector())
            self.y = engine.managed(self.to_layer.inputs.get_vector())

            # Binary context masks for source and target regions
            self.ctx_x = engine.managed(self.x, dtype=np.int32)
            self.ctx_y = engine.managed(self.y, dtype=np.int32)

            # Index lists for source and target regions
            # Used by fast CUDA kernels (fast_dot, fast_learn, fast_converge)
            self.index_x = engine.managed(self.x, dtype=np.int32)
            self.index_y = engine.managed(self.y, dtype=np.int32)

    def __repr__(self):
        return "[to_layer : %s, from_layer : %s, name : %s]" % (
            self.to_layer.name, self.from_layer.name, self.name)


    # GPU kernels use transposed weight matrix
    # This function will safely return correct weight matrix
    #     regardless of whether CUDA is being used
    def get_weights(self):
        engine.set_device(self.device)
        #engine.synchronize(self.stream)
        engine.synchronize()
        return engine.get_weights(self.weights)

    def get_num_weights(self):
        return self.to_layer.size * self.from_layer.size

    def get_num_bytes(self):
        return self.get_num_weights() * np.dtype(engine.get_weight_dtype()).itemsize

    def activate(self):
        self.activ_count += 1

        if self.emulate:
            em_out = self.emulate_activate()
            #self.activ_history.append(em_out)
            return em_out
        else:
            out = self.from_layer.outputs.saturate()

            engine.set_device(self.device)
            #engine.set_stream(self.stream)
            #engine.synchronize(self.stream)
            engine.synchronize()

            self.x[:] = out.get_vector()

            inp = self.to_layer.inputs
            from_mask = out.ctx_state is not None
            to_mask = inp.ctx_state is not None

            FAST = (from_mask or to_mask) and engine.fast()

            if FAST:
                if from_mask:
                    self.ctx_x[:] = out.ctx_state.get_vector()
                if to_mask:
                    self.ctx_y[:] = inp.ctx_state.get_vector()

                engine.fast_dot(self.x, self.y,
                    self.weights,
                    self.ctx_x, self.ctx_y,
                    self.index_x, self.index_y,
                    from_mask, to_mask)
            else:
                engine.dot(self.x, self.y, self.weights)

            #engine.synchronize(self.stream)
            engine.synchronize()
            delta = self.y

            inp = self.to_layer.inputs.add(delta)
            #self.activ_history.append(inp)

            return inp

    def lookup(self, state, to_ctx=None):
        return self.all_mappings.lookup(state, to_ctx)

    def converge(self):
        if self.emulate:
            inp = self.lookup(self.to_layer.outputs)
        else:
            out = self.to_layer.outputs
            self.y[:] = out.get_vector()

            from_mask = out.ctx_state is not None
            if from_mask:
                self.ctx_y[:] = out.ctx_state.get_vector()

            engine.set_device(self.device)
            #engine.set_stream(self.stream)
            #engine.synchronize(self.stream)
            engine.synchronize()

            FAST = from_mask and engine.fast()
            if FAST:
                engine.fast_converge(
                    self.x, self.y,
                    self.weights,
                    self.ctx_x, self.ctx_y,
                    self.index_x, self.index_y,
                    from_mask, self.to_layer.act, 10)
            else:
                engine.converge(self.x, self.y, self.weights,
                    act=self.to_layer.act, timeout=10)

            #engine.synchronize(self.stream)
            engine.synchronize()
            inp = State(self.to_layer, vec=np.copy(self.x))

        self.activ_count += 1
        #self.activ_history.append(inp)
        return inp
        
    def learn(self, immediate=False, inputs=False):
        x = self.from_layer.outputs.saturate()

        # Cannot use both flags
        if immediate and inputs:
            raise RuntimeError
        # Learn based on current target activation
        elif immediate:
            y = self.to_layer.outputs.g()
        # Learn based on current target inputs
        elif inputs:
            y = self.to_layer.inputs.g()
        # Learn based on stashed pattern
        else:
            y = self.to_layer.stash

        # Store mappings if emulated or activity checking is enabled
        if self.emulate or self.to_layer.check:
            self.all_mappings.add(x, y)
            self.online_mappings.add(x, y)

        if self.emulate:
            return

        engine.set_device(self.device)
        #engine.set_stream(self.stream)
        #engine.synchronize(self.stream)
        engine.synchronize()

        self.x[:] = x.get_vector()
        self.y[:] = y.get_vector()

        if self.compare:
            engine.dipole(self.x, self.y, self.weights, self.norm)
        else:
            from_mask = x.ctx_state is not None
            to_mask = y.ctx_state is not None

            FAST = (from_mask or to_mask) and engine.fast()
            if FAST:
                # Context masks are only used if masking is enabled
                if from_mask:
                    self.ctx_x[:] = x.ctx_state.get_vector()
                if to_mask:
                    self.ctx_y[:] = y.ctx_state.get_vector()

                engine.fast_learn("rehebbian",
                    self.x, self.y,
                    self.weights,
                    self.ctx_x, self.ctx_y,
                    self.index_x, self.index_y,
                    from_mask, to_mask,
                    self.norm, self.diag, self.decay)
            else:
                engine.learn("rehebbian", self.x, self.y,
                    self.weights,
                    self.norm, self.diag, self.decay)
        #engine.synchronize(self.stream)
        engine.synchronize()

    def flash(self,
            to_syms, from_syms,
            ctx_layer=None, ctx_syms=None,
            to_ctx_layer=None, to_ctx_syms=None,
            from_ctx_layer=None, from_ctx_syms=None,
            from_mask=True, to_mask=True,
            learning_rule="rehebbian",
            diag=None,
            decay=None,
            count=1,
            erase=False,
            verbose=False):

        if diag is None: diag = self.diag
        if decay is None: decay = self.decay

        if len(from_syms) == 0:
            return

        self.emulate_flash(
            to_syms, from_syms,
            ctx_layer, ctx_syms,
            to_ctx_layer, to_ctx_syms,
            from_ctx_layer, from_ctx_syms,
            from_mask, to_mask)

        if self.emulate:
            return

        X = make_flash_matrix(self.from_layer, from_syms)
        Y = make_flash_matrix(self.to_layer, to_syms, g=True)

        if erase:
            Y *= 0.00000001

        if to_ctx_layer or ctx_layer:
            to_mask = True
            if not to_ctx_layer:
                to_ctx_layer = ctx_layer
                to_ctx_syms = ctx_syms
            CY = make_flash_matrix(to_ctx_layer, to_ctx_syms, ctx=True)
            np.multiply(Y, CY, out=Y)
        else:
            CY = None
            to_mask = False

        if from_ctx_layer or ctx_layer:
            from_mask = True
            if not from_ctx_layer:
                from_ctx_layer = ctx_layer
                from_ctx_syms = ctx_syms
            CX = make_flash_matrix(from_ctx_layer, from_ctx_syms, ctx=True)
            np.multiply(X, CX, out=X)
        else:
            CX = None
            from_mask = False

        engine.set_device(self.device)
        #engine.set_stream(self.stream)
        engine.synchronize()

        FAST = engine.fast()
        if FAST:
            if CX is not None:
                IX = engine.managed(CX, dtype=np.int32)
            else: IX = None

            if CY is not None:
                IY = engine.managed(CY, dtype=np.int32)
            else: IY = None

        if verbose:
            print("Flashing %40s %10d" % (str(self.name), len(to_syms)))
            start_time = perf_counter()

        for _ in range(count):
            if FAST:
                engine.fast_learn(
                    learning_rule,
                    X, Y, self.weights,
                    CX, CY,
                    IX, IY,
                    from_mask, to_mask,
                    norm = self.norm,
                    diag = diag,
                    decay = decay)
            else:
                engine.learn(learning_rule,
                    X, Y, self.weights,
                    norm = self.norm,
                    diag = diag,
                    decay = decay)

        engine.synchronize()

        if verbose:
            elapsed = perf_counter() - start_time
            print("  --> %f" % elapsed)


    def emulate_activate(self):
        inputs = self.to_layer.inputs
        outputs = self.from_layer.outputs

        if self.compare:
            last_learned = self.online_mappings.get_most_recent()[0]

            if outputs.vec is not None:
                acc = outputs.compare(last_learned)
                res = "true" if acc > CMP_THRESH else "false"
            else:
                res = ("true"
                    if (outputs.get_symbol() == last_learned.get_symbol()
                        and outputs.get_symbol() is not None)
                    else "false")

            inp = State(self.to_layer, sym=res)
        else:
            # If accuracy is below threshold, treat it as unrecognized
            if outputs.get_acc() > CMP_THRESH:
                inp = self.lookup(outputs, inputs.ctx_state)
            else:
                inp = State(self.to_layer)

        return inp.mask(inputs.ctx_state)

    def emulate_flash(self,
            to_syms, from_syms,
            ctx_layer=None, ctx_syms=None,
            to_ctx_layer=None, to_ctx_syms=None,
            from_ctx_layer=None, from_ctx_syms=None,
            from_mask=True, to_mask=True):

        if ctx_layer:
            to_ctx_layer = ctx_layer
            from_ctx_layer = ctx_layer
            to_ctx_syms = ctx_syms
            from_ctx_syms = ctx_syms
        elif not (ctx_layer or to_ctx_layer or from_ctx_layer):
            ctx_syms = [None for _ in to_syms]
            to_ctx_syms = ctx_syms
            from_ctx_syms = ctx_syms

        for sym in to_syms:
            self.to_layer.encode(sym)
        for sym in from_syms:
            self.from_layer.encode(sym)

        if to_ctx_layer:
            for sym in to_ctx_syms:
                to_ctx_layer.encode(sym)
        if from_ctx_layer:
            for sym in from_ctx_syms:
                from_ctx_layer.encode(sym)

        for ts,fs,to_cs,from_cs in zip(to_syms,from_syms,to_ctx_syms,from_ctx_syms):
            from_ctx_state = (State(to_ctx_layer, sym=to_cs)
                if (from_mask and from_ctx_layer) else None)
            to_ctx_state = (State(from_ctx_layer, sym=from_cs)
                if (to_mask and to_ctx_layer) else None)

            to_state = State(self.to_layer, sym=ts, ctx_state=to_ctx_state)
            from_state = State(self.from_layer, sym=fs, ctx_state=from_ctx_state)

            self.flashed_mappings.add(from_state, to_state)
            self.all_mappings.add(from_state, to_state)

    def print_stats(self):
        print(" ".join("%15s" % str(x) for x in self.name),
            "  |  ",
            "online: %5d / %5d | " % self.online_mappings.sizes(),
            "flashed: %5d / %5d | " % self.flashed_mappings.sizes(),
            "activ_count:  %5d" % self.activ_count)

    def test(self, to_sym, from_sym,
            to_ctx_sym=None, to_ctx_layer=None,
            from_ctx_sym=None, from_ctx_layer=None,
            converge=False):

        # Set source pattern
        self.from_layer.set_output(from_sym,
            State(from_ctx_layer, sym=from_ctx_sym)
                if from_ctx_layer else None)

        # Set target context and expected output
        if to_ctx_layer:
            to_ctx_state = State(to_ctx_layer, sym=to_ctx_sym)
            self.to_layer.inputs = self.to_layer.inputs.mask(to_ctx_state)
            exp = State(self.to_layer, sym=to_sym, ctx_state=to_ctx_state)
        else:
            exp = State(self.to_layer, sym=to_sym)

        self.to_layer.run_conn_comp("activate", self.from_layer, self.name[-1])
        self.to_layer.run_layer_comp("cycle")
        out = self.to_layer.outputs
        sim = out.compare(exp)

        if converge:
            self.to_layer.run_layer_comp("converge")
            self.to_layer.run_layer_comp("cycle")

            conv_out = self.to_layer.outputs
            conv_exp = State(self.to_layer, sym=to_sym)
            conv_sim = conv_out.compare(conv_exp)
            return (out, exp, sim), (conv_out, conv_exp, conv_sim)
        else:
            return (out, exp, sim), (None, None, 0.)



class Layer:
    def __init__(self, name, act, size,
            compare=False, ortho=False, emulate=False, check=False):
        self.name = name
        self.act = act
        self.size = size
        self.compare = compare
        self.emulate = emulate and not check
        self.check = check

        self.coder = Coder(act, ortho)
        self.malloc_count = 0

        self.gain_flag = True
        self.stash_flag = False
        self.stable_flag = False

        self.connections = {}

        self.inputs = State(self)
        self.outputs = State(self)
        self.stash = State(self)
        if compare:
            self.stash = State(self, vec=self.act.g(self.encode("true")))

    def __repr__(self):
        return "[Layer : %s]" % self.name

    def str_state(self, updated=False, outputs=None):
        if outputs is None:
            outputs = self.outputs
        acc = outputs.get_acc()
        return ((">" if updated else " ") +
            " | ".join("%20s" % x for x in (self.name,
                outputs.get_symbol(),
                acc,
                np.mean(np.abs(outputs.get_vector())))))

    def add_connection(self, from_layer, name, args={}):
        try:
            conn = self.connections[from_layer, name]
        except KeyError:
            conn = Connection(self, from_layer, name, **args)
            self.connections[from_layer, name] = conn
        return conn

    def get_connection(self, from_layer, name):
        return self.connections[from_layer, name]

    def check_state(self, conn):
        # TODO:
        #   comparison works here, but only because bias gate runs
        #   before the main connection computation

        if self.compare:
            return
        out = self.inputs.saturate()
        em_out = conn.emulate_activate().saturate()

        # TODO: what if it's auto?
        #   more generally, what do we do with minor divergences?
        #   we should log them without crashing the program

        # Divergence is emulated as a transition to None
        if em_out.is_none():
            # TODO: is_none?
            pass
        else:
            acc = out.compare(em_out)
            #eq_sym = (out.get_symbol() == em_out.get_symbol())

            #if not eq_sym:
            if acc < CMP_THRESH:
                # Mismatch with emulator
                print()
                print("Emulator mismatch")
                print(conn.name, "( acc:", acc, ")")
                #print(conn.name, "( acc:", acc, ", eq_sym:", eq_sym, ")")
                print("From: ", conn.from_layer.str_state())
                print("To:   ", self.str_state(outputs=out))
                print("Exp:  ", self.str_state(outputs=em_out))

                # TODO: what to do?
                if self.inputs.ctx_state is None:
                    input("Press enter to continue...")
                #raise RuntimeError

    def run_layer_comp(self, typ):
        # TODO: add transit type for convenience
        #     indicate where the context pattern comes from
        #     mem transit mem_ctx
        if typ == "cycle":
            self.cycle()
        elif typ == "converge":
            conn = self.connections[self, "auto"]
            self.inputs = conn.converge().mask(self.inputs.ctx_state)
            if self.check: self.check_state(conn)
            self.gain_flag = False
        elif typ == "noise":
            self.set_input("_%d" % self.malloc_count)
            self.malloc_count += 1
            self.gain_flag = False
        elif typ == "decay":
            self.gain_flag = False
        elif typ == "stash":
            self.stash_flag = True
        else:
            raise RuntimeError("Unrecognized gate: " + str(typ))

    def run_context_comp(self, from_layer):
        self.inputs = self.inputs.mask(from_layer.outputs)

    def run_bias_comp(self, sym):
        if not self.emulate:
            self.inputs = self.inputs.add(
                CMP_THRESH * self.act.g(self.encode(sym)))

    def run_conn_comp(self, typ, from_layer, name):
        conn = self.connections[from_layer, name]

        if typ == "activate":
            self.inputs = conn.activate()
            if self.check: self.check_state(conn)
            self.gain_flag = False
        elif typ == "learn":
            conn.learn(immediate=False, inputs=False)
        elif typ == "learn_immediate":
            conn.learn(immediate=True, inputs=False)
        elif typ == "learn_inputs":
            conn.learn(immediate=False, inputs=True)
        else:
            raise RuntimeError("Unrecognized gate: " + str(typ))

    def encode(self, sym, pattern=None):
        return self.coder.encode(sym, pattern)

    def decode(self, vec=None):
        if self.emulate and vec is None:
            return self.outputs.get_symbol(), 0.
        else:
            if vec is None:
                vec = self.outputs.saturate().get_vector()
            elif type(vec) == State:
                vec = vec.get_vector()
            return self.coder.decode(vec)

    def set_output(self, val=None, ctx_state=None):
        if type(val) == np.ndarray:
            self.outputs = State(self, vec=np.copy(val), ctx_state=ctx_state)
        else:
            self.outputs = State(self, sym=val, ctx_state=ctx_state)

    def set_input(self, sym=None):
        if sym is None:
            self.inputs = State(self)
        else:
            self.inputs = State(self, sym=sym,
                vec= (None if self.emulate
                    else self.act.g(self.encode(sym))))
        self.gain_flag = False

    def cycle(self):
        # Recurrent gain
        if self.gain_flag:
            if not self.stable_flag:
                self.outputs = self.outputs.saturate()
                self.stable_flag = True
            if self.inputs.ctx_state:
                self.outputs = self.outputs.mask(self.inputs.ctx_state)
        else:
            # Output transfer
            self.outputs = self.inputs.f()
            self.stable_flag = False

        # Stash for learning
        if self.stash_flag:
            self.stash = self.outputs.g()

        # Input (context) and flag reset
        self.inputs = State(self)
        self.gain_flag = True
        self.stash_flag = False

class Network:
    def __init__(self):
        self.layers = {}
        self.gate_layer = None
        self.gate_hidden = None
        self.gate_indices = {}

        self.gate_history = {}
        self.gate_time = {}

    def greedy_device_allocation(self):
        if engine.get_num_devices() == 1:
            return

        limits = engine.get_free_memory()
        conns = [c for l in self.layers.values() for c in l.connections.values()]
        sizes = [c.get_num_bytes() for c in conns]

        for i in sorted(range(len(conns)), reverse=True, key = lambda x: sizes[x]):
            size = sizes[i]
            device = limits.index(max(limits))
            conns[i].device = device
            limits[device] -= size

    def add_layer(self, layer):
        self.layers[layer.name] = layer

    def get_layer(self, name):
        return self.layers[name]

    def add_connection(self, to_layer, from_layer, name, args={}):
        return self.layers[to_layer].add_connection(
            from_layer, name, args)

    def get_connection(self, to_layer, from_layer, name):
        return self.layers[to_layer].get_connection(
            self.layers[from_layer], name)

    def set_outputs(self, out_map):
        for k,v in out_map.items():
            self.layers[k].set_output(v)

    def get_outputs(self, name):
        return self.layers[name].outputs.get_vector()

    def make_gate_pattern(self, keys):
        if self.gate_layer is None:
            raise RuntimeError

        p = -np.ones((self.gate_layer.size,1))
        for key in keys:
            p[self.gate_indices[key]] = 1
        return self.gate_layer.act.saturate(p)

    def flash_gates(self, gate_map, from_layer=None, emulate=False, verbose=False):
        self.gates = list(reversed(sorted(set(tuple(g)
            for gs in gate_map.values() for g in gs))))
        self.gate_indices = {
            g : i for i,g in enumerate(self.gates)
        }

        self.gate_layer = Layer("go",
            heaviside_activator(len(self.gates)), len(self.gates),
            emulate=emulate)

        for key, g in gate_map.items():
            self.gate_layer.encode(key,
                self.make_gate_pattern(g))

        if from_layer is not None:
            self.gate_hidden = from_layer
            self.gate_layer.add_connection(from_layer, "hetero").flash(
                gate_map.keys(), gate_map.keys(),
                learning_rule="rehebbian",
                verbose=verbose)

    def compute_gate(self, key):
        # TODO
        # timing is off because there is no synchronize after learn
        start_time = perf_counter()
        self.gate_history[key] = self.gate_history.get(key, 0) + 1

        if len(key) == 2:
            self.layers[key[0]].run_layer_comp(key[1])
        elif len(key) == 3:
            if key[1] == "context":
                self.layers[key[0]].run_context_comp(self.layers[key[2]])
            elif key[1] == "bias":
                self.layers[key[0]].run_bias_comp(key[2])
            else:
                raise RuntimeError("Unrecognized gate: " + str(key))
        else:
            self.layers[key[0]].run_conn_comp(
                key[1], self.layers[key[2]], key[3])

        elapsed = perf_counter() - start_time
        self.gate_time[key] = self.gate_time.get(key, 0.) + elapsed

    def get_active_gates(self):
        start_time = perf_counter()
        self.gate_layer.run_conn_comp("activate", self.gate_hidden, "hetero")
        elapsed = perf_counter() - start_time
        key = ("go", "activate", self.gate_hidden.name, "hetero")
        self.gate_history[key] = self.gate_history.get(key, 0) + 1
        self.gate_time[key] = self.gate_time.get(key, 0.) + elapsed

        start_time = perf_counter()
        self.gate_layer.run_layer_comp("cycle")
        elapsed = perf_counter() - start_time
        key = ("go", "cycle")
        self.gate_history[key] = self.gate_history.get(key, 0) + 1
        self.gate_time[key] = self.gate_time.get(key, 0.) + elapsed

        gates = np.where(self.gate_layer.outputs.get_vector()[:,0] > 0)[0]
        return tuple(self.gates[i] for i in gates)

    def run_manual(self, gate_sequence):
        engine.synchronize()
        for gates in gate_sequence:
            for g in sorted(gates, key=lambda x: (len(x), x[1])):
                self.compute_gate(g)

            for l in self.layers.values():
                self.compute_gate((l.name, "cycle"))

    def run_auto(self, t=100, verbose=False, debug=False, inputs=()):

        # Ensure gate layer exists
        if self.gate_layer is None:
            raise RuntimeError

        curr_t = 0

        # Register SIGINT handler for graceful interruption
        # To avoid distorting runtime analysis, handler flips interrupt flag
        #    and confirmation is requested within the loop below
        interrupted = [False]
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        def signal_handler(sig, frame):
            interrupted[0] = True

        signal.signal(signal.SIGINT, signal_handler)

        # Synchronize CUDA
        engine.synchronize()

        input_stream = iter(inputs)
        output = []

        if debug:
            print("t=", t)
            for layer in self.layers.values():
                print(layer.name, layer.decode())
            print()

        start_time = perf_counter()

        while curr_t < t:
            if debug: print(curr_t)
            updated = set()

            if interrupted[0]:
                print("Current iteration: ", curr_t)
                resp = input("Enter y/Y to interrupt: ")

                if len(resp) > 0 and (resp in "yY"):
                    print()
                    print("===============")
                    print("Interrupting...")
                    print("===============")
                    print()
                    break
                else:
                    interrupted[0] = False

            # Compute active gates
            gates = self.get_active_gates()
            if len(gates) == 0: break

            # Check for EOF before doing anything to the model
            if any(g[1] == "read" for g in gates):
                try: tok = next(input_stream)
                except StopIteration: break

            # Execute each ungated component
            for g in sorted(gates, key=lambda x: (len(x), x[1])):
                if debug: print(g)

                to_layer = self.layers[g[0]]
                updated.add(to_layer)

                if g[1] == "print":
                    out = to_layer.decode()
                    output.append((curr_t,) + out)
                    if verbose: print("Out: ", str(output[-1]), perf_counter() - start_time)

                    if out[0] == "probe_env":
                        print(self.layers["bind"].str_state())

                elif g[1] == "read":
                    # tok already read above during eof check
                    to_layer.set_input(tok)
                    if verbose: print("In: ", tok)
                else:
                    self.compute_gate(g)

            for l in self.layers.values():
                self.compute_gate((l.name, "cycle"))

            if debug:
                for layer in updated:
                    print(layer.str_state(layer in updated))
                print("-" * 91)
                for layer in self.layers.values():
                    if layer not in updated:
                        print(layer.str_state(layer in updated))
                print()

            curr_t += 1

        total_elapsed = perf_counter() - start_time

        if verbose:
            print("output:")
            for o in output:
                print(o)

            print()
            print("learned:")

            conns = [c
                for l in self.layers.values()
                    for c in l.connections.values()]

            # Print statistics on connection learning
            learn_conns = { conn : conn.all_mappings.sizes()
                for conn in conns if not conn.all_mappings.empty()}

            learn_conns = sorted(learn_conns.keys(),
                key = lambda k: learn_conns[k], reverse=True)
            for conn in learn_conns:
                conn.print_stats()

            print()
            print("Final activity states:")
            for name, layer in self.layers.items():
                print("%15s : %s" % (name, "%20s  %f" % layer.decode()))

            print()
            print("Gate history:")
            for key, elapsed in sorted(self.gate_time.items(), key=lambda x:x[1]):
                count = self.gate_history[key]
                print("%-48s : %10d   %20.4f   %20.18f" % (
                    "".join("%12s" % x for x in  key),
                    count,
                    elapsed,
                    elapsed / count))

            print()
            print("Executed %d timesteps in %fs" % (curr_t, total_elapsed))
            print()

        signal.signal(signal.SIGINT, original_sigint_handler)

        #for layer in self.layers.values():
        #    for conn in layer.connections.values():
        #        self.test(conn.name, conn.all_mappings,
        #            converge = conn.name[0] == "mem")

        return curr_t, output

    def flash(self, conn_name, flash_mappings, count=1, verbose=False):
        conn = self.get_connection(*conn_name)

        (to_syms, from_syms,
            to_ctx_syms, from_ctx_syms,
            to_ctx_layer, from_ctx_layer) = flash_mappings.split_list()

        # Make sure to use the layers from the non-emulated network!
        if to_ctx_layer: to_ctx_layer = self.get_layer(to_ctx_layer.name)
        if from_ctx_layer: from_ctx_layer = self.get_layer(from_ctx_layer.name)

        conn.flash(to_syms, from_syms,
            to_ctx_layer=to_ctx_layer,
            to_ctx_syms=to_ctx_syms,
            from_ctx_layer=from_ctx_layer,
            from_ctx_syms=from_ctx_syms,
            count=count,
            verbose=verbose)

    def test(self, conn_name, test_mappings, converge=False):
        conn = self.get_connection(*conn_name)
        start_time = perf_counter()

        (to_syms, from_syms,
            to_ctx_syms, from_ctx_syms,
            to_ctx_layer, from_ctx_layer) = test_mappings.split_dict()

        # Make sure to use the layers from the non-emulated network!
        if to_ctx_layer: to_ctx_layer = self.get_layer(to_ctx_layer.name)
        if from_ctx_layer: from_ctx_layer = self.get_layer(from_ctx_layer.name)

        print("Conn:", conn.name)
        print("Mappings: %d | %d" % (len(to_syms), len(test_mappings.learned)))
        if to_ctx_layer: print("To context:   ", to_ctx_layer)
        if from_ctx_layer: print("From context: ", from_ctx_layer)

        if len(to_syms) == 0:
            return

        results = []
        conv_results = []
        for to_sym, from_sym, to_ctx_sym, from_ctx_sym in \
                zip(to_syms, from_syms, to_ctx_syms, from_ctx_syms):
            (out, exp, sim), (conv_out, conv_exp, conv_sim) = conn.test(
                to_sym, from_sym,
                to_ctx_sym, to_ctx_layer,
                from_ctx_sym, from_ctx_layer,
                converge=converge)
            results.append(sim)
            conv_results.append(conv_sim)

            if (not converge) and (sim < CMP_THRESH):
                print("FAILURE")
                print(len(results), sim, to_sym, from_sym, to_ctx_sym, from_ctx_sym)
                print("   ", conn.to_layer.decode(out.get_vector()))
            if converge and (conv_sim < CMP_THRESH):
                print("CONV FAILURE")
                print(len(results), conv_sim, to_sym, from_sym, to_ctx_sym, from_ctx_sym)
                print("   ", conn.to_layer.decode(conv_out.get_vector()))

        print("Mean:", mean(results))
        print("Min:", min(results))
        print("Max:", max(results))
        try: print("Std:", stdev(results))
        except: pass

        if converge:
            print("  === Conv ===")
            print("  Mean:", mean(conv_results))
            print("  Min:", min(conv_results))
            print("  Max:", max(conv_results))
            try: print("  Std:", stdev(conv_results))
            except: pass
        elapsed = perf_counter() - start_time
        print("  --> %f" % elapsed)
        print()


'''
TODO:

* streamline passage of learning parameters into network during construction
*
* what ordering constraints need to be maintained?
*   - if they are loosened, the program is simplified
'''
