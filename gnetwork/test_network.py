import argparse

from gnetwork.activator import *
from gnetwork.network import *
from random import sample, choice

def test_a(args={}):
    print("Test A")

    net = Network()

    x = Layer("x", sign_activator(32), 32, emulate=args.emulate, check=args.check)
    y = Layer("y", sign_activator(32), 32, emulate=args.emulate, check=args.check)
    ctx = Layer("ctx", heaviside_activator(32, 0.5), 32, emulate=args.emulate, check=args.check)

    net.add_layer(x)
    net.add_layer(y)
    net.add_layer(ctx)

    x.add_connection(x, "auto", { "diag" : False }).flash(
        ["a"], ["a"],
        diag=False)

    x.add_connection(y, "hetero").flash(
        ["a"], ["b"],
        ctx_layer=ctx, ctx_syms=["c"],
        from_mask=False, to_mask=True,
        learning_rule="rehebbian",
        diag=True)

    net.set_outputs({
        "y" : "b",
        "ctx" : "c",
    })

    net.run_manual(
        [
            [("x", "context", "ctx"),
             ("x", "activate", "y", "hetero")],
        ]
    )

    target = np.multiply(x.encode("a"), ctx.encode("c"))
    actual = x.outputs.get_vector()
    assert(np.array_equal(target, actual))

    net.run_manual(
        [
            [("x", "activate", "x", "auto")],
        ]
    )

    target = x.encode("a")
    actual = x.outputs.get_vector()
    assert(np.array_equal(target, actual))

def test_b(args={}):
    print("Test B")
    net = Network()

    size = 16*16
    gh = Layer("gh", sign_activator(size), size, emulate=args.emulate, check=args.check)
    x = Layer("x", sign_activator(size), size, emulate=args.emulate, check=args.check)
    y = Layer("y", sign_activator(size), size, emulate=args.emulate, check=args.check)
    z = Layer("z", sign_activator(size), size, emulate=args.emulate, check=args.check)

    net.add_layer(gh)
    net.add_layer(x)
    net.add_layer(y)
    net.add_layer(z)

    # x <- y connection
    x.add_connection(y, "hetero").flash(
        ["a"], ["b"],
        learning_rule="rehebbian",
        diag=True)

    # y <- z connection
    y.add_connection(z, "hetero").flash(
        ["b"], ["c"],
        learning_rule="rehebbian",
        diag=True)

    # gh <- gh connection
    gh.add_connection(gh, "hetero").flash(
        ["x<-y"], ["y<-z"],
        learning_rule="rehebbian",
        diag=True)

    # gates
    net.flash_gates({
        "y<-z" : [("y", "activate", "z", "hetero"),
                  ("gh", "activate", "gh", "hetero")],
        "x<-y" : [("x", "activate", "y", "hetero")],
    }, gh, emulate=args.emulate)

    # init and execute
    net.set_outputs({
        "z" : "c",
        "gh" : "y<-z",
    })
    net.run_auto(2)

    target = x.encode("a")
    actual = x.outputs.get_vector()
    assert(np.array_equal(target, actual))

def gen_gh_mappings(gate_sequences, gh):
    maps = []

    # Add gate sequences
    sym_gates = {}
    for start_key,term_key,seq in gate_sequences:
        names = [("%s_%02d" % (start_key,i)) for i,gs in enumerate(seq)]
        names[0] = start_key

        # add in gh<-gh
        for name, gs in zip(names, seq):
            sym_gates[name] = (gs
                if any(g[:2] == (gh, "activate") for g in gs)
                else gs + ((gh, "activate", gh, "hetero"),))

        for prev,nxt in zip(names, names[1:]):
            maps.append((nxt, prev))

        if term_key:
            maps.append((term_key, names[-1]))

    sym_gates.update({"halt" : ()})
    return { (gh, gh, "hetero") : maps }, sym_gates

def gen_op_mappings(operations, macros, op_layer, lex_layer, gh_layer, auto_op=False):
    mappings = {
        (op_layer, op_layer, "hetero") : [],
        (op_layer, op_layer, "auto") : [],
        (op_layer, lex_layer, "hetero") : [],
        (gh_layer, op_layer, "hetero") : [],
        (lex_layer, op_layer, "hetero") : [],
    }

    # Extract macros, add None for missing args
    macros = {
        gh : tuple(op if len(op) == 2 else op+(None,) for op in ops)
        for gh,ops in macros
    }

    def expand(gh,var):
        if gh in macros:
            # Arguments are marked by integers
            # Values are strings
            ops = tuple(
                (new_gh, (var if type(arg) is int else arg))
                    for new_gh,arg in macros[gh])
            return tuple(op
                for g,v in ops
                    for op in expand(g,v))
        else:
            return ((gh,var),)

    for start_key,term_key,ops in operations:
        # Add None for missing args
        ops = [
            op if len(op) == 2 else op+(None,)
            for op in ops
        ]

        # Expand macros
        ops = tuple(op for g,v in ops for op in expand(g,v))

        # Create name mapping (for op_exec)
        mappings[op_layer, lex_layer, "hetero"].append(
            (start_key, start_key))

        names = [
            "%s_%02d" % (start_key,i)
            for i,op in enumerate(ops)]
        names[0] = start_key

        for name,(gh,var) in zip(names, ops):
            mappings[gh_layer, op_layer, "hetero"].append(
                (gh, name))

            if var:
                mappings[lex_layer, op_layer, "hetero"].append(
                    (var, name))
        if auto_op:
            mappings[op_layer, op_layer, "auto"].append(
                (start_key, start_key))

        for prev,nxt in zip(names, names[1:]):
            mappings[op_layer, op_layer, "hetero"].append(
                (nxt, prev))

        if term_key:
            mappings[op_layer, op_layer, "hetero"].append(
                (term_key, names[-1]))

    return mappings

def test_c(args={}):
    print("Test C")

    macros = [
        ("foo",
         (("pre","bar"),)),
    ]
    operations = [
        ("test", None,
         (("foo",),)),
    ]

    gate_sequence = [
        ("load", "halt", [
            (("gh", "activate", "op", "hetero"),),
        ]),
        ("pre", "learn", [
            (("y", "activate", "z", "hetero"),),
            (("x", "activate", "y", "hetero"),("x", "context", "ctx"),),
        ]),
        ("learn", "post", [
            (("x", "stash"),),
            (("x", "learn", "z", "hetero"),),
        ]),
        ("post", "halt", [
            (("x", "decay"),),
            (("x", "activate", "z", "hetero"), ("x", "context", "ctx"),),
            (("op", "activate", "op", "hetero"),),
        ]),
    ]

    mappings, sym_gates = gen_gh_mappings(gate_sequence, "gh")
    mappings.update(
        gen_op_mappings(
            operations, macros, "op", "lex", "gh"))
    gates = set(tuple(g) for gs in sym_gates.values() for g in gs)

    layers = {}

    size = 16*16
    layers["x"] = Layer("x", sign_activator(size), size, emulate=args.emulate, check=args.check)
    layers["y"] = Layer("y", sign_activator(size), size, emulate=args.emulate, check=args.check)
    layers["z"] = Layer("z", sign_activator(size), size, emulate=args.emulate, check=args.check)
    layers["gh"] = Layer("gh", sign_activator(size), size, emulate=args.emulate, check=args.check)
    layers["op"] = Layer("op", sign_activator(size), size, emulate=args.emulate, check=args.check)
    layers["lex"] = Layer("lex", sign_activator(size), size, emulate=args.emulate, check=args.check)
    layers["ctx"] = Layer("ctx", heaviside_activator(size,lam=0.5), size, emulate=args.emulate, check=args.check)

    for g in gates:
        if len(g) == 4:
            #print(g)
            to_layer, typ, from_layer, name = g
            to_layer, from_layer = layers[to_layer], layers[from_layer]

            to_layer.add_connection(from_layer, name)

    net = Network()
    for layer in layers.values():
        net.add_layer(layer)

    mappings["x", "y", "hetero"] = [("a", "b")]
    mappings["y", "z", "hetero"] = [("b", "c")]

    for k,v in mappings.items():
        #print(k)
        if len(v):
            if len(k) == 3:
                to_layer, from_layer, name = k
                to_sym, from_sym = zip(*v)
                ctx_layer = None
                ctx_sym = []
            elif len(k) == 4:
                to_layer, from_layer, name, ctx_layer = k
                to_sym, from_sym, ctx_sym = zip(*v)
                ctx_layer = layers[ctx_layer]

            #print(to_layer, from_layer, name)
            layers[to_layer].add_connection(layers[from_layer], name).flash(
                to_sym, from_sym,
                ctx_layer=ctx_layer, ctx_syms=ctx_sym,
                learning_rule="rehebbian",
                diag=True)

    # gates
    net.flash_gates(sym_gates, layers["gh"], emulate=args.emulate)

    # init and execute
    net.set_outputs({
        "z" : "c",
        "ctx" : "c",
        "gh" : "load",
        "op" : "test",
    })
    net.run_auto()

    context = layers["ctx"].encode("c")
    target = np.multiply(layers["x"].encode("a"), context)
    actual = layers["x"].outputs.get_vector()
    #print(context)
    #print(target)
    assert(np.array_equal(target, actual))

    target = layers["gh"].encode("halt")
    actual = layers["gh"].outputs.get_vector()
    assert(np.array_equal(target, actual))


def gen_inst_mappings(inst_layer, inst_ctx_layer, op_layer, lex_layer):
    mappings = {
        (inst_ctx_layer, inst_layer, "hetero") : [],
        (inst_layer, inst_layer, "auto") : [],
        (inst_layer, inst_layer, "hetero", inst_ctx_layer) : [],
        (op_layer, inst_layer, "hetero") : [],
        (lex_layer, inst_layer, "hetero") : [],
    }

    # (begin (print x) (print y) (halt))
    parent = "main0"
    insts = [
        ("main0", "begin", None, "main1"),
        ("main1", "print", "x", "main2"),
        ("main2", "print", "y", "main3"),
        ("main3", "break", None, None),
    ]

    for source, op, sym, target in insts:
        mappings[inst_ctx_layer, inst_layer, "hetero"].append((source, source))
        mappings[inst_layer, inst_layer, "auto"].append((source, source))
        mappings[op_layer, inst_layer, "hetero"].append((op, source))

        if sym is not None:
            mappings[lex_layer, inst_layer, "hetero"].append((sym, source))

        if target is not None:
            mappings[inst_layer, inst_layer, "hetero", inst_ctx_layer].append((
                target, source, parent))

    return mappings

def gen_stack_mappings(stack_layer, size):
    mappings = {
        (stack_layer, stack_layer, "fwd") : [],
        (stack_layer, stack_layer, "bwd") : [],
    }

    for i in range(size):
        curr = str(i)
        successor = str((i+1) % size)

        mappings[stack_layer, stack_layer, "fwd"].append((successor, curr))
        mappings[stack_layer, stack_layer, "bwd"].append((curr, successor))

    return mappings

def gen_io_mappings(lex_layer, io_layer, syms):
    return {
        (lex_layer, io_layer, "hetero") : [(sym,sym) for sym in syms],
        (io_layer, lex_layer, "hetero") : [(sym,sym) for sym in syms],
    }


def test_d(args={}):
    print("Test D")

    macros = []

    operations = [
        ("halt", "halt",
         (("halt",),)),

        ("return", None,
         (("get_stack_inst",),
          ("adv_stack_prev",),
          ("get_stack_op",))),

        ("break", "return",
         (("adv_stack_prev",),)),

        # Executes an unbounded list of instructions
        # (begin insts...)
        ("begin", "begin",
         (("adv_inst_inst",),
          ("exec",)),),

        # Transfers symbol to I/O layer
        # (print lex)
        ("print", "return",
         (("print",),)),
    ]

    gate_sequence = [
        # Begins a new gate sequence based on operation
        ("final_gh_op", None,
         ((("gh", "activate", "op", "hetero"),),)),

        # Executes a child instruction
        # Binds the current operation to the current stack value
        # (final_op_inst)
        # (final_gh_op)
        ("exec", None,
         (
             (("op", "stash"),),
             (("op", "learn", "stack", "hetero"),),
             (("stack", "activate", "stack", "fwd"),
              ("op", "activate", "inst", "hetero"),
              ("inst_ctx", "activate", "inst", "hetero"),
              ("inst", "stash"),
              ("inst_ctx", "stash"),),
             (("inst_ctx", "learn", "stack", "hetero"),
              ("inst", "learn", "stack", "hetero"),
              ("gh", "activate", "op", "hetero"),),
         )),

        # Executes a child >operation<
        ("op_exec", "final_gh_op",
         (
             (("op", "stash"),),
             (("op", "learn", "stack", "hetero"),
              ("stack", "activate", "stack", "fwd"),),
             (("inst_ctx", "activate", "stack", "hetero"),
              ("inst", "activate", "stack", "hetero"),
              ("lex", "activate", "op", "hetero"),),
             (), # stabilize lex
             (("op", "activate", "lex", "hetero"),),
         )),

        # Stabilizes the instruction layer
        # (final_op_op)
        ("stabil_inst", "final_gh_op",
         (
             (("inst", "activate", "inst", "auto"),),
             (("inst", "activate", "inst", "auto"),),
             (("inst", "activate", "inst", "auto"),),
             (("inst", "activate", "inst", "auto"),),
             (("inst", "activate", "inst", "auto"),
              ("op", "activate", "op", "hetero"),),
         )),

        # Advances the instruction layer using instruction context
        ("adv_inst", "stabil_inst",
         (
             (("inst", "context", "inst_ctx"),),
             (("inst", "activate", "inst", "hetero"),
              ("inst", "context", "inst_ctx"),),
         )),

        # Loads parent instruction into instruction context layer
        ("adv_inst_inst", "adv_inst",
         ((("inst_ctx", "activate", "stack", "hetero"),),)),

        #########
        ### STACK
        #########

        # Retrieves the operation bound to the stack
        ("get_stack_op", "final_gh_op",
         (
             (("op", "activate", "stack", "hetero"),),
             (("op", "activate", "op", "hetero"),),
         )),

        # Advances the stack backward
        # (final_op_op)
        ("adv_stack_prev", "final_gh_op",
         (
             (("stack", "activate", "stack", "bwd"),
              ("op", "activate", "op", "hetero"),),
         )),

        # Gets the instruction bound to the current stack value
        # (final_op_op)
        ("get_stack_inst", "final_gh_op",
         (
             (("inst", "activate", "stack", "hetero"),
              ("op", "activate", "op", "hetero"),),
         )),

        # Binds memory to stack
        # (final_op_op)
        ("bind_mem_stack", "final_gh_op",
         (
             (("mem", "stash"),),
             (("mem", "learn", "stack", "hetero"),
              ("op", "activate", "op", "hetero"),),
         )),

        # Gets memory from stack
        # (final_op_op)
        ("get_mem_stack", "final_gh_op",
         (
             (("mem", "activate", "stack", "hetero"),
              ("op", "activate", "op", "hetero"),),
         )),

        ##############
        ### OPERATIONS
        ##############

        # Prints variable by transferring it to I/O layer
        # (final_op_op)
        # (final_gh_op)
        ("print", None,
         (
             (("lex", "activate", "inst", "hetero"),),
             (("op", "activate", "op", "hetero"),), # stabilize lex
             (("lex", "print"),
              ("gh", "activate", "op", "hetero"),),
        )),
    ]

    mappings, sym_gates = gen_gh_mappings(gate_sequence, "gh")
    gates = set(tuple(g) for gs in sym_gates.values() for g in gs)

    mappings.update(
        gen_op_mappings(
            operations, macros, "op", "lex", "gh", auto_op=False))

    mappings.update(
        gen_inst_mappings("inst", "inst_ctx", "op", "lex"))

    mappings.update(
        gen_stack_mappings("stack", 64))

    syms = set(sym for sym,inst in mappings["lex", "inst", "hetero"])

    layers = {}

    size = 32*32
    ctx_lam = 0.25

    #main_activator = sign_activator(size)
    main_activator = tanh_activator(0.0001, size)

    def add_layer(name):
        if name not in layers:
            layers[name] = Layer(name,
                heaviside_activator(size, lam=ctx_lam)
                    if "ctx" in name else main_activator,
                size, emulate=args.emulate, check=args.check)
        return layers[name]

    for g in gates:
        typ = g[1]
        to_layer = g[0]

        if len(g) == 4:
            from_layer = g[2]

            name = g[3]

            conn_args={}

            if name == "auto":
                conn_args["diag"] = False
                conn_args["decay"] = 0.9995

            to_layer, from_layer = add_layer(to_layer), add_layer(from_layer)
            to_layer.add_connection(from_layer, name, conn_args)

    net = Network()
    for layer in layers.values():
        net.add_layer(layer)

    for k,v in mappings.items():
        if len(v):
            if len(k) == 3:
                to_layer, from_layer, name = k
                to_sym, from_sym = zip(*v)
                ctx_layer = None
                ctx_sym = []
            elif len(k) == 4:
                to_layer, from_layer, name, ctx_layer = k
                to_sym, from_sym, ctx_sym = zip(*v)
                ctx_layer = layers[ctx_layer]

            '''
            print(to_layer, from_layer, ctx_layer)
            print(to_sym)
            print(from_sym)
            print(ctx_sym)
            print()
            '''

            auto = (name == "auto")
            layers[to_layer].add_connection(layers[from_layer], name).flash(
                to_sym, from_sym,
                ctx_layer=ctx_layer, ctx_syms=ctx_sym,
                learning_rule="rehebbian",
                diag= not auto,
                decay = 0.9995 if auto else 1.)

    # gates
    net.flash_gates(sym_gates, layers["gh"], emulate=args.emulate)

    # init and execute
    net.set_outputs({
        "inst" : "main0",
        "inst_ctx" : "main0",
        "op" : "halt",
        "gh" : "exec",
        "stack" : "0",
    })
    t, out = net.run_auto(1000)
    assert(out[0][1] == "x" and out[1][1] == "y")

    '''
    for i in range(size):
        net.set_outputs({
            "mem" : str(i),
        })
        net.run_manual(
            [
                [("lex", "activate", "mem", "hetero")],
            ]
        )
        print(i, net.layers["lex"].decode())
    '''

def test_e(args={}):
    print("Test E")
    size = 8
    x = Layer("x", sign_activator(size), size, emulate=args.emulate, check=args.check)
    ctx = Layer("ctx", heaviside_activator(size, 0.5), size, emulate=args.emulate, check=args.check)

    st = State(x, sym="foo")
    assert(np.array_equal(
        st.get_vector(),
        x.encode("foo")))

    st = State(x, vec=st.get_vector())
    assert(st.get_symbol() == "foo")

    ctx_st = State(ctx, sym="bar")
    mask_st = State(x, vec=st.get_vector(), ctx_state=ctx_st)
    assert(np.array_equal(
        mask_st.get_vector(),
        x.encode("foo") * (ctx.encode("bar") > 0)))

def test_f(args={}):
    print("Test F")
    net = Network()

    size = 32*32
    mem_lam = 0.25
    env_lam = 0.25
    #env_layer = Layer("env", tanh_activator(0.0001, size), size, emulate=args.emulate, check=args.check)
    env_layer = Layer("env", sign_activator(size), size, emulate=args.emulate, check=args.check)
    mem_layer = Layer("mem", sign_activator(size), size, emulate=args.emulate, check=args.check)
    mem_ctx_layer = Layer("mem_ctx", heaviside_activator(size, mem_lam), size, emulate=args.emulate, check=args.check)
    env_ctx_layer = Layer("env_ctx", heaviside_activator(size, env_lam), size, emulate=args.emulate, check=args.check)

    net.add_layer(env_layer)
    net.add_layer(mem_layer)
    net.add_layer(mem_ctx_layer)
    net.add_layer(env_ctx_layer)

    env_layer.add_connection(env_layer, "auto")
    mem_layer.add_connection(env_layer, "hetero")
    mem_layer.add_connection(mem_layer, "auto")

    num_syms = 64
    num_envs = 16
    num_mems = 64
    bindings_per = 8

    syms = [str(x) for x in range(num_syms)]
    mems = [str(x) for x in range(num_mems)]
    envs = { }

    mem_layer.get_connection(mem_layer, "auto").flash(
        syms, syms,
        from_mask=False, to_mask=False,
        learning_rule="rehebbian",
        diag=False, decay=1.)
        #diag=False, decay=1.)

    for i in range(num_envs):
        envs[str(i)] = dict(
            zip(sample(syms, bindings_per),
                sample(mems, bindings_per)))

        env_layer.get_connection(env_layer, "auto").flash(
            [str(i)] * bindings_per,
            [str(i)] * bindings_per,
            ctx_layer=env_ctx_layer, ctx_syms=envs[str(i)].keys(),
            from_mask=True, to_mask=True,
            learning_rule="rehebbian",
            diag=False, decay=1.)
            #diag=False, decay=1.)

        mem_layer.get_connection(env_layer, "hetero").flash(
            envs[str(i)].values(),
            [str(i)] * bindings_per,
            to_ctx_layer=mem_ctx_layer, to_ctx_syms=envs[str(i)].keys(),
            from_ctx_layer=env_ctx_layer, from_ctx_syms=envs[str(i)].keys(),
            from_mask=True, to_mask=True,
            learning_rule="rehebbian",
            diag=True, decay=1.)
            #diag=False, decay=1.)

        correct = 0
        for env,bindings in envs.items():
            for sym in syms:
            #for sym in bindings.union(sample(syms, 16)):
                net.set_outputs({
                    "env" : env,
                    "env_ctx" : sym,
                    "mem_ctx" : sym,
                })

                net.run_manual(
                    [
                        [("env", "context", "env_ctx")],
                        [("env", "context", "env_ctx"),
                         ("mem", "context", "mem_ctx"),
                         ("env", "activate", "env", "auto"),
                         ("mem", "activate", "env", "hetero")],
                        [("mem", "converge")],
                    ]
                )

                target = np.multiply(env_ctx_layer.encode(sym) > 0, env_layer.encode(env))
                actual = env_layer.outputs.get_vector()
                found = np.array_equal(np.sign(target), np.sign(actual))

                if sym in bindings:
                    retrieved = (mem_layer.decode()[0] == bindings[sym])
                    correct += found and retrieved
                else:
                    correct += not found

        assert(correct ==  (len(envs) * len(syms)))
        #print(correct, correct / (len(envs) * len(syms)))
        #print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store_true', default=False,
                        dest='emulate',
                        help='emulate network activity')
    parser.add_argument('-c', action='store_true', default=False,
                        dest='check',
                        help='check network activity')
    args = parser.parse_args()

    print("args: ")
    for arg in vars(args):
         print("  %10s : %s" % (arg, getattr(args, arg)))
    print()

    test_a(args)
    test_b(args)
    test_c(args)
    test_d(args)
    test_e(args)
    test_f(args)




"""

TODO:
   * parallel implementation
   * parse specification

"""
