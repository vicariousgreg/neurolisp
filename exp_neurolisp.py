import argparse
from pcfg import lispify_pcfg
from unify import gen_trees
import pickle
from random import choice
import os

from build_neurolisp import *
from gnetwork import *

suite_test_cases = [
    # List testing
    ("(cons (quote A) (cons (quote B) NIL))",
     "(A B)"),
    ("(list (quote A) (quote B))",
     "(A B)"),
    ("(quote (A B))",
     "(A B)"),
    ("(car (cons (quote A) NIL))",
     "A"),
    ("(car (cdr (cdr (list (quote A) (quote B) (quote C)))))",
     "C"),
    ("(car (cdr (car (cdr (quote (A (B C) D))))))",
     "C"),
    ("(cadr (quote (A (B C) D)))",
     "(B C)"),

    # eq, atom, listp
    ("(eq 'x 'x)",
     "true"),
    ("(eq 'x 'y)",
     "false"),
    ("(eq 'x (list 'x))",
     "false"),
    ("(atom 'x)",
     "true"),
    ("(atom (list 'x))",
     "false"),
    ("(listp 'x)",
     "false"),
    ("(listp (list 'x))",
     "true"),

    # read, print
    ("(print (read)) A",
     "A A"),
    ("(print (list (read) (read) (read))) A B C",
     "(A B C) (A B C)"),

    # progn, dolist
    ("(progn (print 'foo) (print 'bar) 'baz)",
     "foo bar baz"),
    ("(dolist (x '(A B C) x) (print x))",
     "A B C C"),

    # quote, eval
    ("(eval (quote (print (quote x))))",
     "x x"),
    ("(eval (cons 'print (cdr (list 'foo '(quote x)))))",
     "x x"),

    # cond, if
    ("(if true 'foo 'bar) (if false 'foo 'bar)",
     "foo bar"),
    ("(if (or false (and true true)) 'foo 'bar)",
     "foo"),
    ("""(cond (false 'a)
              ((or false false) 'b)
              ((and true false) 'c)
              ((not true) 'd)
              ((eq 'x 'y) 'e)
              (true 'f))""",
     "f"),

    # lambda, label, defun
    ("((lambda (x y) (list x y)) 'foo 'bar)",
     "(foo bar)"),
    ("((lambda (f x y) (f x y)) (lambda (x y) (list x y)) 'foo 'bar)",
     "(foo bar)"),
    ("((label f (lambda (x) (if x (progn (print (car x)) (f (cdr x)))))) (list 'foo 'bar))",
     "foo bar NIL"),
    ("(defun f (x y) (list x y)) (defun g (x y) (f x y)) (g 'foo 'bar)",
     "#FUNCTION #FUNCTION (foo bar)"),

    # let, setq
    ("(let ((x 'foo)) (progn (print x) (let ((x 'bar)) (print x)) x))",
     "foo bar foo"),
    ("(let ((x 'foo) (y 'bar)) (list x y))",
     "(foo bar)"),
    ("(let ((x 'foo) (y 'bar)) (progn (defun f (x) (print (list x y))) (f 'baz) x))",
     "(baz bar) foo"),
    ("(progn (setq x 'foo) x)",
     "foo"),
    ("(let ((x 'foo)) (progn (setq x 'bar) x))",
     "bar"),
    ("(let ((x 'foo)) (progn (let ((x 'bar)) (setq x 'baz)) x))",
     "foo"),
    ("(let ((x 'foo)) (progn (defun f (x) (setq x 'bar)) (f 'baz) x))",
     "foo"),
    ("(defun f () (setq x 'foo)) (f) x",
     "#FUNCTION foo foo"),

    # hashing
    ("""(let ((hash (makehash))) (progn
            (sethash 'key1 'val1 hash)
            (sethash 'key2 'val2 hash)
            (print (and
                (checkhash 'key1 hash)
                (checkhash 'key2 hash)
                (not (checkhash 'key3 hash))))
            (print (list
                (gethash 'key1 hash)
                (gethash 'key2 hash)))
            (remhash 'key1 hash)
            (print (checkhash 'key1 hash))
            (sethash 'key1 'foo hash)
            (print (checkhash 'key1 hash))
            (print (gethash 'key1 hash))
            hash
        ))""",
        "true (val1 val2) false true foo #HASH"),

    # y-combinator
    ("""(progn
          (((lambda (le)
              ((lambda (g) (g g))
                   (lambda (h)
                          (le (lambda (x) ((h h) x))))))
                (lambda (f)
                    (lambda (x) (cond
                        (x (progn
                            (print x)
                            (f (cdr x))
                            (print (car x)))
                        (true x))))))
                '(a b c))
            'complete)""",
     "(a b c) (b c) (c) c b a complete"),
]

pcfg_tests = lispify_pcfg("./test_data/pcfg_data/pcfg_source.txt",
    "./test_data/pcfg_data/pcfg_target.txt")

pcfg_tests_sample_mem = [pcfg_tests[i] for i in
    pickle.load(open("./test_data/pcfg_data/mem_filtered_200_indices.p", "rb"))]
pcfg_tests_sample_bind = [pcfg_tests[i] for i in
    pickle.load(open("./test_data/pcfg_data/bind_filtered_200_indices.p", "rb"))]

'''
unify_tests = [
    ("""(A (var x) (var x) C)
        (A B B C)
        (x)""",
     "(B)"),
    ("""(A (B (var x) (var y)) (var y) (C (var x)))
        (A (B F G) G (C F))
        (x y)""",
     "(F G)"),
    ("""(A (B (var x) (var y)) (var y) (C (var x)))
        (A (B F G) G (C H))
        (x y)""",
     "NO_MATCH"),
]
'''

unify_tests = []
for tree_size in [6, 8, 10, 12, 14]:
    for i in range(20):
        mismatch = random() < 0.2
        pat1, pat2, subs = gen_trees(
            tree_size=tree_size,
            syms="abcdefghij",
            var_names="VWXYZ",
            num_subs=3,
            max_var_size=5,
            mismatch=mismatch)
        variables = tuple(k for k in subs)
        values = tuple(subs[v] for v in variables)
        ref = "NO_MATCH" if mismatch else ("( %s )" % " ".join(values))
        unify_tests.append(
            ("%s %s ( %s )" % (pat1, pat2, " ".join(variables)), ref))

list_tests = []
syms = "ABCDEFGHIJ"
for size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    for j in range(20):
        l = "(%s)" % " ".join(choice(syms) for i in range(size))
        list_tests.append((l,l))


pcfg_prog = '''
    (defun append (x y)
        (if x
            (cons (car x)
                (append (cdr x) y))
            y))
    (defun prepend (x y) (append y x))
    (defun remove_first (x y) y)
    (defun remove_second (x y) x)

    (defun last (x) (dolist (e x e)))
    (defun copy (x) x)

    (defun reverse (pre)
        (let ((post NIL))
            (dolist (x pre post)
                (setq post (cons x post)))))

    (defun shift (x) (append (cdr x) (list (car x))))

    (defun swap-helper (first mid)
        (if (cdr mid)
            (cons (car mid) (swap-helper first (cdr mid)))
            (list first)))
    (defun swap_first_last (x)
        (cons (last x)
            (swap-helper (car x) (cdr x))))

    (defun repeat (x) (append x x))
    (defun echo (x) (append x (list (last x))))
'''

unify_prog = '''
    (defun var? (x)
        (and
            (listp x)
            (eq (car x) 'var)))

    (defun match-var (var pat subs)
        (cond
            ((and (var? pat) (eq var (cadr pat))) subs)
            ((checkhash var subs)
                (unify (gethash var subs) pat subs))
            (true (sethash var pat subs))))

    (defun unify (pat1 pat2 subs)
        (cond
            ((not subs) subs)
            ((var? pat1) (match-var (cadr pat1) pat2 subs))
            ((var? pat2) (match-var (cadr pat2) pat1 subs))
            ((atom pat1)
                (if (eq pat1 pat2) subs NIL))
            ((atom pat2) NIL)
            (true
                (unify (cdr pat1) (cdr pat2)
                    (unify (car pat1) (car pat2) subs)))))

    (defun get-subs (vars subs)
        (if vars
            (cons
                (gethash (car vars) subs)
                (get-subs (cdr vars) subs))
            NIL))

    (let ((rule (read))
          (pat (read))
          (targets (read))
          (subs (unify rule pat (makehash))))
        (if subs
            (get-subs targets subs)
            'NO_MATCH))
    '''

list_prog = """ (read) """

def make_bind_one_test(args_list, val_list):
    return (" ".join("(setq %s '%s)" % (a,v)
            for a,v in zip(args_list,val_list))
        + " " + " ".join(args_list))

bind_one_tests = []
syms = "ABCDEFGHIJ"
for size in [70, 80, 90, 100, 110, 120]:
    for j in range(20):
        args_list = ["v%d" % i for i in range(size)]
        val_list = [choice(syms) for i in range(size)]
        prog = make_bind_one_test(args_list, val_list)
        bind_one_tests.append((
            prog,
            #" ".join(val_list)))
            "{0} {0}".format(" ".join(val_list))))

bind_many_prog = """
    (progn
        (defun f (x)
            (if x
                (progn
                    (f (cdr x))
                    (print (car x)))))
        (f (read))
        'NIL)
"""

bind_many_tests = []
syms = "ABCDEFGHIJ"
for size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    for j in range(20):
        l = [choice(syms) for i in range(size)]
        bind_many_tests.append((
            "(%s)" % " ".join(l),
            " ".join(reversed(l)) + " NIL"))

def run_test(prog, io_pairs, args, layer_sizes, ctx_lam):
    print("RUNNING TESTS")
    print("Prog: ", prog)
    print("Args: ")
    for arg in vars(args):
         print("  %10s : %s" % (arg, getattr(args, arg)))
    print()

    print("Layer sizes: ")
    for reg,size in layer_sizes.items():
         print("  %10s : %s" % (reg, size))
    print()

    print("Contexts sizes: ")
    for reg,size in ctx_lam.items():
         print("  %10s : %s" % (reg, size))
    print()
    print()

    results = []

    preprocessed_prog = preprocess(prog)
    for test_index, (test_input, test_ref) in enumerate(io_pairs):
        preprocessed_test_input = preprocess(test_input)

        print("Running test %d" % test_index)
        print("Test: ", test_input)
        print("Ref:  ", test_ref)

        # Emulate program
        # Get program output
        em_net, (em_timesteps, em_output) = test(
            inputs=preprocessed_prog,
            t=1000000,
            verbose=args.verbose,
            debug=args.debug,
            capacity = {
                "mem": 16,
                "lex": 16,
                "bind": 16,
                "stack": 1024,
                "data_stack": 1024,
            },
            ortho=True,
            emulate=True)
        prog_output = [o for t,o,acc in em_output]

        # Emulate test
        em_timesteps_test, em_output_test = em_net.run_auto(
            t=10000000,
            verbose=args.verbose,
            debug=args.debug,
            inputs=preprocessed_test_input)

        # Append outputs, adjust test output timestamps
        em_output += [(t+em_timesteps, out, acc) for t,out,acc in em_output_test]
        em_timesteps += em_timesteps_test

        print("Emulated test in %d timesteps" % em_timesteps)
        print("Emulator output:")
        print(" ".join(o[1] for o in em_output))

        # Inspect learned associations
        mem_auto_conn = em_net.get_connection("mem", "mem", "auto")
        mem_hetero_conn = em_net.get_connection("mem", "mem", "hetero")
        lex_mem_conn = em_net.get_connection("mem", "lex", "hetero")
        bind_mem_conn = em_net.get_connection("mem", "bind", "hetero")
        bind_hetero_conn = em_net.get_connection("bind", "bind", "hetero")
        stack_op_conn = em_net.get_connection("op", "stack", "hetero")
        data_op_conn = em_net.get_connection("mem", "data_stack", "hetero")
        print("Memories:            ", len(mem_auto_conn.online_mappings.mappings))
        print("Transits:            ", len(mem_hetero_conn.online_mappings.mappings))
        print("Symbols:             ", len(lex_mem_conn.online_mappings.mappings))
        print("Bindings:            ", len(bind_mem_conn.online_mappings.mappings))
        print("Namespaces:          ", len(bind_hetero_conn.online_mappings.mappings))
        print("Runtime stack depth: ", len(stack_op_conn.online_mappings.mappings))
        print("Data stack depth:    ", len(data_op_conn.online_mappings.mappings))


        if args.emulate:
            net = em_net
            timesteps = em_timesteps
            output = em_output
        else:
            # Execute program and inputs on real network
            net, (timesteps, output) = test(
                inputs=preprocessed_prog + preprocessed_test_input,
                t=em_timesteps + 1000,
                verbose=args.verbose,
                debug=args.debug,
                layer_sizes = layer_sizes,
                ctx_lam = ctx_lam,
                ortho=args.ortho,
                emulate=args.emulate,
                check=args.check,
                decay=args.decay)

        print("Ran test in %d timesteps" % timesteps)
        print("Actual output:")
        print(" ".join(o[1] for o in output))

        # Compare with emulator output
        em_out_syms = tuple(o[:2] for o in em_output)
        out_syms = tuple(o[:2] for o in output)
        print("Output matches? %s" % (out_syms == em_out_syms))
        print("Timesteps match? %s" % (timesteps == em_timesteps))

        # Check reference output
        ref = prog_output + preprocess(test_ref, strip_comments=False)
        out_syms = [o[1] for o in out_syms]

        if ref == out_syms:
            print("PASSED")
        else:
            print(" ===== FAILED ===== ")

        print()

        # Test results
        # 1. Passed?
        # 2. Number of timesteps
        # 3. Number of mappings per connection
        results.append((
            test_input,
            test_ref,
            out_syms,
            ref == out_syms,
            timesteps,
            dict((conn.name, len(conn.online_mappings.mappings))
                for layer in em_net.layers.values()
                for conn in layer.connections.values()),
        ))

    layer_sizes = { layer.name : layer.size for layer in net.layers.values() }
    return (args, layer_sizes, ctx_lam, prog, results)

def suite_test(args):
    to_run = suite_test_cases

    mem_size = args.mem_size if args.mem_size != 0 else 2048
    bind_size = args.bind_size if args.bind_size != 0 else 1024
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    path = args.path if args.path else "./test_data/suite_data/"
    filename = "%s/suite.p" % path

    dump = args.dump
    if dump and os.path.exists(filename):
        return

    (args, layer_sizes, ctx_lam, prog, results) = \
        run_test("", to_run, args,
            layer_sizes = {
                "mem": mem_size,
                "bind": bind_size,
                "lex": lex_size,
                "stack": 256,
                "data_stack": 256,
            },
            ctx_lam = {
                "mem_ctx" : 0.25,
                "bind_ctx" : args.bind_ctx_lam,
            })
    if dump:
        pickle.dump((args, layer_sizes, prog, results), open(filename, "wb"))

def tree_test(args):
    filenames = ["./test_data/tree_data/tree%d.lisp" % x for x in range(8)]
    to_run = [(" ".join(open(f).readlines()), "ALL_TESTS_PASSED") for f in filenames]

    mem_size = args.mem_size if args.mem_size != 0 else 6000
    bind_size = args.bind_size if args.bind_size != 0 else 1024
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    path = args.path if args.path else "./test_data/tree_data/"
    filename = "%s/tree.p" % path

    dump = args.dump
    if dump and os.path.exists(filename):
        return

    (args, layer_sizes, ctx_lam, prog, results) = \
        run_test("", to_run, args,
            layer_sizes = {
                "mem": mem_size,
                "bind": bind_size,
                "lex": lex_size,
                "stack": 256,
                "data_stack": 256,
            },
            ctx_lam = {
                "mem_ctx" : 0.25,
                "bind_ctx" : args.bind_ctx_lam,
            })
    if dump:
        pickle.dump((args, layer_sizes, prog, results), open(filename, "wb"))

def pcfg_test_mem(args):
    pcfg_test(args, typ="mem")

def pcfg_test_bind(args):
    pcfg_test(args, typ="bind")

def pcfg_test(args, typ):
    #to_run = pcfg_tests
    #filename = "./test_data/pcfg_data/pcfg_emulate.p"

    # 3000, 3500, 4000, 4500, 5000, 5500
    mem_size = args.mem_size if args.mem_size != 0 else 5500

    # 100, 200, 300, 400, 500, 600
    bind_size = args.bind_size if args.bind_size != 0 else 1024
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    if typ == "mem":
        to_run = pcfg_tests_sample_mem
        path = args.path if args.path else "./test_data/pcfg_data/"
        filename = "%s/pcfg_mem_%d.p" % (path, mem_size)
    elif typ == "bind":
        to_run = pcfg_tests_sample_bind
        path = args.path if args.path else "./test_data/pcfg_data/"
        filename = "%s/pcfg_bind_%d.p" % (path, bind_size)
    else:
        raise ValueError

    dump = args.dump
    if dump and os.path.exists(filename):
        return

    (args, layer_sizes, ctx_lam, prog, results) = \
        run_test(pcfg_prog, to_run, args,
            layer_sizes = {
                "mem": mem_size,
                "bind": bind_size,
                "lex": lex_size,
                "stack": 256,
                "data_stack": 256,
            },
            ctx_lam = {
                "mem_ctx" : 0.25,
                "bind_ctx" : args.bind_ctx_lam,
            })
    if dump:
        pickle.dump((args, layer_sizes, prog, results), open(filename, "wb"))

# TODO adapt layer_sizes
# generate datasets
def unify_test_mem(args):
    unify_test(args, typ="mem")

def unify_test_bind(args):
    unify_test(args, typ="bind")

def unify_test(args, typ):
    #to_run = unify_tests
    #filename = "./test_data/unify_data/unify_emulate.p"

    # 3000, 3500, 4000, 4500, 5000, 5500
    mem_size = args.mem_size if args.mem_size != 0 else 5500

    # 100, 200, 300, 400, 500, 600
    bind_size = args.bind_size if args.bind_size != 0 else 1024
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    if typ == "mem":
        #to_run = unify_tests_sample_mem
        to_run = unify_tests
        path = args.path if args.path else "./test_data/unify_data/"
        filename = "%s/unify_mem_%d.p" % (path, mem_size)
    elif typ == "bind":
        #to_run = unify_tests_sample_bind
        to_run = unify_tests
        path = args.path if args.path else "./test_data/unify_data/"
        filename = "%s/unify_bind_%d.p" % (path, bind_size)
    else:
        raise ValueError

    dump = args.dump
    if dump and os.path.exists(filename):
        return


    '''
    run_test(unify_prog, unify_tests, args,
        capacity = {
            "mem": 400,
            "lex": 64,
            "bind": 128,
            "stack": 256,
            "data_stack": 64,
        })

    '''
    (args, layer_sizes, ctx_lam, prog, results) = \
        run_test(unify_prog, to_run, args,
            layer_sizes = {
                "mem": mem_size,
                "bind": bind_size,
                "lex": lex_size,
                "stack": 256,
                "data_stack": 256,
            },
            ctx_lam = {
                "mem_ctx" : 0.25,
                "bind_ctx" : args.bind_ctx_lam,
            })
    if dump:
        pickle.dump((args, layer_sizes, prog, results), open(filename, "wb"))

def list_test_mem(args):
    list_test(args, typ="mem")

def list_test_lex(args):
    list_test(args, typ="lex")

def list_test(args, typ):
    to_run = list_tests
    #filename = "./test_data/list_data/list_emulate.p"

    # 300, 600, 900, 1200, 1500, 1800
    mem_size = args.mem_size if args.mem_size != 0 else 2048

    bind_size = args.bind_size if args.bind_size != 0 else 256
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    path = args.path if args.path else "./test_data/list_data/"
    filename = "%s/list_mem_%d.p" % (path, mem_size)

    if typ == "mem":
        filename = "%s/list_mem_%d.p" % (path, mem_size)
    elif typ == "lex":
        filename = "%s/list_lex_%d.p" % (path, lex_size)
    else:
        raise ValueError

    dump = args.dump
    if dump and os.path.exists(filename):
        return

    (args, layer_sizes, ctx_lam, prog, results) = \
        run_test(list_prog, to_run, args,
            layer_sizes = {
                "mem": mem_size,
                "bind": bind_size,
                "lex": lex_size,
                "stack": 1024,
                "data_stack": 1024,
            },
            ctx_lam = {
                "mem_ctx" : 0.25,
                "bind_ctx" : args.bind_ctx_lam,
            })
    if dump:
        pickle.dump((args, layer_sizes, prog, results), open(filename, "wb"))

def bind_one_test(args):
    to_run = bind_one_tests

    mem_size = args.mem_size if args.mem_size != 0 else 5000

    # 100, 200, 300, 400, 500, 600
    bind_size = args.bind_size if args.bind_size != 0 else 600
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    path = args.path if args.path else "./test_data/bind_data/bind_one/"
    filename = "%s/bind_bind_%d.p" % (path, bind_size)

    dump = args.dump
    if dump and os.path.exists(filename):
        return

    (args, layer_sizes, ctx_lam, prog, results) = \
        run_test("", to_run, args,
            layer_sizes = {
                "mem": mem_size,
                "bind": bind_size,
                "lex": lex_size,
                "stack": 256,
                "data_stack": 256,
            },
            ctx_lam = {
                "mem_ctx" : 0.25,
                "bind_ctx" : args.bind_ctx_lam,
            })
    if dump:
        pickle.dump((args, layer_sizes, prog, results), open(filename, "wb"))

def bind_many_test(args):
    to_run = bind_many_tests

    mem_size = args.mem_size if args.mem_size != 0 else 2048

    # 1000, 2000, 3000, 4000, 5000
    bind_size = args.bind_size if args.bind_size != 0 else 5000
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    path = args.path if args.path else "./test_data/bind_data/bind_many/"
    filename = "%s/bind_bind_%d.p" % (path, bind_size)

    dump = args.dump
    if dump and os.path.exists(filename):
        return

    (args, layer_sizes, ctx_lam, prog, results) = \
        run_test(bind_many_prog, to_run, args,
            layer_sizes = {
                "mem": mem_size,
                "bind": bind_size,
                "lex": lex_size,
                "stack": 1024,
                "data_stack": 1024,
            },
            ctx_lam = {
                "mem_ctx" : 0.25,
                "bind_ctx" : args.bind_ctx_lam,
            })
    if dump:
        pickle.dump((args, layer_sizes, prog, results), open(filename, "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default="list_mem",
                        help='test to run')
    parser.add_argument('-v', action='store_true', default=False,
                        dest='verbose',
                        help='verbose output during execution')
    parser.add_argument('-x', action='store_true', default=False,
                        dest='debug',
                        help='debugging output of network activity')
    parser.add_argument('-m', action='store_true', default=False,
                        dest='emulate',
                        help='emulate network activity')
    parser.add_argument('-c', action='store_true', default=False,
                        dest='check',
                        help='check network activity')
    parser.add_argument('-o', action='store_true', default=False,
                        dest='ortho',
                        help='orthogonal matrices for interpreter')
    parser.add_argument('-d', type=float, default=1.0,
                        dest="decay",
                        help="weight decay")
    parser.add_argument('--mem_size', type=int, default=0,
                        dest='mem_size',
                        help='size of memory region')
    parser.add_argument('--bind_size', type=int, default=0,
                        dest='bind_size',
                        help='size of bind region')
    parser.add_argument('--lex_size', type=int, default=0,
                        dest='lex_size',
                        help='size of lexicon region')
    parser.add_argument('--bind_ctx_lam', type=float, default=0.25,
                        dest="bind_ctx_lam",
                        help="bind ctx lambda")
    parser.add_argument('--dump', action='store_true', default=False,
                        dest='dump',
                        help='dump results to file')
    parser.add_argument('--path', type=str, default="",
                        help='path for results')
    args = parser.parse_args()

    print("args: ")
    for arg in vars(args):
         print("  %10s : %s" % (arg, getattr(args, arg)))
    print()

    if args.decay < 0.0 or args.decay > 1.0:
        raise ValueError("Decay must be between 0 and 1")

    try:
        {
            "suite": suite_test,
            "tree": tree_test,
            "pcfg_mem": pcfg_test_mem,
            "pcfg_bind": pcfg_test_bind,
            "unify_mem": unify_test_mem,
            "unify_bind": unify_test_bind,
            "list_mem": list_test_mem,
            "list_lex": list_test_lex,
            "bind_one": bind_one_test,
            "bind_many": bind_many_test,
        }[args.t](args)
    except KeyError:
        raise ValueError("Unrecognized test: %s" % args.t)
