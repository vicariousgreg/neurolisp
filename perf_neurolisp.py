import argparse
import pickle
from random import choice
import os

from build_neurolisp import *
from gnetwork import *

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
            verbose=False,
            debug=False,
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
            verbose=False,
            debug=False,
            inputs=preprocessed_test_input)

        # Append outputs, adjust test output timestamps
        em_output += [(t+em_timesteps, out, acc) for t,out,acc in em_output_test]
        em_timesteps += em_timesteps_test

        print("Emulated test in %d timesteps" % em_timesteps)
        print("Emulator output:")
        print(" ".join(o[1] for o in em_output))

        # Inspect learned associations
#        mem_auto_conn = em_net.get_connection("mem", "mem", "auto")
#        mem_hetero_conn = em_net.get_connection("mem", "mem", "hetero")
#        lex_mem_conn = em_net.get_connection("mem", "lex", "hetero")
#        bind_mem_conn = em_net.get_connection("mem", "bind", "hetero")
#        bind_hetero_conn = em_net.get_connection("bind", "bind", "hetero")
#        stack_op_conn = em_net.get_connection("op", "stack", "hetero")
#        data_op_conn = em_net.get_connection("mem", "data_stack", "hetero")
#        print("Memories:            ", len(mem_auto_conn.online_mappings.mappings))
#        print("Transits:            ", len(mem_hetero_conn.online_mappings.mappings))
#        print("Symbols:             ", len(lex_mem_conn.online_mappings.mappings))
#        print("Bindings:            ", len(bind_mem_conn.online_mappings.mappings))
#        print("Namespaces:          ", len(bind_hetero_conn.online_mappings.mappings))
#        print("Runtime stack depth: ", len(stack_op_conn.online_mappings.mappings))
#        print("Data stack depth:    ", len(data_op_conn.online_mappings.mappings))


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

def perf_test(args):
    to_run = [
        ("""(progn
              (print 'executing)
              (defun f (x)
                  (if x (f (cdr x))))
              (f '(1 2 3 4 5 6 7 8 9 10))
              'complete)""",
         "executing complete"),
    ]

    mem_size = args.mem_size if args.mem_size != 0 else 2048
    bind_size = args.bind_size if args.bind_size != 0 else 1024
    lex_size = args.lex_size if args.lex_size != 0 else 2048

    path = args.path if args.path else "./test_data/perf_data/"
    filename = "%s/perf.p" % path

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
                #"mem_ctx" : 0.25,
                "mem_ctx" : args.mem_ctx_lam,
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
    parser.add_argument('--mem_ctx_lam', type=float, default=0.25,
                        dest="mem_ctx_lam",
                        help="mem ctx lambda")
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
            "perf": perf_test,
        }[args.t](args)
    except KeyError:
        raise ValueError("Unrecognized test: %s" % args.t)
