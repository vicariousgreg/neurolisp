import argparse
from build_neurolisp import *
from gnetwork import *

def test_cfg(args={}):
    prog = '''
        (progn
            (defun last (x)
                (if (cdr x)
                    (last (cdr x))
                    (car x)))

            (defun copy (x) x)

            (defun rev-helper (pre post)
                (if pre
                    (rev-helper
                        (cdr pre)
                        (cons (car pre) post))
                    post))

            (defun reverse (x) (rev-helper x NIL))

            (defun shift (x) (append (cdr x) (list (car x))))

            (defun swap-helper (first mid)
                (if (cdr mid)
                    (cons (car mid) (swap-helper first (cdr mid)))
                    (list first)))
            (defun swap (x) (cons (last x) (swap-helper (car x) (cdr x))))

            (defun repeat (x) (append x x))
            (defun echo (x) (append x (list (last x))))

            (defun append (x y)
                (if x
                    (cons
                        (car x)
                        (append (cdr x) y))
                    y))
            (defun prepend (x y) (append y x))
            (defun remove-first (x y) y)
            (defun remove-second (x y) x)

            (defun read-list (x)
                (cond
                    ((eq x (quote ,)) NIL)
                    ((eq x (quote $)) NIL)
                    (true (cons x (read-list (read))))))

            (defun exec-one (x)
                ((eval x)
                    (parse (read))))
            (defun exec-two (x)
                ((eval x)
                    (parse (read))
                    (parse (read))))

            (defun parse (x)
                (progn
                    (print x)
                    (cond
                        ((eq x (quote copy)) (exec-one x))
                        ((eq x (quote reverse)) (exec-one x))
                        ((eq x (quote shift)) (exec-one x))
                        ((eq x (quote swap)) (exec-one x))
                        ((eq x (quote repeat)) (exec-one x))
                        ((eq x (quote echo)) (exec-one x))
                        ((eq x (quote append)) (exec-two x))
                        ((eq x (quote prepend)) (exec-two x))
                        ((eq x (quote remove-first)) (exec-two x))
                        ((eq x (quote remove-second)) (exec-two x))
                        (true (read-list x)))))

            (print (parse (read))))
    '''

    #inputs = ("append", "swap", "f", "g", "h", ",", "repeat", "i", "j", "$")

    inputs = ("append",
        "echo", "copy", "reverse", "repeat", "swap", "f", "g", "h", ",",
        "repeat", "i", "j", "$")

    return test(prog, inputs,
        t=10000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = { #"mem" : 64*64,
        #    #"lex" : 64*64,
        #    "bind": 32*32
        #},
        capacity = {
            "mem": 64,
            "lex": 32,
            "bind": 64,
            "stack": 128,
            "data_stack": 32,
        },
        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_cfg_mini(args={}):
    prog = '''
        (progn
            (defun copy (x) x)

            (defun remove-first (x y) y)
            (defun remove-second (x y) x)

            (defun read-list (x)
                (cond
                    ((eq x (quote ,)) NIL)
                    ((eq x (quote $)) NIL)
                    (true (cons x (read-list (read))))))

            (defun exec-one (x)
                ((eval x)
                    (parse (read))))
            (defun exec-two (x)
                ((eval x)
                    (parse (read))
                    (parse (read))))

            (defun parse (x)
                (progn
                    (print x)
                    (cond
                        ((eq x (quote copy)) (exec-one x))
                        ((eq x (quote remove-first)) (exec-two x))
                        ((eq x (quote remove-second)) (exec-two x))
                        (true (read-list x)))))

            (print (parse (read))))
    '''

    inputs = ("remove-first", "copy", "f", "g", "h", ",", "copy", "i", "j", "$")

    return test(prog, inputs,
        t=10000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = { #"mem" : 64*64,
        #    #"lex" : 64*64,
        #    "bind": 32*32
        #},
        capacity = {
            "mem": 32,
            "lex": 0,
            "bind": 32,
            "stack": 64,
            "data_stack": 32,
        },
        ctx_lam = { "mem_ctx" : 0.25, "bind_ctx" : 0.5 },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_cfg_new(args={}):
    inputs = preprocess('''
        (defun append (x y)
            (if x
                (cons (car x)
                    (append (cdr x) y))
                y))
        (defun prepend (x y) (append y x))
        (defun remove-first (x y) y)
        (defun remove-second (x y) x)

        (defun last (x)
            (if (cdr x)
                (last (cdr x))
                x))

        (defun copy (x) x)

        (defun rev-helper (pre post)
            (if pre
                (rev-helper
                    (cdr pre)
                    (cons (car pre) post))
                post))
        (defun reverse (x) (rev-helper x NIL))

        (defun shift (x) (append (cdr x) (list (car x))))

        (defun swap-helper (first mid)
            (if (cdr mid)
                (cons (car mid) (swap-helper first (cdr mid)))
                (list first)))
        (defun swap (x)
            (cons (car (last x))
                (swap-helper (car x) (cdr x))))

        (defun repeat (x) (append x x))
        (defun echo (x) (append x (last x)))
    ''')

    inputs += preprocess('''
        (append
            (echo
                (copy
                    (reverse
                        (repeat
                            (swap
                                (quote (f g h)))))))
            (repeat
                (quote (i j))))
    ''')

    inputs += preprocess("(halt)")


    return test(inputs=inputs,
        t=10000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = { #"mem" : 64*64,
        #    #"lex" : 64*64,
        #    "bind": 32*32
        #},
        capacity = {
            "mem": 300,
            "lex": 64,
            "bind": 64,
            "stack": 128,
            "data_stack": 32,
        },
        ctx_lam = { "mem_ctx" : 0.25, "bind_ctx" : 0.2 },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_reverse(args={}):
    inputs = preprocess('''
        (defun rev-helper (pre post)
            (if pre
                (rev-helper
                    (cdr pre)
                    (cons (car pre) post))
                post))
        (defun reverse (x) (rev-helper x NIL))

        (reverse (quote (a b c d e)))
    ''')

    inputs += preprocess("(halt)")


    return test(inputs=inputs,
        t=10000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = { #"mem" : 64*64,
        #    #"lex" : 64*64,
        #    "bind": 32*32
        #},
        capacity = {
            "mem": 256+32,
            "lex": 64,
            "bind": 64,
            "stack": 128,
            "data_stack": 32,
        },
        ctx_lam = { "mem_ctx" : 0.25, "bind_ctx" : 0.2 },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)


# Y combinator
def test_y(args={}):
    prog = '''
        (progn
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
                (quote (a b c d e f g h i j)))
            (print (quote complete)))
    '''

    return test(prog,
        t=100000, verbose=args.verbose,
        debug=args.debug,
        capacity = {
            "mem": 32,
            "lex": 0,
            "bind": 128,
            "stack": 64,
            "data_stack": 32,
        },
        ctx_lam = { "mem_ctx" : 0.25, "bind_ctx" : 0.125 },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_stack(depth=4, args={}):
    prog = '''
        (progn
            (defun read-list (x)
                (if (eq x (quote $)) NIL
                    (cons x (read-list (read)))))

            (defun print-list (x)
                (if (not x) NIL
                    (progn
                        (print (car x))
                        (print-list (cdr x)))))

            (print-list (read-list (read))))
    '''

    inputs = tuple(str(x) for x in range(depth)) + ("$",)
    return test(prog, inputs,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = { "mem" : 64*32, "lex" : 16*32 },
        capacity = { "lex" : depth },
        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

    prog = '''
        (progn
            (defun loop ()
                (if (eq (print (read)) (quote $)) NIL (loop)))

            (loop)
            (print (quote complete)))
    '''

    inputs = tuple(str(x) for x in range(depth)) + ("$",)
    return test(prog, inputs,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = { "mem" : 64*32, "lex" : 16*32 },
        capacity = { "lex" : depth },
        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_boolean(args={}):
    prog = '''
    (progn
        (cond
            ((not true) (error))
            ((not (eq false false)) (error))
            ((not (eq false false)) (error))

            ((not (and (eq false false) (eq true true))) (error))
            ((and (eq false false) (eq true false)) (error))
            ((and (eq true false) (eq false false)) (error))

            ((not (or (eq false false) (eq true true))) (error))
            ((not (or (eq true false) (eq true true))) (error))
            ((not (or (eq false false) (eq true false))) (error))
            ((or (eq true false) (eq false true)) (error))

            ((not true) (error))
            ((not (eq (car (list)) NIL)) (error))
            ((not (eq (cdr (list)) NIL)) (error))
            ((not
                (cond
                    (false false)
                    (true true)))
             (error))

            ((not (not false)) (error))
            ((eq false true) (error))

            ((not (eq NIL NIL)) (error))
            ((not (eq NIL (cdr (list (quote a))))) (error)))
        (print (quote complete)))
        '''
    return test(prog,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        layer_sizes = {
           # "mem" : 32*32,
            "lex" : 32*32,
           # "bind" : 32*32
        },
        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_medium(args={}):
    prog = '''
    (progn
        (defun f ()
            (cond
                ((eq (quote a) true)
                    (car (list (quote a) (quote b))))
                (false (quote c))
                (true
                    (cdr (cdr
                        (list (quote d) (quote e) (quote f) (quote g)))))
        ))
        (print (f))
        (print (f)))
        '''
    return test(prog,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = {
        #    "mem" : 32*32,
        #    #"lex" : 32*16,
        #    "bind" : 16*16
        #},

        capacity = {
            "mem": 16,
            "lex": 8,
            "bind": 16,
            "stack": 64,
            "data_stack": 64,
        },

        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

    #for k,v in net.get_connection("lex", "mem", "hetero").mappings.items():
    #    print(v.tuple()[1], "<-", k[1])

def test_small(args={}):
    prog = '''
        (print
            (cond
                ((eq (quote a) true)
                    (car (quote (a b))))
                (false (quote c))
                (true
                    (cdr (cdr (quote (d e f g)))))))
        '''
    return test(prog,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = {
        #    "mem" : 32*32,
        #    #"lex" : 32*16,
        #    "bind" : 16*16
        #},

        capacity = {
            "mem": 16,
            "lex": 8,
            "bind": 16,
            "stack": 64,
            "data_stack": 64,
        },

        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

    #for k,v in net.get_connection("lex", "mem", "hetero").mappings.items():
    #    print(v.tuple()[1], "<-", k[1])

def test_read(args={}):
    prog = '''
        (progn
            (defun f (x)
                (eval (read)))
            (f (quote b)))
        '''

    inputs = preprocess("(print x)")

    return test(prog, inputs,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        capacity = {
            "mem": 0,
            "lex": 0,
            "bind": 8,
            "stack": 16,
            "data_stack": 16,
        },
        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_hash(args={}):
    prog = '''
        (progn
            (defun check (hash key answer)
                (print
                    (if (eq (checkhash key hash) answer)
                        'good
                        'bad)))

            (let ((foo 'bar))
                (progn
                    (let ((hash (makehash)) (foo 'baz))
                        (progn
                            (sethash 'key1 'val1 hash)
                            (sethash 'key2 'val2 hash)
                            (sethash 'key3 'val3 hash)

                            (print (list
                                foo
                                (gethash 'key1 hash)
                                (gethash 'key2 hash)
                                (gethash 'key3 hash)))

                            (check hash 'key1 true)
                            (check hash 'key2 true)
                            (check hash 'key3 true)
                            (check hash 'key4 false)

                            (sethash 'key1 'val4 hash)
                            (check hash 'key1 true)
                            (print (gethash 'key1 hash))

                            (remhash 'key1 hash)
                            (check hash 'key1 false)
                            (sethash 'key1 'val5 hash)
                            (check hash 'key1 true)
                            (print (gethash 'key1 hash))

                            ))
                    (print foo))))
        '''
    return test(prog,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = {
        #    "mem" : 32*32,
        #    #"lex" : 32*16,
        #    "bind" : 16*16
        #},

        capacity = {
            "mem": 32,
            "lex": 32,
            "bind": 32,
            "stack": 64,
            "data_stack": 64,
        },

        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_unify(args={}):
    prog = '''
        (progn
            (defun var? (x)
                (and
                    (not (atom x))
                    (eq (car x) 'var)))

            (defun get-var-sym (var)
                (cadr var))

            (defun occurs (var pat)
                (cond
                    ((atom pat) false)
                    ((var? pat) (eq var (get-var-sym pat)))
                    (true (or
                        (occurs var (car pat))
                        (occurs var (cdr pat))))))

            (defun match-var (var pat subs)
                (if (and (var? pat) (eq var (get-var-sym pat)))
                    subs
                    (cond
                        ((checkhash var subs)
                            (unify (gethash var subs) pat subs))
                        ((occurs var pat) 'failed)
                        (true (sethash var pat subs)))))

            (defun unify (pat1 pat2 subs)
                (cond
                    ((eq subs 'failed) 'failed)
                    ((var? pat1) (match-var (get-var-sym pat1) pat2 subs))
                    ((var? pat2) (match-var (get-var-sym pat2) pat1 subs))
                    ((atom pat1)
                        (if (eq pat1 pat2)
                            subs
                            'failed))
                    ((atom pat2) 'failed)
                    (true
                        (unify (cdr pat1) (cdr pat2)
                            (unify (car pat1) (car pat2) subs)))))

            (defun equal (x y)
                (if (or (atom x) (atom y))
                    (eq x y)
                    (and (eq (car x) (car y))
                        (equal (cdr x) (cdr y)))))

#            (defun check-subs (subs correct)
#                (if (eq correct NIL) true
#                    (let ((var (car (car correct)))
#                          (val (cadr (car correct))))
#                        (if (and (checkhash var subs)
#                                (equal val (gethash var subs)))
#                            (check-subs subs (cdr correct))
#                            false))))
#
#            (defun check-unify (pat1 pat2 correct)
#                (let ((subs (unify pat1 pat2 (makehash))))
#                    (if (eq correct 'failed)
#                        (eq subs 'failed)
#                        (check-subs subs correct))))
#
#            (print
#                (check-unify
#                    '((a (var x) c (d e)))
#                    '((a b c (var y)))
#                    '((x b) (y (d e)))))
#
#            (print
#                (check-unify
#                    '(a (var x))
#                    '(var y)
#                    '((y (a (var x))))))
#
#            (print
#                (check-unify
#                    '(a (var x))
#                    '(var x)
#                    'failed))

            (defun substitute (pat subs)
                (cond
                    ((eq subs 'failed) 'failed)
                    ((atom pat) pat)
                    ((var? pat)
                        (if (checkhash (get-var-sym pat) subs)
                            (gethash (get-var-sym pat) subs)
                            pat))
                    (true
                        (cons (substitute (car pat) subs)
                            (substitute (cdr pat) subs)))))

            (defun match (rule obs)
                (let ((subs (unify (cdr rule) obs (makehash))))
                    (if (eq subs 'failed)
                        'failed
                        (substitute (car rule) subs))))

            (let ((rule
                    '((move (var obj) (var loc))
                      (grasp (var obj))
                      (move-arm (var obj) (var loc))
                      (release))))
                (progn
                    (print
                        (equal '(move x y)
                            (match
                                rule
                                '((grasp x)
                                  (move-arm x y)
                                  (release)))))
                    (print
                        (eq 'failed
                            (match
                                rule
                                '((grasp x)
                                  (move-arm z y)
                                  (release)))))))
        )
        '''
    #return test(prog,
    return test(inputs=preprocess(prog),
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = {
        #    "mem" : 32*32,
        #    #"lex" : 32*16,
        #    "bind" : 16*16
        #},

        capacity = {
            "mem": 500,
            "lex": 64,
            "bind": 128,
            "stack": 256,
            "data_stack": 64,
        },

        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_custom(args={}):
    #prog = input("Enter program: ")
    inputs = preprocess(input("Enter program: "))

    return test(inputs=inputs,
        t=1000000, verbose=args.verbose,
        debug=args.debug,
        #layer_sizes = {
        #    "mem" : 32*32,
        #    #"lex" : 32*16,
        #    "bind" : 16*16
        #},

        capacity = {
            "mem": 256,
            "lex": 128,
            "bind": 64,
            "stack": 256,
            "data_stack": 256,
        },

        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def test_file(args={}):
    filename = input("Enter filename: ")
    inputs = preprocess(" ".join(open(filename).readlines()))

    return test(inputs=inputs,
        t=1000000, verbose=args.verbose,
        debug=args.debug,

        capacity = {
            "mem": 256,
            "lex": 128,
            "bind": 64,
            "stack": 256,
            "data_stack": 256,
        },

        ctx_lam = { "mem_ctx" : 0.25, },
        ortho=args.ortho,
        emulate=args.emulate,
        check=args.check,
        decay=args.decay)

def check_wasted(net):
    print()
    print("Analyzing memory space...")
    print()

    mem = net.get_layer("mem")
    mem_auto_conn = net.get_connection("mem", "mem", "auto")
    mem_hetero_conn = net.get_connection("mem", "mem", "hetero")
    lex_conn = net.get_connection("lex", "mem", "hetero")
    mem_ctx_conn = net.get_connection("mem_ctx", "mem", "hetero")

    mem_states = [k[1] for k in mem_auto_conn.mappings.keys()]
    labels = {
        m : lex_conn.mappings[State(mem, sym=m).tuple()].get_symbol()
        for m in mem_states }

    conses = {}

    for k,v in labels.items():
        if v == "#LIST":
            ctx_state = mem_ctx_conn.lookup(State(mem, sym=k))
            state = State(mem, sym=k, ctx_state=ctx_state)
            car = mem_hetero_conn.lookup(state)
            cdr = mem_hetero_conn.lookup(car)
            if car.get_symbol() is None or cdr.get_symbol() is None:
                raise RuntimeError
            conses[k] = (car.get_symbol(), cdr.get_symbol())

    def trace(k):
        if k in conses:
            car, cdr = conses[k]
            return (trace(car),) + trace(cdr)
        elif labels[k] == "NIL":
            return ()
        else:
            return labels[k]

    traces = { k : trace(k) for k,v in labels.items() if v == "#LIST" }
    inv_traces = { }

    for k,v in traces.items():
        inv_traces[v] = inv_traces.get(v, []) + [k]

    wasted = 0
    for v,k in inv_traces.items():
        if len(k) > 1:
            wasted += len(k) - 1
            print("%3d : %s" % (len(k), v))

    lists = tuple(v for v in labels.values() if v == "#LIST")
    funcs = tuple(v for v in labels.values() if v == "#FUNCTION")
    syms = tuple(v for v in labels.values() if v[0] != "#")

    print("Memory states:", len(labels))
    print("Symbols:", len(syms))
    print("Functions:", len(funcs))
    print("Lists:", len(lists))
    print("Wasted:", wasted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default="y",
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
    parser.add_argument('-w', action='store_true', default=False,
                        dest='wasted',
                        help='analyze memory space and check for redundant states')
    args = parser.parse_args()

    print("test: ", args.t)
    print("args: ")
    for arg in vars(args):
         print("  %10s : %s" % (arg, getattr(args, arg)))
    print()

    if args.decay < 0.0 or args.decay > 1.0:
        raise ValueError("Decay must be between 0 and 1")

    try:
        net, (timesteps, output) = {
            "y": test_y,
            "cfg": test_cfg,
            "cfg_mini": test_cfg_mini,
            "cfg_new": test_cfg_new,
            "reverse": test_reverse,
            "stack": test_stack,
            "boolean": test_boolean,
            "medium": test_medium,
            "small": test_small,
            "read": test_read,
            "hash": test_hash,
            "unify": test_unify,
            "custom": test_custom,
            "file": test_file,
        }[args.t](args=args)
        print("Ran test '%s' in %d timesteps" % (args.t, timesteps))
        print("Output:")
        print(" ".join(o[1] for o in output))
    except KeyError:
        raise ValueError("Unrecognized test: %s" % args.t)

    if args.wasted:
        check_wasted(net)




# Works without ortho stacks!
# Investigate ctx_lam
# Investigate decoding masked patterns
#
# Size of stacks (64 vs 256)
# mem auto malloc for function call cons
# mem_stabil for stack mem retrieval


"""

TODO:
   * parse specification
   *
   * learn auto-associative mappings for masked to unmasked attractors
   *   after each transition is learned
   *
   * use shared ctx for everything?
   *
   * investigate long term memory
   *   - make flashed weights into initial LTM values
   *   - decay STM matrix toward LTM matrix
   *   - optional: drift LTM toward STM slowly
   *   - parameterize STM length
   *      + like reservoir spectral radius
   *      + experiment with limited STM
   *      + chunking combines STM elements into single pointer to LTM element
   *      + ways to identify existing compositions
   *          ~ somehow check prior to cons
   *          ~ learned associations between cons and cdr of proposed cons node?
   *          ~ if memory is immutable, the existing cons cell can be returned
   *
   *
   * separate out ctx_lam and layer size specifications
   *
   *
   * generalize network construction process
   * parameterize everything
   *   - activator
   *   - connection arguments
   *   - comparison circuit
   *   - context circuits
   *
   * language for expressing the gen_*_mappings functions
"""
