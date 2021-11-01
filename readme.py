from build_neurolisp import test, preprocess

# Code for recursive function that reverses a list
code = '''
    (defun rev-helper (pre post)
        (if pre
            (rev-helper
                (cdr pre)
                (cons (car pre) post))
            post))
    (defun reverse (x) (rev-helper x NIL))

    (reverse (quote (a b c d e)))
'''

# Preprocessing step breaks code into sequence of symbols for model input
inputs = preprocess(code)

test(inputs=inputs,
    t=10000000,           # maximum timesteps (timeout)
    verbose=True,         # print verbose output
    capacity = {          # memory capacity by region
        "mem": 256+32,
        "lex": 64,
        "bind": 128,
        "stack": 128,
        "data_stack": 32,
    },
    ctx_lam = {           # context density parameters
        "mem_ctx" : 0.25,
        "bind_ctx" : 0.125,
    })
