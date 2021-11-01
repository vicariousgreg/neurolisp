job_list = [
    "python3 exp_neurolisp.py -o -t suite --dump --path=./test_data/suite_data/",

    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=2000 --dump --path=./test_data/unify_data/mem_test/",
    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=2500 --dump --path=./test_data/unify_data/mem_test/",
    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=3000 --dump --path=./test_data/unify_data/mem_test/",
    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=3500 --dump --path=./test_data/unify_data/mem_test/",
    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=4000 --dump --path=./test_data/unify_data/mem_test/",
    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=4500 --dump --path=./test_data/unify_data/mem_test/",
    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=5000 --dump --path=./test_data/unify_data/mem_test/",
    "python3 exp_neurolisp.py -o -t unify_mem --mem_size=5500 --dump --path=./test_data/unify_data/mem_test/",

    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=100 --bind_ctx_lam 0.25 --dump --path=./test_data/unify_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=200 --bind_ctx_lam 0.25 --dump --path=./test_data/unify_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=300 --bind_ctx_lam 0.25 --dump --path=./test_data/unify_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=400 --bind_ctx_lam 0.25 --dump --path=./test_data/unify_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=500 --bind_ctx_lam 0.25 --dump --path=./test_data/unify_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=600 --bind_ctx_lam 0.25 --dump --path=./test_data/unify_data/bind_test/quarter/",

    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=100 --bind_ctx_lam 0.125 --dump --path=./test_data/unify_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=200 --bind_ctx_lam 0.125 --dump --path=./test_data/unify_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=300 --bind_ctx_lam 0.125 --dump --path=./test_data/unify_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=400 --bind_ctx_lam 0.125 --dump --path=./test_data/unify_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=500 --bind_ctx_lam 0.125 --dump --path=./test_data/unify_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=600 --bind_ctx_lam 0.125 --dump --path=./test_data/unify_data/bind_test/eighth/",

    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=100 --bind_ctx_lam 0.5 --dump --path=./test_data/unify_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=200 --bind_ctx_lam 0.5 --dump --path=./test_data/unify_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=300 --bind_ctx_lam 0.5 --dump --path=./test_data/unify_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=400 --bind_ctx_lam 0.5 --dump --path=./test_data/unify_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=500 --bind_ctx_lam 0.5 --dump --path=./test_data/unify_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t unify_bind --bind_size=600 --bind_ctx_lam 0.5 --dump --path=./test_data/unify_data/bind_test/half/",


    "python3 exp_neurolisp.py -o -t list_mem --mem_size=300 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_mem --mem_size=600 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_mem --mem_size=900 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_mem --mem_size=1200 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_mem --mem_size=1500 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_mem --mem_size=1800 --dump --path=./test_data/list_data/",

    "python3 exp_neurolisp.py -o -t list_lex --lex_size=300 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_lex --lex_size=600 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_lex --lex_size=900 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_lex --lex_size=1200 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_lex --lex_size=1500 --dump --path=./test_data/list_data/",
    "python3 exp_neurolisp.py -o -t list_lex --lex_size=1800 --dump --path=./test_data/list_data/",

    "python3 exp_neurolisp.py -o -t pcfg_mem --mem_size=3000 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/mem_test/",
    "python3 exp_neurolisp.py -o -t pcfg_mem --mem_size=3500 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/mem_test/",
    "python3 exp_neurolisp.py -o -t pcfg_mem --mem_size=4000 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/mem_test/",
    "python3 exp_neurolisp.py -o -t pcfg_mem --mem_size=4500 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/mem_test/",
    "python3 exp_neurolisp.py -o -t pcfg_mem --mem_size=5000 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/mem_test/",
    "python3 exp_neurolisp.py -o -t pcfg_mem --mem_size=5500 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/mem_test/",

    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=100 --dump --bind_ctx_lam 0.5 --path=./test_data/pcfg_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=200 --dump --bind_ctx_lam 0.5 --path=./test_data/pcfg_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=300 --dump --bind_ctx_lam 0.5 --path=./test_data/pcfg_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=400 --dump --bind_ctx_lam 0.5 --path=./test_data/pcfg_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=500 --dump --bind_ctx_lam 0.5 --path=./test_data/pcfg_data/bind_test/half/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=600 --dump --bind_ctx_lam 0.5 --path=./test_data/pcfg_data/bind_test/half/",

    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=100 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=200 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=300 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=400 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=500 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/bind_test/quarter/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=600 --dump --bind_ctx_lam 0.25 --path=./test_data/pcfg_data/bind_test/quarter/",

    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=100 --dump --bind_ctx_lam 0.125 --path=./test_data/pcfg_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=200 --dump --bind_ctx_lam 0.125 --path=./test_data/pcfg_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=300 --dump --bind_ctx_lam 0.125 --path=./test_data/pcfg_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=400 --dump --bind_ctx_lam 0.125 --path=./test_data/pcfg_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=500 --dump --bind_ctx_lam 0.125 --path=./test_data/pcfg_data/bind_test/eighth/",
    "python3 exp_neurolisp.py -o -t pcfg_bind --bind_size=600 --dump --bind_ctx_lam 0.125 --path=./test_data/pcfg_data/bind_test/eighth/",


    "python3 exp_neurolisp.py -o -t bind_many --bind_size=1000 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_many/eighth/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=2000 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_many/eighth/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=3000 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_many/eighth/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=4000 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_many/eighth/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=5000 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_many/eighth/",

    "python3 exp_neurolisp.py -o -t bind_many --bind_size=1000 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_many/half/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=2000 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_many/half/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=3000 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_many/half/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=4000 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_many/half/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=5000 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_many/half/",

    "python3 exp_neurolisp.py -o -t bind_many --bind_size=1000 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_many/quarter/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=2000 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_many/quarter/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=3000 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_many/quarter/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=4000 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_many/quarter/",
    "python3 exp_neurolisp.py -o -t bind_many --bind_size=5000 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_many/quarter/",


    "python3 exp_neurolisp.py -o -t bind_one --bind_size=100 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_one/quarter/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=200 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_one/quarter/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=300 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_one/quarter/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=400 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_one/quarter/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=500 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_one/quarter/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=600 --dump --bind_ctx_lam 0.25 --path=./test_data/bind_data/bind_one/quarter/",

    "python3 exp_neurolisp.py -o -t bind_one --bind_size=100 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_one/eighth/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=200 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_one/eighth/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=300 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_one/eighth/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=400 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_one/eighth/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=500 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_one/eighth/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=600 --dump --bind_ctx_lam 0.125 --path=./test_data/bind_data/bind_one/eighth/",

    "python3 exp_neurolisp.py -o -t bind_one --bind_size=100 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_one/half/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=200 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_one/half/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=300 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_one/half/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=400 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_one/half/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=500 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_one/half/",
    "python3 exp_neurolisp.py -o -t bind_one --bind_size=600 --dump --bind_ctx_lam 0.5 --path=./test_data/bind_data/bind_one/half/",
]

from os import system

if __name__ == '__main__':
    for job in job_list:
        command = "%s >/dev/null" % job
        print(command)
        system(command)
