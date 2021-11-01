import pickle
import argparse
import matplotlib.pyplot as plt
from pcfg import lispify_pcfg
from exp_neurolisp import run_test, pcfg_prog
from random import sample

pcfg_tests = lispify_pcfg("./test_data/pcfg_data/pcfg_source.txt",
    "./test_data/pcfg_data/pcfg_target.txt")

def load_results(filename):
    a, sz, prog, results = pickle.load(open(filename, "rb"))
    return results

def plot_results(results, filtered=None):
    mems = [m["mem", "mem", "auto"] for inp,ref,out,c,t,m in results]
    binds = [m["mem", "bind", "hetero"] for inp,ref,out,c,t,m in results]
    transits = [m["mem", "mem", "hetero"] for inp,ref,out,c,t,m in results]
    symbols = [m["mem", "lex", "hetero"] for inp,ref,out,c,t,m in results]
    namespaces = [m["bind", "bind", "hetero"] for inp,ref,out,c,t,m in results]
    stacks = [m["op", "stack", "hetero"] for inp,ref,out,c,t,m in results]
    data_stacks = [m["mem", "data_stack", "hetero"] for inp,ref,out,c,t,m in results]

    if filtered:
        mems = [mems[i] for i in filtered]
        binds = [binds[i] for i in filtered]
        transits = [transits[i] for i in filtered]
        symbols = [symbols[i] for i in filtered]
        namespaces = [namespaces[i] for i in filtered]
        stacks = [stacks[i] for i in filtered]
        data_stacks = [data_stacks[i] for i in filtered]

    fig, axs = plt.subplots(7)

    axs[0].set_title("Memories")
    axs[0].hist(mems, bins=10)

    axs[1].set_title("Transits")
    axs[1].hist(transits, bins=10)

    axs[2].set_title("Symbols")
    axs[2].hist(symbols, bins=10)

    axs[3].set_title("Namespaces")
    axs[3].hist(namespaces, bins=10)

    axs[4].set_title("Bindings")
    axs[4].hist(binds, bins=10)

    axs[5].set_title("Runtime stack")
    axs[5].hist(stacks, bins=10)

    axs[6].set_title("Data stack")
    axs[6].hist(data_stacks, bins=10)
    #for ax in axs:
    #    ax.set_xlim([0,1000])
    plt.show()

def filter_tests(results):
    mems = [m["mem", "mem", "auto"] for inp,ref,out,c,t,m in results]
    binds = [m["mem", "bind", "hetero"] for inp,ref,out,c,t,m in results]
    transits = [m["mem", "mem", "hetero"] for inp,ref,out,c,t,m in results]
    symbols = [m["mem", "lex", "hetero"] for inp,ref,out,c,t,m in results]
    namespaces = [m["bind", "bind", "hetero"] for inp,ref,out,c,t,m in results]
    stacks = [m["op", "stack", "hetero"] for inp,ref,out,c,t,m in results]
    data_stacks = [m["mem", "data_stack", "hetero"] for inp,ref,out,c,t,m in results]

    return [i for i in range(len(mems))
        if mems[i] >= 250 and mems[i] < 350
        and namespaces[i] <= 128
        and stacks[i] <= 64 and data_stacks[i] <= 64]

def sample_tests(results, indices, num=100, param="mem", rng=[250,350]):
    if param == "mem":
        params = [m["mem", "mem", "auto"] for inp,ref,out,c,t,m in results]
    elif param == "bind":
        params = [m["mem", "bind", "hetero"] for inp,ref,out,c,t,m in results]
    else:
        raise ValueError

    bins = [[] for i in range(10)]
    for i in indices:
        if params[i] >= rng[0] and params[i] < rng[1]:
            bin_index = int(10 * (params[i] - rng[0]) / (rng[1] - rng[0]))
            bins[bin_index].append(i)

    sample_indices = []
    for b in bins:
        sample_indices.extend(sample(b, int(num/10)))
    return sample_indices

def print_results(path_template, sizes, x):
    for index, size in enumerate(sizes):
        try:
            results = load_results(path_template % size)
            #print(size, sum(r[3] for r in results))
            #print(size, "".join("X" if r[3] else " " for r in results))
            blocks = [results[i:i+20] for i in range(0, len(results), 20)]
            block_results = [5*sum(r[3] for r in block) for block in blocks]
            #for i, r in enumerate(results):
            #    if not r[3]:
            #        print(i)
            #        print(r[0])
            #        print(r[1])
            #        print(r[2])
            #        print(r[-1]["bind", "bind", "hetero"])
            #        print(r[-1]["mem", "bind", "hetero"])
            #        #for k,v in r[-1].items():
            #        #    print(k,v)
            #        print()
            print("        %", size, block_results)

            plt.plot(x, block_results)
            for a,b in zip(x, block_results):
                print("        (%s, %s)" % (a,b))

            #x = [i + (0.15 * index) for i in range(len(block_results))]
            #plt.bar(x, block_results, 0.15)
        except FileNotFoundError: pass
        except ValueError: pass
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    '''
    results = load_results("test_data/pcfg_data/pcfg_emulate.p")
    filtered = filter_tests(results)
    mem_filtered = sample_tests(results, filtered, num=200, param="mem", rng=[250,350])
    bind_filtered = sample_tests(results, filtered, num=200, param="bind", rng=[20,120])
    #plot_results(results, None)
    #plot_results(results, filtered)
    #pickle.dump(mem_filtered, open("./test_data/pcfg_data/mem_filtered_200_indices.p", "wb"))
    #pickle.dump(bind_filtered, open("./test_data/pcfg_data/bind_filtered_200_indices.p", "wb"))
    '''

    '''
    pcfg_tests = lispify_pcfg("./test_data/pcfg_data/pcfg_source.txt",
        "./test_data/pcfg_data/pcfg_target.txt")

    pcfg_tests_sample_mem = [pcfg_tests[i] for i in
        pickle.load(open("./test_data/pcfg_data/mem_filtered_200_indices.p", "rb"))]
    pcfg_tests_sample_bind = [pcfg_tests[i] for i in
        pickle.load(open("./test_data/pcfg_data/bind_filtered_200_indices.p", "rb"))]

    results = load_results("test_data/pcfg_data/pcfg_emulate.p")
    plot_results(results, pcfg_tests_sample_mem)
    plot_results(results, pcfg_tests_sample_bind)
    '''

    print("=" * 80)
    print("Unify tests:")
    print()
    print("Memory tests:")
    print_results("test_data/unify_data/mem_test/unify_mem_%d.p",
        [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500],
        [6, 8, 10, 12, 14])

    print()
    print("Binding tests:")
    for lam in ["eighth", "quarter", "half"]:
        print("bind_ctx_lam = ", lam)
        print_results("test_data/unify_data/bind_test/%s/unify_bind_%s.p" % (lam, "%d"),
            [100, 200, 300, 400, 500, 600],
            [6, 8, 10, 12, 14])
        print()

    print("=" * 80)
    print("PCFG tests:")
    print()
    print("Memory tests:")
    print_results("test_data/pcfg_data/mem_test/pcfg_mem_%d.p",
        [3000, 3500, 4000, 4500, 5000, 5500],
        [250, 260, 270, 280, 290, 300, 310, 320, 330, 340])

    print()
    print("Binding tests:")
    for lam in ["eighth", "quarter", "half"]:
        print("bind_ctx_lam = ", lam)
        print_results("test_data/pcfg_data/bind_test/%s/pcfg_bind_%s.p" % (lam, "%d"),
            [100, 200, 300, 400, 500, 600],
            [20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
        print()

    print("=" * 80)
    print("LIST tests:")
    print()
    print("Memory tests:")
    print_results("test_data/list_data/list_mem_%d.p",
        [300, 600, 900, 1200, 1500, 1800],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print()
    print("Lex tests:")
    print_results("test_data/list_data/list_lex_%d.p",
        [300, 600, 900, 1200, 1500, 1800],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    print("=" * 80)
    print("BIND tests:")
    print()
    print("Many tests:")
    for lam in ["eighth", "quarter", "half"]:
        print("bind_ctx_lam = ", lam)
        print_results("test_data/bind_data/bind_many/%s/bind_bind_%s.p" % (lam, "%d"),
            [1000, 2000, 3000, 4000, 5000],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print()
    print("One tests:")
    print()
    for lam in ["eighth", "quarter", "half"]:
        print("bind_ctx_lam = ", lam)
        print_results("test_data/bind_data/bind_one/%s/bind_bind_%s.p" % (lam, "%d"),
            [100, 200, 300, 400, 500, 600],
            [10, 20, 30, 40, 50, 60])
