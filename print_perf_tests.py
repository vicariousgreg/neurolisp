from os import system
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

KEYS = (
    "mem_size",
    "bind_size",
    "mem",
    "mem0",
    "mem1",
    "time",
    "parse",
    "execute",
    "kernel_activate",
    "kernel_learn",
    "num_neurons",
    "num_weights",
    "time_per_mem",
)

HEADER = " ".join("%14s" % k for k in KEYS)

def format_line(out_dict):
    s = []
    for k in KEYS:
        v = out_dict[k]
        if type(v) is float:
            s.append("%14.7f" % v)
        else:
            s.append("%14s" % str(v))
    return " ".join(s)

def extract(dual_devices, half, fast, mem_size, bind_size):
    path = "./test_data/perf_data/%s/%s/%s/" % (
        "dual" if dual_devices else "single",
        "half" if half else "single",
        "fast" if fast else "slow")
    filename = "%s/mem%d_bind%d.txt" % (path, mem_size, bind_size)

    try:
        out = { k : 0 for k in KEYS }
        out["mem_size"] = mem_size
        out["bind_size"] = bind_size

        for line in open(filename, "r").readlines():
            if "GB" in line:
                gb = float(line.split()[-2][1:])
                dev = int(line[0])
                out["mem"+str(dev)] = int(gb * 1024)
                out["mem"] += int(gb * 1024)
            elif "Out:" in line:
                if "executing" in line:
                    out["parse"] = float(line.split()[-1])
                elif "complete" in line:
                    t = float(line.split()[-1])
                    out["time"] = t
                    out["execute"] = t - out["parse"]
            elif line.startswith("Total:"):
                num_neurons, num_weights = line.split()[-2:]
                out["num_neurons"] = int(num_neurons)
                out["num_weights"] = int(num_weights)
            elif len(line.split()) > 5 and line.split()[-4] == ":":
                s = line.split()
                if s[:4] == ["mem", "activate", "mem", "hetero"]:
                    out["kernel_activate"] = float(s[-2])
                elif s[:4] == ["mem", "learn", "mem", "hetero"]:
                    out["kernel_learn"] = float(s[-2])
        if out["parse"] * out["execute"] == 0.:
            return None
        else:
            out["time_per_mem"] = out["time"] / out["mem"]
            return out
    except FileNotFoundError: return None

if __name__ == '__main__':
    mem_sizes = (
        # MEM   BIND
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
        45000,
        50000,
        55000,
        60000,
    )

    bind_sizes = (
        # MEM   BIND
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
        45000,
        50000,
        55000,
        60000,
        65000,
        70000,
    )

    params = {
        (True, True, True) : "all",
        (False, True, True) : "fast half",
        (True, False, True) : "fast dual",
        (False, False, True) : "fast",

        (True, True, False) : "dual half",
        (False, True, False) : "half",
        (True, False, False) : "dual",
        (False, False, False) : "none",
    }
    mem_max = 11800

    mem_data = {}
    bind_data = {}

    for dual_devices, half, fast in params:
        mem_dict = { k : {} for k in KEYS }
        mem_data[dual_devices] = mem_data.get(dual_devices, {})
        mem_data[dual_devices][half] = mem_data[dual_devices].get(half, {})
        mem_data[dual_devices][half][fast] = mem_data[dual_devices][half].get(fast, mem_dict)

        bind_dict = { k : {} for k in KEYS }
        bind_data[dual_devices] = bind_data.get(dual_devices, {})
        bind_data[dual_devices][half] = bind_data[dual_devices].get(half, {})
        bind_data[dual_devices][half][fast] = bind_data[dual_devices][half].get(fast, bind_dict)

        filename = ("%s_%s_%s" % (
            "dual" if dual_devices else "single",
            "half" if half else "single",
            "fast" if fast else "slow"))

        print("\\begin{filecontents}{%s_mem.dat}" % filename)
        print(HEADER)
        for mem_size in mem_sizes:
            out = extract(dual_devices, half, fast, mem_size, 10000)
            if out is not None:
                for k,v in out.items():
                    mem_dict[k][mem_size] = v
                overload = "X" if any(x > mem_max for x in (out["mem0"], out["mem1"])) else ""
                tup = tuple(out[k] for k in ("parse", "execute", "mem0", "mem1"))
                #print("%5d %s   %s" % (mem_size, "%12f %12f    | %6d %6d" % tup, overload))
                print(format_line(out))
        print("\\end{filecontents}")

        print("\\begin{filecontents}{%s_bind.dat}" % filename)
        print(HEADER)
        for bind_size in bind_sizes:
            out = extract(dual_devices, half, fast, 10000, bind_size)
            if out is not None:
                for k,v in out.items():
                    bind_dict[k][bind_size] = v
                overload = "X" if any(x > mem_max for x in (out["mem0"], out["mem1"])) else ""
                tup = tuple(out[k] for k in ("parse", "execute", "mem0", "mem1"))
                #print("%5d %s   %s" % (bind_size, "%12f %12f    | %6d %6d" % tup, overload))
                print(format_line(out))
        print("\\end{filecontents}")
        print()

    fig, axs = plt.subplots(3, 2, sharey="row")
    for ax in axs.flat:
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())

    for (dual_devices, half, fast),label in params.items():
        if fast: continue

        mem_dict = mem_data[dual_devices][half][fast]
        style = "dotted" if fast else "solid"
        marker = "x"
        cap = mem_max * (2 if dual_devices else 1)

        sizes = [x for x in mem_sizes if x in mem_dict["parse"]]
        axs[0,0].plot(sizes, [mem_dict["parse"][s] for s in sizes], label=label, linestyle=style, marker=marker)
        axs[1,0].plot(sizes, [mem_dict["execute"][s] for s in sizes], label=label, linestyle=style, marker=marker)

        #axs[2,0].plot(sizes, [mem_dict["mem"][s] / cap for s in sizes], label=label, marker=marker)
        #axs[2,0].bar([i for i in range(len(sizes))], [mem_dict["mem"][s] / cap for s in sizes], label=label)
        if not dual_devices:
            axs[2,0].plot(sizes, [mem_dict["mem"][s] for s in sizes], label=label, marker=marker)
        else:
            axs[2,0].plot(sizes, [mem_dict["mem0"][s] for s in sizes], label=label, marker=marker)
            axs[2,0].plot(sizes, [mem_dict["mem1"][s] for s in sizes], label=label, marker=marker)

        bind_dict = bind_data[dual_devices][half][fast]

        sizes = [x for x in bind_sizes if x in bind_dict["parse"]]
        axs[0,1].plot(sizes, [bind_dict["parse"][s] for s in sizes], label=label, linestyle=style, marker=marker)
        axs[1,1].plot(sizes, [bind_dict["execute"][s] for s in sizes], label=label, linestyle=style, marker=marker)

        #axs[2,1].plot(sizes, [bind_dict["mem"][s] / cap for s in sizes], label=label, marker=marker)
        #axs[2,1].bar([i for i in range(len(sizes))], [bind_dict["mem"][s] / cap for s in sizes], label=label)
        if not dual_devices:
            axs[2,1].plot(sizes, [bind_dict["mem"][s] for s in sizes], label=label, marker=marker)

    axs[0,0].legend()
    axs[2,0].legend()

    #axs[2,0].axhline(1., linestyle="dotted", color="black")
    #axs[2,1].axhline(1., linestyle="dotted", color="black")

    axs[2,0].axhline(mem_max, linestyle="dotted", color="black")
    axs[2,0].axhline(2*mem_max, linestyle="dotted", color="black")
    axs[2,1].axhline(mem_max, linestyle="dotted", color="black")
    axs[2,1].axhline(2*mem_max, linestyle="dotted", color="black")

    plt.show()
