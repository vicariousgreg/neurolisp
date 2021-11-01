from os import system
from os.path import exists

def ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False):
    path = "./test_data/perf_data/%s/%s/%s/" % (
        "dual" if dual_devices else "single",
        "half" if half else "single",
        "fast" if fast else "slow")
    system("mkdir -p %s" % path)
    filename = "%s/mem%d_bind%d.txt" % (path, mem_size, bind_size)

    devices = "0,1" if dual_devices else "1"

    env = "CUDA_DEVICES=%s CUDA_HALF=%s FAST_KERNELS=%s" % (devices, half, fast)
    pyth =  """ %s python3 perf_neurolisp.py -ovt perf \
        --mem_size %d --bind_size %d >> %s""" % (env, mem_size, bind_size, filename)
    if overwrite:
        system("rm %s" % filename)
    command = """echo %s ; %s""" % (pyth, pyth)
    if not exists(filename):
        #print(filename)
        print(command)
        print()
        system(command)


if __name__ == '__main__':
    sizes_1 = (
        # MEM   BIND
        (10000, 10000),

        (10000, 15000),
        (10000, 20000),
        (10000, 25000),
        (10000, 30000),
        (10000, 35000),
        (10000, 40000),

        (15000, 10000),
        (20000, 10000),
        (25000, 10000),
        (30000, 10000),
    )

    sizes_2 = (
        # MEM   BIND
        (10000, 45000),
        (10000, 50000),

        (35000, 10000),
        (40000, 10000),
    )

    sizes_3 = (
        # MEM   BIND
        (10000, 55000),
        (10000, 60000),

        (45000, 10000),
    )

    sizes_4 = (
        # MEM   BIND
        (50000, 10000),
        (55000, 10000),
    )

    sizes_5 = (
        # MEM   BIND
        (10000, 65000),
        (10000, 70000),
        (60000, 10000),
    )

    params_4 = (
        (True, True, True),
        (True, True, False),
    )

    params_3 = (
        (False, True, True),
        (False, True, False),
    )

    params_2 = (
        (True, False, True),
        (True, False, False),
    )

    params_1 = (
        (False, False, True),
        (False, False, False),
    )

    # FAST TESTS
    for mem_size, bind_size in sizes_1:
        for dual_devices, half, fast in (params_4 + params_3 + params_2 + params_1):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)
    for mem_size, bind_size in sizes_2:
        for dual_devices, half, fast in (params_4 + params_3 + params_2):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)
    for mem_size, bind_size in sizes_3:
        for dual_devices, half, fast in (params_4 + params_3):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)
    for mem_size, bind_size in sizes_4:
        for dual_devices, half, fast in (params_4):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)

    # MODERATE TESTS
    for mem_size, bind_size in sizes_2:
        for dual_devices, half, fast in (params_1):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)
    for mem_size, bind_size in sizes_3:
        for dual_devices, half, fast in (params_2):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)
    for mem_size, bind_size in sizes_4:
        for dual_devices, half, fast in (params_3):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)

    for mem_size, bind_size in sizes_3:
        for dual_devices, half, fast in (params_1):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)

    # SLOW TESTS
    for mem_size, bind_size in sizes_5:
        for dual_devices, half, fast in (params_4):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)

    for mem_size, bind_size in sizes_4:
        for dual_devices, half, fast in (params_2):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)

    for mem_size, bind_size in sizes_4:
        for dual_devices, half, fast in (params_1):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)

    for mem_size, bind_size in sizes_5:
        for dual_devices, half, fast in (params_3 + params_2 + params_1):
            ex(dual_devices, half, fast, mem_size, bind_size, overwrite=False)
