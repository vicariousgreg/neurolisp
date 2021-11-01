import numpy as np

class numpyEngine:
    def __init__(self):
        pass

    def cuda_acc(self):
        return False

    def fast(self):
        return False

    def get_weight_dtype(self):
        return np.float32

    def get_num_devices(self):
        return 1

    def get_free_memory(self):
        # TODO: currently unused, added for correspondence with CUDAEngine
        return 0

    def dot(self, x, y, w):
        y[:] = np.dot(w, x)

    def rehebbian(self, x, y, w, norm, diag):
        nonzero = np.sum(x != 0.) - (not diag)
        denom = norm * nonzero

        delta = y - np.multiply((y != 0.), np.dot(w, x))

        dw = np.dot(delta, x.T) / denom
        if not diag:
            np.fill_diagonal(dw, 0.)

        w += dw

    def hebbian(self, x, y, w, norm, diag):
        nonzero = np.sum(x != 0.) - (not diag)
        denom = norm * nonzero

        dw = np.dot(y, x.T) / denom
        if not diag:
            np.fill_diagonal(dw, 0.)

        w += dw

    def dipole(self, x, y, w, norm):
        nonzero = np.sum(x != 0.)
        denom = norm * nonzero
        w[:] = np.dot(y, x.T) / denom

    def decay_weights(self, x, y, w, decay):
        if decay != 1.:
            mask = np.dot((y != 0.), (x != 0.).T)
            w -= np.multiply(mask, (1 - decay) * w)

    def learn(self, learning_rule, x, y, w, norm, diag, decay):
        if learning_rule == "rehebbian":
            f = self.rehebbian
        elif learning_rule == "hebbian":
            f = self.hebbian
        else:
            raise ValueError("Unrecognized learning rule %s" % learning_rule)

        if y.shape[1] == 1 and x.shape[1] == 1:
            self.decay_weights(x, y, w, decay)
            f(x, y, w, norm, diag)
        else:
            for i in range(y.shape[0]):
                x_slice = x[i,:].reshape(-1,1)
                y_slice = y[i,:].reshape(-1,1)
                self.decay_weights(x_slice, y_slice, w, decay)
                f(x_slice, y_slice, w, norm, diag)

    def converge(self, x, y, w, act, timeout=10):
        if timeout < 1:
            raise ValueError

        for t in range(timeout):
            # Swap x and y
            self.dot(y, x, w)

            identical = np.all(np.sign(x) == np.sign(y))
            y[:] = act.f(x)
            if identical: break

    def fast_dot(self, x, y, w,
            ctx_x, ctx_y,
            index_x, index_y,
            from_mask, to_mask):
        self.dot(x, y, w)

    def fast_learn(self, learning_rule,
            x, y, w,
            ctx_x, ctx_y,
            index_x, index_y,
            from_mask, to_mask,
            norm, diag, decay):
        self.learn(learning_rule, x, y, w, norm, diag, decay)

    def fast_converge(self,
            x, y, w,
            ctx_x, ctx_y,
            index_x, index_y,
            from_mask, act, timeout=10):
        self.converge(x, y, w, act, timeout)

    def managed(self, arr=None, shape=None, copy=False, dtype=np.float32):
        if arr is None and shape is None:
            raise ValueError
        elif arr is not None:
            shape = arr.shape

        man = np.zeros(shape, dtype=dtype)
        if copy and (arr is not None):
            man[:] = arr
        return man

    # GPUs use transposed weight matrix for memory access efficiency
    # Host simply returns
    def get_weights(self, w):
        return w

    def synchronize(self, stream=None):
        pass

    def set_device(self, index):
        pass

    def get_device(self):
        return 0

    def make_stream(self):
        return 0

    def set_stream(self, index):
        pass

    def get_stream(self):
        return 0


##########

cuda_half_preamble = """
    #include <cuda_fp16.h>
"""

cuda_single_preamble = """
    typedef float half;
    __device__ __inline float __half2float(float x) { return x; }
    __device__ __inline float __float2half(float x) { return x; }
"""

cuda_kernel_code = """

    extern "C" {

    __global__ void dot(float *x, float *y, half *w,
        int from_size, int to_size)
    {
        int to_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (to_index < to_size) {
            float sum = 0.;

            for (int from_index = 0; from_index < from_size; ++from_index) {
                sum += x[from_index] * __half2float(
                    w[(size_t)from_index * to_size + to_index]);
            }
            y[to_index] = sum;
        }
    }

    __global__ void fast_dot(float *x, float *y, half *w,
        int *index_x, int *index_y,
        int from_index_size, int to_index_size, int to_size,
        int from_mask, int to_mask)
    {
        int to_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (to_index < to_index_size) {
            int adj_to_index = to_mask
                ? index_y[to_index] : to_index;
            float sum = 0.;

            for (int from_index = 0; from_index < from_index_size; ++from_index) {
                int adj_from_index = from_mask
                    ? index_x[from_index] : from_index;
                sum += x[adj_from_index] * __half2float(
                    w[(size_t)adj_from_index * to_size + adj_to_index]);
            }
            y[adj_to_index] = sum;
        }
    }

    #include <stdio.h>
    __global__ void rehebbian(float *x, float *y, half *w,
        int from_size, int to_size, float norm, int diag,
        float decay, int num_patterns)
    {
        int to_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (to_index < to_size) {

            for (int pat_index = 0 ; pat_index < num_patterns ; ++pat_index) {
                int from_offset = pat_index * from_size;
                int to_offset = pat_index * to_size;

                float target = y[to_offset + to_index];

                if (target == 0.) continue;

                float sum = 0.;
                int nonzero = 0;

                for (int from_index = 0; from_index < from_size; ++from_index) {
                    float val = x[from_offset + from_index];
                    size_t w_index = (size_t)from_index * to_size + to_index;
                    float weight = __half2float(w[w_index]);
                    if (val != 0.) {
                        nonzero += 1;
                        if (decay != 1.) {
                            weight = (decay * weight);
                            w[w_index] = __float2half(weight);
                        }
                    }
                    sum += val * weight;
                }

                if (nonzero == 0) continue;

                float delta = target - sum;
                float denom = norm * (nonzero - (not diag));
                float factor = delta / denom;

                for (int from_index = 0; from_index < from_size; ++from_index) {
                    w[(size_t)from_index * to_size + to_index] +=
                        __float2half(factor * x[from_offset + from_index]);
                }
                if (not diag) {
                    w[(size_t)to_index * to_size + to_index] = 0;
                }
            }
        }
    }

    __global__ void fast_rehebbian(float *x, float *y, half *w,
        int *index_x, int *index_y,
        int *from_index_sizes, int *to_index_sizes,
        int from_size, int to_size,
        int from_mask, int to_mask,
        float norm, int diag,
        float decay,
        int num_patterns)
    {
        int to_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (to_index < to_size) {
            for (int pat_index = 0 ; pat_index < num_patterns ; ++pat_index) {
                int to_index_size = to_index_sizes[pat_index];
                if (to_index >= to_index_size) continue;

                int from_offset = pat_index * from_size;
                int to_offset = pat_index * to_size;

                int adj_to_index = to_mask
                    ? index_y[to_offset + to_index] : to_index;

                float sum = 0.;

                int from_index_size = from_index_sizes[pat_index];
                for (int from_index = 0; from_index < from_index_size; ++from_index) {
                    int adj_from_index = from_mask
                        ? index_x[from_offset + from_index] : from_index;
                    float val = x[from_offset + adj_from_index];
                    size_t w_index = (size_t)adj_from_index * to_size + adj_to_index;
                    float weight = __half2float(w[w_index]);

                    if (decay != 1.) {
                        weight = (decay * weight);
                        w[w_index] = __float2half(weight);
                    }

                    sum += val * weight;
                }

                int nonzero = from_index_size;
                if (nonzero == 0) continue;

                float target = y[to_offset + adj_to_index];
                float delta = target - sum;
                float denom = norm * (nonzero - (not diag));
                float factor = delta / denom;

                for (int from_index = 0; from_index < from_index_size; ++from_index) {
                    int adj_from_index = from_mask
                        ? index_x[from_offset + from_index] : from_index;
                    w[(size_t)adj_from_index * to_size + adj_to_index] +=
                        __float2half(factor * x[from_offset + adj_from_index]);
                }
                if (not diag) {
                    w[(size_t)adj_to_index * to_size + adj_to_index] = 0;
                }
            }
        }
    }

    __global__ void hebbian(float *x, float *y, half *w,
        int from_size, int to_size, float norm, int diag,
        float decay, int num_patterns)
    {
        int to_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (to_index < to_size) {

            for (int pat_index = 0 ; pat_index < num_patterns ; ++pat_index) {
                int from_offset = pat_index * from_size;
                int to_offset = pat_index * to_size;

                float target = y[to_offset + to_index];

                if (target == 0.) continue;

                int nonzero = 0;

                for (int from_index = 0; from_index < from_size; ++from_index) {
                    float val = x[from_offset + from_index];
                    size_t w_index = (size_t)from_index * to_size + to_index;
                    float weight = __half2float(w[w_index]);
                    if (val != 0.) {
                        nonzero += 1;
                        if (decay != 1.) {
                            weight = (decay * weight);
                            w[w_index] = __float2half(weight);
                        }
                    }
                }

                float denom = norm * (nonzero - (not diag));
                float factor = target / denom;

                for (int from_index = 0; from_index < from_size; ++from_index)
                    w[from_index * to_size + to_index] +=
                        __float2half(factor * x[from_offset + from_index]);
                if (not diag) w[to_index * to_size + to_index] = 0;
            }
        }
    }

    __global__ void fast_hebbian(float *x, float *y, half *w,
        int *index_x, int *index_y,
        int *from_index_sizes, int *to_index_sizes,
        int from_size, int to_size,
        int from_mask, int to_mask,
        float norm, int diag,
        float decay,
        int num_patterns)
    {
        int to_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (to_index < to_size) {
            for (int pat_index = 0 ; pat_index < num_patterns ; ++pat_index) {
                int to_index_size = to_index_sizes[pat_index];
                if (to_index >= to_index_size) continue;

                int from_offset = pat_index * from_size;
                int to_offset = pat_index * to_size;

                int adj_to_index = to_mask
                    ? index_y[to_offset + to_index] : to_index;

                int from_index_size = from_index_sizes[pat_index];
                for (int from_index = 0; from_index < from_index_size; ++from_index) {
                    int adj_from_index = from_mask
                        ? index_x[from_offset + from_index] : from_index;
                    size_t w_index = (size_t)adj_from_index * to_size + adj_to_index;
                    float weight = __half2float(w[w_index]);

                    if (decay != 1.) {
                        weight = (decay * weight);
                        w[w_index] = __float2half(weight);
                    }
                }

                int nonzero = from_index_size;
                if (nonzero == 0) continue;

                float target = y[to_offset + adj_to_index];
                float denom = norm * (nonzero - (not diag));
                float factor = target / denom;

                for (int from_index = 0; from_index < from_index_size; ++from_index) {
                    int adj_from_index = from_mask
                        ? index_x[from_offset + from_index] : from_index;
                    w[(size_t)adj_from_index * to_size + adj_to_index] +=
                        __float2half(factor * x[from_offset + adj_from_index]);
                }
                if (not diag) w[(size_t)adj_to_index * to_size + adj_to_index] = 0;
            }
        }
    }

    __global__ void dipole(float *x, float *y, half *w,
        int from_size, int to_size, float norm)
    {
        int to_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (to_index < to_size) {
            float target = y[to_index];

            if (target == 0.) return;

            int nonzero = 0;

            for (int from_index = 0; from_index < from_size; ++from_index)
                if (x[from_index] != 0.)
                    nonzero += 1;
            float denom = norm * nonzero;
            float factor = target / denom;

            for (int from_index = 0; from_index < from_size; ++from_index)
                w[(size_t)from_index * to_size + to_index] =
                    __float2half(factor * x[from_index]);
        }
    }

    }

    """


try:
    from pycuda import compiler, gpuarray, _driver
    import pycuda.driver as drv
    from pycuda.tools import clear_context_caches
    from math import ceil
    from os import environ
    import atexit

    drv.init()

    class CUDAEngine:
        def __init__(self, devices=None, half_precision=True, fast_kernels=True):

            found_devices = range(drv.Device.count())
            if devices is None:
                devices = found_devices
            elif any(d not in found_devices for d in devices):
                raise ValueError
            self.devices = devices

            self.contexts = []
            self.mods = []
            self.streams = []
            self.active_device = None
            self.active_context = None
            self.active_mod = None
            self.active_stream = None

            self.fast_kernels = fast_kernels
            self.half_precision = half_precision
            code = (cuda_half_preamble if half_precision else cuda_single_preamble) + cuda_kernel_code
            for i in self.devices:
                self.contexts.append(drv.Device(i).make_context())
                self.mods.append(compiler.SourceModule(code, no_extern_c=True))

            self.set_device(self.devices[0])
            atexit.register(CUDAEngine.cleanup, self)

        def cuda_acc(self):
            return True

        def fast(self):
            return self.fast_kernels

        def get_weight_dtype(self):
            return np.float16 if self.half_precision else np.float32

        def get_num_devices(self):
            return len(self.devices)

        def get_free_memory(self):
            free = []
            for d in self.devices:
                self.set_device(d)
                free.append(drv.mem_get_info()[0])
            return free

        # Upon deletion, clear CUDA contexts
        def cleanup(self):
            for ctx in self.contexts:
                ctx.detach()
            self.contexts = []
            clear_context_caches()

        def set_device(self, dev):
            if dev not in self.devices:
                raise RuntimeError
            index = self.devices.index(dev)

            self.active_device = dev
            if self.active_context: self.active_context.pop()
            self.active_context = self.contexts[index]
            self.active_context.push()
            self.active_mod = self.mods[index]

        def get_device(self):
            return self.active_device

        def make_stream(self):
            self.streams.append(drv.Stream())
            return len(self.streams) - 1

        def set_stream(self, index):
            if index is None:
                self.active_stream = None
            else:
                self.active_stream = self.streams[index]

        def get_stream(self):
            return self.active_stream

        def synchronize(self, stream=None):
            if stream is not None:
                self.streams[stream].synchronize()
            else:
                self.active_context.synchronize()

        #########


        def dot(self, gpu_x, gpu_y, gpu_w):

            to_size = gpu_y.shape[0] if gpu_y.shape[1] == 1 else gpu_y.shape[1]
            from_size = gpu_x.shape[0] if gpu_x.shape[1] == 1 else gpu_x.shape[1]

            self.active_mod.get_function("dot")(
                gpu_x, gpu_y, gpu_w,
                np.int32(from_size), np.int32(to_size),
                stream = self.active_stream,
                block = (128,1,1), grid = (ceil(to_size / 128),1,1))

        def learn(self, learning_rule, gpu_x, gpu_y, gpu_w, norm, diag=True, decay=1.):
            if gpu_y.shape[1] == 1 and gpu_x.shape[1] == 1:
                num_patterns = 1
                to_size = gpu_y.shape[0]
                from_size = gpu_x.shape[0]
            else:
                num_patterns = gpu_y.shape[0]
                to_size = gpu_y.shape[1]
                from_size = gpu_x.shape[1]

            try:
                func = self.active_mod.get_function(learning_rule)
            except _driver.LogicError:
                raise ValueError("Unrecognized learning rule %s" % learning_rule)

            func(
                gpu_x, gpu_y, gpu_w,
                np.int32(from_size), np.int32(to_size),
                np.float32(norm), np.int32(diag),
                np.float32(decay),
                np.int32(num_patterns),
                stream = self.active_stream,
                block = (128,1,1), grid = (ceil(to_size / 128),1,1))

        def dipole(self, gpu_x, gpu_y, gpu_w, norm):
            self.active_mod.get_function("dipole")(
                gpu_x, gpu_y, gpu_w,
                np.int32(gpu_x.size), np.int32(gpu_y.size),
                np.float32(norm),
                stream = self.active_stream,
                block = (128,1,1), grid = (ceil(gpu_y.shape[0] / 128),1,1))

        def converge(self, gpu_x, gpu_y, gpu_w, act, timeout=10):
            if timeout < 1:
                raise ValueError

            for t in range(timeout):
                # Swap x and y
                self.active_mod.get_function("dot")(
                    gpu_y, gpu_x, gpu_w,
                    np.int32(gpu_y.size), np.int32(gpu_x.size),
                    stream = self.active_stream,
                    block = (128,1,1), grid = (ceil(gpu_x.shape[0] / 128),1,1))

                self.synchronize(self.active_stream)

                identical = np.all(np.sign(gpu_x) == np.sign(gpu_y))
                gpu_y[:] = act.f(gpu_x)
                if identical: break

        def _red(self, ctx, indices):
            if ctx.shape[1] == 1:
                ndx = np.where(ctx[:,0] != 0.)[0]
                count = ndx.shape[0]
                indices[:count,0] = ndx[:]
                return [count]
            else:
                counts = []
                for i in range(ctx.shape[0]):
                    ndx = np.where(ctx[i,:] != 0.)[0]
                    count = ndx.shape[0]
                    indices[i,:count] = ndx[:]
                    counts.append(count)
                return counts

        def fast_dot(self, gpu_x, gpu_y, gpu_w,
                gpu_ctx_x, gpu_ctx_y,
                gpu_index_x, gpu_index_y,
                from_mask, to_mask):

            from_index_size = self._red(gpu_ctx_x, gpu_index_x)[0] if from_mask else gpu_x.shape[0]
            to_index_size = self._red(gpu_ctx_y, gpu_index_y)[0] if to_mask else gpu_y.shape[0]

            if to_index_size > 0 and from_index_size > 0:
                to_size = gpu_y.shape[0]
                self.active_mod.get_function("fast_dot")(
                    gpu_x, gpu_y, gpu_w,
                    gpu_index_x, gpu_index_y,
                    np.int32(from_index_size), np.int32(to_index_size), np.int32(to_size),
                    np.int32(from_mask), np.int32(to_mask),
                    stream = self.active_stream,
                    block = (128,1,1), grid = (ceil(to_index_size / 128),1,1))

        def fast_learn(self, learning_rule,
                gpu_x, gpu_y, gpu_w,
                gpu_ctx_x, gpu_ctx_y,
                gpu_index_x, gpu_index_y,
                from_mask, to_mask,
                norm, diag, decay):
            try:
                func = self.active_mod.get_function("fast_" + learning_rule)
            except _driver.LogicError:
                raise ValueError("Unrecognized learning rule %s" % learning_rule)

            if gpu_y.shape[1] == 1 and gpu_x.shape[1] == 1:
                num_patterns = 1
                to_size = gpu_y.shape[0]
                from_size = gpu_x.shape[0]
            else:
                num_patterns = gpu_y.shape[0]
                to_size = gpu_y.shape[1]
                from_size = gpu_x.shape[1]

            from_index_sizes = (self._red(gpu_ctx_x, gpu_index_x)
                if from_mask else [from_size for _ in range(num_patterns)])
            to_index_sizes = (self._red(gpu_ctx_y, gpu_index_y)
                if to_mask else [to_size for _ in range(num_patterns)])

            kernel_size = max(to_index_sizes)

            if kernel_size > 0 and max(from_index_sizes) > 0:
                from_index_sizes = gpuarray.to_gpu(np.array(from_index_sizes, dtype=np.int32))
                to_index_sizes = gpuarray.to_gpu(np.array(to_index_sizes, dtype=np.int32))

                # If mask flag is disabled, index arrays dont get accessed
                # This is simply to placate the kernel call with real pointers
                if not from_mask: gpu_index_x = from_index_sizes
                if not to_mask: gpu_index_y = to_index_sizes

                func(
                    gpu_x, gpu_y, gpu_w,

                    gpu_index_x, gpu_index_y,
                    from_index_sizes, to_index_sizes,
                    np.int32(from_size), np.int32(to_size),
                    np.int32(from_mask), np.int32(to_mask),
                    np.float32(norm), np.int32(diag),
                    np.float32(decay),
                    np.int32(num_patterns),
                    stream = self.active_stream,
                    block = (128,1,1), grid = (ceil(kernel_size / 128),1,1))

        def fast_converge(self,
                gpu_x, gpu_y, gpu_w,
                gpu_ctx_x, gpu_ctx_y,
                gpu_index_x, gpu_index_y,
                from_mask, act, timeout=10):
            # NOTE: x and y are flipped here from activate
            self.fast_dot(gpu_y, gpu_x,
                gpu_w,
                gpu_ctx_y, gpu_ctx_x,
                gpu_index_y, gpu_index_x,
                from_mask, False)
            #synchronize(self.stream)
            self.synchronize()
            gpu_y[:] = act.f(gpu_x[:])

            if from_mask or np.any(np.sign(gpu_x) != np.sign(gpu_y)):
                self.converge(gpu_x, gpu_y, gpu_w, act=act, timeout=timeout-1)

        def managed(self, arr=None, shape=None, copy=False, dtype=np.float32):
            if arr is None and shape is None:
                raise ValueError
            elif arr is not None:
                shape = arr.shape

            man = drv.managed_zeros(shape, dtype=dtype,
                mem_flags=drv.mem_attach_flags.GLOBAL)
            if copy and (arr is not None):
                man[:] = arr
            return man

        # GPUs use transposed weight matrix for memory access efficiency
        def get_weights(self, w):
            return w.T

    try: _CUDA_ACC = environ["CUDA_ACC"] == "True"
    except KeyError: _CUDA_ACC = True

    try: _CUDA_HALF = environ["CUDA_HALF"] == "True"
    except KeyError: _CUDA_HALF = True

    try: _CUDA_DEVICES = environ["CUDA_DEVICES"]
    except KeyError: _CUDA_DEVICES = None

    try: _FAST_KERNELS = environ["FAST_KERNELS"] == "True"
    except KeyError: _FAST_KERNELS = True

    if _CUDA_DEVICES:
        _CUDA_DEVICES = [int(s) for s in _CUDA_DEVICES.split(",")]

    if _CUDA_ACC: engine = CUDAEngine(_CUDA_DEVICES, _CUDA_HALF, _FAST_KERNELS)
    else: engine = numpyEngine()

except ImportError as e:
    engine = numpyEngine()

if __name__ == "__main__":
    from time import perf_counter

    from_size = 20000
    to_size = 20000
    from_lam = 0.25
    to_lam = 0.25
    from_mask = True
    to_mask = True

    print("from_size = ", from_size)
    print("to_size = ", to_size)
    print("from_lam = ", from_lam)
    print("to_lam = ", to_lam)
    print("from_mask = ", from_mask)
    print("to_mask = ", to_mask)

    def gen_random(sz, num_patterns=1):
        if num_patterns > 1:
            return engine.managed(
                np.sign(np.random.random((num_patterns,sz)) - 0.5),
                copy=True)
        else:
            return engine.managed(
                np.sign(np.random.random((sz,num_patterns)) - 0.5),
                copy=True)
    def gen_ctx(vec, thresh):
        ctx = np.random.random(vec.shape) < thresh
        return engine.managed(ctx, copy=True, dtype=np.int32)

    x = gen_random(from_size)
    y = gen_random(to_size)
    ctx_x = gen_ctx(x, from_lam if from_mask else 1.0)
    ctx_y = gen_ctx(y, to_lam if to_mask else 1.0)
    w = engine.managed(np.random.random((from_size,to_size)) < 0, copy=True)

    index_x = engine.managed(x, dtype=np.int32)
    index_y = engine.managed(y, dtype=np.int32)

    if from_mask: x *= ctx_x

    ################################
    ################################
    ################################

    print()
    print("Dot test")

    ################################
    # PRE RUN
    engine.synchronize()
    engine.dot(x, y, w)
    engine.synchronize()
    ###
    y[:] = 0.

    start_time = perf_counter()
    engine.dot(x, y, w)
    engine.synchronize()
    elapsed = perf_counter() - start_time
    dot_elapsed = elapsed
    print("Regular: ", elapsed)

    ref = np.multiply(y, ctx_y)
    y[:] = 0

    ################################
    # PRE RUN
    engine.synchronize()
    engine.fast_dot(x, y, w,
        ctx_x, ctx_y,
        index_x, index_y,
        from_mask=from_mask, to_mask=to_mask)
    engine.synchronize()
    ###

    y[:] = 0.

    start_time = perf_counter()
    engine.fast_dot(x, y, w,
        ctx_x, ctx_y,
        index_x, index_y,
        from_mask=from_mask, to_mask=to_mask)
    engine.synchronize()
    elapsed = perf_counter() - start_time
    fast_dot_elapsed = elapsed
    print("Fast:    ", elapsed)

    #print("Diff:", np.sum(np.abs(y - ref)))
    fast_rate = fast_dot_elapsed / dot_elapsed
    print("Fast speed: %f%%" % (100. * fast_rate))

    #print(y)



    ################################
    ################################
    ################################

    w_original = w.copy()

    num_patterns = 2
    x = gen_random(from_size, num_patterns)
    y = gen_random(to_size, num_patterns)
    ctx_x = gen_ctx(x, from_lam if from_mask else 1.0)
    ctx_y = gen_ctx(y, to_lam if to_mask else 1.0)

    index_x = engine.managed(x, dtype=np.int32)
    index_y = engine.managed(y, dtype=np.int32)

    if from_mask: x *= ctx_x
    if to_mask: y *= ctx_y


    print()
    print("Learn test")

    ################################
    # PRE RUN
    engine.synchronize()
    engine.learn("rehebbian",
        x, y, w,
        norm=1., diag=True, decay=1.)
    engine.synchronize()
    ###

    start_time = perf_counter()
    engine.learn("rehebbian",
        x, y, w,
        norm=1., diag=True, decay=1.)
    engine.synchronize()
    elapsed = perf_counter() - start_time
    rehebbian_elapsed = elapsed
    print("Regular: ", elapsed)

    ref = w.copy()
    w[:] = w_original

    ################################
    # PRE RUN
    engine.synchronize()
    engine.fast_learn("rehebbian",
        x, y, w,
        ctx_x, ctx_y,
        index_x, index_y,
        from_mask=from_mask, to_mask=to_mask,
        norm=1., diag=True, decay=1.)
    engine.synchronize()
    ###

    start_time = perf_counter()
    engine.fast_learn("rehebbian",
        x, y, w,
        ctx_x, ctx_y,
        index_x, index_y,
        from_mask=from_mask, to_mask=to_mask,
        norm=1., diag=True, decay=1.)
    engine.synchronize()
    elapsed = perf_counter() - start_time
    fast_rehebbian_elapsed = elapsed
    print("Fast:    ", elapsed)

    #print()
    #diff = w - ref
    #net = np.sum(np.abs(diff))
    #count = np.sum(w != ref)
    #print("Diff", net, net / w.size, np.max(diff), np.min(diff), count)
    #print(w - ref)
    print()

    #print("ORIGINAL")
    #print(ref)
    #print()
    #print("FAST")
    #print(w)
    fast_rate = fast_rehebbian_elapsed / rehebbian_elapsed
    print("Fast speed: %f%%" % (100. * fast_rate))
