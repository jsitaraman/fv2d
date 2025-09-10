import time

class Timer:
    """
    Timer class for CPU and GPU code (handles CuPy synchronization).
    
    Usage:
        timer = Timer(use_gpu=True)
        timer.start()
        # code to time
        timer.stop()
        print(timer.elapsed)  # elapsed time in seconds
    """
    def __init__(self, use_gpu=False, verbose=True):
        """
        Args:
            use_gpu (bool): True if timing GPU code (CuPy arrays)
            verbose (bool): whether to print elapsed time on stop()
        """
        self.use_gpu = use_gpu
        if use_gpu:
           import cupy as cp
           self.cp=cp
        self.verbose = verbose
        self.t0 = None
        self.elapsed = None
    
    def start(self):
        """Start the timer."""
        if self.use_gpu:
            self.cp.cuda.Stream.null.synchronize()  # sync GPU before timing
        self.t0 = time.perf_counter()
        self.elapsed = None
    
    def stop(self):
        """Stop the timer and compute elapsed time."""
        if self.t0 is None:
            raise RuntimeError("Timer not started. Call start() first.")
        if self.use_gpu:
            self.cp.cuda.Stream.null.synchronize()  # sync GPU after code finishes
        t1 = time.perf_counter()
        self.elapsed = t1 - self.t0
        if self.verbose:
            device = "GPU" if self.use_gpu else "CPU"
            print(f"[{device}] Elapsed time: {self.elapsed:.6f} s")
        return self.elapsed
