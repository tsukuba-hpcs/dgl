import torch as th
import dgl

if __name__ == "__main__":
  mcontext = dgl.hpc.ManagerContext()
  print('mcontext.rank', mcontext.rank)
  print('mcontext.size', mcontext.size)
  init_func = lambda shape, dtype: th.zeros(shape, dtype)
  dgl.hpc.serveTensor(mcontext, "foo", (100,100), th.float64, dgl.hpc.ModuloPolicy, init_func)
  mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")
  mcontext.serve()
