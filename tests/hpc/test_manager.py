import torch as th
import dgl

if __name__ == "__main__":
  mcontext = dgl.hpc.ManagerContext()
  print('mcontext.rank', mcontext.rank)
  print('mcontext.size', mcontext.size)
  fooshard = dgl.hpc.createTensor(mcontext, "foo", (100,100), th.float64, dgl.hpc.ModuloPolicy)
  mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")
  mcontext.serve()
