import torch as th
import dgl

if __name__ == "__main__":
  mcontext = dgl.hpc.ManagerContext()
  print('mcontext.rank', mcontext.rank)
  print('mcontext.size', mcontext.size)
  shard = dgl.hpc.Shard()
  fooshard = dgl.hpc.createTensor(shard, "foo", (100,100), th.float64, dgl.hpc.ModuloPolicy)
  mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")
  mcontext.serve()
