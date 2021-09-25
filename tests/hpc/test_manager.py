import torch as th
import dgl

if __name__ == "__main__":
  mcontext = dgl.hpc.ManagerContext()
  shard = dgl.hpc.Shard(mcontext.rank, mcontext.size)
  fooshard = dgl.hpc.createTensor(shard, "foo", (101,100), th.float64, dgl.hpc.ModuloPolicy)
  assert fooshard.id == 0
  barshard = dgl.hpc.createTensor(shard, "bar", (102,100), th.float64, dgl.hpc.ModuloPolicy)
  assert barshard.id == 1
  mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")
  mcontext.serve(shard)
