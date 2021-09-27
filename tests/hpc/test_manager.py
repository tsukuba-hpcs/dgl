import torch as th
import dgl

if __name__ == "__main__":
  mcontext = dgl.hpc.ManagerContext()
  shard = dgl.hpc.Shard(mcontext.rank, mcontext.size)
  foo = dgl.hpc.createTensor(shard, "foo", (11,10), th.float64, dgl.hpc.ModuloPolicy)
  assert foo.id == 0
  foo.local_tensor.uniform_(-1, 1)
  print('foo.local_tensor[0, :]', foo.local_tensor[0, :])
  bar = dgl.hpc.createTensor(shard, "bar", (12,10), th.float64, dgl.hpc.ModuloPolicy)
  assert bar.id == 1
  bar.local_tensor.fill_(0)
  print('bar.local_tensor[0, :]', bar.local_tensor[0, :])
  mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")
  mcontext.serve(shard)
