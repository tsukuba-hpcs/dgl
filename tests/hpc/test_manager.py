import torch as th
import dgl

if __name__ == "__main__":
  with dgl.hpc.ManagerContext() as mcontext:
    with dgl.hpc.Shard(mcontext.rank, mcontext.size) as shard:
      foo = shard.createTensor("foo", (11,10), th.float64, dgl.hpc.ModuloPolicy)
      assert foo.id == 0
      foo.local_tensor.uniform_(-1, 1)
      print('foo.local_tensor[0, :]', foo.local_tensor[0, :])
      bar = shard.createTensor("bar", (12,10), th.float64, dgl.hpc.ModuloPolicy)
      assert bar.id == 1
      bar.local_tensor.fill_(0)
      print('bar.local_tensor[0, :]', bar.local_tensor[0, :])
      mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")
      mcontext.serve(shard)
