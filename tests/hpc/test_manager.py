import torch as th
import dgl

if __name__ == "__main__":
    with dgl.hpc.ManagerContext() as mcontext:
        with dgl.hpc.Shard(mcontext.rank, mcontext.size) as shard:
            foo = shard.createTensor("foo", (11,10), th.float64, dgl.hpc.ModuloPolicy)
            assert foo.id == 0
            foo.local_tensor.uniform_(-1, 1)
            print('foo.local_tensor.shape', foo.local_tensor.shape)
            print('foo.local_tensor[0, :]', foo.local_tensor[0, :])
            mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")
            mcontext.serve(shard)
