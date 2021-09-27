import dgl

if __name__ == "__main__":
  with dgl.hpc.WorkerContext() as wcontext:
    print('wcontext.rank', wcontext.rank)
    print('wcontext.size', wcontext.size)
