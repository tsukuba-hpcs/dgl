import dgl

if __name__ == "__main__":
  wcontext = dgl.hpc.WorkerContext()
  print('wcontext.rank', wcontext.rank)
  print('wcontext.size', wcontext.size)
