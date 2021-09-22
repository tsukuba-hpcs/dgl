import dgl

if __name__ == "__main__":
  mcontext = dgl.hpc.ManagerContext()
  print('mcontext.rank', mcontext.rank)
  print('mcontext.size', mcontext.size)
  mcontext.launchWorker(1, "python", "tests/hpc/test_worker.py")