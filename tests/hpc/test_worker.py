import dgl

if __name__ == "__main__":
    with dgl.hpc.WorkerContext() as wcontext:
        with dgl.hpc.ShardClient(wcontext) as client:
            client.getMetadata("foo")
            print('wcontext.rank', wcontext.rank)
            print('wcontext.size', wcontext.size)
