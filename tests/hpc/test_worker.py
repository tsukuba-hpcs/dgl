import dgl

if __name__ == "__main__":
    with dgl.hpc.WorkerContext() as wcontext:
        with dgl.hpc.ShardClient(wcontext) as client:
            foo = client.getMetadata("foo")
            print(vars(foo))
            fooslice = client.fetchSlice(foo, 0, 0)
            print('fooslice', fooslice.tensor)