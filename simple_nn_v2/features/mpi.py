class DummyMPI(object):
    def __init__(self):
        self.MPI  = None
        self.comm = None
        self.rank = 0
        self.size = 1

    def barrier(self):
        pass

    def disconnet(self):
        pass

    def free(self):
        pass

    def gather(self, data, root=0):
        return [data]

    def allreduce_max(self, data):
        return data

    def bcast(self, data, root=0):
        return data

    def scatter(self, data, root=0):
        return data

    def allgather(self, data):
        return [data]

    def Allgatherv(self, sendbuf, recvbuf, count, displ, dtype):
        if sendbuf.size != recvbuf.size:
            assert False
        for i in range(sendbuf.size):
            recvbuf[i] = sendbuf[i]
        return recvbuf

class MPI4PY(object):
    def __init__(self):
        from mpi4py import MPI
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def barrier(self):
        self.comm.barrier()

    def disconnect(self):
        self.comm.Disconnect()

    def free(self):
        self.comma.Free()

    def gather(self, data, root=0):
        return self.comm.gather(data, root=root)

    def allreduce_max(self, data):
        return self.comm.allreduce(data, op=self.MPI.MAX)

    def bcast(self, data, root=0):
        return self.comm.bcast(data, root=root)

    def scatter(self, data, root=0):
        return self.comm.scatter(data, root=root)

    def allgather(self, data):
        return self.comm.allgather(data)

    def Allgatherv(self, sendbuf, recvbuf, count, displ, dtype):
        if dtype == "double":
            return self.comm.Allgatherv(sendbuf, [recvbuf, count, displ, self.MPI.DOUBLE])
        elif dtype == "int":
            return self.comm.Allgatherv(sendbuf, [recvbuf, count, displ, self.MPI.INT])
        else:
            assert False
