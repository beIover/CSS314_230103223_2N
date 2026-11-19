
from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(
    f"Rank {rank}/{size} | "
    f"Host: {socket.gethostname()}",
    flush=True
)

# Test communication between all ranks
value = rank + 1
total = comm.allreduce(value, op=MPI.SUM)

comm.Barrier()

if rank == 0:
    print("MPI ALLREDUCE RESULT:", total)
    print("EXPECTED RESULT:", 10)
    print("MPI TEST:", "PASSED" if total == 10 else "FAILED")
