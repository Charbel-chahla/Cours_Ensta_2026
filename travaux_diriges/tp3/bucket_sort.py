import numpy as np
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

N = 200000
seed = 123

if N % nbp != 0:
    if rank == 0:
        print("Erreur: N doit être divisible par nbp")
    raise SystemExit

Nloc = N // nbp


if rank == 0:
    np.random.seed(seed)
    data = np.random.rand(N).astype(np.float64)
else:
    data = None

local = np.empty(Nloc, dtype=np.float64)

comm.Barrier()
t0 = time()

comm.Scatter(data, local, root=0)


buckets = [[] for _ in range(nbp)]
for x in local:
    b = int(x * nbp)
    buckets[b].append(x)


send_counts = np.array([len(buckets[i]) for i in range(nbp)], dtype=np.int32)
send_displs = np.zeros(nbp, dtype=np.int32)
send_displs[1:] = np.cumsum(send_counts[:-1])

sendbuf = np.concatenate([np.array(buckets[i], dtype=np.float64) for i in range(nbp)]) \
          if send_counts.sum() > 0 else np.empty(0, dtype=np.float64)

recv_counts = np.empty(nbp, dtype=np.int32)
comm.Alltoall(send_counts, recv_counts)

recv_displs = np.zeros(nbp, dtype=np.int32)
recv_displs[1:] = np.cumsum(recv_counts[:-1])
recv_total = int(np.sum(recv_counts))
recvbuf = np.empty(recv_total, dtype=np.float64)

comm.Alltoallv(
    [sendbuf, send_counts, send_displs, MPI.DOUBLE],
    [recvbuf, recv_counts, recv_displs, MPI.DOUBLE]
)


recvbuf.sort()


counts = comm.gather(recvbuf.size, root=0)

if rank == 0:
    displs = np.zeros(nbp, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    out = np.empty(np.sum(counts), dtype=np.float64)
else:
    displs = None
    out = None

comm.Gatherv(recvbuf, [out, counts, displs, MPI.DOUBLE], root=0)

comm.Barrier()
t1 = time()

tmax = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

if rank == 0:
    print(f"[BucketSort] N={N} nbp={nbp} temps={tmax:.6f}s")
    print("Tri correct ?", np.all(out[:-1] <= out[1:]))

'''

Référence : T(1) = 0.290515 s

p (nbp) | Temps T(p) (s) | Speedup S(p) = T(1)/T(p)
----------------------------------------------------
  1     | 0.290515       | 1.000
  2     | 0.183155       | 1.586
  4     | 0.082472       | 3.522
 16     | 0.046129       | 6.297
 32     | 0.041151       | 7.060
 64     | 0.168744       | 1.721

Interprétation :
- On observe une bonne accélération jusqu'à 32 processus : le temps diminue
  fortement (0.29 s -> 0.041 s) et le speedup atteint ~7.06.
- Le speedup n'est pas linéaire (on est loin de S(p)=p) car l'algorithme
  comporte des communications MPI (Scatter/Alltoallv/Gatherv) et des
  synchronisations dont le coût ne diminue pas avec p.
- À 64 processus, la performance s'effondre (temps 0.168 s) : ici l'overhead
  de communication et de gestion (beaucoup de petits messages + surcoût MPI)
  devient dominant par rapport au calcul local, et/ou la machine ne dispose
  pas réellement de 64 cœurs efficaces pour ce job (surcharge, partage CPU).
- Conclusion : pour N=200000, le meilleur compromis est autour de 16-32
  processus. Au-delà, augmenter p gaspille des ressources et peut ralentir.

'''