# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI

# Dimension du problème (peut-être changé)
dim = 120
# Initialisation de la matrice
'''
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
print(f"u = {u}")

# Produit matrice-vecteur
v = A.dot(u)
print(f"v = {v}")
'''
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

if dim % size != 0:
    if rank == 0:
        print("Le nombre de processus doit diviser la dimension de la matrice.")
    MPI.Finalize()
    exit()

Nloc=dim // size

j0= rank * Nloc
j1= (rank + 1) * Nloc

u = np.array([i + 1.0 for i in range(dim)], dtype=np.double)

comm.Barrier()
to=MPI.Wtime()

A_block = np.empty((dim, Nloc), dtype=np.double)
for i in range(dim):
    for jj, j in enumerate(range(j0, j1)):
        A_block[i, jj] = ((i + j) % dim) + 1.0


v_partial = A_block.dot(u[j0:j1])
v = np.empty((dim,), dtype=np.double)
comm.Allreduce(v_partial, v, op=MPI.SUM)


t1=MPI.Wtime()
t_max=comm.reduce(t1-to, op=MPI.MAX, root=0)

if rank == 0:
    print(f"[Q1] dim={dim} nbp={size} Nloc={Nloc} temps={t_max:.6f}s")

# ============================================================
# Partie 2.a — Produit matrice-vecteur parallèle (découpage par colonnes)
#
# Rappel :
# - Nloc = N / nbp  (N divisible par nbp)
# - Chaque processus possède un bloc de colonnes A[:, j0:j1]
# - Chaque processus calcule v_partial = A_block * u_local
# - Puis on somme les contributions : v = sum(v_partial) via MPI_Allreduce
#
# Mesures expérimentales (temps max sur les processus) :
#
# N = 120
# p (nbp) | Temps T(p) (s)        | Speedup S(p) = T(1)/T(p)
# ----------------------------------------------------------
#   1     | 0.0080487010          | 1.00
#   2     | 0.0048368620          | 1.66
#   4     | 0.0025046470          | 3.21
#   8     | 0.0031049280          | 2.59
#  12     | 0.0019503220          | 4.13
#
# Interprétation :
# - Le speedup n'est pas monotone (ex: p=8 plus lent que p=4) car le problème est
#   très petit (N=120) et le temps est dominé par l'overhead MPI (communications,
#   synchronisations, MPI_Allreduce) et le lancement des processus.
# - Le gain dépend aussi du bruit de mesure (temps très courts => variation).
# - Pour des matrices plus grandes, on s'attend à un speedup plus régulier
#   (le calcul devient dominant par rapport au coût de communication).
# ============================================================
i0 = rank * Nloc
i1 = (rank + 1) * Nloc


u = np.array([i + 1.0 for i in range(dim)], dtype=np.double)

comm.Barrier()
t0 = MPI.Wtime()


A_rows = np.empty((Nloc, dim), dtype=np.double)
for ii, i in enumerate(range(i0, i1)):
    for j in range(dim):
        A_rows[ii, j] = ((i + j) % dim) + 1.0


v_local = A_rows.dot(u)


v = np.empty((dim,), dtype=np.double)
comm.Allgather(v_local, v)

t1 = MPI.Wtime()
t_max = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

if rank == 0:
    print(f"[Q2] dim={dim} nbp={size} Nloc={Nloc} temps={t_max:.6f}s")

# ============================================================
# Partie 2 — Produit matrice-vecteur MPI : speedup
# (Q1 = découpage par COLONNES, Q2 = découpage par LIGNES)
#
# Rappel :
# - Nloc = N / nbp  (ici N=120)
#
# Temps mesurés (en secondes) :
#
# p (nbp) | T_Q1 colonnes | T_Q2 lignes
# -------------------------------------
#   1     |   0.008496    |  0.007225
#   2     |   0.005689    |  0.004424
#   3     |   0.004283    |  0.003407
#   4     |   0.002545    |  0.002757
#   8     |   0.001809    |  0.001265
#  12     |   0.001960    |  0.001585
#
# Speedup (référence = T(1) de chaque approche) :
#
# p (nbp) | S_Q1 = T1_Q1/Tp_Q1 | S_Q2 = T1_Q2/Tp_Q2
# -------------------------------------------------
#   1     |        1.00        |        1.00
#   2     |        1.49        |        1.63
#   3     |        1.98        |        2.12
#   4     |        3.34        |        2.62
#   8     |        4.70        |        5.71
#  12     |        4.33        |        4.56
#
# Interprétation :
# - Les speedups ne sont pas parfaitement monotones car N=120 est petit :
#   l'overhead MPI (synchronisations, communications) et le bruit de mesure
#   deviennent importants par rapport au temps de calcul.
# - Q1 (colonnes) nécessite un MPI_Allreduce (somme de contributions sur tout v),
#   Q2 (lignes) nécessite un MPI_Allgather (rassembler des morceaux de v).
# - Selon p, l'un ou l'autre peut être plus performant (ex : Q2 meilleur à p=8),
#   mais globalement le coût de communication limite l'accélération.
# ============================================================
