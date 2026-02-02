# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) :
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations
'''
# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height
convergence = np.empty((width, height), dtype=np.double)
# Calcul de l'ensemble de mandelbrot :
deb = time()
for y in range(height):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        convergence[x, y] = mandelbrot_set.convergence(c, smooth=True)
fin = time()
print(f"Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}")

# Constitution de l'image résultante :
deb = time()
image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
fin = time()
print(f"Temps de constitution de l'image : {fin-deb}")
image.show()

'''

def block_rows(rank, size, height):
    base = height // size
    rem  = height % size
    y0 = rank * base + min(rank, rem)
    nloc = base + (1 if rank < rem else 0)
    return y0, y0 + nloc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3.0/width
scaleY = 2.25/height

y_start, y_end = block_rows(rank, size, height)
nloc= y_end - y_start
convergence_loc = np.empty((nloc, width), dtype=np.double)


comm.Barrier()
t0 = MPI.Wtime()

for j,y in enumerate(range(y_start, y_end)):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        convergence_loc[j, x] = mandelbrot_set.convergence(c, smooth=True)


t1=MPI.Wtime()

t_max=comm.reduce(t1-t0, op=MPI.MAX, root=0)

sendbuf=convergence_loc.ravel()
counts = np.array([
    (block_rows(r, size, height)[1] - block_rows(r, size, height)[0]) * width
    for r in range(size)], dtype=np.int64)

displs = np.zeros(size, dtype=np.int64)
displs[1:] = np.cumsum(counts[:-1])

if rank == 0:
    full = np.empty((height * width,), dtype=np.double)
else:
    full = None

comm.Gatherv(sendbuf, [full, counts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    convergence = full.reshape((height, width))
    print(f"[Q1] nbp={size} temps={t_max:.6f}s")

    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    image.save("mandelbrot_p1_q1.png")
    #image.show()

# ============================================================
# Partie 1 – Question 1 : Temps d'exécution et speedup
#
# Mesures expérimentales :
#
# nbp (p) | Temps T(p) (s) | Speedup S(p)=T(1)/T(p)
# ------------------------------------------------------------
#    1    |   6.916713    |   1.00
#    4    |   2.100641    |   3.29
#    8    |   1.157017    |   5.98
#   16    |   0.653610    |  10.58
#
# Interprétation :
# - Le speedup augmente avec le nombre de processus, ce qui montre
#   que la parallélisation est efficace.
# - Le speedup reste inférieur au speedup idéal (S(p)=p) à cause :
#     * du déséquilibre de charge (certaines lignes coûtent plus cher),
#     * du coût des communications (MPI_Gatherv),
#     * d'une partie séquentielle incompressible
#       (création et sauvegarde de l'image sur le rang 0),
#     * de l'overhead MPI.
# - Les rendements sont décroissants lorsque p augmente, ce qui est
#   conforme à


ys = list(range(rank, height, size))   
nloc = len(ys)
local = np.empty((nloc, width), dtype=np.double)

comm.Barrier()
t0 = MPI.Wtime()

for j,y in enumerate(ys):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        local[j, x] = mandelbrot_set.convergence(c, smooth=True)


t1=MPI.Wtime()

t_max=comm.reduce(t1-t0, op=MPI.MAX, root=0)

counts = comm.gather(nloc, root=0)

if rank == 0:
    convergence = np.empty((height, width), dtype=np.double)

    
    y0 = np.array(ys, dtype=np.int32)
    convergence[y0, :] = local

    for src in range(1, size):
        nsrc = counts[src]
        y_src = np.empty((nsrc,), dtype=np.int32)
        pix_src = np.empty((nsrc, width), dtype=np.double)

        comm.Recv(y_src, source=src, tag=10)
        comm.Recv(pix_src, source=src, tag=11)

        convergence[y_src, :] = pix_src

    print(f"[Q2] nbp={size} temps={t_max:.6f}s")

    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    image.save("mandelbrot_P1_q2.png")
    

else:
    y_arr = np.array(ys, dtype=np.int32)
    comm.Send(y_arr, dest=0, tag=10)
    comm.Send(local, dest=0, tag=11)

# ============================================================
# Partie 1 – Question 2 : Répartition statique améliorée (cyclique)
# Comparaison avec Q1 (blocs contigus)
#
# Temps mesurés :
#
# p (nbp) | T_Q1 bloc (s) | T_Q2 cyclique (s)
# -------------------------------------------
#   1     |   6.722879    |   6.485679
#   4     |   2.107625    |   1.726538
#   8     |   1.142764    |   0.882137
#  16     |   0.637176    |   0.486749
#
# Speedup (référence = T(1) de chaque méthode) :
#
# p (nbp) | S_Q1 = T1_Q1/Tp_Q1 | S_Q2 = T1_Q2/Tp_Q2
# -------------------------------------------------
#   1     |        1.00        |        1.00
#   4     |        3.19        |        3.76
#   8     |        5.88        |        7.35
#  16     |       10.55        |       13.32
#
# Comparaison / Conclusion :
# - La répartition cyclique est plus rapide que la répartition par blocs
#   (T_Q2 < T_Q1 pour p=4,8,16) et donne un meilleur speedup.
# - Explication : le coût par ligne n'est pas uniforme dans Mandelbrot.
#   Avec des blocs contigus, un processus peut recevoir beaucoup de lignes "dures"
#   (près de la frontière) => déséquilibre de charge.
#   En cyclique, les lignes "dures" sont réparties entre tous les processus
#   => meilleur équilibrage => meilleure performance.
#
# Problèmes possibles avec cette stratégie :
# - Rassemblement plus compliqué : les lignes ne sont plus contiguës, on doit
#   envoyer les indices y + les lignes, ou faire un assemblage plus complexe
#   (donc plus d'overhead communication).
# - Localité mémoire moins bonne (lignes espacées) => parfois moins efficace cache.
# - Stratégie moins robuste : si on change la zone/zoom, la répartition des lignes
#   difficiles/faciles change, donc l'intérêt peut varier.
# ============================================================
TAG_WORK = 1
TAG_STOP = 2
TAG_RESULT = 3

def compute_row(y: int) -> np.ndarray:
    row = np.empty((width,), dtype=np.double)
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
        row[x] = mandelbrot_set.convergence(c, smooth=True)
    return row


comm.Barrier()
t0 = MPI.Wtime()

if rank == 0:
    
    convergence = np.empty((height, width), dtype=np.double)

    next_y = 0
    done = 0

    
    for dst in range(1, size):
        if next_y < height:
            comm.send(next_y, dest=dst, tag=TAG_WORK)
            next_y += 1
        else:
            comm.send(None, dest=dst, tag=TAG_STOP)

    
    status = MPI.Status()
    while done < height:
        y, row = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        src = status.Get_source()

        convergence[y, :] = row
        done += 1

        if next_y < height:
            comm.send(next_y, dest=src, tag=TAG_WORK)
            next_y += 1
        else:
            comm.send(None, dest=src, tag=TAG_STOP)

    comm.Barrier()
    t1 = MPI.Wtime()

    print(f"[Q3 master-worker] nbp={size} temps={t1 - t0:.6f}s")

    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    image.save("mandelbrot_P1_q3.png")
    

else:
    
    status = MPI.Status()
    while True:
        y = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            break

        row = compute_row(y)
        comm.send((y, row), dest=0, tag=TAG_RESULT)

    comm.Barrier()


    # ============================================================
# Partie 1 – Question 3 : Stratégie maître-esclave (master-worker)
# ============================================================
#
# Principe :
# - Le processus 0 joue le rôle de maître et distribue dynamiquement
#   les lignes de l'image à calculer.
# - Les autres processus (esclaves) reçoivent une ligne y, calculent
#   la ligne correspondante et renvoient le résultat au maître.
# - Cette stratégie permet un équilibrage dynamique de la charge,
#   car chaque processus reçoit du travail dès qu'il termine le précédent.
#
# ------------------------------------------------------------
# Temps mesurés (en secondes)
#
# p (nbp) | Q1 blocs | Q2 cyclique | Q3 maître-esclave
# ----------------------------------------------------
#   4     | 2.062851 | 1.730668    | 2.065389
#   8     | 1.168697 | 0.894811    | 0.914765
#  16     | 0.601573 | 0.452065    | 0.433304
#  32     | 0.560827 | 0.438712    | 0.412979
#
# ------------------------------------------------------------
# Speedup (référence T(1) ≈ 6.72 s)
#
# p (nbp) | S_Q1 bloc | S_Q2 cyclique | S_Q3 maître-esclave
# ----------------------------------------------------------
#   4     |   3.26    |    3.88       |      3.25
#   8     |   5.75    |    7.51       |      7.35
#  16     |  11.17    |   14.86       |     15.51
#  32     |  11.99    |   15.33       |     16.28
#
# (S(p) = T(1) / T(p))
#
# ------------------------------------------------------------
# Comparaison des stratégies
#
# - Q1 (blocs contigus) :
#   * Implémentation simple.
#   * Déséquilibre de charge possible car certaines lignes de Mandelbrot
#     sont plus coûteuses (zones proches de la frontière).
#   * Speedup limité lorsque p augmente.
#
# - Q2 (répartition cyclique statique) :
#   * Meilleur équilibrage que Q1, car les lignes "dures" et "faciles"
#     sont réparties entre tous les processus.
#   * Temps plus faibles et speedup nettement meilleur que Q1.
#   * Overhead de communication encore raisonnable.
#
# - Q3 (maître-esclave, dynamique) :
#   * Meilleur équilibrage de charge : aucun processus ne reste inactif.
#   * Performances comparables à Q2 pour p=8, et meilleures pour p>=16.
#   * Donne le meilleur speedup pour un grand nombre de processus.
#
# ------------------------------------------------------------
# Limites et problèmes potentiels de la stratégie maître-esclave
#
# - Surcoût de communication important : chaque ligne entraîne
#   un envoi + une réception.
# - Le processus maître peut devenir un goulot d'étranglement
#   lorsque le nombre de processus augmente.
# - La scalabilité est donc limitée par la capacité du maître
#   à distribuer et collecter les tâches.
#
# ------------------------------------------------------------
# Conclusion générale
#
# La stratégie maître-esclave fournit le meilleur équilibrage de charge
# et les meilleurs speedups pour un nombre élevé de processus, en
# particulier lorsque le coût des tâches est très variable.
# Cependant, son overhead de communication et le rôle central du maître
# peuvent limiter ses performances à grande échelle.
# Dans ce contexte, la répartition cyclique statique (Q2) représente
# un excellent compromis entre simplicité, équilibrage et efficacité.
# ============================================================
