import numpy as np
from scipy import stats, optimize

# calcola la sezione a mu fissato della banda,
# cioè trova il kmin massimo tale che:
# \sum_{kmin=0}^\infty poisson(k;mu) >= CL
def kmin(mu, CL):
    coverage = 1 # partiamo con tutti i k, quindi la somma è 1
    kmin = -1
    while coverage >= CL: # andiamo avanti finché non scendiamo sotto il CL
        kmin += 1
        coverage -= stats.poisson.pmf(kmin, mu) # togliamo kmin dalla somma
    # quando il ciclo si ferma il coverage è appena troppo basso e
    # kmin è appena stato tolto dalla somma, quindi vuol dire che
    # kmin è quello che cerchiamo a questo punto
    return kmin

# stima rozza in eccesso di mumax
# ottenuta con l'approssimazione gaussiana + il risultato per k = 0
def mumax_bound(k, CL):
    k_big = k + np.ceil(np.sqrt(k) * stats.norm.ppf((1 + CL) / 2))
    k_0 = - np.log(1 - CL)
    margin = np.exp(-1)
    return k_big + k_0 + margin

# calcola la sezione a k fissato della banda,
# cioè trova il massimo mu tale che k = kmin(mu)
def mumax(k, CL):
    f = lambda mu: kmin(mu, CL) - k - 1/2
    # f è la funzione di cui cercare lo zero.
    # ci servirebbe il massimo zero di kmin - k,
    # lo troviamo con il trucco di aggiungere -1/2
    mumax = optimize.bisect(f, 0, mumax_bound(k, CL), rtol=1e-5) # bisezione
    return mumax
