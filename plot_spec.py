import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
from copy import deepcopy as dc

sigma = 17.16141009
alpha = 0.703617558
mu = -25.9113993 # mean-energy

def f(E, mu, sigma, alpha):
    fs = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(E-mu)**2/(2*sigma**2))*(
        erfc(-alpha*(E-mu)/(np.sqrt(2)*sigma)))
    return fs/fs.sum()

postfix = '3'

data = np.loadtxt(f'GBEs{postfix}.txt')
data = data[data[:, 0]!=0]
ids = data[:, 0]
Es = data[:, 1]
data = np.loadtxt(f'bulkEs{postfix}.txt')
if len(data.shape)==1:
    E0 = data[1]
else:
    E0 = np.mean(data[:,1])

eV2kJmol = 96.485
kT = 0.025*eV2kJmol

Eseg0 = (Es-E0)*eV2kJmol
srt = np.argsort(Eseg0)
Eseg = Eseg0[srt]
ids = ids[srt]
plt.hist(Eseg,density=True, bins=40)
plt.ylim((0, plt.gca().get_ylim()[1]))
plt.twinx()
plt.plot(Eseg, f(Eseg, mu, sigma, alpha), color='red')
plt.ylim((0, plt.gca().get_ylim()[1]))
plt.show()
ts = []
i = 0
while i<len(Eseg):
    t=np.sum(np.abs(Eseg-Eseg[i])<kT/20)
    ts.append(t)
    i += t
#print(ts)
#print(len(ts))

data = np.loadtxt(f'GBEs_int{postfix}.txt')
data = data[data[:, 0]!=0]
ids_c = data[:, 0]
Eint = data[:, 1:]*eV2kJmol

ids_n = np.zeros_like(Eint)

# with open(f'neigbors{postfix}.txt', 'r') as f:
#     for i, line in enumerate(f):
#         if i > 0 and i-1 < len(ids_c):
#             df = line.replace('\n', '').split(' ')
            
#             if int(df[1])>0:
#                 assert (int(df[0])==ids_c[i-1])
#                 df.remove('')
#                 ids_n[i-1, :len(df)-2] = np.array(df[2:]).astype(int) 
                
with open(f'neigbors{postfix}.txt', 'r') as f:
    k = 0
    for i, line in enumerate(f):
        if i > 0 and i-k-1 < len(ids_c):
            df = line.replace('\n', '').split(' ')
            
            if int(df[1])>0:
                assert (int(df[0])==ids_c[i-1-k])
                df.remove('')
                ids_n[i-1-k, :len(df)-2] = np.array(df[2:]).astype(int) 
            else:
                k += 1

        
Epure = float(np.loadtxt(f'pureE{postfix}.txt'))*eV2kJmol

w = Eint + Epure - 2*E0*eV2kJmol
w[Eint==0] = 0
for i in range(len(w)):
    for j in range(len(w[i][w[i]!=0])):
        i1 = np.where(ids==ids_c[i])
        j1 = np.where(ids==ids_n[i][j])
        w[i, j] -= (Eseg[i1]+Eseg[j1])
    
wavg = np.sum(w, axis = 1)
plt.hist(wavg, bins=100)
plt.show()
m = np.isin(ids, ids_c)
y = Eseg[m]
plt.plot(y, wavg, '.')
plt.show()
zs = np.sum(np.abs(w)>kT, axis=1)

Eselected = dc(y)
wselected = dc(w)
ids_c_selected = dc(ids_c)
ids_n_selected = dc(ids_n)

Er = dc(y)
for i in range(1, len(Er)):
    ids_filled = ids_c_selected[:i]
    t = Eselected[i:] + np.sum(wselected[i:]*np.isin(ids_n_selected[i:], 
                                                     ids_filled), axis=1)
    i0 = np.argmin(t)
    t0 = t[i0]
    Er[i] = t0
    wselected[i0], wselected[i] = wselected[i], wselected[i0]
    Eselected[i0], Eselected[i] = Eselected[i], Eselected[i0]
    ids_c_selected[i0], ids_c_selected[i] = ids_c_selected[i], ids_c_selected[i0]
    ids_n_selected[i0], ids_n_selected[i] = ids_n_selected[i], ids_n_selected[i0]
    
#%%
E_s = dc(Er)
F_s = np.ones(Er.shape, dtype=int)
cnt = 0
ids_c_s = dc(ids_c_selected)
ids_n_s = dc(ids_n_selected)
w_s = dc(wselected) 

for i in range(1, len(Er)):
    if E_s[i] < E_s[i-1]:
        k = 1 # lenght of anomaleus sequense: E_(i+k-1) < E_(i+k-1) < ... < E_(i-1) [k elements]
        while i+k<len(E_s) and E_s[i+k] < E_s[i+k-1]:
            k+=1
        E2 = (E_s[i+k-1] + E_s[i+k-2])/2
        n = 2
        cluster = [ids_c[i+k-1], ids_c[i+k-2]]
        Erow = np.zeros(k)
        Erow[-1] = E2
        Frow = np.ones_like(Erow)
        Frow[-1] = 2
        #print(Erow)
        for j in range(k-1):
            ind = i+k-2-j
            mask = np.isin(ids_n_s[ind], cluster) # is neighbour of [i+k-2-j] the member of cluster?
            O = np.sum(w_s[ind]*mask) # bonds with current site
            if E2 - O/n < E_s[ind] + O: # case when solutes does not form a cluster
            """
            !!!!!!!!!!!
            что делать с несколькими кластерами в серии???????????????
            !!!!!!!!!!
            """
                Erow[k-2-j] = E_s[ind] + O
                Erow[k-3-j] = E2 - O/n
            else: # solutes form a cluster
                E2 = (E2*n + E_s[ind])/(n+1)
                n += 1
                holes = n-2 # number of removed [individual] sites -1 (because array has lenght shorter by 1) 
                Erow[k-2-j] = E2
                Frow[k-2-j] = n
                for ii in range(k-1-j, k-holes): # shift
                    Erow[ii] = Erow[ii+1] # shift
                    Frow[ii] = Frow[ii+1] # shift
                Erow[-1] = 0 # shift
                Frow[-1] = 0 # shift
                cluster.append(ids_c[ind])
        
        print(f'{k} {n} {Frow}')
        
        E_s[i-1:i+k-1] = Erow # write calculated series to E_s
        # shift
        F_s[i-1:i+k-1] = Frow # write nuber of clusters

#plt.hist(Er, bins=50, density=True, label='interaction')
plt.hist(E_s, bins=50, density=True, alpha=0.4, label='interaction s')
#plt.hist(y, bins=40, alpha=0.4, density=True, label='dilute')
plt.xlabel('$E_{seg}$')
plt.ylabel('probability')
plt.legend()
plt.show()










