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
#w_s = dc(wselected) 

ids_c_s = -np.ones((len(Er), len(Er)), dtype=int)
ids_c_s[:, 0] = ids_c_selected

cnt = 0

#ids_n_s = dc(ids_n_selected)

change_flag = True
iteration = 0
i0 = len(Er)
while change_flag:
    print(f'iter {iteration}, last step on {i0}/{len(Er)}')
    iteration += 1
    change_flag = False
    for i in range(len(Er)-1-cnt, 1, -1): # reverse order
        if E_s[i] < E_s[i-1]:
            change_flag = True
            
            O = 0
            lst = ids_c_s[i-1]
            lst = lst[lst!=-1]
            for ind in lst:
                pind = np.where(ids_c_selected==ind)
                mask = np.isin(ids_n_selected[pind], ids_c_s[i]) # is neighbours of [i-1] the member of cluster [i]?
                O += np.sum(wselected[pind]*mask) # bonds with current site
                
            E2 = (E_s[i-1]*F_s[i-1] + E_s[i]*F_s[i])/(F_s[i-1]+F_s[i])
            
            if E2 - O/F_s[i] < E_s[i-1] + O: # case when solutes does not form a cluster
                # replace it with corresponding changes in bonds energy
                
                # E_s
                E_s[i] = E_s[i-1] + O
                E_s[i-1] = E2 - O/F_s[i]
                
                # F_s
                t = F_s[i]
                F_s[i] = F_s[i-1]
                F_s[i-1] = t
                
                # ids_c_s
                t = ids_c_s[i]
                ids_c_s[i] = ids_c_s[i-1]
                ids_c_s[i-1] = t
                
            else: # solutes form a cluster
                cnt += 1
                # combine elements into one and shift right side of array
                
                # Combine
                E_s[i-1] = E2 # E_s
                F_s[i-1] += F_s[i] # F_s
                
                # ids_c_s
                ids3 = np.concatenate((ids_c_s[i-1], ids_c_s[i]))
                ids3 = ids3[ids3!=-1]
                ids_c_s[i-1, :len(ids3)] = ids3 
                
                # Shift
                
                # E_s
                E_s[i:] = np.roll(E_s[i:], -1)
                E_s[-1] = 0
                
                # F_s
                F_s[i:] = np.roll(F_s[i:], -1)
                F_s[-1] = 0
                
                # ids_c_s
                ids_c_s[i:] = np.roll(ids_c_s[i:], -1, axis=0)
                ids_c_s[-1] = -np.ones(len(Er))
        elif change_flag:
            i0 = len(Er)-i
            break
     
    plt.plot(E_s)
    plt.savefig(f'plots/plot_{iteration}.png')
    plt.show()
    
#plt.hist(Er, bins=50, density=True, label='interaction')
plt.hist(E_s[F_s!=0], bins=50, density=True, alpha=0.4, label='interaction s')
#plt.hist(y, bins=40, alpha=0.4, density=True, label='dilute')
plt.xlabel('$E_{seg}$')
plt.ylabel('probability')
plt.legend()
plt.show()





