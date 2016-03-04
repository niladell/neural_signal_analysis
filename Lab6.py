import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = np.load('Python_data/emu018Pdata.pkl')

coherences = data['cohe'].astype(np.float) * (data['stimulus'].astype(np.float) >0 ) +\
 -1 * data['cohe'].astype(np.float) * (data['stimulus'].astype(np.float)  == 0 )
#print(coherences)

regres = linear_model.LinearRegression()

spikes1 = data['nspikes1'].astype(np.float)/2
coherences = coherences.reshape(-1,1)
print(np.shape(spikes1))
print(np.shape(coherences))

regres.fit(coherences, spikes1)

print("Coeficients", regres.coef_, " -- Intercept (w0):  ", regres.intercept_)

plt.plot(coherences, regres.predict(coherences))

plt.scatter(coherences, spikes1)

plt.show()