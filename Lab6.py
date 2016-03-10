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

print("R2", regres.score(coherences, spikes1))

plot_label = str(regres.coef_[0]) + 'x + ' + str(regres.intercept_)
plt.plot(coherences, regres.predict(coherences), label=plot_label)
plt.scatter(coherences, spikes1)
plt.title('Linear regression')
plt.xlabel('Coherence (%))')
plt.ylabel('Firing rate (Hz)')
plt.legend(loc='best') #(str(regres.coef) + 'x + ' str(regres.interccept_))

plt.show()


