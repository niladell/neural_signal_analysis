import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from extended_linearRegression import LinearRegression
from scipy.stats import ttest_1samp as tt


data = np.load('Python_data/emu018Pdata.pkl')

cohe = data['cohe'].astype(np.float)
stimulus = data['stimulus'].astype(np.int)

# We convert 
coherences = cohe * (stimulus >0 ) + -1 * cohe * (stimulus  == 0 )


regres = linear_model.LinearRegression()

spikes1 = data['nspikes1'].astype(np.float)/2
coherences = coherences.reshape(-1,1)

regres.fit(coherences, spikes1)

print("Coeficients", regres.coef_, " -- Intercept (w0):  ", regres.intercept_)

print("R2", regres.score(coherences, spikes1))

# plot_label = str(regres.coef_[0]) + 'x + ' + str(regres.intercept_)
# plt.plot(coherences, regres.predict(coherences), label=plot_label)
# plt.scatter(coherences, spikes1)
# plt.title('Linear regression')
# plt.xlabel('Coherence (%))')
# plt.ylabel('Firing rate (Hz)')
# plt.legend(loc='best') #(str(regres.coef) + 'x + ' str(regres.interccept_))
# 
# plt.show()


# Part 2

stimulus = (stimulus > 0 ) + -1 * (stimulus  == 0 )

decision = data['decision'].astype(np.int)
decision = (decision > 0 ) + -1 * (decision  == 0 )

spikes2 = data['nspikes2'].astype(np.float)/2


regressors = np.array([stimulus[1:], stimulus[:-1], decision[1:], decision[:-1]]).T
regres.fit(regressors, spikes2[1:])

regres2 = LinearRegression()
regres2.fit(regressors, spikes2[1:])
print(regres2.betasPValue)

for i in range(4):
    print(tt(regressors[:,i],regres.coef_[i]))

print("Coeficients", regres.coef_, " -- Intercept (w0):  ", regres.intercept_)
print("R2", regres.score(regressors, spikes2[1:]))
print(np.shape(regressors))

