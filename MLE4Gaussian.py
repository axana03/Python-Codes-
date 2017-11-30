import numpy as np
import matplotlib.pyplot as plt

# compute gaussian probability for a set of input values
# using a fixed parameter set
def compute_prob(X,mu,sigma2):
    Y = - 0.5 * ((X - mu * np.ones(X.shape))**2) / sigma2
    P = np.exp(Y) / np.sqrt(2 * np.pi * sigma2)
    return P

# compute log-likelihood for a set of parameters
def compute_log_likelihood(X, Mu, Sigma2):
    N = len(X)

    # for all elements in X, evaluate the likelihood regard the whole Mu matrix
    E = np.zeros(Mu.shape)
    myones = np.ones(Mu.shape)
    for i in range(N):
        E = E + np.square(X[i] * myones - Mu)

    logLH = - 0.5 * np.divide(E, Sigma2) - 0.5 * N * np.log(Sigma2) - 0.5 * N * np.log(2 * np.pi)

    return logLH

# compute log-likelihood for a set of parameters
def compute_log_likelihood_single(X, mu, sigma2):
    N = len(X)

    # for all elements in X, evaluate the likelihood regard the whole Mu matrix
    E = 0.0
    for i in range(N):
        E = E + np.square(X[i] - mu)

    logLH = - 0.5 * E / sigma2 - 0.5 * N * np.log(sigma2) - 0.5 * N * np.log(2 * np.pi)

    return logLH

# compute log-likelihood for a set of parameters
def MLE(X):
    N = len(X)
    mu = sum(X) / N
    Y = (X - mu * np.ones(X.shape))**2
    sigma = np.sqrt(sum(Y)/N)
    return mu, sigma

np.random.seed(100)

mu_true = 1.0
sigma_true = 0.5
sigma_true2 = 0.5 ** 2

mu_1 = 0.9
mu_2 = 1.2
sigma_1 = 0.9
sigma_2 = 0.45

X10 = np.random.normal(mu_true,sigma_true,10)
X100 = np.random.normal(mu_true,sigma_true,100)
X1000 = np.random.normal(mu_true,sigma_true,1000)

mu_mle10,sigma_mle10 = MLE(X10)
mu_mle100,sigma_mle100 = MLE(X100)
mu_mle1000,sigma_mle1000 = MLE(X1000)

X = np.linspace(-1, 3, num=101, endpoint=True)
P1 = compute_prob(X, mu_1, sigma_1**2)
P1_X = compute_prob(X10, mu_1, sigma_1**2)
P2 = compute_prob(X, mu_2, sigma_2**2)
P2_X = compute_prob(X10, mu_2, sigma_2**2)
PTrue = compute_prob(X, mu_true, sigma_true**2)
PTrue_X = compute_prob(X10, mu_true, sigma_true**2)
PMLE = compute_prob(X, mu_mle10, sigma_mle10**2)
PMLE_X = compute_prob(X10, mu_mle10, sigma_mle10**2)

# probabilities

fig = plt.figure()
plt.subplot(4,1,1)
for i in range(len(X10)):
    plt.plot([X10[i],X10[i]], [0.0,P1_X[i]], c='skyblue')
plt.scatter(X10,np.zeros(len(X10)), marker='x',s=20, c='red')
plt.plot(X, P1, c='darkred',label='setting 1')
plt.axis('equal')
plt.legend()
plt.axis([-1, 3, -0.1, 1.0])

plt.subplot(4,1,2)
for i in range(len(X10)):
    plt.plot([X10[i],X10[i]], [0.0,P2_X[i]], c='skyblue')
plt.scatter(X10,np.zeros(len(X10)), marker='x',s=20, c='red')
plt.plot(X, P2, c='green',label='setting 2')
plt.axis('equal')
plt.legend()
plt.axis([-1, 3, -0.1, 1.0])

plt.subplot(4,1,3)
for i in range(len(X10)):
    plt.plot([X10[i],X10[i]], [0.0,PTrue_X[i]], c='skyblue')
plt.scatter(X10,np.zeros(len(X10)), marker='x',s=20, c='red')
plt.plot(X, PTrue, c='blue',label='true setting')
plt.axis('equal')
plt.legend()
plt.axis([-1, 3, -0.1, 1.0])

plt.subplot(4,1,4)
for i in range(len(X10)):
    plt.plot([X10[i],X10[i]], [0.0,PMLE_X[i]], c='skyblue')
plt.scatter(X10,np.zeros(len(X10)), marker='x',s=20, c='red')
plt.plot(X, PMLE, c='red',label='MLE Setting')
plt.axis('equal')
plt.legend()
plt.axis([-1, 3, -0.1, 1.0])

plt.show()
fig.savefig('Probabilities.pdf')

plt.clf()


Mu, Sigma = np.meshgrid(np.linspace(0.3, 1.7, num=201, endpoint=True),
                        np.linspace(0.3,0.9, num=101, endpoint=True))
LLH10 = compute_log_likelihood(X10, Mu, Sigma**2)
LLH100 = compute_log_likelihood(X100, Mu, Sigma**2)
LLH1000 = compute_log_likelihood(X1000, Mu, Sigma**2)

# likelihood of 10 samples
fig=plt.figure()
zmin = np.min(LLH10)
zmax = np.max(LLH10)
CS = plt.contourf(Mu, Sigma, LLH10, 10, cmap=plt.cm.inferno,
                  vmax=zmax, vmin=zmin)
plt.xlabel('mu')
plt.ylabel('sigma')
plt.colorbar()
# computing the llh for different settings (on the 10 samples)
llh_true = compute_log_likelihood_single(X10,mu_true,sigma_true**2)
llh_1 = compute_log_likelihood_single(X10,mu_1,sigma_1**2)
llh_2 = compute_log_likelihood_single(X10,mu_2,sigma_2**2)
llh_mle = compute_log_likelihood_single(X10,mu_mle10,sigma_mle10**2)

plt.scatter([mu_1],[sigma_1],c='darkred',label='setting 1, llh = '+str(round(llh_1,1)))
plt.scatter([mu_2],[sigma_2],c='green',label='setting 2, llh = '+str(round(llh_2,1)))
plt.scatter([mu_true],[sigma_true],c='blue',label='true setting, llh = '+str(round(llh_true,1)))
plt.scatter([mu_mle10],[sigma_mle10],c='red',label='MLE setting, llh = '+str(round(llh_mle,1)))
plt.legend()
plt.title('log-likelihood map of 10 samples')
plt.show()
fig.savefig('llh-S10.png')

# likelihood of 100 samples
plt.clf()
fig=plt.figure()
zmin = np.min(LLH100)
zmax = np.max(LLH100)
CS = plt.contourf(Mu, Sigma, LLH100, 10, cmap=plt.cm.inferno,
                  vmax=zmax, vmin=zmin)
plt.xlabel('mu')
plt.ylabel('sigma')
plt.colorbar()

plt.scatter([mu_1],[sigma_1],c='darkred',label='setting 1')
plt.scatter([mu_2],[sigma_2],c='green',label='setting 2')
plt.scatter([mu_true],[sigma_true],c='blue',label='true setting')
plt.scatter([mu_mle100],[sigma_mle100],c='red',label='MLE setting')
plt.legend()
plt.title('log-likelihood map of 100 samples')
plt.show()
fig.savefig('llh-S100.png')
#


# likelihood of 1000 samples
plt.clf()
fig=plt.figure()
zmin = np.min(LLH1000)
zmax = np.max(LLH1000)
CS = plt.contourf(Mu, Sigma, LLH1000, 10, cmap=plt.cm.inferno,
                  vmax=zmax, vmin=zmin)
plt.xlabel('mu')
plt.ylabel('sigma')
plt.colorbar()

plt.scatter([mu_1],[sigma_1],c='darkred',label='setting 1')
plt.scatter([mu_2],[sigma_2],c='green',label='setting 2')
plt.scatter([mu_true],[sigma_true],c='blue',label='true setting')
plt.scatter([mu_mle1000],[sigma_mle1000],c='red',label='MLE setting')
plt.legend()
plt.title('log-likelihood map of 1000 samples')
plt.show()
fig.savefig('llh-S1000.png')
#
