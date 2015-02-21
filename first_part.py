import numpy as np
import matplotlib.pyplot as plt
import math

#EXCERCISE 1.2.1 Univariate Gaussian Distribution
def UniGaussian(x, mean, std):

    normalize = 1/(std * math.sqrt(2*math.pi))
    exponent = -((x-mean)**2/(2*(std**2)))
    y = normalize*np.exp(exponent)
    return y

#EXCERCISE I.2.1 normal distributions
#parameters
x = np.linspace(-10, 10, 150)
parameters = [(-1, 1), (0, 2), (2, 3)]

#get sequence of distributions and colors
distributions = [UniGaussian(x, i, j)for i, j in parameters]
colors = ["r", "g", "b"]
sequence = zip(distributions, colors)

#iterate through distributions and colors to plot
for d, color in sequence:
        plt.plot(d, color)

plt.grid(True)
plt.show()



#EXCERCISE I.2.22 sampling over a multivariate normal distribution
def sample(mean, L, z):
    y = np.dot(L, z) + np.array(mean).reshape(2,1)
    return y

#size of sample
N = 100

#param of the distribution
mean = [(1.0, 2.0)]
cov = ([0.3, 0.2], [0.2, 0.2])

#apply cholesky
L = np.linalg.cholesky(cov)

#set random generator to get consistent plot
np.random.seed(3)

#get sample
z = np.random.randn(2, N)
data = sample(mean, L, z)

#EXCERCISE I.2.3 get the ML estimate of the mean of the sample
X = data[0]
Y = data[1]

meanX = np.mean(X)
meanY = np.mean(Y)

muML = (meanX, meanY)

#EXCERCISE I.2.4 Maximum likelihood covariance and eigenvectors

#cov maximum likelihood
M = np.zeros((2,2))
diff = [(X - meanX), (Y - meanY)]
Tdiff = np.transpose(diff)
sample_cov = M + (np.dot(diff, Tdiff))/N

#compute eigenvalues, eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)

#scale and translate eigenvectors
e1 = mean + (math.sqrt(eigenvalues[0]) * eigenvectors[0])
e2 = mean + (math.sqrt(eigenvalues[1]) * eigenvectors[1])

#coordinates
e1x = (1, e1[0][0])
e1y = (2, e1[0][1])
e2x = (1, e2[0][0])
e2y = (2, e2[0][1])

#rotation
#define a rotation function
def rotate(estimated_cov, theta):
    R_theta = ([np.cos(np.radians(theta)) , -np.sin(np.radians(theta))], [np.sin(np.radians(theta)), np.cos(np.radians(theta))])
    R_theta_inverse = np.linalg.inv(R_theta)
    R_cov1 =  np.dot(R_theta, estimated_cov)
    R_cov = np.dot(R_cov1, R_theta_inverse)
    L = np.linalg.cholesky(R_cov)
    data1 = sample(mean, L, z)
    return data1


#30 degrees
points1 = rotate(sample_cov, 30)

#60 degrees
points2 = rotate(sample_cov, 60)

#90 degrees
points3 = rotate(sample_cov, 90)

#my_angle

vector = eigenvectors[0]

angle = (math.atan2(vector[0], vector[1]) * 180)/math.pi

points4 = rotate(sample_cov, angle)


#PLOTS

print "Value of the mean: ", mean
print "Value of Maximum likelihood mean: ", muML

#plot rotated covariance
dist30 = plt.scatter(points1[0], points1[1], c='magenta', marker='x', label='theta = 30 degrees')
dist60 = plt.scatter(points2[0], points2[1], c='black', marker='x', label='theta = 60 degrees')
dist90 = plt.scatter(points3[0], points3[1], c='yellow', marker='x', label='theta = 90 degrees')
dist_my_angle = plt.scatter(points4[0], points4[1], c='#FF8000', marker='x', label='rotation that matches x-axis')
plt.legend(handles=[dist30, dist60, dist90, dist_my_angle])
plt.grid(True)
plt.show()

#plot sample distribution and eigenvectors
dist = plt.scatter(X, Y, c='cyan', marker='x', label='Distribution')
a, = plt.plot(1, 2, c='blue', marker='o', ms= 5.0, label='Distribution Mean') #plot mean of the distribution
b, = plt.plot(e1x, e1y, c='red',label='eigenvector1' )
c, = plt.plot(e2x, e2y, c='green', label='eigenvector2')
plt.legend(handles=[dist, a, b, c])
plt.grid(True)
plt.show()

#plot sample distribution and mean
dist = plt.scatter(X, Y, c='cyan', marker='x', label='Distribution')
b, = plt.plot(1, 2, c='blue', marker='o', ms= 5.0, label='Distribution Mean') #plot mean of the distribution
c, = plt.plot(meanX, meanY, c='green', marker='o', ms= 5.0, label='Sample Mean') #plot sample mean
plt.legend(handles=[dist, b, c])
plt.grid(True)
plt.show()
