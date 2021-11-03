import random
import matplotlib.pyplot as plt
from ADF_test import Stationarity_test

random.seed(5)
non_stationary_series = [random.random()+(i*.02) for i in range(100)]
plt.plot(non_stationary_series)
plt.title('Non-Stationary Time Series')
plt.show()

# Check non stationarity
sTest = Stationarity_test()
sTest.ADF_test(non_stationary_series, printResults = True)
print("Is the time series stationary? {0}".format(sTest.is_stationary))


# stationarity
stationary_series = [random.random() for _ in range(100)]
plt.plot(stationary_series)
plt.title('Stationary Time Series')
plt.show()

sTest.ADF_test(stationary_series, printResults = True)
print("Is the time series stationary? {0}".format(sTest.is_stationary))