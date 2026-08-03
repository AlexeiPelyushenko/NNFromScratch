import numpy as np
import math

def erf(x):
    """
    Abramowitz & Stegun approximation for the error function
    X can be a single number or numpy array
    """
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    
    sign = np.where(x >= 0, 1, -1)
    x = np.abs(x)
    t = 1 / (1 + p*x)
    
    output = 1 - (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * np.power(np.e, -x**2)
    return sign * output

def standard_normal_pdf(x):
    return 1/(2 * np.pi)**0.5 * np.power(np.e, -x**2/2)

def standard_normal_cdf(x):
    return 0.5 * (1 + erf(x / 2**0.5))

def truncated_variance(x):
    """
    Calculates variance of a standard normal Gaussian distribution if it were truncated after x standard deviations.
    
    x: points that are x standard deviations away are truncated
    """
    return 1 - (2 * x * standard_normal_pdf(x)) / (2 * standard_normal_cdf(x) - 1)
    

if __name__ == "__main__":
    print(standard_normal_pdf(3))
    print(standard_normal_cdf(3))
    print(truncated_variance(3))
    