from sympy import factorint
from functools import reduce
from fractions import Fraction
import numpy as np


# code from https://codefights.com/feed/oJoQuXaP2uT2gPjyM
def modinv(a, m):
    MMI = lambda A, n, s=1, t=0, N=0: (n < 2 and t % N or MMI(n, A % n, t, s - A // n * t, N or n), -1)[n < 1]
    return MMI(a,m)

def binary(x):
    return [int(i) for i in bin(x)[2:]]

def repeated_square(base, exp, mod):
    binaryRep = binary(exp)
    print('{} in binaty is {}'.format(exp,''.join([str(x) for x in binaryRep])))
    rev = binaryRep[::-1]
    factors = [2**i for i,u in list(enumerate(rev)) if u == 1]
    print('{} = {}'.format(exp,'+'.join([str(x) for x in factors])))
    result = []
    k = 2
    res = (base**2) % mod
    print('{0}^{1} mod {2} = {3}'.format(base, k, mod,res))
    if k in factors:
        result.append(res)
    while k != max(factors):
        k = k*2
        print('{0}^{1} = {2}^2 mod {3} = '.format(base, k, res,mod), end="")
        res =  (res**2) % mod
        print('{0}.'.format(res))
        if k in factors:
            result.append(res)

    print('*'.join([str(x) for x in result]),end="")
    answer = reduce(lambda x,y: x*y, result)
    print('= {}'.format(answer))
    m = answer % mod
    print('{0} mod {1} = {2} '.format(answer, mod, m))


def is_EC_valid(A,B):
    res = 4*(A**3) + 27*(B**2)
    print('4({})^3 + 27({})^2 = {}'.format(A,B,res))
    return False if res == 0 else True

def is_on_EC(x,y,f):
    y_sq = y**2
    x_fun = f(x)
    print('{}^2 = {}'.format(y,x_fun))
    return True if y_sq == x_fun else False

def doublePoint(x,y,A):
    a = (3*(x**2) + A)/(2*y)
    print('a = ({}^2 + {})/2{} = {}'.format(x,A,y,a))
    x3 = a**2 - (2*x)
    print('x3 = {}^2 - 2({}) = {}'.format(a,x,x3))
    y3 = -y -(a)*(x3-x)
    print('y3 = -{} - ({})({}-{}) = {}'.format(y,a,x3,x,y3))
    return x3, y3

def addPoint(x1,y1,x2,y2):
    a = Fraction((y2 - y1),(x2-x1))
    print('a = ({} - {})/({} - {}) = {}'.format(y2,y1,x2,x1,a))
    x3 = a**2 - x2 - x1
    print('x3 = ({})^2 - {} - {} = {}'.format(a,x2,x1,x3))
    y3 = -y1 -(a)*(x3 - x1)
    print('y3 = -{} - ({})({}-{}) = {}'.format(y1,a,x3,x1,y3))
    return x3, y3


def addPointMod(x1,y1,x2,y2,p):
    x_inv = modinv(x2-x1,p)
    print('({}-{})^-1 = {}'.format(x2,x1,x_inv))
    a = (y2 - y1)*(x_inv) % p
    print('a = ({} - {})*{} mod {} = {}'.format(y2,y1,x_inv,p,a))
    x3 = (a ** 2 - x2 - x1) % p
    print('x3 = ({})^2 - {} - {} mod {} = {}'.format(a,x2,x1,p,x3))
    y3 = (-y1 -(a)*(x3 - x1)) % p
    print('y3 = -{} - ({})({}-{}) mod {} = {}'.format(y1,a,x3,x1,p,y3))
    return x3, y3

def doublePointMod(x,y,A,p):
    y_inv = modinv(2*y,p)
    print('2({})^-1 = {}'.format(y,y_inv))
    a = ((3*(x**2) + A)*y_inv) % p
    print('a = ({}^2 + {})*({}) mod {} = {}'.format(x,A,y_inv,p,a))
    x3 = (a**2 - (2*x)) % p
    print('x3 = {}^2 - 2({}) mod {} = {}'.format(a,x,p,x3))
    y3 = (-y -(a)*(x3-x)) % p
    print('y3 = -{} - ({})({}-{}) mod {} = {}'.format(y,a,x3,x,p,y3))
    return x3, y3

def kPointMod(k,x,y,A,p):
    res = doublePointMod(x,y,A,p)
    if k > 2:
        for i in range(3,k+1):
            print("Hi",res[0])
            res = addPointMod(x,y,res[0],res[1],p)
    return res

def latticeReduce(vec1, vec2):
    i = 1
    t = float("inf")
    # print("Iteration {}".format(i))
    #
    # print("V1 is {}",vec1)
    # print("V2 is {}",vec2)
    #
    # mag_v1 = np.linalg.norm(vec1)
    # mag_v2 = np.linalg.norm(vec2)
    #
    # print("||v1|| = {}, ||V2|| = {}".format(mag_v1,mag_v2))
    #
    # if (mag_v2 < mag_v1):
    #     tmp = vec1
    #     vec1 = vec2
    #     vec2 = tmp
    #     print("Swaped")
    # else:
    #     print("Not Swaped")
    #
    #
    # print("New V1 is {}", vec1)
    # print("New V2 is {}", vec2)
    #
    # u = np.dot(vec1,vec2)/(np.dot(vec1,vec1))
    # print('u = ({}*{})/(({}*{}) = {}.'.format(vec1,vec2,vec1,vec1,u))
    #
    # t = round(u)
    # print('t = {}'.format(t))
    #
    # if t == 0:
    #     return vec1,vec2
    # else:
    #     nextV1 = vec1
    #     nextV2 = vec2 - (t*vec1)
    #
    #     print("Next vec1 is {}".format(nextV1))
    #     print("Next vec2 is {}".format(nextV2))
    #     i = 2

    while t != 0:
        print("Iteration {}".format(i))

        print("V1 is {}", vec1)
        print("V2 is {}", vec2)

        mag_v1 = np.linalg.norm(vec1)
        mag_v2 = np.linalg.norm(vec2)

        print("||v1|| = {}, ||V2|| = {}".format(mag_v1, mag_v2))

        if (mag_v2 < mag_v1):
            tmp = vec1
            vec1 = vec2
            vec2 = tmp
            print("Swaped")
        else:
            print("Not Swaped")

        print("New V1 is {}", vec1)
        print("New V2 is {}", vec2)

        u = np.dot(vec1, vec2) / (np.dot(vec1, vec1))
        print('u = ({}*{})/(({}*{}) = {}.'.format(vec1, vec2, vec1, vec1, u))

        t = round(u)
        print('t = {}'.format(t))

        if t == 0:
            return vec1, vec2
        else:
            vec1 = vec1
            vec2 = vec2 - (t * vec1)

            print("Next vec1 is {}".format(vec1))
            print("Next vec2 is {}".format(vec2))
            i +=1
        print()



print(latticeReduce(np.array([90,123]),np.array([56,76])))



