# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

from math import pow

def function_value(x, coefficient):
    value=0
    for power in range(len(coefficient)):
        value = (pow(x,power))*coefficient[-(power+1)]
    return value

def steffensen_method(coefficient, p0, tol, num_iteration):
    
    while num_iteration >= 0:
        
        p1 = p0+ function_value(p0,coefficient)
        p2 = p1 +function_value(p1,coefficient)
        
        p = p2 - ((pow((p2-p1),2))/(p2-2*p1+p0))
        
        if (abs(p-p0)<tol):
            print("answer after {} iteration".format(num_iteration))
            print(p)
            break        
        else:
            p0 = p
        num_iteration-=1
    if(num_iteration <= 0):
            print('Method failed after {}'.format(num_iteration))
    print("Function value a convergence point",function_value(p,coefficient))
    return
        
def main():
    coefficient = [float(x) for x in input("Enter the coefficeint of the function, space seperated.\nAnd place 0 if the intermediate power is not required for function\n").split()]
    p0 = float(input("Enter the initial guess"))
    tol = float(input("Enter the tolerance vlaue"))
    num_iteration = float(input("Enter the number of iteration"))
    '''
    coefficient=[1,-2,-56]
    p0=0.7
    tol=0.0005
    num_iteration = 100000
    '''
    steffensen_method(coefficient, p0, tol, num_iteration)
    
    
if __name__ == "__main__" :
    main()
    
        



































