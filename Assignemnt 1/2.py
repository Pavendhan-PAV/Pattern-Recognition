import numpy as np 
import math as m 

def q_form(Ht,A,H):
    result = 0
    result = ((Ht @A ) @ H)
    return result

def main():
    A = ([1,0.135,0.195,0.137,0.157],
            [0.135,1,0.2,0.309,0.143],
            [0.195,0.2,1,0.157,0.122], 
            [0.137,0.309,0.157,1,0.195], 
            [0.157,0.143,0.122,0.195,1])
  
    Ht = np.array([0.5,0.5,-0.5,-0.25,-0.25])
   
    H = Ht.T

    print("Quadratic form distance: "+str(q_form(Ht,A,H)))
if __name__=="__main__":
    main() 