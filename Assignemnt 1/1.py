import math

def kl_dist(h1,h2,size):
    sum1=0
    sum2=0
    for i in range(size):
        sum1+=h1[i]*math.log(h1[i]/h2[i],2)
        sum2+=h2[i]*math.log(h2[i]/h1[i],2)
    return sum1,sum2

def bh_dist(h1,h2,size):
    sum=0
    for i in range(size):
        sum+=math.sqrt(h1[i]*h2[i])
    return (-1*math.log(sum,2))

def main():
        size=8
        h1=[ 0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04]
        h2=[ 0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02]
        print("\nKL-Distance between (H1 and H2) & (H2 and H1) [BITS - BASE 2]: "+str(kl_dist(h1,h2,size)))
        print("\nBhattacharyya Distance between (H1 and H2) [BITS - BASE 2]: "+str(bh_dist(h1,h2,size))+"\n")
    

if __name__=="__main__":
    main()