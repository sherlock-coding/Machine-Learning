#LMS算法
import math
import matplotlib.pyplot as plt

#求h(x,theta)
def GetHThetaX(x,theta):
    theta_1 = theta[1:]
    h = theta[0] + sum([xi*thetai for xi,thetai in zip(x,theta_1)])
    return h

#求J(theta)
def GetJ(X,Y,theta):
        J = sum(pow(GetHThetaX(x,theta)-y,2) for x,y in zip(X,Y))/2
        return J
        

#求梯度
#j表示对thetaj求偏导数
def GetGradient(X,Y,theta,j):
        h = []
        for x in X:
                h.append(GetHThetaX(x,theta))
        g = 0;
        if j==0:
                g = sum([hi-yi for hi,yi in zip(h,Y)])
        else:
                g = sum([(hi-yi)*x[j-1] for hi,yi,x in zip(h,Y,X)])
        return g


#n表示n维的特征向量
def LMS_Prediction(X,X1,Y,n,year):
        #theta全都初始化为0
        #theta = [i for i in range(0,n)]
        theta = [-1500,1]
        
        plt.plot(X1,Y,'ro')
        plt.axis([2000,2015,1,15])
        plt.plot(X1,[theta[0]+theta[1]*x for x in X1],'y-')

        #learning rate设置
        alpha = 0.00000001

        steps = 0

        curJ = GetJ(X,Y,theta)
        preJ = curJ


                
        while True:
                temp = []
                for i in range(0,n):
                        g = GetGradient(X,Y,theta,i)
                        #print(g)
                        temp.append(theta[i]-alpha*g)
                #print(theta)
                theta = temp
                print(steps,":",temp)
                steps += 1
                preJ = curJ
                curJ = GetJ(X,Y,theta)
                if curJ>preJ:
                        break
                plt.plot(X1,[theta[0]+theta[1]*x for x in X1],'b-')
        plt.plot(X1,[theta[0]+theta[1]*x for x in X1],'r-')
        plt.show()
        
        

if __name__ == '__main__':
    X = [[2000],[2001],[2002],[2003],[2004],[2005],[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013]]
    Y = [2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,6.853,7.971,8.561,10.000,11.280,12.900]
    X1 = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]
    LMS_Prediction(X,X1,Y,2,2014)
    #X = [[1],[2],[5],[7], [10], [15]]
    #Y = [2, 6, 7, 9, 14, 19]
    #X = [[2000],[2001],[2002],[2003],[2004],[2005],[2006],[2007],[2008],[2009],[2010],[2011],[2012],[2013]]
    #Y = [2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,6.853,7.971,8.561,10.000,11.280,12.900]
    #X1 = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]
    #pylab.scatter(X1,Y)
    #pylab.show()
	
	
