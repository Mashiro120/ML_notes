__arthor__='Mashiro'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import pylab
'''
data form (no headers and index, the following shows the form only):
    v1  v2  v3  v4  v5 ...  #---[attribution]
0
1
2
3
4
5
...
[index]
'''
class data_plot(object):
    def __init__(self,filename=None):
        filename = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")
        self.data=pd.read_csv(filename,header=None, prefix="V")
        #if there is headers
        #self.data.columns=self.data.keys()
        self.data.columns = ['Sex', 'Length', 'Diameter', 'Height',
                   'Whole weight', 'Shucked weight',
                   'Viscera weight', 'Shell weight', 'Rings']
        #some treatment for the abalone.data to shift every data into digit
        for i in range(len(self.data.index)):
            if self.data.iloc[i,0]=='M':
                self.data.iloc[i,0]=1
            else:
                self.data.iloc[i,0]=0

    def corMar(self):
        corMatdata=pd.DataFrame(self.data.iloc[:,:].corr())
        plt.pcolor(corMatdata)
        plt.show()
    
    def parallelplot(self):# the last attribution decides the color 
        #get summary to use for scaling
        summary = self.data.describe()
        min_attr = summary.iloc[3,-1]
        max_attr = summary.iloc[7,-1]
        mean_attr = summary.iloc[1,-1]
        sd_attr = summary.iloc[2,-1]
        nrows = len(self.data.index)
        for i in range(nrows):
            #plot rows of data as if they were series data
            dataRow=self.data.iloc[i,:-1]
            labelColor = (self.data.iloc[i,-1] - min_attr) / (max_attr - min_attr)
            #normTarget = (self.data.iloc[i,attr_i] - mean_attr)/sd_attr
            #labelColor = 1.0/(1.0 + math.exp(-normTarget))  
            dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)
        plt.xlabel("Attribute Index")
        plt.ylabel(("Attribute Values"))
        plt.show()
    
    def summary(self):
        # if values ranges closly
        array=self.data.iloc[:,:].values
        plt.boxplot(array)
        plt.xlabel("Attribute Index")
        plt.ylabel(("Quartile Ranges"))
        plt.show()

        # else
        summary=self.data.describe()
        mcolumns=len(self.data.keys())
        data_normalized=self.data.iloc[:,:]
        for i in range(mcolumns):
            mean=summary.iloc[1,i]
            sd=summary.iloc[2,i]
            data_normalized.iloc[:,i]=(data_normalized.iloc[:,i]-mean)/sd
        array2=data_normalized.values
        plt.boxplot(array2)
        plt.xlabel("Attribute Index")
        plt.ylabel(("Quartile Ranges - Normalized "))
        plt.show()
    def qqplot(self,col=1):
        stats.probplot(self.data[self.data.keys()[col]],dist='norm',plot=pylab,rvalue=True)
        pylab.show()
    def scatter_plot(self,attr1=1,attr2=2):
        plt.scatter(self.data[self.data.keys()[attr1]],self.data[self.data.keys()[attr2]])
        plt.xlabel(str(attr1+1)+" attri")
        plt.ylabel(str(attr2+1)+" attri")
        plt.show()
    def full_graph(self):
        self.scatter_plot()
        self.summary()
        self.qqplot()
        self.parallelplot()
        self.corMar()
def main():
    data_show=data_plot()
    data_show.full_graph()

if __name__ == "__main__":
    main()
