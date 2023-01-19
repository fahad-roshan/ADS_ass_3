# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:59:23 2023

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn import cluster
import err_ranges as err
from sklearn.preprocessing import normalize

def data_(filename):
    data = pd.read_csv(filename)
    data = pd.DataFrame(data)
    print(data) 
    print(data.columns)
    # drop null values in rows as part of data cleaning
    data = data.drop(["Indicator Code","Indicator Name","Country Code","2021"],axis=1)
    print(data.shape)
    data = data.replace(np.NaN,0)
    print(data)
    print(data.dropna(axis=0).shape)
    print(data.shape)
    u = ["Benin","Bangladesh","Bahrain","Brazil","Colombia","Canada"]
    dt = data["Country Name"].isin(u)
    # transposing the dataframe
    print(data[dt])
    data = data[dt]
    d_t = np.transpose(data)
    d_t = d_t.reset_index()
    d_t = d_t.rename(columns={"index":"year"})
    d_t = d_t.drop(0,axis=0)
    #data = data.iloc[:,0:10]
    d_t = d_t.rename(columns={18:"Benin",20:"Bangladesh",22:"Bahrain",29:"Brazil",35:"Colombia",45:"Canada"})
    d_t["year"] = pd.to_numeric(d_t["year"])
    d_t["Bahrain"] = pd.to_numeric(d_t["Bahrain"])
    return d_t
def linfunc(x, a, b):
    y = a*x + b
    return y
def exp_(t,n0, g):
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f
def log_(t, scale, growth, t0):
    """ Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f
bdata = data_("D:\\applaid_last_prj\\data_set\\Birth rate, crude (per 1,000 people).csv")
pop = data_("D:\\applaid_last_prj\\data_set\\Population, total.csv")
school_pri = data_("D:\\applaid_last_prj\\data_set\\School enrollment, primary (% gross).csv")
trained_teacher = data_("D:\\applaid_last_prj\\data_set\\Trained teachers in primary education (% of total teachers).csv")
persistence_to_last = data_("D:\\applaid_last_prj\\data_set\\Persistence to last grade of primary, total (% of cohort).csv")
Bahrain = pd.DataFrame()
Bahrain["Year"] = bdata["year"]
Bahrain["Birth rate"] = bdata["Bahrain"]
Bahrain["Population_total"] = pop["Bahrain"]
Bahrain["school_primary_enrol"] = school_pri["Bahrain"]
Bahrain["trained_teacher"] = trained_teacher["Bahrain"]
Bahrain["persistence_to_last_grade"] = persistence_to_last["Bahrain"]
print(Bahrain)

Brazil = pd.DataFrame()
Brazil["Year"] = bdata["year"]
Brazil["Birth rate"] = pd.to_numeric(bdata["Brazil"])
Brazil["Population_total"] = pd.to_numeric(pop["Brazil"])
Brazil["school_primary_enrol"] = pd.to_numeric(school_pri["Brazil"])
Brazil["trained_teacher"] = pd.to_numeric(trained_teacher["Brazil"])
Brazil["persistence_to_last_grade"] = pd.to_numeric(persistence_to_last["Brazil"])
print(Brazil)

Canada = pd.DataFrame()
Canada["Year"] = bdata["year"]
Canada["Birth rate"] = pd.to_numeric(bdata["Canada"])
Canada["Population_total"] = pd.to_numeric(pop["Canada"])
Canada["school_primary_enrol"] = pd.to_numeric(school_pri["Canada"])
Canada["trained_teacher"] = pd.to_numeric(trained_teacher["Canada"])
Canada["persistence_to_last_grade"] = pd.to_numeric(persistence_to_last["Canada"])
print(Canada)

"""
cor = Bahrain.corr()
# plotting the heatmap
sns.heatmap(data=cor,annot=False,cmap="jet")

pd.plotting.scatter_matrix(Bahrain, figsize=(14.0, 12.0))
plt.tight_layout()
plt.show()

# clustering

kmean = cluster.KMeans(n_clusters=4)
ptlg = np.array(Bahrain["persistence_to_last_grade"]).reshape(-1,1)
spe = np.array(Bahrain["school_primary_enrol"]).reshape(-1,1)
cl = np.concatenate((ptlg,spe),axis=1)
kmean = kmean.fit(cl)
label = kmean.labels_
km_c = kmean.cluster_centers_
col = ["persistence_to_last_grade","school_primary_enrol"]
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)
res = pd.concat((result,labels),axis=1)

print(res)
print(km_c)
plt.figure()
plt.title("Bahrain(persistence_to_last_grade vs school_primary_enrol)")
plt.scatter(res["persistence_to_last_grade"],res["school_primary_enrol"],c=label,cmap="jet")
plt.xlabel("persistence_to_last_grade")
plt.ylabel("school_primary_enrol")
for ic in range(4):
    xc, yc = km_c[ic,:]
    plt.plot(xc, yc, "dk", markersize=7,c="black")
plt.show()


# clustering

kmean = cluster.KMeans(n_clusters=4)
ptlg = np.array(Brazil["persistence_to_last_grade"]).reshape(-1,1)
spe = np.array(Brazil["school_primary_enrol"]).reshape(-1,1)
cl = np.concatenate((ptlg,spe),axis=1)
kmean = kmean.fit(cl)
label = kmean.labels_
km_c = kmean.cluster_centers_
col = ["persistence_to_last_grade","school_primary_enrol"]
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)
res = pd.concat((result,labels),axis=1)

print(res)
print(km_c)
plt.figure()
plt.title("Brazil(persistence_to_last_grade vs school_primary_enrol)")
plt.scatter(res["persistence_to_last_grade"],res["school_primary_enrol"],c=label,cmap="jet")
plt.xlabel("persistence_to_last_grade")
plt.ylabel("school_primary_enrol")
for ic in range(4):
    xc, yc = km_c[ic,:]
    plt.plot(xc, yc, "dk", markersize=7,c="black")
plt.show()

# clustering

kmean = cluster.KMeans(n_clusters=4)
ptlg = np.array(Canada["persistence_to_last_grade"]).reshape(-1,1)
spe = np.array(Canada["school_primary_enrol"]).reshape(-1,1)
cl = np.concatenate((ptlg,spe),axis=1)
kmean = kmean.fit(cl)
label = kmean.labels_
km_c = kmean.cluster_centers_
col = ["persistence_to_last_grade","school_primary_enrol"]
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)
res = pd.concat((result,labels),axis=1)

print(res)
print(km_c)
plt.figure()
plt.title("Canada(persistence_to_last_grade vs school_primary_enrol)")
plt.scatter(res["persistence_to_last_grade"],res["school_primary_enrol"],c=label,cmap="jet")
plt.xlabel("persistence_to_last_grade")
plt.ylabel("school_primary_enrol")
for ic in range(4):
    xc, yc = km_c[ic,:]
    plt.plot(xc, yc, "dk", markersize=7,c="black")
plt.show()
"""

Bahrain["NORM_birth_rate"] = Bahrain["Birth rate"]/Bahrain["Birth rate"].abs().max() 
print(Bahrain)
# Bahrain
param,cparm = opt.curve_fit(exp_,Bahrain["Year"],Bahrain["NORM_birth_rate"],p0=[4e8,0.2])
sigma = np.sqrt(np.diag(cparm))
low,up = err.err_ranges(Bahrain["Year"],exp_,param,sigma)
Bahrain["Birth rate_fit"] = exp_(Bahrain["Year"],*param)
plt.figure()
Bahrain.plot("Year",["NORM_birth_rate","Birth rate_fit"])
plt.fill_between(Bahrain["Year"],low,up,alpha=0.9)
plt.legend()
plt.show()
plt.figure()
plt.title("Birth rate of Bahrain")
plt.plot(Bahrain["Year"],Bahrain["NORM_birth_rate"],label="benin")
pred_year = np.arange(1960,2030)
bpred = exp_(pred_year,*param)
plt.plot(pred_year,bpred,label="prediction")
plt.legend()
plt.show()


# BRAZIL
Brazil["NORM_birth_rate"] = Brazil["Birth rate"]/Brazil["Birth rate"].abs().max() 
print(Bahrain)
param,cparm = opt.curve_fit(exp_,Brazil["Year"],Brazil["NORM_birth_rate"],p0=[4e8,0.1])
print(*param)
Brazil["Brazil_fit"] = exp_(Brazil["Year"],*param)
plt.figure()
Brazil.plot("Year",["NORM_birth_rate","Brazil_fit"])
plt.fill_between(Bahrain["Year"],low,up,alpha=0.9)
plt.show()
plt.figure()
plt.title("Birth rate of BRAZIL")
plt.plot(Brazil["Year"],Brazil["NORM_birth_rate"],label="Brazil")
pred_year = np.arange(1960,2030)
brapred = exp_(pred_year,*param)
plt.plot(pred_year,brapred,label="prediction")
plt.legend()
plt.show()



# CANADA
Canada["NORM_birth_rate"] = Canada["Birth rate"]/Canada["Birth rate"].abs().max()
param,cparm = opt.curve_fit(exp_,Canada["Year"],Canada["NORM_birth_rate"],p0=(73233967692.102798,0.04))
print(*param)
Canada["Canada_fit"] = exp_(Canada["Year"],*param)
plt.figure()
Canada.plot("Year",["NORM_birth_rate","Canada_fit"])
plt.fill_between(Bahrain["Year"],low,up,alpha=0.9)
plt.legend()
plt.show()
plt.figure()
plt.title("Birth rate of CANADA")
plt.plot(Canada["Year"],Canada["NORM_birth_rate"],label="Canada")
pred_year = np.arange(1960,2030)
capred = exp_(pred_year,*param)
plt.plot(pred_year,capred,label="prediction")
plt.legend()
plt.show()
