# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:56:01 2023

@author: User
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn import cluster
import err_ranges as err

def data_(filename):
    data = pd.read_csv(filename)
    data = pd.DataFrame(data)
    print(data) 
    print(data.columns)
    # drop null values in rows as part of data cleaning
    data = data.drop(["Indicator Code","Indicator Name","Country Code","2021","2020"],axis=1)
    data = data.replace(np.NaN,0)
    u = ["Benin","Bangladesh","Bahrain","Brazil","Colombia","Canada"]
    dt = data["Country Name"].isin(u)
    data = data[dt]
    print(data)
    d_t = np.transpose(data)
    d_t = d_t.reset_index()
    d_t = d_t.rename(columns={"index":"year"})
    d_t = d_t.drop(0,axis=0)
    #data = data.iloc[:,0:10]
    d_t = d_t.rename(columns={18:"Benin",20:"Bangladesh",22:"Bahrain",29:"Brazil",35:"Colombia",45:"Canada"})
    d_t["year"] = pd.to_numeric(d_t["year"])
    d_t["Bahrain"] = pd.to_numeric(d_t["Bahrain"])
    d_t["Brazil"] = pd.to_numeric(d_t["Brazil"])
    d_t["Canada"] = pd.to_numeric(d_t["Canada"])
    d_t = d_t.dropna()
    return d_t,data
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
bdata,borg = data_("D:\\applaid_last_prj\\data_set\\Birth rate, crude (per 1,000 people).csv")
pop,poporg = data_("D:\\applaid_last_prj\\data_set\\Population, total.csv")
EPC,EPC_org = data_("D:\\applaid_last_prj\\data_set\\Electric power consumption (kWh per capita).csv")
co2_emmission,co2_emmission_org = data_("D:\\applaid_last_prj\\data_set\\CO2 emissions (kt).csv")


Bahrain = pd.DataFrame()
Bahrain["Year"] = bdata["year"]
Bahrain["Birth rate"] = bdata["Bahrain"]
Bahrain["Population_total"] = pop["Bahrain"]
Bahrain["Electric_power_consumption"] = EPC["Bahrain"]
Bahrain["co2_emission"] = co2_emmission["Bahrain"]
Bahrain = Bahrain.iloc[30:60,:]


Brazil = pd.DataFrame()
Brazil["Year"] = bdata["year"]
Brazil["Birth rate"] = pd.to_numeric(bdata["Brazil"])
Brazil["Population_total"] = pd.to_numeric(pop["Brazil"])
Brazil["Electric_power_consumption"] = EPC["Brazil"]
Brazil["co2_emission"] = co2_emmission["Brazil"]
Brazil = Brazil.iloc[30:69,:]


Canada = pd.DataFrame()
Canada["Year"] = bdata["year"]
Canada["Birth rate"] = pd.to_numeric(bdata["Canada"])
Canada["Population_total"] = pd.to_numeric(pop["Canada"])
Canada["Electric_power_consumption"] = EPC["Canada"]
Canada["co2_emission"] = co2_emmission["Canada"]
Canada = Canada.iloc[30:60,:]


plt.figure()
plt.title("Correlation map of Bahrain")
cor = Bahrain.corr()
sns.heatmap(data=cor,annot=False,cmap="jet")
plt.show()    


def set_mat(country):
    pd.plotting.scatter_matrix(country, figsize=(14.0, 12.0))
    plt.tight_layout()
    plt.show()
set_mat(Bahrain)
set_mat(Brazil)
set_mat(Canada)
    

# clustering

kmean = cluster.KMeans(n_clusters=2,max_iter=30)
ptlg = np.array(Bahrain["Electric_power_consumption"]).reshape(-1,1)
spe = np.array(Bahrain["co2_emission"]).reshape(-1,1)
cl = np.concatenate((ptlg,spe),axis=1)
kmean = kmean.fit(cl)
label = kmean.labels_
km_c = kmean.cluster_centers_
col = ["Electric_power_consumption","co2_emission"]
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)
res = pd.concat((result,labels),axis=1)

print(res)
print(km_c)
plt.figure()
plt.title("BAHRAIN Electric_power_consumption vs co2_emission ")
plt.scatter(res["Electric_power_consumption"],res["co2_emission"],c=label,cmap="jet")
plt.xlabel("Electric_power_consumption")
plt.ylabel("co2_emission")
for ic in range(2):
    xc, yc = km_c[ic,:]
    plt.plot(xc, yc, "dk", markersize=7,c="black")
plt.show()


# clustering

kmean = cluster.KMeans(n_clusters=2,max_iter=30)
ptlg = np.array(Brazil["Electric_power_consumption"]).reshape(-1,1)
spe = np.array(Brazil["co2_emission"]).reshape(-1,1)
cl = np.concatenate((ptlg,spe),axis=1)
kmean = kmean.fit(cl)
label = kmean.labels_
km_c = kmean.cluster_centers_
col = ["Electric_power_consumption","co2_emission"]
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)
res = pd.concat((result,labels),axis=1)
print(res)
print(km_c)
plt.figure()
plt.title("BRAZIL Electric_power_consumption vs co2_emission ")
plt.scatter(res["Electric_power_consumption"],res["co2_emission"],c=label,cmap="jet")
plt.xlabel("Electric_power_consumption")
plt.ylabel("co2_emission")
for ic in range(2):
    xc, yc = km_c[ic,:]
    plt.plot(xc, yc, "dk", markersize=7,c="black")
plt.show()

# clustering

kmean = cluster.KMeans(n_clusters=2,max_iter=30)
ptlg = np.array(Canada["Electric_power_consumption"]).reshape(-1,1)
spe = np.array(Canada["co2_emission"]).reshape(-1,1)
cl = np.concatenate((ptlg,spe),axis=1)
kmean = kmean.fit(cl)
label = kmean.labels_
km_c = kmean.cluster_centers_
col = ["Electric_power_consumption","co2_emission"]
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)
res = pd.concat((result,labels),axis=1)

print(res)
print(km_c)
plt.figure()
plt.title("CANADA Electric_power_consumption vs co2_emission ")
plt.scatter(res["Electric_power_consumption"],res["co2_emission"],c=label,cmap="jet")
plt.xlabel("Electric_power_consumption")
plt.ylabel("co2_emission")
for ic in range(2):
    xc, yc = km_c[ic,:]
    plt.plot(xc, yc, "dk", markersize=7,c="black")
plt.show()

# Bahrain
Bahrain["NORM_CO2_emission"] = Bahrain["co2_emission"]/Bahrain["co2_emission"].abs().max() 
print(Bahrain)
param,cparm = opt.curve_fit(exp_,Bahrain["Year"],Bahrain["NORM_CO2_emission"],p0=[4e8,0.2])
sigma = np.sqrt(np.diag(cparm))
low,up = err.err_ranges(Bahrain["Year"],exp_,param,sigma)
Bahrain["fit"] = exp_(Bahrain["Year"],*param)
plt.figure()
Bahrain.plot("Year",["NORM_CO2_emission","fit"])
plt.fill_between(Bahrain["Year"],low,up,alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.title("CO2 emission of Bahrain")
plt.plot(Bahrain["Year"],Bahrain["NORM_CO2_emission"],label="benin")
pred_year = np.arange(1990,2030)
bpred = exp_(pred_year,*param)
plt.plot(pred_year,bpred,label="prediction")
plt.legend()
plt.show()


# BRAZIL
Brazil["NORM_CO2_emission"] = Brazil["co2_emission"]/Brazil["co2_emission"].abs().max() 
param,cparm = opt.curve_fit(exp_,Brazil["Year"],Brazil["NORM_CO2_emission"],p0=[4e8,0.1])
print(*param)
low,up = err.err_ranges(Brazil["Year"],exp_,param,sigma)
Brazil["fit"] = exp_(Brazil["Year"],*param)
plt.figure()
Brazil.plot("Year",["NORM_CO2_emission","fit"])
plt.fill_between(Brazil["Year"],low,up,alpha=0.5)
plt.show()
plt.figure()
plt.title("CO2 emmission of BRAZIL")
plt.plot(Brazil["Year"],Brazil["NORM_CO2_emission"],label="Brazil")
pred_year = np.arange(1990,2040)
brapred = exp_(pred_year,*param)
plt.plot(pred_year,brapred,label="prediction")
plt.legend()
plt.show()



# CANADA
Canada["NORM_CO2_emission"] = Canada["co2_emission"]/Canada["co2_emission"].abs().max()
param,cparm = opt.curve_fit(exp_,Canada["Year"],Canada["NORM_CO2_emission"],p0=(73233967692.102798,0.04))
print(*param)
low,up = err.err_ranges(Canada["Year"],exp_,param,sigma)
Canada["fit"] = exp_(Canada["Year"],*param)
plt.figure()
Canada.plot("Year",["NORM_CO2_emission","fit"])
plt.fill_between(Canada["Year"],low,up,alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.title("CO2 emmission of CANADA")
plt.plot(Canada["Year"],Canada["NORM_CO2_emission"],label="Canada")
pred_year = np.arange(1990,2040)
capred = exp_(pred_year,*param)
plt.plot(pred_year,capred,label="prediction")
plt.legend()
plt.show()


# clustering

comp = pd.DataFrame()
comp["Bahrain_co2"] = Bahrain["Population_total"]
comp["Canada_co2"] = Canada["Population_total"]
#comp["brazil_co2"] = Brazil["Population_total"]
print(comp)

"""
kmean = cluster.KMeans(n_clusters=2)
kmean = kmean.fit(comp)
label = kmean.labels_
km_c = kmean.cluster_centers_
col = ["Bahrain_co2","Canada_co2"]
labels = pd.DataFrame(label,columns=['Cluster ID'])
result = pd.DataFrame(cl,columns=col)
res = pd.concat((result,labels),axis=1)
plt.title("Bahrain_co2 vs Canada_co2")
plt.scatter(res["Bahrain_co2"],res["Canada_co2"],c=label,cmap="jet")
plt.xlabel("Bahrain_co2")
plt.ylabel("Canada_co2")
for ic in range(2):
    xc, yc = km_c[ic,:]
    plt.plot(xc, yc, "dk", markersize=7,c="black")
plt.show()
"""


