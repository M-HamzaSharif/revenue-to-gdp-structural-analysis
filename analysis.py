    #The Start

    #Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


    #Pulling Data from relevant csv file
data = pd.read_csv("data/Revenueasapctofgdpdata.csv")

    #Data Validity & Checks
data = data.sort_values("FY")
print(data.head())
print(data.tail())
print(data.isnull().sum())

    #Plotting
plt.figure()
plt.plot(data["FY"], data["rev_per_gdp"])
plt.axvline(x=2010, linestyle='--')  # boundary year (adjust if your FY labels differ)
plt.xlabel("FY")
plt.ylabel("Revenue (% of GDP)")
plt.title("Revenue as % of GDP (1990–2024)")
plt.show()
    #Breaking
pre = data[data["FY"] <= 2009]
post = data[data["FY"] >= 2011]

print("Pre mean:", pre["rev_per_gdp"].mean())
print("Post mean:", post["rev_per_gdp"].mean())
print("Difference (Post - Pre):", post["rev_per_gdp"].mean() - pre["rev_per_gdp"].mean())

    #Training on Pre-2010 rev%gdp
train = data[data["FY"] <= 2009].copy()
X_train = train[["FY"]]
y_train = train["rev_per_gdp"]

lin = LinearRegression()
lin.fit(X_train, y_train)

    #Complete Prediction
X_all = data[["FY"]]
data["lin_pred"] = lin.predict(X_all)

    #Plotting graphs over each other
plt.figure()
plt.plot(data["FY"], data["rev_per_gdp"], label="Actual")
plt.plot(data["FY"], data["lin_pred"], label="Linear (trained on <=2009)")
plt.axvline(x=2010, linestyle='--')
plt.xlabel("FY")
plt.ylabel("Revenue (% of GDP)")
plt.title("Actual vs Linear Counterfactual")
plt.legend()
plt.show()

    #Post-period gaps
post = data[data["FY"] >= 2011].copy()
post["gap"] = post["rev_per_gdp"] - post["lin_pred"]

print("Average gap:", post["gap"].mean())
print("Median gap:", post["gap"].median())
print("Cumulative gap:", post["gap"].sum())
print("% years below prediction:", (post["gap"] < 0).mean() * 100)

    #Some Features of the predicted Line
print("Intercept:", lin.intercept_)
print("Slope per year:", lin.coef_[0])

    #The End