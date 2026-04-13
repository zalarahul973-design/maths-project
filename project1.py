import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
n = 150
np.random.seed(42)

data={
    "Household_ID":["HH"+str(i).zfill(4) for i in range(1,n+1)],
    "Age_of_house_head":np.random.randint(21,70, n),
    "Household_income":np.random.randint(10000,100000, n),
    "Education_level":np.random.choice(["Primary","Secondary","Graduate","PostGrad"], n),
    "Family_size":np.random.randint(1, 8, n),
    "Own_house":np.random.choice(["Yes", "No"],n),
    "Urban_rural":np.random.choice(["Urban","Rural"],n)
}

df=pd.DataFrame(data)
print(df)

# types of data
print(df.dtypes)

# Numerical columns
print("Numerical Columns:")
print(df.select_dtypes(include=['int64', 'float64']).columns)

# Categorical columns
print("\nCategorical Columns:")
print(df.select_dtypes(include=['object']).columns)


#central tendency
print("Mean:")
print(df[["Age_of_house_head","Household_income"]].mean())

print("\nMedian:")
print(df[["Age_of_house_head", "Household_income"]].median())

print("\nMode:")
print(df[["Age_of_house_head", "Household_income"]].mode().iloc[0])


#measures of Dispersion
print("\nRange:")
range_value=df["Household_income"].max() - df["Household_income"].min()
print(range_value)

print("\nvar:")
print(df[["Household_income"]].var())

print("\nstd:")
print(df[["Household_income"]].std)



print("\nIQR (Interquartile Range):")

Q1 = df["Household_income"].quantile(0.25)
Q3 = df["Household_income"].quantile(0.75)

IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)


if IQR > 50000:
    print("Income spread is High")
elif IQR > 20000:
    print("Income spread is Moderate")
else:
    print("Income spread is Low")
    

# Distribution
plt.hist(df["Household_income"],bins=5,color='red',rwidth=0.5)

plt.title("Histogram of household income")
plt.xlabel("Household income")
plt.ylabel("Frequency")
plt.show()



plt.hist(df["Household_income"], bins=10, density=True)

mean = df["Household_income"].mean()
std = df["Household_income"].std()

x = np.linspace(df["Household_income"].min(),
                df["Household_income"].max(), 100)

y = norm.pdf(x, mean, std)

plt.plot(x, y)

plt.title("Income Distribution with Normal Curve")
plt.xlabel("Income")
plt.ylabel("Density")
plt.show()

print("\nSkewness  Kurtosis")
print("Skewness:", df["Household_income"].skew())
print("Kurtosis:", df["Household_income"].kurt())


sns.histplot(df['Household_income'], kde=True)
plt.title("Income Distribution with KDE")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()


sns.boxplot(x="Education_level", y="Family_size",data=df)
plt.title("Family Size by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Family Size")
plt.show()

sns.kdeplot(x=df[ "Age_of_house_head"], y=df['Household_income'], cmap="Blues", fill=True)
plt.title("Age vs Income Distribution")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()
