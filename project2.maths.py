import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
z = 200

df = pd.DataFrame({
    "study_hours": np.random.randint(1, 10, z),
    "attendance": np.random.randint(60, 100, z),
    "group_discussion": np.random.choice(["Yes", "No"], z),
    "previous_test_score": np.random.randint(28, 100, z)
})
print(df)

def result(row):
    if row["study_hours"] > 7 and row["attendance"] > 75 and row["previous_test_score"] > 50:
        return "Pass"
    else:
        return "Fail"

df["final_exam_pass"] = df.apply(result, axis=1)

print("\n===Dataset Preview===")
print(df.head())

df["high_study"] = df["study_hours"] > 7
df["high_attendance"] = df["attendance"] > 80
df["group_yes"] = df["group_discussion"] == "Yes"


P_pass = df["final_exam_pass"].value_counts(normalize=True)["Pass"]

print("\nProbability of Passing:", P_pass)

print("\nEvent Probabilities:")
print("P(Study > 7):", df["high_study"].mean())
print("P(Attendance > 80):", df["high_attendance"].mean())
print("P(Group Discussion = Yes):", df["group_yes"].mean())


P_empirical = df["high_study"].mean()
P_theoretical = 2 / 9
print("\nEmpirical Probability:", P_empirical)
print("Theoretical Probability:", P_theoretical)


p = P_pass
x_vals = [0, 1, 2, 3]

prob_values = []
for x in x_vals:
    from math import comb
    prob = comb(3, x) * (p ** x) * ((1 - p) ** (3 - x))
    prob_values.append(prob)

dist_df = pd.DataFrame({"X": x_vals, "P(X)": prob_values})
print("\n===Probability Distribution===")
print(dist_df)


Mean = np.dot(x_vals, prob_values)
Var= np.dot((np.array(x_vals) - Mean) ** 2, prob_values)
print("\nMean:",Mean)
print("Var:",Var)


P_A = df["high_study"].mean()
P_B = df["high_attendance"].mean()
P_A_and_B = (df["high_study"] & df["high_attendance"]).mean()
print("\nVenn Probabilities:")
print("P(A):", P_A)
print("P(B):", P_B)
print("P(A ∩ B):", P_A_and_B)


a= pd.crosstab(df["group_discussion"], df["final_exam_pass"])
print("\n===Contingency table===")
print(a)

total = len(df)

P_joint = a.loc["Yes", "Pass"] / total
P_pass_marginal = P_pass
P_conditional = a.loc["Yes", "Pass"] / a.loc["Yes"].sum()
print("\nJoint Probability:", P_joint)
print("Marginal Probability:", P_pass_marginal)
print("Conditional Probability:", P_conditional)


print("\n===Check independence===")

if np.isclose(P_conditional,P_pass_marginal,atol=0.05):
    print("Approximately independent")
else:
    print("Dependent events")


P_pass_given_attendance = (
    ((df["high_attendance"]) & (df["final_exam_pass"] == "Pass")).mean()
    / df["high_attendance"].mean())
print("\n===Result===")
print("P(Pass | High attendance):",P_pass_given_attendance)


plt.figure()
sns.histplot(data=df, x="attendance", hue="final_exam_pass",kde=True)
plt.title("Attendance distribution")
plt.show()

plt.figure()
sns.boxplot(data=df, x="final_exam_pass", y="study_hours",color='red')
plt.title("Study hours vs Pass/Fail")
plt.show()

plt.figure()
sns.scatterplot(data=df, x="previous_test_score", y="study_hours",
                hue="final_exam_pass")
plt.title("Score vs Study Hours")
plt.show()

plt.figure()
sns.countplot(data=df, x="final_exam_pass",color='green')
plt.title("Pass & fail count")
plt.show()

plt.figure()
sns.countplot(data=df,x="group_discussion",hue="final_exam_pass")
plt.title("Group discussion VS Result")
plt.show()

plt.figure()
sns.histplot(data=df, x="attendance", hue="final_exam_pass")
plt.title("Attendance distribution")
plt.show()


