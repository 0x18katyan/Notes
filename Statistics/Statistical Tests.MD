# Notes

References:

1. [Machine Learning Mastery - Chi-Squared Test for Machine Learning](https://machinelearningmastery.com/chi-squared-test-for-machine-learning/)

2. [Statistics How To - What is a Contingency Table?](https://www.statisticshowto.com/what-is-a-contingency-table/)

## Chi-Squared Test

Chi-Squared Test is used to find how relevant are input features for a model. Pearson's chi-squared statistical test is method for determining the independence between two categorical variables.

### Contingency Table

A contingency table (also called crosstabs or two-way tables) is the summary of occurrences between two categorical columns such that co-occurrences between values of one column to another is captured in a table. A contingency table could also be thought of as a special type of frequency distribution table where 2 variables are simultaneously displayed.

Example: How many times does Holi fall on a Saturday within the last 100 years?

### Pearson's Chi-Squared Test

$c^2 =$ $\sum_{n=1}^k(\frac{{O_i - E_i}^2}{E_i})$ ,

where $O$ is the *observed* value, $E$ is the *expected* value, and *i* is the *ith* position in the table.

A **small** $chi^2$ value means that there is little relationship between the categorical variables while a **larger** $chi^2$ means that there is a definite correlation between the variables.

