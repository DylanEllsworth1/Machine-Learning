# Assignment 1 - Implement KNN

## Description:

1. funtion `pickUnseenData(df, num)` split the data into two sets T and U
2. We use euclidean distance as our distance metric in function `getDistance(ins1, ins2)`

## Performance:

_We set the size for U=60_

|  k  | average_accuracy |
| :-: | :--------------: |
| 200 |      56.67%      |
| 100 |      47.33%      |
| 50  |      59.0%       |
| 20  |      51.0%       |
| 10  |      56.0%       |
|  5  |      55.33%      |

## How to test our code

## specification

python version: 3.8.6 <br>
libraries we used: pandas, sys, random
