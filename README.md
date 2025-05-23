# CS205 Project 2 Feature Selection with Nearest Neighbor
### Student Name: Hugo Wan
### NetID: twan012
### SID: 862180666

## How to play?
* Run command to compile: 
```bash
g++ -std=c++17 -o main main.cpp 
```
* Run command to play test my code: 
```bash
./main
```
* Run command to open the CSV graph of the results after running the search: 
```bash
python3 result_graph.py
```

## Input Summary
* Enter test filename to input a dataset
* Enter choice 1 to select Forward Selection Function
* Enter choice 2 to select Backward Elimination Function

## Result Summary
* Output the best feature subset with the best accuracy when the accuracy is starting to decrease
* A bar plot of the best accuracy of each iteration is shown. Only plot it up to the first decreasing accuarcy.