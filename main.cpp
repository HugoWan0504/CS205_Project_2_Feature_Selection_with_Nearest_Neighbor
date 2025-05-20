#include <bits/stdc++.h>
#include "header_functions.h"
using namespace std;

int main() {
    cout << "Welcome to Hugo Wan's Feature Selection Algorithm.\n";
    cout << "Type in the name of the file to test: ";
    
    string filename;
    cin >> filename;

    vector<vector<double>> features;
    vector<int> labels;
    load_data(filename, features, labels);

    int choice;
    cout << "Type the number of the algorithm you want to run.\n";
    cout << "1) Forward Selection\n";
    cout << "2) Backward Elimination\n";
    cin >> choice;

    if (choice == 1) {
        forward_selection(features, labels);
    } else {
        cout << "Backward Elimination not yet implemented.\n";
    }

    return 0;
}
