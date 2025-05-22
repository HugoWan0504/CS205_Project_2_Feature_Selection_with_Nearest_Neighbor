#include <bits/stdc++.h>
#include "header_functions.h"
using namespace std;

int main() {
    cout << "Welcome to Hugo Wan's Feature Selection Algorithm!\n";
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
    cout << "Your choice: ";
    cin >> choice;

    if (choice == 1) {
        forward_selection(features, labels, 1);
    } else if (choice == 2) {
        backward_elimination(features, labels, 1);
    } else {
        cout << "Invalid option.\n";
    }


    return 0;
}
