#ifndef header_functions
#define header_functions

#include <bits/stdc++.h>
using namespace std;

void load_data(const string &filename, vector<vector<double>> &features, vector<int> &labels) {
    ifstream infile(filename);
    if (!infile.is_open()) { // output a error message if the file doesn't open properly
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(infile, line)) {
        istringstream ss(line);
        double value;
        vector<double> feature_row;

        ss >> value; // First value label
        labels.push_back(static_cast<int>(value));

        while (ss >> value) {
            feature_row.push_back(value);
        }

        features.push_back(feature_row);
    }
    infile.close();
}


double leave_one_out_cross_validation_stub(const vector<vector<double>> &, const vector<int> &, const vector<int> &) {                                    
    return rand() % 100 + 1;  // Accuracy between 1 and 100 %
}


void forward_selection(const vector<vector<double>> &features, const vector<int> &labels) {
    int numF = features[0].size();  // Feature numbers
    vector<int> currF;              // Current set of the features

    cout << "Beginning search." << endl;

    for (int i = 0; i < numF; ++i) {
        int addF = -1; // Feature to add at this level
        double bestAccuracy = 0; // Best accuracy so far

        for (int k = 0; k < numF; ++k) {
            if (find(currF.begin(), currF.end(), k) != currF.end()) {
                continue; // Skip if already added
            }

            vector<int> tempSet = currF;
            tempSet.push_back(k);

            // Use stub accuracy for now
            double accuracy = leave_one_out_cross_validation_stub(features, labels, tempSet);

            cout << "--Considering adding feature " << k + 1 << " (accuracy: " << accuracy << "%)\n";

            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                addF = k;
            }
        }

        if (addF != -1) {
            currF.push_back(addF);
            cout << "On level " << i + 1 << ", added feature " << addF + 1
                 << " to current set (accuracy: " << bestAccuracy << "%)\n";
        }
    }

    cout << "Finished search! The best feature subset is { ";
    for (int f : currF)
        cout << f + 1 << " ";
    cout << "}\n";
}

/*
void backward_elimination(const vector<vector<double>> &features,
                          const vector<int> &labels);


void normalize(vector<vector<double>> &features);
*/


#endif /* header_functions */
