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

// Replace stub with a real NNC
double nearest_neighbor_classification(const vector<vector<double>> &features,
                                       const vector<int> &labels,
                                       const vector<int> &selectF) { // selected features
    int total = features.size();
    int correct = 0;

    for (int i = 0; i < total; ++i) {
        double nrDist = DBL_MAX;    // nearest distance
        int nrIndex = -1;      // nearest index

        for (int j = 0; j < total; ++j) {
            if (i == j) continue;

            double dist = 0.0;
            for (int f : selectF) {
                double diff = features[i][f] - features[j][f];
                dist += diff * diff;
            }
            dist = sqrt(dist);

            if (dist < nrDist) {
                nrDist = dist;
                nrIndex = j;
            }
        }

        if (labels[i] == labels[nrIndex]) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / total * 100.0;
}


void forward_selection(const vector<vector<double>> &features,
                       const vector<int> &labels,
                       int LMT = 1) {   // local minimum threshold
    int numF = features[0].size();      // number of features
    int numR = features.size();         // number of records

    vector<int> selectF;        // selected features
    vector<int> bestFSet;       // best feature set
    double bestAccuracy = 0.0;  // best accuracy

    // Short summary of the input dataset
    cout << "This dataset has " << numR << " records and " << numF << " features." << endl;
    cout << "Beginning search." << endl;

    int LM = LMT; // set local minimum threshold to the local minimum

    for (int level = 0; level < numF; ++level) {
        double currBestAccuracy = 0.0;      // track the current best accuracy
        vector<int> currBestSet;            // track the current best set

        for (int f = 0; f < numF; ++f) {
            if (find(selectF.begin(), selectF.end(), f) != selectF.end()) continue;

            vector<int> trialF = selectF;   // features' trials
            trialF.push_back(f);

            double accuracy = nearest_neighbor_classification(features, labels, trialF);
            cout << "     Current feature(s) { ";
            for (int i : trialF) cout << i + 1 << " ";
            cout << "} with accuracy " << fixed << setprecision(2) << accuracy << "%" << endl;

            if (accuracy > currBestAccuracy) {
                currBestAccuracy = accuracy;
                currBestSet = trialF;
            }
        }

        if (currBestSet != selectF) {
            selectF = currBestSet;
            if (currBestAccuracy > bestAccuracy) {
                bestAccuracy = currBestAccuracy;
                bestFSet = selectF;
                LM = LMT;
                cout << "Current best overall is { ";
                for (int i : bestFSet) cout << i + 1 << " ";
                cout << "} with accuracy " << fixed << setprecision(2) << bestAccuracy << "%" << endl;
            } else {
                LM--;
                cout << "The accuracy is decreasing!" << endl;
                cout << "Current round feature(s): { ";
                for (int i : selectF) cout << i + 1 << " ";
                cout << "} with accuracy " << currBestAccuracy << "%, lower than best "
                     << bestAccuracy << "%" << endl;

                if (LM == 0) break;
            }
        }
    }

    cout << "Best feature subset is { ";
    for (int i : bestFSet) cout << i + 1 << " ";
    cout << "} with accuracy " << fixed << setprecision(2) << bestAccuracy << "%" << endl;
}


/*
void backward_elimination(const vector<vector<double>> &features,
                          const vector<int> &labels);


void normalize(vector<vector<double>> &features);
*/


#endif /* header_functions */
