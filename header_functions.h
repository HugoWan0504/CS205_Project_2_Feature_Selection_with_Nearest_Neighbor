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
        int nrIndex = -1;           // nearest index

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

// Part 1.1 Forward Selection Search
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


// Part 1.2 Backward Elimination Search
void backward_elimination(const vector<vector<double>> &features,
                          const vector<int> &labels,
                          int LMT = 1) {        // local minimum threshold
    int numF = features[0].size();              // number of features
    int numR = features.size();                 // number of records

    vector<int> selectF(numF);                  // selected features (starting with all)
    iota(selectF.begin(), selectF.end(), 0);    // fill with 0 to numF - 1

    double bestAccuracy = nearest_neighbor_classification(features, labels, selectF); // accuracy with all features
    vector<int> bestFSet = selectF;   // best feature set

    // Short summary of the input dataset
    cout << "This dataset has " << numR << " records and " << numF << " features." << endl;
    cout << "Initial accuracy with all features: " << fixed << setprecision(2) << bestAccuracy << "%" << endl;
    cout << "Beginning search." << endl;
    int LM = LMT;  // set local minimum threshold

    while (selectF.size() > 1) {
        double currBestAccuracy = 0.0;    // track the current best accuracy
        int removeF = -1;                 // track which feature to remove

        for (int f : selectF) {
            vector<int> trialF = selectF; // copy current feature set
            // remove feature f from the trial set
            trialF.erase(remove(trialF.begin(), trialF.end(), f), trialF.end());

            double accuracy = nearest_neighbor_classification(features, labels, trialF);
            cout << "     Current feature(s) { ";
            for (int i : trialF) cout << i + 1 << " ";
            cout << "} with accuracy " << fixed << setprecision(2) << accuracy << "%" << endl;

            if (accuracy > currBestAccuracy) {
                currBestAccuracy = accuracy;
                removeF = f;
            }
        }

        // Actually remove the feature that gives the best accuracy improvement
        selectF.erase(remove(selectF.begin(), selectF.end(), removeF), selectF.end());

        if (currBestAccuracy > bestAccuracy) {
            bestAccuracy = currBestAccuracy;
            bestFSet = selectF;
            LM = LMT; // reset local minimum threshold
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

    cout << "Best feature subset is { ";
    for (int i : bestFSet) cout << i + 1 << " ";
    cout << "} with accuracy " << fixed << setprecision(2) << bestAccuracy << "%" << endl;
}




#endif /* header_functions */
