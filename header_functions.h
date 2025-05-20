#ifndef header_functions
#define header_functions

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


double leave_one_out_accuracy(const vector<vector<double>> &features,
                              const vector<int> &labels,
                              const vector<int> &selected_features);

void forward_selection(const vector<vector<double>> &features,
                       const vector<int> &labels);

                       
void backward_elimination(const vector<vector<double>> &features,
                          const vector<int> &labels);


void normalize(vector<vector<double>> &features);

#endif /* header_functions */
