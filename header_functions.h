#ifndef header_functions
#define header_functions

void load_data(const string &filename, vector<vector<double>> &features, vector<int> &labels);

double leave_one_out_accuracy(const vector<vector<double>> &features,
                              const vector<int> &labels,
                              const vector<int> &selected_features);

void forward_selection(const vector<vector<double>> &features,
                       const vector<int> &labels);

                       
void backward_elimination(const vector<vector<double>> &features,
                          const vector<int> &labels);


void normalize(vector<vector<double>> &features);

#endif /* header_functions */
