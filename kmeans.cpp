// Josh Morris
// CSCI 4350
// Dr. Phillips
// kmeans.cpp
// December 5, 2017

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace std;

void loadData(string filePath, vector<vector<double> >& data);
void printData(const vector<vector<double> >& data);
void initCenters(vector<vector<double> >& centers, vector<vector<double> >& data);
int closest(const vector<double>& curPoint, const vector<vector<double> >& centers);
void train(vector<vector<double> >& centers, vector<vector<double> >& data, vector<double>& label);
double majClass(vector<double>& classes);
int classify(const vector<vector<double> >& centers, const vector<vector<double> >& data, const vector<double>& label);

int main(int argc, char* argv[]){
	int randSeed;
	int numCluster;
	int numFeat;
	int numCorrect;
	string trainPath;
	string testPath;
	vector<vector<double> > data;
	vector<vector<double> > centers;
	vector<double> label;
	

	if (argc != 6) {
		cout << "usage: ./kmeans randSeed numCluster numFeat trainPath testPath" << endl;
		exit(-1);
	}

	// read arguments 
	randSeed = atoi(argv[1]);
	numCluster = atoi(argv[2]);
	numFeat = atoi(argv[3]);
	trainPath = argv[4];
	testPath = argv[5];

	//seed random function
	srand(randSeed);

	loadData(trainPath, data);

	// assign center vectors
	centers.assign(numCluster, vector<double>(data[0].size() - 1, 0));
	label.assign(numCluster, 0);

	// train
	train(centers, data, label);

	// load test data
	data.clear();
	loadData(testPath, data);

	// classify data
	numCorrect = classify(centers, data, label);

	// print the number of correct classifications
	cout << numCorrect << endl;

	
	return 0;
}

void loadData(string filePath, vector<vector<double> >& data){
	fstream fs(filePath);
	string line;
	double value;
	
	getline(fs, line);

	while(!fs.eof()) {
		vector<double> temp;
		stringstream parsed(line);

		while(!parsed.eof()) {
			parsed >> value;
			temp.push_back(value);
		}

		data.push_back(temp);

		getline(fs, line);
	}
}

void printData(const vector<vector<double> >& data) {
	for (auto i = data.begin(); i != data.end(); ++i){
		for (auto j = i->begin(); j != i->end(); ++j){
			cout << *j << " ";
		}
		cout << endl;
	}
}

void initCenters(vector<vector<double> >& centers, vector<vector<double> >& data){
	// for each center
	for (auto center = centers.begin(); center != centers.end(); ++center) {
		int cur = (data.size() * random()) / (RAND_MAX + 1.0);

		center->assign(data[cur].begin(), data[cur].end() - 1);
	}
}

void train(vector<vector<double> >& centers, vector<vector<double> >& data, vector<double>& label) {
	vector<int> centerCount(centers.size(), 0);
	vector<vector<double> > prevCenters(centers.size(), vector<double>(data[0].size() - 1, 0));
	vector<vector<double> > labels(centers.size(), vector<double>(0,0));

	initCenters(centers, data);

	while (prevCenters != centers) {
		// move centers to prev centers
		prevCenters = centers;

		// zero centers
		for (auto i = centers.begin(); i != centers.end(); ++i){
			for (auto j = i->begin(); j != i->end(); ++j){
				*j = 0;
			}
		}

		// zero count
		for (auto i = centerCount.begin(); i != centerCount.end(); ++i){
			*i = 0;
		}

		// for every data point find the closest point
		for (auto point = data.begin(); point != data.end(); ++point) {
			// find closest cluster
			int close = closest(*point, prevCenters);

			// add to closest new center
			for (int i = 0; i < centers[close].size(); ++i) {
				centers[close][i] += (*point)[i];
			}

			// add label to labels vector
			labels[close].push_back(point->back());

			++centerCount[close];
		} // for point

		
		for (int i = 0; i < centers.size(); ++i){
			if (centerCount[i] != 0) {
				// divide sum of closest points by counts
				for (auto j = centers[i].begin(); j != centers[i].end(); ++j){
							*j /= centerCount[i];
				}
			}
			else {
				// keep point the same as prev point
				centers[i] = prevCenters[i];
			}
		}
	} // while (prevCenters != centers)

	//label the points
	for (int i = 0; i < centers.size(); ++i){
		if (!labels[i].empty()){ 
			double majLabel = majClass(labels[i]);

			label[i] = majLabel;
		}
	}
}

// returns index of closest cluster
int closest(const vector<double>& curPoint, const vector<vector<double> >& centers) {
	int closestPoint;
	double curDist = 0;
	double shortDist = numeric_limits<double>::max();

	// for each center
	for (int i = 0; i < centers.size(); ++i){
		curDist = 0;

		// calculate distance to current center
		for (int j = 0; j < centers[i].size(); ++j){
			curDist += pow(curPoint[j] - centers[i][j], 2);
		}

		curDist = sqrt(curDist);

		// save if center is closer than current closest center
		if(curDist < shortDist){
			closestPoint = i;
			shortDist = curDist;
		}
	}

	return closestPoint;
}

// function to find the majority classification
// returns the majority classification
double majClass(vector<double>& classes) {
	int majClass = classes.front();
	int majClassCount = 1;
	int curClass = classes.front();
	int curClassCount = 1;

	sort(classes.begin(), classes.end());

	for (auto i = classes.begin() + 1; i != classes.end(); ++i) {
		if (curClass != *i) {
			curClass = *i;
			curClassCount = 1;
		}
		else {
			curClassCount++;
			if (curClassCount > majClassCount) {
				majClassCount = curClassCount;
				majClass = curClass;
			}
		}
	} //end for (auto i = classes...)

	return majClass;
}

// classify data 
// return the number of correctly classified data points
int classify(const vector<vector<double> >& centers, const vector<vector<double> >& data, const vector<double>& label){
	int count = 0;

	for (auto point = data.begin(); point != data.end(); ++point){
		int close = closest(*point, centers);

		if (point->back() == label[close]) {
			++count;
		}
	}

	return count;
}