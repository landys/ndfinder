#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
using namespace std;
namespace po = boost::program_options; 

class Node;

// represent non-leaf nodes.
class Node
{
public:
	// if its sons are non-leaf nodes: only minimum and maximum index in the NonLeaves, so the size of sons is 2
	// else: offset order(index) of data file for leaf node
	vector<int> sons;
	// the center of the cluster
	float* center;
	// number of keypoints in the cluster. if a non-leaf node, the number is sum of keypoints number of sub-clusters.
	int npoints;

	Node() : center(0), npoints(0) {}
};

const int HeaderSize = 12;
const int RowNumOffset = 4;
int L = 5;
int N = 0;
int BegClus = 0;
int NClus = 0;
int Dim = 0;
string DataFile;
string PreClusFile;
string WordsFile;

// the first element is root
deque<Node> NonLeaves;

void readPartInfoFromData(const string& dataFile, int& n);
//void readInfoFromDatas();
string getClusterName(const string& preClusFile, int level);
void readInterLevelCluster(FILE* fc, int& curClusterId);
void readLastLevelCluster(FILE* fc, int begN, int n, int& begClus, int& nClus);
void genWordsFile();
void calcPartClusCenters(const string& dataFile, int begClus, int nClus, int begN, int n);
void calcClusCenters();
void printWordsFile();
void releaseMemory();

void readPartInfoFromData(const string& dataFile, int& n)
{
	FILE* fd = fopen(dataFile.c_str(), "rb");
	if (fd == 0)
	{
		printf("Cannot open %s\n", dataFile.c_str());
		return;
	}

	// read data summery information
	unsigned int eleSize = 0;
	unsigned int rows = 0;
	unsigned int columns = 0;
	fread(&eleSize, sizeof(unsigned int), 1, fd);
	fread(&rows, sizeof(unsigned int), 1, fd);
	fread(&columns, sizeof(unsigned int), 1, fd);
	printf("%s: eleSize=%d, rows=%d, columns=%d\n", dataFile.c_str(), eleSize, rows, columns);
	n = rows;
	Dim = columns;

	fclose(fd);
}

//void readInfoFromDatas()
//{
//	readPartInfoFromData(DataFile, N);
//}

// level begin from 1
string getClusterName(const string& preClusFile, int level)
{
	//if (level == 1)
	//{
	//	return preClusFile+"_1.txt";
	//}
	//else
	//{
	//	return str(boost::format("%s_1_%d.txt") % preClusFile.c_str() % (level-1));
	//}
	return str(boost::format("%s_%d.txt") % preClusFile.c_str() % level); 
}

void readInterLevelCluster(FILE* fc, int& curClusterId)
{
	int np = 0;
	int cid, p;
	fscanf(fc, "%d", &np);
	for (int i=0; i<np; ++i)
	{
		fscanf(fc, "%d %d", &cid, &p);
		NonLeaves.push_back(Node());
		int ni = NonLeaves.size() - 1;
		NonLeaves[ni].sons.push_back(curClusterId);
		NonLeaves[ni].sons.push_back(curClusterId + p -1);
		curClusterId += p;
	}
}

void readLastLevelCluster(FILE* fc, int begN, int n, int& begClus, int& nClus)
{
	begClus = NonLeaves.size();

	int np = 0;
	int cid, p;
	// read partitions info
	fscanf(fc, "%d", &np);
	for (int i=0; i<np; ++i)
	{
		// the read data will be ignored
		fscanf(fc, "%d %d", &cid, &p);
		// push the node with its sons empty first, it will be added when reading each keypoints' clusters.
		NonLeaves.push_back(Node());
	}
	nClus = np;

	printf("readLastLevelCluster: begN=%d, n=%d, begClus=%d, nClus=%d\n", begN, n, begClus, nClus);

	// read keyponts' clusters
	for (int i=0; i<n; ++i)
	{
		fscanf(fc, "%d", &cid);
		NonLeaves[begClus+cid].sons.push_back(begN+i);
	}
}

void genWordsFile()
{
	//readInfoFromDatas();
	readPartInfoFromData(DataFile, N);

	// put root, it has two sons
	//NonLeaves.push_back(Node());
	//NonLeaves[0].sons.push_back(1);
	//NonLeaves[0].sons.push_back(2);
	int curClusterId = 1;
	// put non-leaves without calculate k-means center
	for (int l=1; l<=L; ++l)
	{
		// open cluster files
		string clusFile = getClusterName(PreClusFile, l);
		FILE* fc = fopen(clusFile.c_str(), "r");
		if (fc == 0)
		{
			printf("Cannot open %s\n", clusFile.c_str());
			return;
		}

		// read cluster information
		readInterLevelCluster(fc, curClusterId);

		if (l == L)
		{
			readLastLevelCluster(fc, 0, N, BegClus, NClus);
			//readLastLevelCluster(fc2, N1, N2, BegClus2, NClus2);
		}


		fclose(fc);
		//fclose(fc2);
	}
	
}

void calcPartClusCenters(const string& dataFile, int begClus, int nClus, int begN, int n)
{
	FILE* fd = fopen(dataFile.c_str(), "rb");
	if (fd == 0)
	{
		printf("Cannot open %s\n", dataFile.c_str());
		return;
	}
	fseek(fd, HeaderSize, SEEK_SET);

	// read data
	float** data = new float*[n];
	for (int i=0; i<n; ++i)
	{
		data[i] = new float[Dim];
		int ws = fread(data[i], sizeof(float), Dim, fd);
		if (ws < Dim)
		{
			// end of file
			printf("ERROR: Only read %d records from %s.\n", i, dataFile.c_str());
			break;
		}
	}
	fclose(fd);
	
	int endClust = begClus + nClus;
	for (int i=begClus; i<endClust; ++i)
	{
		NonLeaves[i].center = new float[Dim];
		fill(NonLeaves[i].center, NonLeaves[i].center+Dim, 0.0f);
		int sn = NonLeaves[i].sons.size();
		for (int j=0; j<sn; ++j)
		{
			for (int d=0; d<Dim; ++d)
			{
				NonLeaves[i].center[d] += data[NonLeaves[i].sons[j]-begN][d];
			}
		}
		for (int d=0; d<Dim; ++d)
		{
			NonLeaves[i].center[d] /= sn;
		}
		NonLeaves[i].npoints = sn;
	}

	// release memory
	for (int i=0; i<n; ++i)
	{
		delete[] data[i];
	}
	delete[] data;
}

void calcClusCenters()
{
	printf("begin calculate leaf nodes...\n");
	calcPartClusCenters(DataFile, BegClus, NClus, 0, N);

	//printf("begin calculate part2 leaf nodes...\n");
	//calcPartClusCenters(DataFile2, BegClus2, NClus2, N1, N2);

	printf("Begin calculate non-leaf nodes...\n");
	for (int i=BegClus-1; i>=0; --i)
	{
		NonLeaves[i].center = new float[Dim];
		fill(NonLeaves[i].center, NonLeaves[i].center+Dim, 0.0f);
		int sn = 0;
		for (int j=NonLeaves[i].sons[0]; j<=NonLeaves[i].sons[1]; ++j)
		{
			for (int d=0; d<Dim; ++d)
			{
				NonLeaves[i].center[d] += NonLeaves[j].center[d] * NonLeaves[j].npoints;
			}
			sn += NonLeaves[j].npoints;
		}
		for (int d=0; d<Dim; ++d)
		{
			NonLeaves[i].center[d] /= sn;
		}
		NonLeaves[i].npoints = sn;
	}
}

void printWordsFile()
{
	FILE* fw = fopen(WordsFile.c_str(), "w");
	if (fw == 0)
	{
		printf("Cannot open %s\n", WordsFile.c_str());
		return;
	}

	fprintf(fw, "%d %d %d %d\n", N, NonLeaves.size(), NClus, Dim);
	for (int i=0; i<NonLeaves.size(); ++i)
	{
		fprintf(fw, "%d %d %d ", i, NonLeaves[i].npoints, NonLeaves[i].sons.size());
		for (int j=0; j<NonLeaves[i].sons.size(); ++j)
		{
			fprintf(fw, "%d ", NonLeaves[i].sons[j]);
		}
		for (int j=0; j<Dim; ++j)
		{
			fprintf(fw, "%g ", NonLeaves[i].center[j]);
		}
		fprintf(fw, "\n");
	}

	fclose(fw);
}

void releaseMemory()
{
	for (int i=0; i<NonLeaves.size(); ++i)
	{
		delete[] NonLeaves[i].center;
	}
}

int main(int argc, char* argv[])
{
	// args
	string partFile;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("siftdata,d", po::value<string>(&DataFile), "sift data.")
		("clusdata,c", po::value<string>(&PreClusFile), "prefix of cluster files.")
		("result,r", po::value<string>(&WordsFile), "sift visual words index file.")
		("level,v", po::value<int>(&L)->default_value(5), "the max level of current hierarchy k-means.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("siftdata") == 0
		|| vm.count("clusdata") == 0  || vm.count("result") == 0)
	{
		cout << desc;
		return 1;
	}

	// generate words file
	printf("genWordsFile...\n");
	genWordsFile();

	printf("calcClusCenters...\n");
	calcClusCenters();

	printf("printWordsFile...\n");
	printWordsFile();

	releaseMemory();

	return 0;
}
