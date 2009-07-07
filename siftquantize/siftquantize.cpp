#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
#include <cxcore.h>
#include <vector>
using namespace std;
namespace po = boost::program_options; 

#pragma comment(lib, "cxcore.lib")

const int HeaderSize = 12;
const int RowNumOffset = 4;
int N = 0;
int K = 10;
int L = 5;
int A = 30;
int E = 100;

FILE* Fsd = 0;
FILE* Fparts = 0;
FILE* Fre = 0;
FILE* Flog = 0;

int Dim;

void hierKMeans();

void hierKMeans()
{
	float* fbuf = new float[Dim];
	vector<CvMat*> data;
	
	for (int l=0; l<L; ++l)
	{
		int bp = Parts[p];
		int ep = Parts[p+1];
		int n = ep - bp;
		printf("begin k-means [%d, %d)...", bp, ep);

		CvMat* points = cvCreateMat( n, Dim, CV_32FC1 );
		CvMat* clusters = cvCreateMat( n, 1, CV_32SC1 );

		// read data for hierarchy k-means
		for (int i=0; i<n; ++i)
		{
			int ws = fread(fbuf, sizeof(float), Dim, Fsd);
			if (ws < Dim)
			{
				// end of file
				break;
			}

			for (int j=0; j<ws; ++j)
			{
				float* ptr = (float*)(points->data.ptr + (i *  points->step + j));
				*ptr = fbuf[j];
			}
		}

		// do hierarchy k-means
		cvKMeans2( points, K, clusters,
			cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ));

		for (int i=0; i<N; ++i)
		{
			fprintf(Fre, "%d\n", CV_MAT_ELEM(*clusters, int, i, 0));
		}

		// release matrices
		cvReleaseMat(&points);
		cvReleaseMat(&clusters);
	}
	delete[] fbuf;
}

int main(int argc, char* argv[])
{
	// args
	string dataFile;
	string resultFile;
	string logFile;
	string partFile;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("siftdata,d", po::value<string>(&dataFile), "sift data for index.")
		("result,r", po::value<string>(&resultFile), "cluster result file, also called visual words file.")
		("logfile,l", po::value<string>(&logFile), "log file for clustering.")
		("partitions, p", po::value<string>(&partFile), "partition file for clustering. Only one partition if not provided.")
		("keypoints,n", po::value<int>(&N)->default_value(0), "the keypoints used for k-means clustering. 0 for all keypoints in the data file.")
		("clusters,k", po::value<int>(&K)->default_value(10), "the number of clusters in each tree level.")
		("level,v", po::value<int>(&L)->default_value(5), "the max level of current hierarchy k-means.")
		("kpc,a", po::value<int>(&A)->default_value(30), "when the size of one cluster, the cluster will NOT be divided any more")
		("epsilon,e", po::value<int>(&E)->default_value(100), "the epsilon of k-means, which is the distance limit between the center and the farthest points in the cluster.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("siftdata") == 0 || vm.count("result") == 0
		|| vm.count("logfile") == 0)
	{
		cout << desc;
		return 1;
	}

	// open files
	Fsd = fopen(dataFile.c_str(), "rb");
	if (Fsd == 0)
	{
		printf("Cannot open %s\n", dataFile.c_str());
		return 2;
	}
	Fre = fopen(resultFile.c_str(), "w");
	if (Fre == 0)
	{
		printf("Cannot open %s\n", resultFile.c_str());
		return 3;
	}
	Flog = fopen(logFile.c_str(), "w");
	if (Flog == 0)
	{
		printf("Cannot open %s\n", logFile.c_str());
		return 4;
	}
	if (vm.count("partitions") > 0)
	{
		Fparts = fopen(partFile.c_str(), "r");
		if (Fparts == 0)
		{
			printf("Cannot open %s\n", partFile.c_str());
			return 5;
		}
	}

	// read data summery information
	unsigned int eleSize = 0;
	unsigned int rows = 0;
	unsigned int columns = 0;
	fread(&eleSize, sizeof(unsigned int), 1, Fsd);
	fread(&rows, sizeof(unsigned int), 1, Fsd);
	fread(&columns, sizeof(unsigned int), 1, Fsd);

	printf("eleSize=%d\n", eleSize);
	printf("rows=%d\n", rows);
	printf("columns=%d\n", columns);

	// assign global values for clustering
	Dim = columns;
	if (N == 0)
	{
		N = rows;
	}

	// print out some information
	printf("keyponts=%d, clusters=%d, level=%d\n", N, K, L);
	
	// do cluster
	hierKMeans();

	// close files
	fclose(Fsd);
	fclose(Fre);
	fclose(Flog);
	if (Fparts != 0)
	{
		fclose(Fparts);
	}

	return 0;
}

