#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
#include <cxcore.h>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
using namespace std;
namespace po = boost::program_options; 

#pragma comment(lib, "cxcore_64.lib")

const int HeaderSize = 12;
const int RowNumOffset = 4;
int N = 0;
int K = 10;
int L = 5;
int A = 30;
int E = 100;
int Iter = 15;

FILE* Fsd = 0;
FILE* Fparts = 0;
FILE* Flog = 0;
string PreResultFile;

int Dim;

// value - size of cluster, it will be modified in each lever cluster.
vector<int> Parts;

// all points cluster info, the index is the offset in the file, the value if cluster id.
int* PCinfo = 0;

void hierKMeans();
void printLevelClusters(int level, const vector<int>& partClusters);
int myKMeans2( const CvArr* samples_arr, int cluster_count,
			  CvArr* labels_arr, CvTermCriteria termcrit );

void hierKMeans()
{
	// index - cid, value - points in the cluster
	float* fbuf = new float[Dim];
	for (int l=0; l<L; ++l)
	{
		printf("begin k-means %d level...\n", l+1);
		
		// allocate memory
		deque<CvMat*> data;
		deque<vector<int> > posis;
		for (int i=0; i<Parts.size(); ++i)
		{
			data.push_back(cvCreateMat(Parts[i], Dim, CV_32FC1));
			posis.push_back(vector<int>());
		}

		// read data for hierarchy k-means
		fseek(Fsd, HeaderSize, SEEK_SET);
		for (int i=0; i<N; ++i)
		{
			int ws = fread(fbuf, sizeof(float), Dim, Fsd);
			if (ws < Dim)
			{
				// end of file
				break;
			}

			int cid = PCinfo[i];
			CvMat* points = data[cid];
			float* ptr = (float*)(points->data.ptr + posis[cid].size() *  points->step);
			for (int j=0; j<ws; ++j)
			{
				*(ptr+j) = fbuf[j];
			}
			posis[cid].push_back(i);
		}

		printf("load data finished...\n");
		// do k-means for all partitions(clusters)
		deque<int> newParts;
		vector<int> partClusters(Parts.size(), 0);
		for (int i=0; i<Parts.size(); ++i)
		{
			int curCluster = newParts.size();
			if (Parts[i] <= A)
			{
				for (int j=0; j<Parts[i]; ++j)
				{
					PCinfo[posis[i][j]] = curCluster;
				}
				newParts.push_back(Parts[i]);
				partClusters[i] = 1;
			}
			else
			{
				CvMat* clusters = cvCreateMat(Parts[i], 1, CV_32SC1);
				int k = (Parts[i] + A - 1) / A;
				k = k > K ? K : k;
				int realK = myKMeans2(data[i], k, clusters,
					cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, Iter, E));
				
				newParts.resize(curCluster+realK, 0);
				for (int j=0; j<Parts[i]; ++j)
				{
					int cid = curCluster + clusters->data.i[j]; //CV_MAT_ELEM(*clusters, int, j, 0);
					PCinfo[posis[i][j]] = cid;
					++newParts[cid];
				}

				cvReleaseMat(&clusters);

				partClusters[i] = realK;
			}
		}

		// release matrices
		for (int i=0; i<Parts.size(); ++i)
		{
			cvReleaseMat(&data[i]);
		}

		// copy newParts to Parts for next level cluster.
		Parts.resize(newParts.size(), 0);
		copy(newParts.begin(), newParts.end(), Parts.begin());

		// print result to file
		printLevelClusters(l + 1, partClusters);

		printf("success, clusters=%d\n", Parts.size());
	}
	delete[] fbuf;
}

void printLevelClusters(int level, const vector<int>& partClusters)
{
	string resultFile = str(boost::format("%s_%d.txt") % PreResultFile.c_str() % level);
	FILE* fre = fopen(resultFile.c_str(), "w");
	if (fre == 0)
	{
		printf("Cannot open %s\n", resultFile.c_str());
		return;
	}

	// print cluster number of each partitions
	fprintf(fre, "%d\n", partClusters.size());
	for (int i=0; i<partClusters.size(); ++i)
	{
		fprintf(fre, "%d %d\n", i, partClusters[i]);
	}

	// print information of all clusters
	fprintf(fre, "%d\n", Parts.size());
	for (int i=0; i<Parts.size(); ++i)
	{
		fprintf(fre, "%d %d\n", i, Parts[i]);
	}

	for (int i=0; i<N; ++i)
	{
		fprintf(fre, "%d\n", PCinfo[i]);
	}

	fclose(fre);
}

int main(int argc, char* argv[])
{
	// args
	string dataFile;
	string logFile;
	string partFile;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("siftdata,d", po::value<string>(&dataFile), "sift data for index.")
		("result,r", po::value<string>(&PreResultFile), "prefix of each level cluster result files.")
		//("logfile,l", po::value<string>(&logFile), "log file for clustering.")
		("partitions,p", po::value<string>(&partFile), "partition file for clustering. Only one partition if not provided.")
		("keypoints,n", po::value<int>(&N)->default_value(0), "the keypoints used for k-means clustering. 0 for all keypoints in the data file.")
		("clusters,k", po::value<int>(&K)->default_value(10), "the number of clusters in each tree level.")
		("Iterater,i", po::value<int>(&Iter)->default_value(15), "the number of max iterators in k-means.")
		("level,v", po::value<int>(&L)->default_value(1), "the max level of current hierarchy k-means.")
		("kpc,a", po::value<int>(&A)->default_value(15), "when the size of one cluster, the cluster will NOT be divided any more.")
		("epsilon,e", po::value<int>(&E)->default_value(100), "the epsilon of k-means, which is the distance limit between the center and the farthest points in the cluster.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("siftdata") == 0 || vm.count("result") == 0)
		//|| vm.count("logfile") == 0)
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
	/*Flog = fopen(logFile.c_str(), "a");
	if (Flog == 0)
	{
		printf("Cannot open %s\n", logFile.c_str());
		return 4;
	}*/
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

	// input clusters(partitions) information
	PCinfo = new int[N];
	fill(PCinfo, PCinfo+N, 0);
	if (Fparts != 0)
	{
		// ignore last cluster infor
		int np = 0;
		int cid, p;
		fscanf(Fparts, "%d", &np);
		for (int i=0; i<np; ++i)
		{
			fscanf(Fparts, "%d %d", &cid, &p);
		}

		// read partitions infor
		fscanf(Fparts, "%d", &np);
		Parts.resize(np, 0);
		for (int i=0; i<np; ++i)
		{
			fscanf(Fparts, "%d %d", &cid, &p);
			Parts[cid] = p;
		}

		// read cluster info of all keypoints
		for (int i=0; i<N; ++i)
		{
			fscanf(Fparts, "%d", &cid);
			PCinfo[i] = cid;
		}
	}
	else
	{
		Parts.push_back(N);
	}
	printf("Input partitions information(%d clusters):\n", Parts.size());
	for (int i=0; i<Parts.size(); ++i)
	{
		printf("(%d, %d),", i, Parts[i]);
	}
	printf("\n");
	
	// do cluster
	hierKMeans();

	// close files
	fclose(Fsd);
	//fclose(Flog);
	if (Fparts != 0)
	{
		fclose(Fparts);
	}
	delete[] PCinfo;

	return 0;
}

// l2 norm, the farthest point to the center point is less than termcrit.epsilon, the k-means will stop.
// return the number of real clusters, which excluding empty clusters.
// only allowed empty clusters, comparing to cvKMeans;
int myKMeans2( const CvArr* samples_arr, int cluster_count,
			  CvArr* labels_arr, CvTermCriteria termcrit )
{
	CvMat* centers = 0;
	CvMat* old_centers = 0;
	CvMat* counters = 0;
	CvMat* relabels = 0;
	int real_count = 0;

	CV_FUNCNAME( "cvKMeans2" );

	__BEGIN__;

	CvMat samples_stub, labels_stub;
	CvMat* samples = (CvMat*)samples_arr;
	CvMat* labels = (CvMat*)labels_arr;
	CvMat* temp = 0;
	CvRNG rng = CvRNG(-1);
	int i, j, k, sample_count, dims;
	int ids_delta, iter;
	double max_dist;

	if( !CV_IS_MAT( samples ))
		CV_CALL( samples = cvGetMat( samples, &samples_stub ));

	if( !CV_IS_MAT( labels ))
		CV_CALL( labels = cvGetMat( labels, &labels_stub ));

	if( cluster_count < 1 )
		CV_ERROR( CV_StsOutOfRange, "Number of clusters should be positive" );

	if( CV_MAT_DEPTH(samples->type) != CV_32F || CV_MAT_TYPE(labels->type) != CV_32SC1 )
		CV_ERROR( CV_StsUnsupportedFormat,
		"samples should be floating-point matrix, cluster_idx - integer vector" );

	if( labels->rows != 1 && (labels->cols != 1 || !CV_IS_MAT_CONT(labels->type)) ||
		labels->rows + labels->cols - 1 != samples->rows )
		CV_ERROR( CV_StsUnmatchedSizes,
		"cluster_idx should be 1D vector of the same number of elements as samples' number of rows" );

	CV_CALL( termcrit = cvCheckTermCriteria( termcrit, 1e-6, 100 ));

	termcrit.epsilon *= termcrit.epsilon;
	sample_count = samples->rows;

	if( cluster_count > sample_count )
		cluster_count = sample_count;

	dims = samples->cols*CV_MAT_CN(samples->type);
	ids_delta = labels->step ? labels->step/(int)sizeof(int) : 1;

	CV_CALL( centers = cvCreateMat( cluster_count, dims, CV_64FC1 ));
	CV_CALL( old_centers = cvCreateMat( cluster_count, dims, CV_64FC1 ));
	CV_CALL( counters = cvCreateMat( 1, cluster_count, CV_32SC1 ));
	CV_CALL( relabels = cvCreateMat( 1, cluster_count, CV_32SC1 ));

	// init centers
	for( i = 0; i < sample_count; i++ )
		labels->data.i[i] = cvRandInt(&rng) % cluster_count;

	counters->cols = cluster_count; // cut down counters
	max_dist = termcrit.epsilon*2;

	for( iter = 0; iter < termcrit.max_iter; iter++ )
	{
		// computer centers
		cvZero( centers );
		cvZero( counters );

		for( i = 0; i < sample_count; i++ )
		{
			float* s = (float*)(samples->data.ptr + i*samples->step);
			k = labels->data.i[i*ids_delta];
			double* c = (double*)(centers->data.ptr + k*centers->step);
			for( j = 0; j <= dims - 4; j += 4 )
			{
				double t0 = c[j] + s[j];
				double t1 = c[j+1] + s[j+1];

				c[j] = t0;
				c[j+1] = t1;

				t0 = c[j+2] + s[j+2];
				t1 = c[j+3] + s[j+3];

				c[j+2] = t0;
				c[j+3] = t1;
			}
			for( ; j < dims; j++ )
				c[j] += s[j];
			counters->data.i[k]++;
		}

		if( iter > 0 )
			max_dist = 0;

		for( k = 0; k < cluster_count; k++ )
		{
			double* c = (double*)(centers->data.ptr + k*centers->step);
			if( counters->data.i[k] != 0 )
			{
				double scale = 1./counters->data.i[k];
				for( j = 0; j < dims; j++ )
					c[j] *= scale;
			}
			else
			{
				i = cvRandInt( &rng ) % sample_count;
				float* s = (float*)(samples->data.ptr + i*samples->step);
				for( j = 0; j < dims; j++ )
					c[j] = s[j];
			}

			if( iter > 0 )
			{
				double dist = 0;
				double* c_o = (double*)(old_centers->data.ptr + k*old_centers->step);
				for( j = 0; j < dims; j++ )
				{
					double t = c[j] - c_o[j];
					dist += t*t;
				}
				if( max_dist < dist )
					max_dist = dist;
			}
		}

		// assign labels
		for( i = 0; i < sample_count; i++ )
		{
			float* s = (float*)(samples->data.ptr + i*samples->step);
			int k_best = 0;
			double min_dist = DBL_MAX;

			for( k = 0; k < cluster_count; k++ )
			{
				double* c = (double*)(centers->data.ptr + k*centers->step);
				double dist = 0;

				j = 0;
				for( ; j <= dims - 4; j += 4 )
				{
					double t0 = c[j] - s[j];
					double t1 = c[j+1] - s[j+1];
					dist += t0*t0 + t1*t1;
					t0 = c[j+2] - s[j+2];
					t1 = c[j+3] - s[j+3];
					dist += t0*t0 + t1*t1;
				}

				for( ; j < dims; j++ )
				{
					double t = c[j] - s[j];
					dist += t*t;
				}

				if( min_dist > dist )
				{
					min_dist = dist;
					k_best = k;
				}
			}

			labels->data.i[i*ids_delta] = k_best;
		}

		if( max_dist < termcrit.epsilon )
			break;

		CV_SWAP( centers, old_centers, temp );
	}

	cvZero( counters );
	for( i = 0; i < sample_count; i++ )
		counters->data.i[labels->data.i[i]]++;

	// calculate real clusters
	real_count = cluster_count;
	for( k = 0; k < cluster_count; k++ )
	{
		if( counters->data.i[k] == 0 )
		{
			--real_count;
		}
		else
		{
			relabels->data.i[k] = k - (cluster_count - real_count);
		}
	}

	// relabel after eliminate empty clusters
	if (real_count < cluster_count)
	{
		for( i = 0; i < sample_count; i++ )
		{
			labels->data.i[i] = relabels->data.i[labels->data.i[i]];
		}
	}

	__END__;

	cvReleaseMat( &centers );
	cvReleaseMat( &old_centers );
	cvReleaseMat( &counters );
	cvReleaseMat( &relabels );

	return real_count;
}
