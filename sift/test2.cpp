#include "siftfeat.h"
#include "imgfeatures.h"
#include <string>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
using namespace std;
namespace po = boost::program_options; 

void printKeypoints(bool lessLog);

//string BaseDir = "E:\\testsift\\mm270k\\";
string ImgsFile;// = BaseDir + "mm270k.txt";
string SiftBinFile;// = BaseDir + "mm270k.sift";
//string SiftTxtFile = BaseDir + "mm270k.sift.txt";
string SiftDataFile;// = BaseDir + "mm270k.sift.data";
string SiftIdsFile;// = BaseDir + "mm270k.ids.txt";
string SiftLogFile;// = BaseDir + "mm270k.sift.log";

// -i E:\testsift\mm270k\sift_rpc\test.txt  -r E:\testsift\mm270k\sift_rpc\mm270k_c1_2000_20_f.sift
//-d E:\testsift\mm270k\sift_rpc\mm270k_c1_2000_20_f.sift.data -s E:\testsift\mm270k\sift_rpc\mm270k_c1_2000_20.info.txt 
//-l E:\testsift\mm270k\sift_rpc\mm270k_c1_2000_20.sift.log -c 0 -p 100 -w 0 -m 10
#ifndef MERGE_TEST
int main(int argc, char* argv[])
{
	// args
	int doubleImg;
	double contrThr;
	double curThr;
	double contrWeight;
	int maxNkps;
	int lessLog;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("imgs,i", po::value<string>(&ImgsFile), "image list file.")
		("rawsift,r", po::value<string>(&SiftBinFile), "raw sift result data in binary format. If imgs not provided, it is the input raw sift data file.")
		("siftdata,d", po::value<string>(&SiftDataFile), "sift data for index.")
		("info,s", po::value<string>(&SiftIdsFile), "Inverse index of id and files, also position/scale/orientation/contract/ratiopc.")
		("log,l", po::value<string>(&SiftLogFile), "sift log file.")
		("double,b", po::value<int>(&doubleImg)->default_value(1), "Double image before sift.")
		("contr,c", po::value<double>(&contrThr)->default_value(0.03), "low contract threshold.")
		("rpc,p", po::value<double>(&curThr)->default_value(10), "ratio of principal curvatures.")
		("contrw,w", po::value<double>(&contrWeight)->default_value(1), "weight of contract, should be in [0,1].")
		("max,m", po::value<int>(&maxNkps)->default_value(3000), "max keypoints per image.")
		("lesslog,g", po::value<int>(&lessLog)->default_value(1), "log less information, or it will be as large as rawsift.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("rawsift") == 0 || vm.count("siftdata") == 0 
		|| vm.count("info") == 0 || vm.count("log") == 0)
	{
		cout << desc;
		return 1;
	}

	bool issift = false;
	if (vm.count("imgs") > 0)
	{
		issift = true;
	}

	if (issift)
	{
		showSift(ImgsFile.c_str(), SiftBinFile.c_str(), doubleImg, contrThr, maxNkps, curThr, contrWeight);
	}
	
	printKeypoints(lessLog != 0);
	return 0;
}
#endif //MERGE_TEST

void printKeypoints(bool lessLog)
{
	FILE* fp = fopen(SiftBinFile.c_str(), "rb");
	//FILE* out = fopen(SiftTxtFile.c_str(), "w");
	FILE* fdata = fopen(SiftDataFile.c_str(), "wb");
	FILE* fids = fopen(SiftIdsFile.c_str(), "w");
	FILE* flog = fopen(SiftLogFile.c_str(), "a");
	
	long long fid;
	int sid;
	int index;
	const int nDetail = 6;
	float buf[nDetail]; // x, y, scl, ori, contr, rpc
	elem_t data[FEATURE_MAX_D];
	int pointLimit = 0;
	int nFiles = 0;
	long long allPointsNum = 0;

	fprintf(flog, "/***********************************************************/\n");
	fread(&allPointsNum, sizeof(long long), 1, fp);
	printf("allPointsNum=%lld\n", allPointsNum);
	fprintf(flog, "allPointsNum=%lld\n", allPointsNum);
	fread(&pointLimit, sizeof(int), 1, fp);
	printf("pointLimit=%d\n", pointLimit);
	fprintf(flog, "pointLimit=%d\n", pointLimit);
	nFiles = allPointsNum / pointLimit + 1;
	printf("nFiles=%d\n", nFiles);
	fprintf(flog, "nFiles=%d\n", nFiles);

	assert(nFiles == 1);
	
	int entrySize = sizeof(elem_t);
	fwrite(&entrySize, sizeof(int), 1, fdata);
	int num = allPointsNum;
	fwrite(&num, sizeof(int), 1, fdata);
	int dim = FEATURE_MAX_D;
	fwrite(&dim, sizeof(int), 1, fdata);

	long long curId = 0;
	long long lastId = curId;
	long long lastFid = -1;
	while (fread(&fid, sizeof(long long), 1, fp) > 0)
	{
		if (!lessLog)
		{
			fprintf(flog, "%lld", fid);
		}
		fread(&sid, sizeof(int), 1, fp);
		if (!lessLog)
		{
			fprintf(flog, "_%d", sid);
		}
		// ids file
		fprintf(fids, "%lld %lld", curId, fid);

		fread(&buf, sizeof(float), nDetail, fp);
		for (int i=0; i<nDetail; i++)
		{
			if (!lessLog)
			{
				fprintf(flog, " %g", buf[i]);
			}
			fprintf(fids, " %g", buf[i]);
		}

		fread(&data, sizeof(elem_t), FEATURE_MAX_D, fp);
		if (!lessLog)
		{
			for (int i=0; i<FEATURE_MAX_D; i++)
			{
				fprintf(flog, " %g", data[i]);
			}
		}
		fwrite(data, sizeof(elem_t), FEATURE_MAX_D, fdata);
		if (!lessLog)
		{
			fprintf(flog, "\n");
		}
		fprintf(fids, "\n");
		//fprintf(out, "\n");

		if (lastFid == -1)
		{
			// first time, no last fid
			lastFid = fid;
		}
		else if (lastFid != fid)
		{
			fprintf(flog, "%lld %lld %lld %lld\n", lastFid, lastId, curId-1, curId-lastId);
			lastFid = fid;
			lastId = curId;
		}

		++curId;
	}

	fclose(fp);
	//fclose(out);
	fclose(fdata);
	fclose(fids);
	fclose(flog);
}