#include "siftfeat.h"
#include <string>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
using namespace std;
namespace po = boost::program_options; 

void printKeypoints();

//string BaseDir = "E:\\testsift\\mm270k\\";
string ImgsFile;// = BaseDir + "mm270k.txt";
string SiftBinFile;// = BaseDir + "mm270k.sift";
//string SiftTxtFile = BaseDir + "mm270k.sift.txt";
string SiftDataFile;// = BaseDir + "mm270k.sift.data";
string SiftIdsFile;// = BaseDir + "mm270k.ids.txt";
string SiftLogFile;// = BaseDir + "mm270k.sift.log";

int main(int argc, char* argv[])
{
	// args
	int doubleImg;
	double contrThr;
	int maxNkps;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("imgs,i", po::value<string>(&ImgsFile), "image list file.")
		("rawsift,r", po::value<string>(&SiftBinFile), "raw sift result data in binary format.")
		("siftdata,d", po::value<string>(&SiftDataFile), "sift data for index.")
		("ids,s", po::value<string>(&SiftIdsFile), "Inverse index of id and files, also position/scale/orientation.")
		("log,l", po::value<string>(&SiftLogFile), "sift log file.")
		("double,b", po::value<int>(&doubleImg)->default_value(1), "Double image before sift.")
		("contr,c", po::value<double>(&contrThr)->default_value(0.03), "low contract threshold.")
		("max,m", po::value<int>(&maxNkps)->default_value(3000), "max keypoints per image.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("imgs") == 0 || vm.count("rawsift") == 0 || vm.count("siftdata") == 0 
		|| vm.count("ids") == 0 || vm.count("log") == 0)
	{
		cout << desc;
		return 1;
	}

	showSift(ImgsFile.c_str(), SiftBinFile.c_str(), doubleImg, contrThr, maxNkps);

	printKeypoints();
	return 0;
}

void printKeypoints()
{
	FILE* fp = fopen(SiftBinFile.c_str(), "rb");
	//FILE* out = fopen(SiftTxtFile.c_str(), "w");
	FILE* fdata = fopen(SiftDataFile.c_str(), "wb");
	FILE* fids = fopen(SiftIdsFile.c_str(), "w");
	FILE* flog = fopen(SiftLogFile.c_str(), "w");

	long long fid;
	int sid;
	int index;
	double buf[4];
	unsigned short data[128];
	int pointLimit = 0;
	int nFiles = 0;
	long long allPointsNum = 0;

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
	
	int entrySize = sizeof(unsigned short);
	fwrite(&entrySize, sizeof(int), 1, fdata);
	int num = allPointsNum;
	fwrite(&num, sizeof(int), 1, fdata);
	int dim = 128;
	fwrite(&dim, sizeof(int), 1, fdata);

	int curId = 0;
	//float t[128];
	while (fread(&fid, sizeof(long long), 1, fp) > 0)
	{
		fprintf(flog, "%lld", fid);
		fread(&sid, sizeof(int), 1, fp);
		fprintf(flog, "_%d", sid);
		// ids file
		fprintf(fids, "%d %lld", curId, fid);

		fread(&buf, sizeof(double), 4, fp); 
		for (int i=0; i<4; i++)
		{
			fprintf(flog, " %lg", buf[i]);
			fprintf(fids, " %lg", buf[i]);
		}

		fread(&data, sizeof(unsigned short), 128, fp);
		for (int i=0; i<128; i++)
		{
			fprintf(flog, " %hu", data[i]);
			//t[i] = data[i];
			//fprintf(out, i==0?"%lg":" %lg", data[i]);
		}
		fwrite(data, sizeof(unsigned short), 128, fdata);
		fprintf(flog, "\n");
		fprintf(fids, "\n");
		//fprintf(out, "\n");

		++curId;
	}

	fclose(fp);
	//fclose(out);
	fclose(fdata);
	fclose(fids);
	fclose(flog);
}