#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <numeric>
#include <functional>
#include <queue>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
using namespace std;
namespace po = boost::program_options;

// suit for old info file: no contr and rpc

class KeyPoint;

string DataFile;
string InfoFile;
string OutDataFile;
string OutInfoFile;
// Number of keypoints per each file
int N;

const int MaxBuf = 10240;
const int HeaderSize = 12;
const int RowNumOffset = 4;
char buf[MaxBuf];

class KeyPoint
{
public:
	long long id;
	long long fid;
	float x;
	float y;
	float scl;
	float ori;
	float contr;
	float rpc;

	KeyPoint() : id(0), fid(0), x(0.0f), y(0.0f), scl(0.0f), ori(0.0f), contr(0.0f), rpc(0.0f) {}
};

int randomPickKeypoints(const vector<KeyPoint>& kps, bool* fk)
{
	int k = kps.size();
	if (k <= N)
	{
		fill(&fk[0], &fk[k], true);
	}
	else
	{
		srand(time(NULL));
		fill(&fk[0], &fk[k], false);

		for (int i=0; i<N; ++i)
		{
			int t = (int)(rand() / (double)(RAND_MAX + 1) * k);
			while (fk[t])
			{
				t = ++t % k;
			}
			fk[t] = true;
		}
		k = N;
	}

	return k;
}

// return the next keypoint unused
void saveKeypoints(const vector<KeyPoint>& kps, const bool* fk, FILE* fdata, FILE* foutdata, FILE* foutinfo, int entrySize, int curId)
{
	int k = kps.size();
	for (int i=0; i<k; ++i)
	{
		if (fread(buf, entrySize, 1, fdata) != 1)
		{
			printf("read data file wrong\n");
			break;
		}

		if (fk[i])
		{
			fwrite(buf, entrySize, 1, foutdata);
			fprintf(foutinfo, "%d %lld %g %g %g %g %g %g\n", curId++, kps[i].fid, kps[i].x, kps[i].y, kps[i].scl, kps[i].ori, kps[i].contr, kps[i].rpc);
		}
	}
}

void selectDataByRandom()
{
	// do
	FILE* fdata = fopen(DataFile.c_str(), "rb");
	if (fdata == 0)
	{
		printf("Cannot open %s\n", DataFile.c_str());
		return;
	}
	FILE* foutdata = fopen(OutDataFile.c_str(), "wb");
	if (foutdata == 0)
	{
		printf("Cannot open %s\n", OutDataFile.c_str());
		return;
	}
	FILE* finfo = fopen(InfoFile.c_str(), "r");
	if (finfo == 0)
	{
		printf("Cannot open %s\n", InfoFile.c_str());
		return;
	}
	FILE* foutinfo = fopen(OutInfoFile.c_str(), "w");
	if (foutinfo == 0)
	{
		printf("Cannot open %s\n", OutInfoFile.c_str());
		return;
	}

	unsigned int eleSize = 0;
	unsigned int rows = 0;
	unsigned int columns = 0;
	fread(&eleSize, sizeof(unsigned int), 1, fdata);
	fread(&rows, sizeof(unsigned int), 1, fdata);
	fread(&columns, sizeof(unsigned int), 1, fdata);

	printf("eleSize=%d\n", eleSize);
	printf("rows=%d\n", rows);
	printf("columns=%d\n", columns);

	int entrySize = eleSize * columns;

	fseek(fdata, 0, SEEK_SET);
	fread(buf, sizeof(char), HeaderSize, fdata);
	fwrite(buf, sizeof(char), HeaderSize, foutdata);
	KeyPoint kp;
	long long lastFid = -1;
	unsigned int count = 0;
	vector<KeyPoint> kps;
	while (fscanf(finfo, "%lld %lld %f %f %f %f", &kp.id, &kp.fid, &kp.x, &kp.y, &kp.scl, &kp.ori) != EOF)
	{
		if (lastFid == -1)
		{
			lastFid = kp.fid;
		}
		else if (lastFid != kp.fid)
		{
			bool* fk = new bool[kps.size()];
			fill(&fk[0], &fk[kps.size()], false);
			int k = randomPickKeypoints(kps, fk);
			saveKeypoints(kps, fk, fdata, foutdata, foutinfo, entrySize, count);
			delete[] fk;

			count += k;
			lastFid = kp.fid;
			kps.clear();
		}

		kps.push_back(kp);
	}

	if (!kps.empty())
	{
		bool* fk = new bool[kps.size()];
		fill(&fk[0], &fk[kps.size()], false);
		int k = randomPickKeypoints(kps, fk);
		saveKeypoints(kps, fk, fdata, foutdata, foutinfo, entrySize, count);
		delete[] fk;

		count += k;
	}

	printf("%u keypoints copied\n", count);
	fseek(foutdata, RowNumOffset, SEEK_SET);
	fwrite(&count, sizeof(unsigned int), 1, foutdata);

	fclose(fdata);
	fclose(foutdata);
	fclose(finfo);
	fclose(foutinfo);
}

int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("data,d", po::value<string>(&DataFile), "input keypoints data file.")
		("info,i", po::value<string>(&InfoFile), "input keypoints information file.")
		("outdata,o", po::value<string>(&OutDataFile), "output random selected keypoins data file.")
		("outinfo,f", po::value<string>(&OutInfoFile), "output random seleted keypoints information file.")
		("nmax,n", po::value<int>(&N), "number of keypoints per image by random selected if possible.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("data") == 0 || vm.count("info") == 0 
		|| vm.count("outdata") == 0 || vm.count("outinfo") == 0 || vm.count("nmax") == 0)
	{
		cout << desc;
		return 1;
	}

	selectDataByRandom();

	return 0;
}
