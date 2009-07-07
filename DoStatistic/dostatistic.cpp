#include <string>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <limits>
#include <lshkit.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <deque>
#include <map>
#include <set>
#include <utility>
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <vector>
using namespace std;
using namespace lshkit;
namespace po = boost::program_options; 


const float Br = 50.0f;
const int Nbr = 10; // number of R
const int Bk = 1;
int Nbk = 10;
const int TotalC = 1000;

const string ImgListFile = "E:\\testsift\\mm270k\\mm270k_c1_2000_ids.txt";
const string QueryListFile = "E:\\testsift\\mm270k\\mm270k_200_20_ids.txt";

string ImgList[2000];
string QueryList[20];

string BaseDir = "E:\\testsift\\mm270k\\sift_wall_result_pr";
string InFile = BaseDir + "\\mm270k_c1_2000_w%d_n%d_result.txt_dist.txt";
string OutFile = BaseDir + "\\statistic\\n%d.txt";
int N;
typedef vector<vector<pair<float, float> > > Result;
Result result;

void initQuery()
{
	FILE* fimg = fopen(ImgListFile.c_str(), "r");
	char buf[256];
	int fid;
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		ImgList[fid-1] = string(buf);
	}
	fclose(fimg);
	cout << "load image list finished." << endl;

	// query list
	fimg = fopen(QueryListFile.c_str(), "r");
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		QueryList[fid-1] = string(buf);
	}
	fclose(fimg);
	cout << "load query list finished." << endl;
}

string getFileNameNoExt(const string& fn)
{
	int i = fn.rfind('.');
	int j = fn.rfind('/');
	if (j == -1)
	{
		j = fn.rfind('\\');
	}
	if (i == -1)
	{
		i = fn.length();
	}

	return fn.substr(j+1, i-j-1);
}

bool checkMatchByFileName(const string& q, const string& r)
{
	string rr = getFileNameNoExt(r);
	return (rr.length() > q.length() && rr[q.length()] == '_' && rr.find(q) == 0);
}

void dostatistic()
{
	FILE* fout = fopen(str(boost::format(OutFile.c_str()) % N).c_str(), "w");
	for (int i=0; i<=10; ++i)
	{
		fprintf(fout, "p_w%d,r_w%d,", i, i);
	}

	// do
	int fid1, fid2;
	float dist;
	char buf[1024];
	int maxN = 0;
	for (int w=0; w<=10; ++w)
	{
		printf("w=%d\n", w);
		FILE* fin = fopen(str(boost::format(InFile.c_str()) % w % N).c_str(), "r");
		if (fin == 0)
		{
			printf("open file error%s\n", str(boost::format(InFile.c_str()) %w %N).c_str());
			break;
		}
		vector<vector<map<int, int> > > stat;
		// init
		for (int i=0; i<20; ++i) // 20 - query images
		{
			stat.push_back(vector<map<int, int> >());
			for (int j=0; j<Nbr; ++j) 
			{
				stat[i].push_back(map<int, int>());
			}
		}

		fgets(buf, 1024, fin); // ignore first line
		while (fscanf(fin, "%d,%d,%f", &fid1, &fid2, &dist) != EOF)
		{
			int t = int(dist/Br);
			for (int i=0; i<=t && i<Nbr; ++i)
			{
				stat[fid1-1][i][fid2]++;
			}
			
		}
		
		fclose(fin);

		// statistic again
		int* corrects[Nbr];
		int* wrongs[Nbr];
		for (int i=0; i<Nbr; ++i)
		{
			corrects[i] = new int[Nbk];
			wrongs[i] = new int[Nbk];
		}
		for (int i=0; i<20; ++i) // 20 - query images
		{
			for (int j=0; j<Nbr; ++j) 
			{
				const string testPicPart = getFileNameNoExt(QueryList[i]);
				for (map<int, int>::const_iterator it=stat[i][j].begin(); it!=stat[i][j].end(); ++it)
				{
					if (checkMatchByFileName(testPicPart, ImgList[it->first-1]))
					{
						for (int ii=0; ii<it->second-1 && ii < Nbk; ++i)
						{
							corrects[j][ii]++;
						}
					}
					else
					{
						for (int ii=0; ii<it->second-1 && ii < Nbk; ++i)
						{
							wrongs[j][ii]++;
						}
					}

				}
			}
		} 

		// calc results
		result.push_back(vector<pair<float, float> >());
		int ii = result.size() - 1;
		for (int i=0; i<Nbr; ++i)
		{
			for (int j=0; j<Nbk; ++j)
			{
				result[ii].push_back(pair<float, float>((float)corrects[i][j] / (corrects[i][j] + wrongs[i][j]), (float)corrects[i][j]/TotalC));
			}
		}

		if (maxN < result[ii].size())
		{
			maxN = result[ii].size();
		}


		// free
		for (int i=0; i<Nbr; ++i)
		{
			delete[] corrects;
			delete[] wrongs;
		}
	}
	
	// output results
	for (int i=0; i<maxN; ++i)
	{
		for (int w=0; w<=10; ++w)
		{
			if (i < maxN)
			{
				fprintf(fout, "%g,%g,", result[w][i]);
			}
			else
			{
				fprintf(fout, ",,");
			}
		}
		fprintf(fout, "\n");
	}

	// free
	fclose(fout);
}

int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("num,n", po::value<int>(&N), "Number limit of keypoints per image.")
		("upkplimit,k", po::value<int>(&Nbk)->default_value(10), "Number limit of keypoints for a near-duplicate match.");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0  || vm.count("num") == 0 || vm.count("upkplimit") == 0)
	{
		cout << desc;
		return 0;
	}

	// 
	dostatistic();

	return 0;
}
