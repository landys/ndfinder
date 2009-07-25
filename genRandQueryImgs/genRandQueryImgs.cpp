#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <queue>
#include <numeric>
#include <utility>
#include <ctime>
#include <cmath>
#include <functional>
using namespace std;
namespace po = boost::program_options;

string ImgsFile;
string ResultFile;
int Nc;
// key - category type, value - image list(first-file id, second file name).
map<string, deque<pair<int, string> > > Imgs;
//// 1, 2
//map<string, int > Types;

//int TypeBound = 4402;

char buf[256];

// the parameter fn should has the result.
string getObjectTypeFromPath(const string& fn)
{
	int i = fn.rfind('/');
	if (i == -1)
	{
		i = fn.rfind('\\');
	}

	int j = fn.rfind('/', i-1);
	if (j == -1)
	{
		j = fn.rfind('\\', i-1);
	}

	string re = fn.substr(j+1, i-j-1);

/*	i = re.find('_');
	if (i != -1)
	{
		re = re.substr(0, i);
	}*/

	return re;
}

void initQuery()
{
	FILE* fimg = fopen(ImgsFile.c_str(), "r");
	int fid;
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		string fname = string(buf);
		string type = getObjectTypeFromPath(fname);
		//if (Imgs.find(type) == Imgs.end())
		//{
		//	Imgs[type] = deque<string>();
		//	if (fid <= TypeBound)
		//	{
		//		Types[type] = 1;
		//	}
		//	else
		//	{
		//		Types[type] = 2;
		//	}
		//}
		Imgs[type].push_back(pair<int, string>(fid, fname));
	}
	fclose(fimg);
	cout << "load image list finished." << endl;
}

void pickAndSaveQueryImgs()
{
	FILE* fre = fopen(ResultFile.c_str(), "w");
	for (map<string, deque<pair<int, string> > >::const_iterator it=Imgs.begin(); it!=Imgs.end(); ++it)
	{
		int n = it->second.size();
		bool* flags = new bool[n];
		if (n <= Nc)
		{
			fill(flags, flags+n, true);
		}
		else
		{
			fill(flags, flags+n, false);
			srand(time(0));
			for (int i=0; i<Nc; ++i)
			{
				int k = (int)(rand() / (double)RAND_MAX * n);
				while (flags[k])
				{
					k = (k + 1) % n;
				}
				flags[k] = true;
			}
		}
		
		for (int i=0; i<n; ++i)
		{
			if (flags[i])
			{
				//fprintf(fre, "%d %s\n", Types[it->first], it->second[i].c_str());
				fprintf(fre, "%d %s\n", it->second[i].first, it->second[i].second.c_str());
			}
			
		}
		delete[] flags;
	}
	fclose(fre);
}

int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("result,r", po::value<string>(&ResultFile), "the result query image list.")
		("nc,n", po::value<int>(&Nc)->default_value(10), "images per category.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("imgslist") == 0 || vm.count("result") == 0)
	{
		cout << desc;
		return 1;
	}

	initQuery();
	pickAndSaveQueryImgs();
	
	return 0;
}