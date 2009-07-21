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
#include "../sift/siftfeat.h"
#include "../sift/imgfeatures.h"
using namespace std;
namespace po = boost::program_options;

string ImgsFile;
string ResultFile;
string QueryImgsFile;
// key - image file name, value - image id
map<string, int > Imgs;

char buf[256];

void init()
{
	FILE* fimg = fopen(ImgsFile.c_str(), "r");
	int fid;
	while (fscanf(fimg, "%d %s", &fid, buf) != EOF)
	{
		Imgs[string(buf)] = fid;
	}
	fclose(fimg);
	cout << "load image list finished." << endl;
}

void pickQueryImgIds()
{
	FILE* fqimg = fopen(QueryImgsFile.c_str(), "r");
	FILE* fre = fopen(ResultFile.c_str(), "w");
	int type;
	while (fscanf(fqimg, "%d %s", &type, buf) != EOF)
	{
		fprintf(fre, "%d %d %s\n", type, Imgs[string(buf)], buf);
	}

	fclose(fqimg);
	fclose(fre);
}

int main(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("imgslist,a", po::value<string>(&ImgsFile), "the map file between pictures and their ids.")
		("querylist,q", po::value<string>(&QueryImgsFile), "the query image list file with type.")
		("result,r", po::value<string>(&ResultFile), "the result query image list file with type and ids.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("imgslist") == 0 || vm.count("querylist") == 0
		|| vm.count("result") == 0)
	{
		cout << desc;
		return 1;
	}

	init();
	pickQueryImgIds();

	return 0;
}