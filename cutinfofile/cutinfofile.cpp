#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
using namespace std;
namespace po = boost::program_options; 

const int MaxBuf = 1024 * 1024;
char buf[MaxBuf];

int main(int argc, char* argv[])
{
	// args
	string infoFile;
	string resultFile;
	int nkps;
	int offset = 0;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("datamap,m", po::value<string>(&infoFile), "the map file between pictures and their keypoints.")
		("partinfo,p", po::value<string>(&resultFile), "output part of sift info file.")
		("kepoints,n", po::value<int>(&nkps), "the n keypoints begin from the offset will be in the prt of sift data.")
		("offset,o", po::value<int>(&offset)->default_value(0), "the offset of the sift data file to be copied.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("datamap") == 0 || vm.count("partinfo") == 0 
		|| vm.count("kepoints") == 0)
	{
		cout << desc;
		return 1;
	}

	// do
	FILE* fin = fopen(infoFile.c_str(), "r");
	if (fin == 0)
	{
		printf("Cannot open %s\n", infoFile.c_str());
		return 2;
	}
	FILE* fre = fopen(resultFile.c_str(), "w");
	if (fre == 0)
	{
		printf("Cannot open %s\n", resultFile.c_str());
		return 3;
	}

	printf("Copy %d keypoints info begin from offset %d.\n", nkps, offset);
	
	for (int i=0; i<offset; ++i)
	{
		if (fgets(buf, MaxBuf, fin) == NULL)
		{
			break;
		}
	}
	int id = 0;
	int i = 0;
	for (i=0; i<nkps; ++i)
	{
		if (fscanf(fin, "%d", &id) == EOF)
		{
			break;
		}

		if (fgets(buf, MaxBuf, fin) == NULL)
		{
			break;
		}

		fprintf(fre, "%d%s", i, buf);
	}

	printf("%u keypoints copied\n", i);

	fclose(fin);
	fclose(fre);

	return 0;
}

