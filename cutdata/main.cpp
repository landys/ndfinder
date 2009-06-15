#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
using namespace std;
namespace po = boost::program_options; 

const int MaxBuf = 1024 * 1024;
const int HeaderSize = 12;
const int RowNumOffset = 4;
char buf[MaxBuf];

int main(int argc, char* argv[])
{
	// args
	string dataFile;
	string resultFile;
	int nkps;
	int offset = 0;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("siftdata,d", po::value<string>(&dataFile), "sift data for index.")
		("partdata,p", po::value<string>(&resultFile), "output part of sift data.")
		("kepoints,n", po::value<int>(&nkps), "the n keypoints begin from the offset will be in the prt of sift data.")
		("offset,o", po::value<int>(&offset)->default_value(0), "the offset of the sift data file to be copied.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("siftdata") == 0 || vm.count("partdata") == 0 
		|| vm.count("kepoints") == 0)
	{
		cout << desc;
		return 1;
	}

	// do
	FILE* fsd = fopen(dataFile.c_str(), "rb");
	if (fsd == 0)
	{
		printf("Cannot open %s\n", dataFile.c_str());
		return 2;
	}
	FILE* fre = fopen(resultFile.c_str(), "wb");
	if (fre == 0)
	{
		printf("Cannot open %s\n", resultFile.c_str());
		return 3;
	}

	unsigned int eleSize = 0;
	unsigned int rows = 0;
	unsigned int columns = 0;
	fread(&eleSize, sizeof(unsigned int), 1, fsd);
	fread(&rows, sizeof(unsigned int), 1, fsd);
	fread(&columns, sizeof(unsigned int), 1, fsd);

	printf("eleSize=%d\n", eleSize);
	printf("rows=%d\n", rows);
	printf("columns=%d\n", columns);

	int entrySize = eleSize * columns;

	fseek(fsd, 0, SEEK_SET);
	fread(buf, sizeof(char), HeaderSize, fsd);
	fwrite(buf, sizeof(char), HeaderSize, fre);
	if (offset > 0)
	{
		fseek(fsd, offset*entrySize, SEEK_CUR);
	}

	int bp = MaxBuf / entrySize;
	unsigned int count = 0;
	for (int i=0; i<nkps; i+=bp)
	{
		int k = bp>(nkps-i) ? (nkps-i) : bp;
		int ws = fread(buf, entrySize, k, fsd);
		fwrite(buf, entrySize, ws, fre);
		count += ws;
		if (ws < k)
		{
			// end of file
			break;
		}
	}

	printf("%u keypoints copied\n", count);
	fseek(fre, RowNumOffset, SEEK_SET);
	fwrite(&count, sizeof(unsigned int), 1, fre);

	fclose(fsd);
	fclose(fre);

	return 0;
}

