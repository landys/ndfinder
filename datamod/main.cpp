#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <cstdio>
#include <string>
#include <iostream>
using namespace std;
namespace po = boost::program_options; 

const int HeaderSize = 12;
const int RowNumOffset = 4;

int main(int argc, char* argv[])
{
	// args
	string dataFile;
	string resultFile;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("siftdata,d", po::value<string>(&dataFile), "source sift data for index.")
		("outputdata,o", po::value<string>(&resultFile), "output sift data which element type is in user defined.")
		("us2f,f", "change element type from unsigned short to float. The only function in current implementation, default is true.");
		//("f2us,s", "change element type from float to unsigned short, can be inactived ty us2f.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("siftdata") == 0 || vm.count("outputdata") == 0)
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

	unsigned int newEleSize = sizeof(float);

	fwrite(&newEleSize, sizeof(unsigned int), 1, fre);
	fwrite(&rows, sizeof(unsigned int), 1, fre);
	fwrite(&columns, sizeof(unsigned int), 1, fre);

	unsigned short* entry = new unsigned short[columns];
	float* newEntry = new float[columns];
	for (unsigned i=0; i<rows; ++i)
	{
		int c = fread(entry, eleSize, columns, fsd);
		if (c != columns)
		{
			printf("Read less than a column, something error.\n");
			break;
		}
		for (unsigned j=0; j<columns; ++j)
		{
			newEntry[j] = entry[j];
		}
		c = fwrite(newEntry, newEleSize, columns, fre);
		if (c != columns)
		{
			printf("Write less than a column, something error.\n");
			break;
		}
	}

	delete[] entry;
	delete[] newEntry;

	fclose(fsd);
	fclose(fre);

	return 0;
}

