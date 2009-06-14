#include <iostream>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <deque>
#include <algorithm>
using namespace std;
namespace po = boost::program_options;

void selectFiles(const string& srcFile, const string& outFile, const string& leftFile, int n)
{
	deque<string> all;
	string s;

	ifstream src(srcFile.c_str());
	while (getline(src, s))
	{
		all.push_back(s);
	}
	src.close();

	int nall = all.size();
	if (nall < n)
	{
		return;
	}

	string f1 = outFile;
	string f2 = leftFile;
	if (n > nall / 2)
	{
		n = nall - n;
		f1 = leftFile;
		f2 = outFile;
	}

	srand(time(NULL));
	int* fi = new int[nall];
	fill(fi, fi+nall, 0);

	ofstream out(f1.c_str());
	ofstream left(f2.c_str());

	for (int i=0; i<n; ++i)
	{
		int k = rand() % nall;
		while (fi[k] == 1)
		{
			k = (k + 1) % nall;
		}
		fi[k] = 1;
	}

	for (int i=0; i<nall; ++i)
	{
		if (fi[i] == 1)
		{
			out << all[i] << endl;
		}
		else
		{
			left << all[i] << endl;
		}
	}

	delete[] fi;
	out.close();
	left.close();
}

int main(int argc, char* argv[])
{
	// parse args
	string srcFile;
	string outFile;
	string leftFile;
	int n;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("source,s", po::value<string>(&srcFile), "source image list file.")
		("output,o", po::value<string>(&outFile), "output image list file.")
		("left,l", po::value<string>(&leftFile), "left image list file, which is (source - output).")
		("number,n", po::value<int>(&n), "number to be select by random.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0 || vm.count("source") == 0 || vm.count("left") == 0 || vm.count("output") == 0 
		|| vm.count("number") == 0)
	{
		cout << desc;
		return 1;
	}

	selectFiles(srcFile, outFile, leftFile, n);

	return 0;
}
