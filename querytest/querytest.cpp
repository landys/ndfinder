//#define PCASIFT

#ifdef PCASIFT
#include "../pcasift/siftfeat.h"
#include "../pcasift/imgfeatures.h"
#else
#include "../sift/siftfeat.h"
#include "../sift/imgfeatures.h"
#endif // PCASIFT

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
using namespace std;
using namespace lshkit;
namespace po = boost::program_options; 

#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "highgui.lib")

#ifdef PCASIFT
#pragma comment(lib, "pcasift.lib")
#else
#pragma comment(lib, "sift.lib")
#endif // PCASIFT

string BaseDir = "E:\\testsift\\";
string IndexFile = BaseDir + "t84.index";
string DataFile = BaseDir + "t84.sift.data";
string IdsFile = BaseDir + "t84.ids.txt";
string ImgFile = BaseDir + "t84.txt";
string QueryLog = BaseDir + "t84.query.log";

string PicDir = "E:\\testsift\\pics\\";
string TestPic = PicDir + "69.jpg";

string ResultDir = BaseDir + "result_100\\";

#ifdef PCASIFT
string EigsFile = "E:\\testsift\\tools\\gpcavects_eig.txt";
const float DistLimit = 0.5f;
const int DIM = PCASIZE;
#else
const float DistLimit = 100;
const int DIM = 128;
#endif // PCASIFT

const int TopKPics = 60;
const int Interval = 50;
int K = 50;
float R = 1e30;//numeric_limits<float>::max();
unsigned T = 20;
//const int MatchKpLimit = 10;

class KeyPoint
{
public:
	int id; // in fact, id is same as line number beginning from 0 in ids file.
	int fid;
	//string fname;
	float x;
	float y;
	float scl;
	float ori;
	
	KeyPoint() : id(-1), fid(-1), x(-1.0f), y(-1.0f), scl(-1.0f), ori(-1.0f)
	{
	}

	KeyPoint(const KeyPoint& kp) : id(kp.id), fid(kp.fid), x(kp.x), y(kp.y), scl(kp.scl), ori(kp.ori)
	{
	}

	KeyPoint(int id, int fid, float x, float y, float scl, float ori) : id(id), fid(fid), x(x), y(y), scl(scl), ori(ori)
	{
	}
};

class MatchKp
{
public:
	KeyPoint kp1;
	KeyPoint kp2;
	float dist;

	MatchKp()
	{
		dist = 1e10;// numeric_limits<float>::max();
	}

	MatchKp(const MatchKp& mk) : kp1(mk.kp1), kp2(mk.kp2), dist(mk.dist)
	{

	}

	MatchKp(const KeyPoint& kp1, const KeyPoint& kp2, float dist) : kp1(kp1), kp2(kp2), dist(dist)
	{

	}
};

bool operator<(const MatchKp& mk1, const MatchKp& mk2)
{
	return mk1.dist < mk2.dist;
}


// all keypoints in dataset
deque<KeyPoint> KeyPoints;
// key - file id, value - file name
map<int, string> Fnames;
// key - file id, value - if the most similar key is found in this file for a certain keypoint.
map<int, bool> FFound;
// key - file id, value - keypoint id, which is the index of KeyPoints
map<int, deque<int> > Fkps;
// key - pic file id, value - matched keypoints sored by distance. multiset allows the same distance of different matched keypoints.
map<int, multiset<MatchKp> > Results;
// for draw merged file

void initQuery()
{
	FILE* fimg = fopen(ImgFile.c_str(), "r");
	char buf[256];
	int id;
	while (fscanf(fimg, "%d %s", &id, buf) != EOF)
	{
		Fnames[id] = string(buf);
		FFound[id] = false;
	}
	fclose(fimg);
	cout << "load image list finished." << endl;

	FILE* fids = fopen(IdsFile.c_str(), "r");
	KeyPoint kp;
	while (fscanf(fids, "%d %d %f %f %f %f", &kp.id, &kp.fid, &kp.x, &kp.y, &kp.scl, &kp.ori) != EOF)
	{
		KeyPoints.push_back(kp);

		map<int, deque<int> >::const_iterator it = Fkps.find(kp.fid);
		if (it == Fkps.end())
		{
			Fkps[kp.fid] = deque<int>();
		}
		Fkps[kp.fid].push_back(kp.id);
	}

	fclose(fids);
	cout << "load file info finished." << endl;
}

string& findPicByKpId(int key)
{
	return Fnames[KeyPoints[key].fid];
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

string getResultPicName(const string& s1, const string& s2, int n, int index)
{
	return str(boost::format("%s%d_%d_%s_%s.jpg") % ResultDir % index % n % getFileNameNoExt(s1) % getFileNameNoExt(s2));
}

IplImage* mergeImages(const IplImage* img1, const IplImage* img2)
{
	CvSize size1 = cvGetSize(img1);
	CvSize size2 = cvGetSize(img2);
	CvSize mergeSize = cvSize(size1.width + size2.width + Interval, max(size1.height, size2.height));

	IplImage* imgMerge = cvCreateImage(mergeSize, img1->depth, img1->nChannels);
	cvSet(imgMerge, cvScalar(255, 255, 255));

	cvSetImageROI(imgMerge, cvRect(0, 0, size1.width, size1.height));
	cvCopyImage(img1, imgMerge);
	cvSetImageROI(imgMerge, cvRect(size1.width + Interval, 0, size2.width, size2.height));
	cvCopyImage(img2, imgMerge);
	cvResetImageROI(imgMerge);

	return imgMerge;
}

void drawMatchLines(IplImage* img, const CvSize& size1, const multiset<MatchKp>& mks, ofstream& qlog)
{
	CvScalar color = cvScalar(255, 0, 0);

	for (multiset<MatchKp>::const_iterator it=mks.begin(); it!=mks.end(); ++it)
	{
		cvLine(img, cvPoint(it->kp1.x, it->kp1.y), cvPoint(it->kp2.x+size1.width+Interval, it->kp2.y), color);

		qlog << boost::format("%g: %g %g %g %g <==> %d %d %g %g %g %g") % it->dist % it->kp1.x % it->kp1.y % it->kp1.scl % it->kp1.ori % it->kp2.id % it->kp2.fid % it->kp2.x % it->kp2.y % it->kp2.scl % it->kp2.ori << endl;
	}
}

void draw_features( IplImage* img, const deque<int>& kids)
{
	CvScalar color = CV_RGB( 255, 255, 255 );
	int i;

	if( img-> nChannels > 1 )
	{
		color = FEATURE_LOWE_COLOR;
	}
	for( i = 0; i < kids.size(); i++ )
	{
		//draw_lowe_feature( img, feat + i, color );
		int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
		double scl, ori;
		double scale = 5.0;
		double hscale = 0.75;
		CvPoint start, end, h1, h2;

		/* compute points for an arrow scaled and rotated by feat's scl and ori */
		start_x = cvRound( KeyPoints[kids[i]].x );
		start_y = cvRound( KeyPoints[kids[i]].y );
		scl = KeyPoints[kids[i]].scl;
		ori = KeyPoints[kids[i]].ori;
		len = cvRound( scl * scale );
		hlen = cvRound( scl * hscale );
		blen = len - hlen;
		end_x = cvRound( len *  cos( ori ) ) + start_x;
		end_y = cvRound( len * -sin( ori ) ) + start_y;
		h1_x = cvRound( blen *  cos( ori + CV_PI / 18.0 ) ) + start_x;
		h1_y = cvRound( blen * -sin( ori + CV_PI / 18.0 ) ) + start_y;
		h2_x = cvRound( blen *  cos( ori - CV_PI / 18.0 ) ) + start_x;
		h2_y = cvRound( blen * -sin( ori - CV_PI / 18.0 ) ) + start_y;
		start = cvPoint( start_x, start_y );
		end = cvPoint( end_x, end_y );
		h1 = cvPoint( h1_x, h1_y );
		h2 = cvPoint( h2_x, h2_y );

		cvLine( img, start, end, color, 1, 8, 0 );
		cvLine( img, end, h1, color, 1, 8, 0 );
		cvLine( img, end, h2, color, 1, 8, 0 );
	}
	
}

void showResultPerImage(IplImage* img1, IplImage* img2, int fid2, const multiset<MatchKp>& mks, ofstream& qlog, int hide, const string& mergedFile)
{
	if (hide == 0)
	{
		cvNamedWindow("siftMerge");
	}

	CvSize size1 = cvGetSize(img1);
	CvSize size2 = cvGetSize(img2);

	draw_features(img2, Fkps[fid2]);

	IplImage* mergeImg = mergeImages(img1, img2);

	// draw matched lines
	drawMatchLines(mergeImg, size1, mks, qlog);

	if (hide == 0)
	{
		cvShowImage("siftMerge", mergeImg);
	}
	cvSaveImage(mergedFile.c_str(), mergeImg);

	if (hide == 0)
	{
		cvWaitKey();
		cvDestroyAllWindows();
	}

	cvReleaseImage(&mergeImg);
}

void showResult(IplImage* img, const string& testPic, ofstream& qlog, int hide)
{
	int count = 0;
	multimap<int, int, greater<int> > topkM;
	for (map<int, multiset<MatchKp> >::const_iterator it=Results.begin(); it!=Results.end(); ++it)
	{
		topkM.insert(pair<int, int>(it->second.size(), it->first));
		/*if (it->second.size() >= MatchKpLimit)
		{
			++count;
			string& s = Fnames[it->first];
			IplImage* img2 = cvLoadImage(s.c_str());

			showResultPerImage(img, img2, it->first, it->second, qlog, hide, getResultPicName(testPic, s, it->second.size(), count));

			cvReleaseImage(&img2);
		}*/
	}

	for (multimap<int, int, greater<int> >::const_iterator it=topkM.begin(); it!=topkM.end(); ++it)
	{
		if (count >= TopKPics)
		{
			break;
		}
		++count;

		string& s = Fnames[it->second];
		IplImage* img2 = cvLoadImage(s.c_str());

		showResultPerImage(img, img2, it->second, Results[it->second], qlog, hide, getResultPicName(testPic, s, it->first, count));

		cvReleaseImage(&img2);
	}

	cout << "total real match files: " << count << endl;
	qlog << "total real match files: " << count << endl;

	cvReleaseImage(&img);
};

void testMpLsh(int argc, char* argv[])
{
	// args
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message.")
		("query,q", po::value<string>(&TestPic), "the query picture.")
#ifdef PCASIFT
		("eigs,e", po::value<string>(&EigsFile), "the eigespace file to initialize PCA-SIFT.")
#endif // PCASIFT
		("index,i", po::value<string>(&IndexFile), "the mplsh index file.")
		("data,d", po::value<string>(&DataFile), "the data of the index.")
		("datamap,m", po::value<string>(&IdsFile), "the map file between pictures and their keypoints.")
		("imgslist,a", po::value<string>(&ImgFile), "the map file between pictures and their ids.")
		("resultdir,r", po::value<string>(&ResultDir), "the directory result pictures saved.")
		("log,l", po::value<string>(&QueryLog), "the query result log.")
		("show,s", "if show the result in windows, default is hide.");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") > 0  || vm.count("query") == 0 || vm.count("index") == 0 
		 || vm.count("data") == 0 || vm.count("datamap") == 0 || vm.count("imgslist") == 0 || vm.count("log") == 0)
	{
		cout << desc;
		return;
	}

	int hide = 1;
	if (vm.count("show") > 0)
	{
		hide = 0;
	}

	if (ResultDir[ResultDir.length()-1] != '\\' && ResultDir[ResultDir.length()-1] != '/')
	{
		ResultDir += '\\';
	}
	
	// load data
	cout << "LOADING DATA..." << endl;
	FloatMatrix data(DataFile);
	typedef MultiProbeLshIndex<FloatMatrix::Accessor> Index;

	FloatMatrix::Accessor accessor(data);
	Index index(accessor);

	// load index
	ifstream is(IndexFile.c_str(), ios_base::binary);
	if (is) {
		is.exceptions(ios_base::eofbit | ios_base::failbit | ios_base::badbit);
		cout << "LOADING INDEX..." << endl;
		//timer.restart();
		index.load(is);
		verify(is);
		//cout << boost::format("LOAD TIME: %1%s.") % timer.elapsed() << endl;
	}

	//Stat recall;
	Stat cost;
	Topk<unsigned> topk;
	feature* feat = 0;
#ifdef PCASIFT
	// init for pca-sift, never forget again!!!!
	initialeigs(EigsFile.c_str());
#endif // PCASIFT
	int Q = siftFeature(TestPic.c_str(), &feat, 1, 0.03, 100);

	cout << "init query..." << endl;
	initQuery();
	cout << "finish init query." << endl;

	boost::progress_display progress(Q);
	ofstream qlog;
	qlog.open(QueryLog.c_str(), ios_base::app);
	cout << "Query pic: " << TestPic.c_str() << endl;
	qlog << "Query pic: " << TestPic.c_str() << endl;
	float qd[DIM];
	for (int i = 0; i < Q; ++i)
	{
		unsigned cnt;
		topk.reset(K, R);
		for (int j=0; j<DIM; ++j)
		{
#ifdef PCASIFT
			qd[j] = feat[i].PCAdescr[j];
#else
			qd[j] = feat[i].descr[j];
#endif // PCASIFT
		}
		index.query(qd, &topk, T, &cnt);

		KeyPoint ckp(-1, -1, feat[i].x, feat[i].y, feat[i].scl, feat[i].ori);
		for (map<int, bool>::iterator it=FFound.begin(); it!=FFound.end(); ++it)
		{
			it->second = false;
		}

		// show result
		for (int j=0; j<topk.size(); ++j)
		{
			if (topk[j].key >= KeyPoints.size())
			{
				qlog << boost::format("wrong match: (%g, %g, %g, %g) == (%d, %g)") % feat[i].x % feat[i].y % feat[i].scl % feat[i].ori % topk[j].key % topk[j].dist << endl;
				break;
			}
			int fid = KeyPoints[topk[j].key].fid;
			// already get the closest keypoint in the file
			if (FFound[fid] || topk[j].dist > DistLimit)
			{
				continue;
			}
			map<int, multiset<MatchKp> >::const_iterator it = Results.find(fid);
			if (it == Results.end())
			{
				Results[fid] = multiset<MatchKp>();
			}

			Results[fid].insert(MatchKp(ckp, KeyPoints[topk[j].key], topk[j].dist));
			FFound[fid] = true;
		}

		//recall << bench.getAnswer(i).recall(topk);

		cost << double(cnt)/double(data.getSize());
		++progress;
	}

	//qlog << "[RECALL] " << recall.getAvg() << " +/- " << recall.getStd() << endl;
	qlog << "[COST] " << cost.getAvg() << " +/- " << cost.getStd() << endl;
	cout << "[COST] " << cost.getAvg() << " +/- " << cost.getStd() << endl;

	qlog << "Raw matched Results: " << Results.size() << endl;
	cout << "Raw matched Results: " << Results.size() << endl;

	IplImage* img = cvLoadImage(TestPic.c_str());
	// draw sift points
	draw_features(img, feat, Q);
	showResult(img, TestPic, qlog, hide);

	cvReleaseImage(&img);
	free(feat);
	qlog.close();
}


int main(int argc, char* argv[])
{
	//test();
	//generateTestData();
	testMpLsh(argc, argv);
	return 0;
}
