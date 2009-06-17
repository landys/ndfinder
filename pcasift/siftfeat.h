#ifdef WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define MERGE_TEST

extern "C" DLL_EXPORT void initialeigs(const char* eigsfile);
// contr_weight means: the n_max is determined by contract and ratio of principle curvatures,
// contr_weight is the weight of contract. Its value is in [0, 1],
// and 1 means n_max is only depended on contract.


// each line of imagenamefile is:
// "id file_name"
// contr_thr is [0, 1], img_dbl is 0/1, curv_thr is >0, contr_weight is [0, 1].
extern "C" DLL_EXPORT int showSift(const char* imagenamefile, const char* out_file_name, int img_dbl=1, 
								   double contr_thr=0.03, int n_max=3000, double curv_thr=10.0, double contr_weight=1.0);
extern "C" DLL_EXPORT int siftImage(const char* imagename, const char* out_file_name, int img_dbl=1, 
									double contr_thr=0.03, long long id=0, int n_max=3000, double curv_thr=10.0, double contr_weight=1.0);
extern "C" DLL_EXPORT int siftFeature(const char* imagename, struct feature** fp, int img_dbl=1, 
									  double contr_thr=0.03, int n_max=3000, double curv_thr=10.0, double contr_weight=1.0);

