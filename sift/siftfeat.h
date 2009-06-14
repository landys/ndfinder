#ifdef WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" DLL_EXPORT int showSift(const char* imagenamefile, const char* out_file_name, int img_dbl, double contr_thr, int n_max=0);
extern "C" DLL_EXPORT int siftImage(const char* imagename, const char* out_file_name, int img_dbl, double contr_thr, long long id=0, int n_max=0);
extern "C" DLL_EXPORT int siftFeature(const char* imagename, struct feature** fp, int img_dbl, double contr_thr, int n_max=0);

