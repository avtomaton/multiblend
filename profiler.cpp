#include "profiler.h"

#include <algorithm>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef min
#undef max
#else
#define uint64 cv_uint64
#define int64 cv_int64
#include <opencv2/core/core.hpp>
#undef uint64
#undef int64
#endif

static long getTickCount_()
{
	#ifdef WIN32
		return GetTickCount();
	#else
		return cv::getTickCount();
	#endif
}
static long getTickFrequency_()
{
	#ifdef WIN32
		return 1000;
	#else
		return cv::getTickFrequency();
	#endif
}

void Profiler::Params::start()
{
	start_time_ = getTickCount_();
	++count_;
}
void Profiler::Params::stop()
{
	timer_ += getTickCount_() - start_time_;
}

void Profiler::start(const std::string &str)
{
	auto it = lst.find(str);
	if (it == lst.end())
	{
		lst.emplace(str, Params(number++));
	}
	else
	{
		(*it).second.start();
	}
	if (!parent_stack.empty())
	{
		auto ig = graph.find(parent_stack.back());
		if (ig == graph.end())
		{
			auto par = graph.emplace(parent_stack.back(), std::set<std::string>());
			par.first->second.emplace(str);
		}
		else
		{
			ig->second.emplace(str);
		}
	}
	parent_stack.push_back(str);
}
void Profiler::stop(const std::string &str)
{
	auto it = lst.find(str);
	if (it != lst.end())
	{
		(*it).second.stop();
	}
	parent_stack.pop_back();
}

void Profiler::report()
{
	printf("-------------------\n");
	printf("report\n");
	int N = lst.size();
	std::vector<std::string> knots;
	knots.reserve(N);
	for (size_t i = 0; i < lst.bucket_count(); ++i)
	{
		for (auto it = lst.begin(i); it != lst.end(i); ++it)
			knots.push_back((*it).first);
	}
	std::sort(knots.begin(), knots.end());
	std::map<std::string, int> pairs;
	for (int i = 0; i < N; ++i)
		pairs.emplace(knots[i], i);

	std::vector < std::list<int> > g(N);
	for (auto it = graph.begin(); it != graph.end(); ++it)
	{
		printf("%s: ", (*it).first.c_str());
		int k = pairs[(*it).first];
		for (auto jt = it->second.begin(); jt != it->second.end(); ++jt)
		{
			printf("%s ", (*jt).c_str());
			g[k].push_back(pairs[*jt]);
		}
		printf("\n");
	}

	if (pairs.find("root") == pairs.end())
	{
		printf("\"root\" not found\n");
		printf("-------------------\n");
		printf("name | count | time\n");
		printf("-------------------\n");
		for (int k = 0; k < N; ++k)
			printf("%s | %d | %f\n", knots[k].c_str(), lst.at(knots[k]).count(), ((double)lst.at(knots[k]).timer()) / getTickFrequency_());
	    printf("-------------------\n");
		return;
	}
	FDS searcher(&g, N, pairs["root"]);
	std::vector<int> indecies;
	std::vector<int> levels;
	searcher.compute(indecies, levels);

	printf("-------------------\n");
	printf("name | count | time | %%part | %%funcs\n");
	printf("-------------------\n");

	double tfull = ((double)lst.at(knots[indecies[0]]).timer()) / getTickFrequency_();

	for (int i = 0; i < N; ++i)
	{
		int k = indecies[i];
		for (int t = 0; t < levels[i]; ++t)
			printf("\t");
		long summ = 0;
		for (auto ig = g[k].begin(); ig != g[k].end(); ++ig)
			summ += lst.at(knots[*ig]).timer();
		double t = ((double)lst.at(knots[k]).timer()) / getTickFrequency_();
    	if (!g[k].empty())
			printf("%-8s | %d | %.3f | %.1f%% | (%.1f%%)\n", knots[k].c_str(), lst.at(knots[k]).count(), t, t / tfull * 100, ((double)summ) / getTickFrequency_() / t * 100);
    	else
    		printf("%-8s | %d | %.3f | %.1f%% | \n", knots[k].c_str(), lst.at(knots[k]).count(), t, t / tfull * 100);

    }
    printf("-------------------\n");
}

