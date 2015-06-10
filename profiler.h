#ifndef PROFILER_H
#define PROFILER_H

#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <list>
#include <vector>

class Profiler
{
public:
	Profiler() : number(0) {}
	struct Params
	{
	public:
		Params(int num) : timer_(0), count_(0), number_(num)
		{
			start();
		}
		void start();
		void stop();
		long count() const {return count_;}
		long timer() const {return timer_;}
		int number() const {return number_;}
	private:
		long timer_;
		long start_time_;
		int count_;
		int number_;
	};
	void start(const std::string &str);
	void stop(const std::string &str);
	void report();
private:
	std::unordered_map<std::string, Params> lst;
	std::list<std::string> parent_stack;
	std::map<std::string, std::set<std::string> > graph;//parents, children
	int number;
};

class Proftimer
{
public:
#ifdef USE_PROFILER
	Proftimer(Profiler *prof_, const std::string &str) : prof(prof_), name(str)
	{
		recording = true;
		prof->start(name);
	}
	~Proftimer()
	{
		if (recording)
			prof->stop(name);
	}
	void stop()
	{
		recording = false;
		prof->stop(name);
	}
#else
	Proftimer(Profiler *, const std::string &) {}
	void stop() {}
#endif
private:
	Profiler *prof;
	std::string name;
	bool recording;
};

class FDS
{
	typedef std::list<int> child_list;
	typedef std::vector< child_list > graph_t;
public:
	FDS(graph_t *g_, int N, int root_)
	{
		setup(g_, N, root_);
	}
	void setup(graph_t *g_, int N, int root_)
	{
		g = g_;
		used.clear();
		used.resize(N, false);
		indecies.resize(N);
		levels.resize(N);
		counter = 0;
		level = 0;
		root = root_;
	}
	void fds(int v) {
		used[v] = true;
		levels[counter] = level++;
		indecies[counter++] = v;
		for (auto it = (*g)[v].begin(); it != (*g)[v].end(); ++it)
		{
			if (!used[*it])
				fds(*it);
		}
		--level;
	}
	void compute(std::vector<int> &out_ind, std::vector<int> &out_lev)
	{
		fds(root);
		std::swap(out_ind, indecies);
		std::swap(out_lev, levels);
	}
private:
	graph_t *g;
	std::vector<char> used;
	std::vector<int> indecies;
	std::vector<int> levels;
	int counter;
	int level;
	int root;
};

#endif
