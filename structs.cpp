#include "structs.h"
#include "globals.h"

#ifdef WIN32
#include "functions.h"
#endif

#ifdef WIN32
void my_timer::set() {
	QueryPerformanceCounter(&t1);
}

double my_timer::read() {
	QueryPerformanceCounter(&t2);
	return (double)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart;
}

my_timer::my_timer() {
	QueryPerformanceFrequency(&frequency);
}

void my_timer::report(const char* name) {
	if (g_timing) output(0,"%s: %.3fs\n",name,this->read());
}

#else

void my_timer::set() { }
double my_timer::read() { return 0; }
void my_timer::report(const char* name) { if (g_timing) printf("Timing not available\n"); }

#endif
