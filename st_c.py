#jh:mix-base,lazy-type

#a.py
import ctypes

lib = ctypes.CDLL("ca.so")

lib.solve()



#g++ ca.cpp -shared -fPIC -o ca.so
#ca.cpp
'''
//sometimes you can't use iostream
#include<cstdio>
using namespace std;

extern "C"{

void solve(){
	printf("Hello,world!\n");
}

}

'''


#int
lib.solve(23)
#void solve(int a);

#float
lib.solve(ctypes.c_float(2.333))
#void solve(float a);

#int-array
ta = (ctypes.c_int*5)(2,3,4,5,6)
lib.solve(ta, 5)
#void solve(int ta[], int len);

#char-array
lib.solve(b'23333')

