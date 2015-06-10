#ifndef DEFINES_H
#define DEFINES_H

#define I g_images[i]
#define MASKON 0
#define MASKOFF 0x80000000
#define NEXTiMASK(i) { temp=*g_images[i].binary_mask.pointer++; maskcount[i]=temp&0x7fffffff; mask[i]=~temp&0x80000000; }
#define PREViMASK(i) { temp=*--g_images[i].binary_mask.pointer; maskcount[i]=temp&0x7fffffff; mask[i]=~temp&0x80000000; }

#define PY(i,l) g_images[i].pyramid[l]

#endif
