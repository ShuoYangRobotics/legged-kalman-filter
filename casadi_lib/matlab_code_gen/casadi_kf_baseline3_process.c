/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) casadi_kf_baseline3_process_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_c3 CASADI_PREFIX(c3)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

static const casadi_int casadi_s0[26] = {22, 1, 0, 22, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
static const casadi_int casadi_s1[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};

static const casadi_real casadi_c0[3] = {0., 0., 1.};
static const casadi_real casadi_c1[3] = {0., 1., 0.};
static const casadi_real casadi_c2[3] = {1., 0., 0.};
static const casadi_real casadi_c3[3] = {0., 0., 9.8000000000000007e+00};

/* process:(i0[22],i1[7],i2[7],i3)->(o0[22]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cr, *cs;
  casadi_real *w0=w+3, w1, w2, *w3=w+27, *w4=w+30, *w5=w+33, *w6=w+42, w7, w8, w9, w10, *w11=w+55, *w12=w+58, *w13=w+61, *w14=w+64, *w15=w+73, *w16=w+82, *w17=w+91, w18, w19, *w20=w+100, *w21=w+103, *w22=w+115, *w23=w+137, *w24=w+159, *w25=w+166, *w26=w+173;
  /* #0: @0 = input[0][0] */
  casadi_copy(arg[0], 22, w0);
  /* #1: @1 = 0.166667 */
  w1 = 1.6666666666666666e-01;
  /* #2: @2 = input[3][0] */
  w2 = arg[3] ? arg[3][0] : 0;
  /* #3: @1 = (@1*@2) */
  w1 *= w2;
  /* #4: @3 = @0[3:6] */
  for (rr=w3, ss=w0+3; ss!=w0+6; ss+=1) *rr++ = *ss;
  /* #5: @4 = zeros(3x1) */
  casadi_clear(w4, 3);
  /* #6: @5 = zeros(3x3) */
  casadi_clear(w5, 9);
  /* #7: @6 = zeros(3x3) */
  casadi_clear(w6, 9);
  /* #8: @7 = @0[8] */
  for (rr=(&w7), ss=w0+8; ss!=w0+9; ss+=1) *rr++ = *ss;
  /* #9: @8 = cos(@7) */
  w8 = cos( w7 );
  /* #10: @9 = sin(@7) */
  w9 = sin( w7 );
  /* #11: @9 = (-@9) */
  w9 = (- w9 );
  /* #12: @10 = 0 */
  w10 = 0.;
  /* #13: @11 = horzcat(@8, @9, @10) */
  rr=w11;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  /* #14: @11 = @11' */
  /* #15: @8 = sin(@7) */
  w8 = sin( w7 );
  /* #16: @7 = cos(@7) */
  w7 = cos( w7 );
  /* #17: @9 = 0 */
  w9 = 0.;
  /* #18: @12 = horzcat(@8, @7, @9) */
  rr=w12;
  *rr++ = w8;
  *rr++ = w7;
  *rr++ = w9;
  /* #19: @12 = @12' */
  /* #20: @13 = [[0, 0, 1]] */
  casadi_copy(casadi_c0, 3, w13);
  /* #21: @13 = @13' */
  /* #22: @14 = horzcat(@11, @12, @13) */
  rr=w14;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #23: @15 = @14' */
  for (i=0, rr=w15, cs=w14; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #24: @8 = @0[7] */
  for (rr=(&w8), ss=w0+7; ss!=w0+8; ss+=1) *rr++ = *ss;
  /* #25: @7 = cos(@8) */
  w7 = cos( w8 );
  /* #26: @9 = 0 */
  w9 = 0.;
  /* #27: @10 = sin(@8) */
  w10 = sin( w8 );
  /* #28: @11 = horzcat(@7, @9, @10) */
  rr=w11;
  *rr++ = w7;
  *rr++ = w9;
  *rr++ = w10;
  /* #29: @11 = @11' */
  /* #30: @12 = [[0, 1, 0]] */
  casadi_copy(casadi_c1, 3, w12);
  /* #31: @12 = @12' */
  /* #32: @7 = sin(@8) */
  w7 = sin( w8 );
  /* #33: @7 = (-@7) */
  w7 = (- w7 );
  /* #34: @9 = 0 */
  w9 = 0.;
  /* #35: @8 = cos(@8) */
  w8 = cos( w8 );
  /* #36: @13 = horzcat(@7, @9, @8) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w9;
  *rr++ = w8;
  /* #37: @13 = @13' */
  /* #38: @14 = horzcat(@11, @12, @13) */
  rr=w14;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #39: @16 = @14' */
  for (i=0, rr=w16, cs=w14; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #40: @6 = mac(@15,@16,@6) */
  for (i=0, rr=w6; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w15+j, tt=w16+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #41: @11 = [[1, 0, 0]] */
  casadi_copy(casadi_c2, 3, w11);
  /* #42: @11 = @11' */
  /* #43: @7 = 0 */
  w7 = 0.;
  /* #44: @9 = @0[6] */
  for (rr=(&w9), ss=w0+6; ss!=w0+7; ss+=1) *rr++ = *ss;
  /* #45: @8 = cos(@9) */
  w8 = cos( w9 );
  /* #46: @10 = sin(@9) */
  w10 = sin( w9 );
  /* #47: @10 = (-@10) */
  w10 = (- w10 );
  /* #48: @12 = horzcat(@7, @8, @10) */
  rr=w12;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w10;
  /* #49: @12 = @12' */
  /* #50: @7 = 0 */
  w7 = 0.;
  /* #51: @8 = sin(@9) */
  w8 = sin( w9 );
  /* #52: @9 = cos(@9) */
  w9 = cos( w9 );
  /* #53: @13 = horzcat(@7, @8, @9) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #54: @13 = @13' */
  /* #55: @15 = horzcat(@11, @12, @13) */
  rr=w15;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #56: @16 = @15' */
  for (i=0, rr=w16, cs=w15; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #57: @5 = mac(@6,@16,@5) */
  for (i=0, rr=w5; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w6+j, tt=w16+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #58: @17 = input[1][0] */
  casadi_copy(arg[1], 7, w17);
  /* #59: @11 = @17[3:6] */
  for (rr=w11, ss=w17+3; ss!=w17+6; ss+=1) *rr++ = *ss;
  /* #60: @4 = mac(@5,@11,@4) */
  for (i=0, rr=w4; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w5+j, tt=w11+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #61: @11 = [0, 0, 9.8] */
  casadi_copy(casadi_c3, 3, w11);
  /* #62: @4 = (@4-@11) */
  for (i=0, rr=w4, cs=w11; i<3; ++i) (*rr++) -= (*cs++);
  /* #63: @11 = zeros(3x1) */
  casadi_clear(w11, 3);
  /* #64: @7 = 1 */
  w7 = 1.;
  /* #65: @8 = @0[7] */
  for (rr=(&w8), ss=w0+7; ss!=w0+8; ss+=1) *rr++ = *ss;
  /* #66: @9 = sin(@8) */
  w9 = sin( w8 );
  /* #67: @10 = @0[6] */
  for (rr=(&w10), ss=w0+6; ss!=w0+7; ss+=1) *rr++ = *ss;
  /* #68: @18 = sin(@10) */
  w18 = sin( w10 );
  /* #69: @9 = (@9*@18) */
  w9 *= w18;
  /* #70: @18 = cos(@8) */
  w18 = cos( w8 );
  /* #71: @9 = (@9/@18) */
  w9 /= w18;
  /* #72: @18 = cos(@10) */
  w18 = cos( w10 );
  /* #73: @19 = sin(@8) */
  w19 = sin( w8 );
  /* #74: @18 = (@18*@19) */
  w18 *= w19;
  /* #75: @19 = cos(@8) */
  w19 = cos( w8 );
  /* #76: @18 = (@18/@19) */
  w18 /= w19;
  /* #77: @12 = horzcat(@7, @9, @18) */
  rr=w12;
  *rr++ = w7;
  *rr++ = w9;
  *rr++ = w18;
  /* #78: @12 = @12' */
  /* #79: @7 = 0 */
  w7 = 0.;
  /* #80: @9 = cos(@10) */
  w9 = cos( w10 );
  /* #81: @18 = sin(@10) */
  w18 = sin( w10 );
  /* #82: @18 = (-@18) */
  w18 = (- w18 );
  /* #83: @13 = horzcat(@7, @9, @18) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w9;
  *rr++ = w18;
  /* #84: @13 = @13' */
  /* #85: @7 = 0 */
  w7 = 0.;
  /* #86: @9 = sin(@10) */
  w9 = sin( w10 );
  /* #87: @18 = cos(@8) */
  w18 = cos( w8 );
  /* #88: @9 = (@9/@18) */
  w9 /= w18;
  /* #89: @10 = cos(@10) */
  w10 = cos( w10 );
  /* #90: @8 = cos(@8) */
  w8 = cos( w8 );
  /* #91: @10 = (@10/@8) */
  w10 /= w8;
  /* #92: @20 = horzcat(@7, @9, @10) */
  rr=w20;
  *rr++ = w7;
  *rr++ = w9;
  *rr++ = w10;
  /* #93: @20 = @20' */
  /* #94: @5 = horzcat(@12, @13, @20) */
  rr=w5;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<3; ++i) *rr++ = *cs++;
  /* #95: @6 = @5' */
  for (i=0, rr=w6, cs=w5; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #96: @12 = @17[:3] */
  for (rr=w12, ss=w17+0; ss!=w17+3; ss+=1) *rr++ = *ss;
  /* #97: @11 = mac(@6,@12,@11) */
  for (i=0, rr=w11; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w6+j, tt=w12+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #98: @21 = zeros(12x1) */
  casadi_clear(w21, 12);
  /* #99: @7 = 1 */
  w7 = 1.;
  /* #100: @22 = vertcat(@3, @4, @11, @21, @7) */
  rr=w22;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w4; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<12; ++i) *rr++ = *cs++;
  *rr++ = w7;
  /* #101: @23 = (@2*@22) */
  for (i=0, rr=w23, cs=w22; i<22; ++i) (*rr++)  = (w2*(*cs++));
  /* #102: @7 = 2 */
  w7 = 2.;
  /* #103: @23 = (@23/@7) */
  for (i=0, rr=w23; i<22; ++i) (*rr++) /= w7;
  /* #104: @23 = (@0+@23) */
  for (i=0, rr=w23, cr=w0, cs=w23; i<22; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #105: @3 = @23[3:6] */
  for (rr=w3, ss=w23+3; ss!=w23+6; ss+=1) *rr++ = *ss;
  /* #106: @4 = zeros(3x1) */
  casadi_clear(w4, 3);
  /* #107: @6 = zeros(3x3) */
  casadi_clear(w6, 9);
  /* #108: @5 = zeros(3x3) */
  casadi_clear(w5, 9);
  /* #109: @7 = @23[8] */
  for (rr=(&w7), ss=w23+8; ss!=w23+9; ss+=1) *rr++ = *ss;
  /* #110: @9 = cos(@7) */
  w9 = cos( w7 );
  /* #111: @10 = sin(@7) */
  w10 = sin( w7 );
  /* #112: @10 = (-@10) */
  w10 = (- w10 );
  /* #113: @8 = 0 */
  w8 = 0.;
  /* #114: @11 = horzcat(@9, @10, @8) */
  rr=w11;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w8;
  /* #115: @11 = @11' */
  /* #116: @9 = sin(@7) */
  w9 = sin( w7 );
  /* #117: @7 = cos(@7) */
  w7 = cos( w7 );
  /* #118: @10 = 0 */
  w10 = 0.;
  /* #119: @12 = horzcat(@9, @7, @10) */
  rr=w12;
  *rr++ = w9;
  *rr++ = w7;
  *rr++ = w10;
  /* #120: @12 = @12' */
  /* #121: @13 = [[0, 0, 1]] */
  casadi_copy(casadi_c0, 3, w13);
  /* #122: @13 = @13' */
  /* #123: @16 = horzcat(@11, @12, @13) */
  rr=w16;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #124: @15 = @16' */
  for (i=0, rr=w15, cs=w16; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #125: @9 = @23[7] */
  for (rr=(&w9), ss=w23+7; ss!=w23+8; ss+=1) *rr++ = *ss;
  /* #126: @7 = cos(@9) */
  w7 = cos( w9 );
  /* #127: @10 = 0 */
  w10 = 0.;
  /* #128: @8 = sin(@9) */
  w8 = sin( w9 );
  /* #129: @11 = horzcat(@7, @10, @8) */
  rr=w11;
  *rr++ = w7;
  *rr++ = w10;
  *rr++ = w8;
  /* #130: @11 = @11' */
  /* #131: @12 = [[0, 1, 0]] */
  casadi_copy(casadi_c1, 3, w12);
  /* #132: @12 = @12' */
  /* #133: @7 = sin(@9) */
  w7 = sin( w9 );
  /* #134: @7 = (-@7) */
  w7 = (- w7 );
  /* #135: @10 = 0 */
  w10 = 0.;
  /* #136: @9 = cos(@9) */
  w9 = cos( w9 );
  /* #137: @13 = horzcat(@7, @10, @9) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w10;
  *rr++ = w9;
  /* #138: @13 = @13' */
  /* #139: @16 = horzcat(@11, @12, @13) */
  rr=w16;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #140: @14 = @16' */
  for (i=0, rr=w14, cs=w16; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #141: @5 = mac(@15,@14,@5) */
  for (i=0, rr=w5; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w15+j, tt=w14+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #142: @11 = [[1, 0, 0]] */
  casadi_copy(casadi_c2, 3, w11);
  /* #143: @11 = @11' */
  /* #144: @7 = 0 */
  w7 = 0.;
  /* #145: @10 = @23[6] */
  for (rr=(&w10), ss=w23+6; ss!=w23+7; ss+=1) *rr++ = *ss;
  /* #146: @9 = cos(@10) */
  w9 = cos( w10 );
  /* #147: @8 = sin(@10) */
  w8 = sin( w10 );
  /* #148: @8 = (-@8) */
  w8 = (- w8 );
  /* #149: @12 = horzcat(@7, @9, @8) */
  rr=w12;
  *rr++ = w7;
  *rr++ = w9;
  *rr++ = w8;
  /* #150: @12 = @12' */
  /* #151: @7 = 0 */
  w7 = 0.;
  /* #152: @9 = sin(@10) */
  w9 = sin( w10 );
  /* #153: @10 = cos(@10) */
  w10 = cos( w10 );
  /* #154: @13 = horzcat(@7, @9, @10) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w9;
  *rr++ = w10;
  /* #155: @13 = @13' */
  /* #156: @15 = horzcat(@11, @12, @13) */
  rr=w15;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #157: @14 = @15' */
  for (i=0, rr=w14, cs=w15; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #158: @6 = mac(@5,@14,@6) */
  for (i=0, rr=w6; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w5+j, tt=w14+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #159: @24 = input[2][0] */
  casadi_copy(arg[2], 7, w24);
  /* #160: @25 = (@17+@24) */
  for (i=0, rr=w25, cr=w17, cs=w24; i<7; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #161: @7 = 2 */
  w7 = 2.;
  /* #162: @25 = (@25/@7) */
  for (i=0, rr=w25; i<7; ++i) (*rr++) /= w7;
  /* #163: @11 = @25[3:6] */
  for (rr=w11, ss=w25+3; ss!=w25+6; ss+=1) *rr++ = *ss;
  /* #164: @4 = mac(@6,@11,@4) */
  for (i=0, rr=w4; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w6+j, tt=w11+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #165: @11 = [0, 0, 9.8] */
  casadi_copy(casadi_c3, 3, w11);
  /* #166: @4 = (@4-@11) */
  for (i=0, rr=w4, cs=w11; i<3; ++i) (*rr++) -= (*cs++);
  /* #167: @11 = zeros(3x1) */
  casadi_clear(w11, 3);
  /* #168: @7 = 1 */
  w7 = 1.;
  /* #169: @9 = @23[7] */
  for (rr=(&w9), ss=w23+7; ss!=w23+8; ss+=1) *rr++ = *ss;
  /* #170: @10 = sin(@9) */
  w10 = sin( w9 );
  /* #171: @8 = @23[6] */
  for (rr=(&w8), ss=w23+6; ss!=w23+7; ss+=1) *rr++ = *ss;
  /* #172: @18 = sin(@8) */
  w18 = sin( w8 );
  /* #173: @10 = (@10*@18) */
  w10 *= w18;
  /* #174: @18 = cos(@9) */
  w18 = cos( w9 );
  /* #175: @10 = (@10/@18) */
  w10 /= w18;
  /* #176: @18 = cos(@8) */
  w18 = cos( w8 );
  /* #177: @19 = sin(@9) */
  w19 = sin( w9 );
  /* #178: @18 = (@18*@19) */
  w18 *= w19;
  /* #179: @19 = cos(@9) */
  w19 = cos( w9 );
  /* #180: @18 = (@18/@19) */
  w18 /= w19;
  /* #181: @12 = horzcat(@7, @10, @18) */
  rr=w12;
  *rr++ = w7;
  *rr++ = w10;
  *rr++ = w18;
  /* #182: @12 = @12' */
  /* #183: @7 = 0 */
  w7 = 0.;
  /* #184: @10 = cos(@8) */
  w10 = cos( w8 );
  /* #185: @18 = sin(@8) */
  w18 = sin( w8 );
  /* #186: @18 = (-@18) */
  w18 = (- w18 );
  /* #187: @13 = horzcat(@7, @10, @18) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w10;
  *rr++ = w18;
  /* #188: @13 = @13' */
  /* #189: @7 = 0 */
  w7 = 0.;
  /* #190: @10 = sin(@8) */
  w10 = sin( w8 );
  /* #191: @18 = cos(@9) */
  w18 = cos( w9 );
  /* #192: @10 = (@10/@18) */
  w10 /= w18;
  /* #193: @8 = cos(@8) */
  w8 = cos( w8 );
  /* #194: @9 = cos(@9) */
  w9 = cos( w9 );
  /* #195: @8 = (@8/@9) */
  w8 /= w9;
  /* #196: @20 = horzcat(@7, @10, @8) */
  rr=w20;
  *rr++ = w7;
  *rr++ = w10;
  *rr++ = w8;
  /* #197: @20 = @20' */
  /* #198: @6 = horzcat(@12, @13, @20) */
  rr=w6;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<3; ++i) *rr++ = *cs++;
  /* #199: @5 = @6' */
  for (i=0, rr=w5, cs=w6; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #200: @12 = @25[:3] */
  for (rr=w12, ss=w25+0; ss!=w25+3; ss+=1) *rr++ = *ss;
  /* #201: @11 = mac(@5,@12,@11) */
  for (i=0, rr=w11; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w5+j, tt=w12+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #202: @21 = zeros(12x1) */
  casadi_clear(w21, 12);
  /* #203: @7 = 1 */
  w7 = 1.;
  /* #204: @23 = vertcat(@3, @4, @11, @21, @7) */
  rr=w23;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w4; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<12; ++i) *rr++ = *cs++;
  *rr++ = w7;
  /* #205: @26 = (2.*@23) */
  for (i=0, rr=w26, cs=w23; i<22; ++i) *rr++ = (2.* *cs++ );
  /* #206: @22 = (@22+@26) */
  for (i=0, rr=w22, cs=w26; i<22; ++i) (*rr++) += (*cs++);
  /* #207: @23 = (@2*@23) */
  for (i=0, rr=w23, cs=w23; i<22; ++i) (*rr++)  = (w2*(*cs++));
  /* #208: @7 = 2 */
  w7 = 2.;
  /* #209: @23 = (@23/@7) */
  for (i=0, rr=w23; i<22; ++i) (*rr++) /= w7;
  /* #210: @23 = (@0+@23) */
  for (i=0, rr=w23, cr=w0, cs=w23; i<22; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #211: @3 = @23[3:6] */
  for (rr=w3, ss=w23+3; ss!=w23+6; ss+=1) *rr++ = *ss;
  /* #212: @4 = zeros(3x1) */
  casadi_clear(w4, 3);
  /* #213: @5 = zeros(3x3) */
  casadi_clear(w5, 9);
  /* #214: @6 = zeros(3x3) */
  casadi_clear(w6, 9);
  /* #215: @7 = @23[8] */
  for (rr=(&w7), ss=w23+8; ss!=w23+9; ss+=1) *rr++ = *ss;
  /* #216: @10 = cos(@7) */
  w10 = cos( w7 );
  /* #217: @8 = sin(@7) */
  w8 = sin( w7 );
  /* #218: @8 = (-@8) */
  w8 = (- w8 );
  /* #219: @9 = 0 */
  w9 = 0.;
  /* #220: @11 = horzcat(@10, @8, @9) */
  rr=w11;
  *rr++ = w10;
  *rr++ = w8;
  *rr++ = w9;
  /* #221: @11 = @11' */
  /* #222: @10 = sin(@7) */
  w10 = sin( w7 );
  /* #223: @7 = cos(@7) */
  w7 = cos( w7 );
  /* #224: @8 = 0 */
  w8 = 0.;
  /* #225: @12 = horzcat(@10, @7, @8) */
  rr=w12;
  *rr++ = w10;
  *rr++ = w7;
  *rr++ = w8;
  /* #226: @12 = @12' */
  /* #227: @13 = [[0, 0, 1]] */
  casadi_copy(casadi_c0, 3, w13);
  /* #228: @13 = @13' */
  /* #229: @14 = horzcat(@11, @12, @13) */
  rr=w14;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #230: @15 = @14' */
  for (i=0, rr=w15, cs=w14; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #231: @10 = @23[7] */
  for (rr=(&w10), ss=w23+7; ss!=w23+8; ss+=1) *rr++ = *ss;
  /* #232: @7 = cos(@10) */
  w7 = cos( w10 );
  /* #233: @8 = 0 */
  w8 = 0.;
  /* #234: @9 = sin(@10) */
  w9 = sin( w10 );
  /* #235: @11 = horzcat(@7, @8, @9) */
  rr=w11;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #236: @11 = @11' */
  /* #237: @12 = [[0, 1, 0]] */
  casadi_copy(casadi_c1, 3, w12);
  /* #238: @12 = @12' */
  /* #239: @7 = sin(@10) */
  w7 = sin( w10 );
  /* #240: @7 = (-@7) */
  w7 = (- w7 );
  /* #241: @8 = 0 */
  w8 = 0.;
  /* #242: @10 = cos(@10) */
  w10 = cos( w10 );
  /* #243: @13 = horzcat(@7, @8, @10) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w10;
  /* #244: @13 = @13' */
  /* #245: @14 = horzcat(@11, @12, @13) */
  rr=w14;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #246: @16 = @14' */
  for (i=0, rr=w16, cs=w14; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #247: @6 = mac(@15,@16,@6) */
  for (i=0, rr=w6; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w15+j, tt=w16+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #248: @11 = [[1, 0, 0]] */
  casadi_copy(casadi_c2, 3, w11);
  /* #249: @11 = @11' */
  /* #250: @7 = 0 */
  w7 = 0.;
  /* #251: @8 = @23[6] */
  for (rr=(&w8), ss=w23+6; ss!=w23+7; ss+=1) *rr++ = *ss;
  /* #252: @10 = cos(@8) */
  w10 = cos( w8 );
  /* #253: @9 = sin(@8) */
  w9 = sin( w8 );
  /* #254: @9 = (-@9) */
  w9 = (- w9 );
  /* #255: @12 = horzcat(@7, @10, @9) */
  rr=w12;
  *rr++ = w7;
  *rr++ = w10;
  *rr++ = w9;
  /* #256: @12 = @12' */
  /* #257: @7 = 0 */
  w7 = 0.;
  /* #258: @10 = sin(@8) */
  w10 = sin( w8 );
  /* #259: @8 = cos(@8) */
  w8 = cos( w8 );
  /* #260: @13 = horzcat(@7, @10, @8) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w10;
  *rr++ = w8;
  /* #261: @13 = @13' */
  /* #262: @15 = horzcat(@11, @12, @13) */
  rr=w15;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #263: @16 = @15' */
  for (i=0, rr=w16, cs=w15; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #264: @5 = mac(@6,@16,@5) */
  for (i=0, rr=w5; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w6+j, tt=w16+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #265: @17 = (@17+@24) */
  for (i=0, rr=w17, cs=w24; i<7; ++i) (*rr++) += (*cs++);
  /* #266: @7 = 2 */
  w7 = 2.;
  /* #267: @17 = (@17/@7) */
  for (i=0, rr=w17; i<7; ++i) (*rr++) /= w7;
  /* #268: @11 = @17[3:6] */
  for (rr=w11, ss=w17+3; ss!=w17+6; ss+=1) *rr++ = *ss;
  /* #269: @4 = mac(@5,@11,@4) */
  for (i=0, rr=w4; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w5+j, tt=w11+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #270: @11 = [0, 0, 9.8] */
  casadi_copy(casadi_c3, 3, w11);
  /* #271: @4 = (@4-@11) */
  for (i=0, rr=w4, cs=w11; i<3; ++i) (*rr++) -= (*cs++);
  /* #272: @11 = zeros(3x1) */
  casadi_clear(w11, 3);
  /* #273: @7 = 1 */
  w7 = 1.;
  /* #274: @10 = @23[7] */
  for (rr=(&w10), ss=w23+7; ss!=w23+8; ss+=1) *rr++ = *ss;
  /* #275: @8 = sin(@10) */
  w8 = sin( w10 );
  /* #276: @9 = @23[6] */
  for (rr=(&w9), ss=w23+6; ss!=w23+7; ss+=1) *rr++ = *ss;
  /* #277: @18 = sin(@9) */
  w18 = sin( w9 );
  /* #278: @8 = (@8*@18) */
  w8 *= w18;
  /* #279: @18 = cos(@10) */
  w18 = cos( w10 );
  /* #280: @8 = (@8/@18) */
  w8 /= w18;
  /* #281: @18 = cos(@9) */
  w18 = cos( w9 );
  /* #282: @19 = sin(@10) */
  w19 = sin( w10 );
  /* #283: @18 = (@18*@19) */
  w18 *= w19;
  /* #284: @19 = cos(@10) */
  w19 = cos( w10 );
  /* #285: @18 = (@18/@19) */
  w18 /= w19;
  /* #286: @12 = horzcat(@7, @8, @18) */
  rr=w12;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w18;
  /* #287: @12 = @12' */
  /* #288: @7 = 0 */
  w7 = 0.;
  /* #289: @8 = cos(@9) */
  w8 = cos( w9 );
  /* #290: @18 = sin(@9) */
  w18 = sin( w9 );
  /* #291: @18 = (-@18) */
  w18 = (- w18 );
  /* #292: @13 = horzcat(@7, @8, @18) */
  rr=w13;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w18;
  /* #293: @13 = @13' */
  /* #294: @7 = 0 */
  w7 = 0.;
  /* #295: @8 = sin(@9) */
  w8 = sin( w9 );
  /* #296: @18 = cos(@10) */
  w18 = cos( w10 );
  /* #297: @8 = (@8/@18) */
  w8 /= w18;
  /* #298: @9 = cos(@9) */
  w9 = cos( w9 );
  /* #299: @10 = cos(@10) */
  w10 = cos( w10 );
  /* #300: @9 = (@9/@10) */
  w9 /= w10;
  /* #301: @20 = horzcat(@7, @8, @9) */
  rr=w20;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #302: @20 = @20' */
  /* #303: @5 = horzcat(@12, @13, @20) */
  rr=w5;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<3; ++i) *rr++ = *cs++;
  /* #304: @6 = @5' */
  for (i=0, rr=w6, cs=w5; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #305: @12 = @17[:3] */
  for (rr=w12, ss=w17+0; ss!=w17+3; ss+=1) *rr++ = *ss;
  /* #306: @11 = mac(@6,@12,@11) */
  for (i=0, rr=w11; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w6+j, tt=w12+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #307: @21 = zeros(12x1) */
  casadi_clear(w21, 12);
  /* #308: @7 = 1 */
  w7 = 1.;
  /* #309: @23 = vertcat(@3, @4, @11, @21, @7) */
  rr=w23;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w4; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<12; ++i) *rr++ = *cs++;
  *rr++ = w7;
  /* #310: @26 = (2.*@23) */
  for (i=0, rr=w26, cs=w23; i<22; ++i) *rr++ = (2.* *cs++ );
  /* #311: @22 = (@22+@26) */
  for (i=0, rr=w22, cs=w26; i<22; ++i) (*rr++) += (*cs++);
  /* #312: @23 = (@2*@23) */
  for (i=0, rr=w23, cs=w23; i<22; ++i) (*rr++)  = (w2*(*cs++));
  /* #313: @23 = (@0+@23) */
  for (i=0, rr=w23, cr=w0, cs=w23; i<22; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #314: @3 = @23[3:6] */
  for (rr=w3, ss=w23+3; ss!=w23+6; ss+=1) *rr++ = *ss;
  /* #315: @4 = zeros(3x1) */
  casadi_clear(w4, 3);
  /* #316: @6 = zeros(3x3) */
  casadi_clear(w6, 9);
  /* #317: @5 = zeros(3x3) */
  casadi_clear(w5, 9);
  /* #318: @2 = @23[8] */
  for (rr=(&w2), ss=w23+8; ss!=w23+9; ss+=1) *rr++ = *ss;
  /* #319: @7 = cos(@2) */
  w7 = cos( w2 );
  /* #320: @8 = sin(@2) */
  w8 = sin( w2 );
  /* #321: @8 = (-@8) */
  w8 = (- w8 );
  /* #322: @9 = 0 */
  w9 = 0.;
  /* #323: @11 = horzcat(@7, @8, @9) */
  rr=w11;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #324: @11 = @11' */
  /* #325: @7 = sin(@2) */
  w7 = sin( w2 );
  /* #326: @2 = cos(@2) */
  w2 = cos( w2 );
  /* #327: @8 = 0 */
  w8 = 0.;
  /* #328: @12 = horzcat(@7, @2, @8) */
  rr=w12;
  *rr++ = w7;
  *rr++ = w2;
  *rr++ = w8;
  /* #329: @12 = @12' */
  /* #330: @13 = [[0, 0, 1]] */
  casadi_copy(casadi_c0, 3, w13);
  /* #331: @13 = @13' */
  /* #332: @16 = horzcat(@11, @12, @13) */
  rr=w16;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #333: @15 = @16' */
  for (i=0, rr=w15, cs=w16; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #334: @7 = @23[7] */
  for (rr=(&w7), ss=w23+7; ss!=w23+8; ss+=1) *rr++ = *ss;
  /* #335: @2 = cos(@7) */
  w2 = cos( w7 );
  /* #336: @8 = 0 */
  w8 = 0.;
  /* #337: @9 = sin(@7) */
  w9 = sin( w7 );
  /* #338: @11 = horzcat(@2, @8, @9) */
  rr=w11;
  *rr++ = w2;
  *rr++ = w8;
  *rr++ = w9;
  /* #339: @11 = @11' */
  /* #340: @12 = [[0, 1, 0]] */
  casadi_copy(casadi_c1, 3, w12);
  /* #341: @12 = @12' */
  /* #342: @2 = sin(@7) */
  w2 = sin( w7 );
  /* #343: @2 = (-@2) */
  w2 = (- w2 );
  /* #344: @8 = 0 */
  w8 = 0.;
  /* #345: @7 = cos(@7) */
  w7 = cos( w7 );
  /* #346: @13 = horzcat(@2, @8, @7) */
  rr=w13;
  *rr++ = w2;
  *rr++ = w8;
  *rr++ = w7;
  /* #347: @13 = @13' */
  /* #348: @16 = horzcat(@11, @12, @13) */
  rr=w16;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #349: @14 = @16' */
  for (i=0, rr=w14, cs=w16; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #350: @5 = mac(@15,@14,@5) */
  for (i=0, rr=w5; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w15+j, tt=w14+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #351: @11 = [[1, 0, 0]] */
  casadi_copy(casadi_c2, 3, w11);
  /* #352: @11 = @11' */
  /* #353: @2 = 0 */
  w2 = 0.;
  /* #354: @8 = @23[6] */
  for (rr=(&w8), ss=w23+6; ss!=w23+7; ss+=1) *rr++ = *ss;
  /* #355: @7 = cos(@8) */
  w7 = cos( w8 );
  /* #356: @9 = sin(@8) */
  w9 = sin( w8 );
  /* #357: @9 = (-@9) */
  w9 = (- w9 );
  /* #358: @12 = horzcat(@2, @7, @9) */
  rr=w12;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w9;
  /* #359: @12 = @12' */
  /* #360: @2 = 0 */
  w2 = 0.;
  /* #361: @7 = sin(@8) */
  w7 = sin( w8 );
  /* #362: @8 = cos(@8) */
  w8 = cos( w8 );
  /* #363: @13 = horzcat(@2, @7, @8) */
  rr=w13;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w8;
  /* #364: @13 = @13' */
  /* #365: @15 = horzcat(@11, @12, @13) */
  rr=w15;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  /* #366: @14 = @15' */
  for (i=0, rr=w14, cs=w15; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #367: @6 = mac(@5,@14,@6) */
  for (i=0, rr=w6; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w5+j, tt=w14+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #368: @11 = @24[3:6] */
  for (rr=w11, ss=w24+3; ss!=w24+6; ss+=1) *rr++ = *ss;
  /* #369: @4 = mac(@6,@11,@4) */
  for (i=0, rr=w4; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w6+j, tt=w11+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #370: @11 = [0, 0, 9.8] */
  casadi_copy(casadi_c3, 3, w11);
  /* #371: @4 = (@4-@11) */
  for (i=0, rr=w4, cs=w11; i<3; ++i) (*rr++) -= (*cs++);
  /* #372: @11 = zeros(3x1) */
  casadi_clear(w11, 3);
  /* #373: @2 = 1 */
  w2 = 1.;
  /* #374: @7 = @23[7] */
  for (rr=(&w7), ss=w23+7; ss!=w23+8; ss+=1) *rr++ = *ss;
  /* #375: @8 = sin(@7) */
  w8 = sin( w7 );
  /* #376: @9 = @23[6] */
  for (rr=(&w9), ss=w23+6; ss!=w23+7; ss+=1) *rr++ = *ss;
  /* #377: @10 = sin(@9) */
  w10 = sin( w9 );
  /* #378: @8 = (@8*@10) */
  w8 *= w10;
  /* #379: @10 = cos(@7) */
  w10 = cos( w7 );
  /* #380: @8 = (@8/@10) */
  w8 /= w10;
  /* #381: @10 = cos(@9) */
  w10 = cos( w9 );
  /* #382: @18 = sin(@7) */
  w18 = sin( w7 );
  /* #383: @10 = (@10*@18) */
  w10 *= w18;
  /* #384: @18 = cos(@7) */
  w18 = cos( w7 );
  /* #385: @10 = (@10/@18) */
  w10 /= w18;
  /* #386: @12 = horzcat(@2, @8, @10) */
  rr=w12;
  *rr++ = w2;
  *rr++ = w8;
  *rr++ = w10;
  /* #387: @12 = @12' */
  /* #388: @2 = 0 */
  w2 = 0.;
  /* #389: @8 = cos(@9) */
  w8 = cos( w9 );
  /* #390: @10 = sin(@9) */
  w10 = sin( w9 );
  /* #391: @10 = (-@10) */
  w10 = (- w10 );
  /* #392: @13 = horzcat(@2, @8, @10) */
  rr=w13;
  *rr++ = w2;
  *rr++ = w8;
  *rr++ = w10;
  /* #393: @13 = @13' */
  /* #394: @2 = 0 */
  w2 = 0.;
  /* #395: @8 = sin(@9) */
  w8 = sin( w9 );
  /* #396: @10 = cos(@7) */
  w10 = cos( w7 );
  /* #397: @8 = (@8/@10) */
  w8 /= w10;
  /* #398: @9 = cos(@9) */
  w9 = cos( w9 );
  /* #399: @7 = cos(@7) */
  w7 = cos( w7 );
  /* #400: @9 = (@9/@7) */
  w9 /= w7;
  /* #401: @20 = horzcat(@2, @8, @9) */
  rr=w20;
  *rr++ = w2;
  *rr++ = w8;
  *rr++ = w9;
  /* #402: @20 = @20' */
  /* #403: @6 = horzcat(@12, @13, @20) */
  rr=w6;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<3; ++i) *rr++ = *cs++;
  /* #404: @5 = @6' */
  for (i=0, rr=w5, cs=w6; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #405: @12 = @24[:3] */
  for (rr=w12, ss=w24+0; ss!=w24+3; ss+=1) *rr++ = *ss;
  /* #406: @11 = mac(@5,@12,@11) */
  for (i=0, rr=w11; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w5+j, tt=w12+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #407: @21 = zeros(12x1) */
  casadi_clear(w21, 12);
  /* #408: @2 = 1 */
  w2 = 1.;
  /* #409: @23 = vertcat(@3, @4, @11, @21, @2) */
  rr=w23;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w4; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<12; ++i) *rr++ = *cs++;
  *rr++ = w2;
  /* #410: @22 = (@22+@23) */
  for (i=0, rr=w22, cs=w23; i<22; ++i) (*rr++) += (*cs++);
  /* #411: @22 = (@1*@22) */
  for (i=0, rr=w22, cs=w22; i<22; ++i) (*rr++)  = (w1*(*cs++));
  /* #412: @0 = (@0+@22) */
  for (i=0, rr=w0, cs=w22; i<22; ++i) (*rr++) += (*cs++);
  /* #413: output[0][0] = @0 */
  casadi_copy(w0, 22, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int process(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int process_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int process_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void process_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int process_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void process_release(int mem) {
}

CASADI_SYMBOL_EXPORT void process_incref(void) {
}

CASADI_SYMBOL_EXPORT void process_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int process_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int process_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real process_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* process_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* process_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* process_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* process_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int process_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 195;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
