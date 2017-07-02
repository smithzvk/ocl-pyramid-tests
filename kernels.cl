#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel
void
upscale(int scaleFactor,
        __global const float *srcImg,
        int imgWidth,
        int imgHeight,
        __global const float *filter,
        int filterSize,
        __global float *dstImg)
{
   int xout = get_global_id(0);
   int yout = get_global_id(1);

   float sum = 0.0f;
   for (int i = 0, yoff = -filterSize/2; i < filterSize; i++, yoff++)
   {
      int ysrc = max((int) 0, min(imgHeight, yout/scaleFactor + yoff));
      int filterRow = filterSize * i;
      int imgRow = imgWidth * ysrc;
      for (int j = 0, xoff = -filterSize/2; j < filterSize; j++, xoff++)
      {
         int xsrc = max((int) 0, min(imgWidth, xout/scaleFactor + xoff));

         sum += filter[filterRow + j] * srcImg[imgRow + xsrc];
      }
   }

   dstImg[yout * scaleFactor * imgWidth + xout] = sum;
   /* dstImg[yout * scaleFactor * imgWidth + xout] = dstImg[yout/scaleFactor * imgWidth + xout/scaleFactor]; */
}

/* __kernel void sum( */
/*     __global const float *a, */
/*     __global const float *b, */
/*     __global float *c, */
/*     long n) */
/* { */
/*   int gid = get_global_id(0); */
/*   if (gid < n) */
/*     c[gid] = a[gid] + b[gid]; */
/* } */
