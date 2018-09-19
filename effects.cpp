#include "cl-helper.h"
#include <cv.h>
#include <highgui.h>

void applyKernel(cl_command_queue queue, cl_kernel knl,
                 cl_mem d_img, int width, int height,
                 float *filter, cl_mem d_filter, int filterSize,
                 cl_mem d_out, float *out, CvMat **cvOut)
{
   CALL_CL_GUARDED(clEnqueueWriteBuffer, (
                      queue, d_filter, /*blocking*/ CL_TRUE, /*offset*/ 0,
                      filterSize * filterSize * sizeof(float), filter,
                      0, NULL, NULL));

   SET_6_KERNEL_ARGS(knl, d_img, width, height, d_filter, filterSize, d_out);
   size_t ldim[] = {1, 1};
   size_t gdim[] = {width, height};
   CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                   (queue, knl,
                    /*dimensions*/ 2, NULL, gdim, ldim,
                    0, NULL, NULL));

   // --------------------------------------------------------------------------
   // transfer back & check
   // --------------------------------------------------------------------------
   CALL_CL_GUARDED(clEnqueueReadBuffer, (
                      queue, d_out, /*blocking*/ CL_TRUE, /*offset*/ 0,
                      width * height * sizeof(float), out,
                      0, NULL, NULL));

   *cvOut = cvCreateMat(height, width, CV_32FC1);

   for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
         cvSetReal2D(*cvOut, y, x, out[y*width + x]);
}

int main(int argc, char **argv)
{
   cl_context ctx;
   cl_command_queue queue;
   /* create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0); */
   create_context_on("pocl", NULL, 0, &ctx, &queue, 0);

   print_device_info_from_queue(queue);

   // --------------------------------------------------------------------------
   // load kernels
   // --------------------------------------------------------------------------
   cl_kernel knl = kernel_from_string(ctx, read_file("../kernels.cl"), "filter", NULL);

   // --------------------------------------------------------------------------
   // allocate and initialize memory
   // --------------------------------------------------------------------------
   cl_int status;
   IplImage *carImg = cvLoadImage("../cargrill.png", 1);
   int height = carImg->height, width = carImg->width;

   CvMat *graycar8 = cvCreateMat(height, width, CV_8UC1);
   CvMat *graycar = cvCreateMat(height, width, CV_32FC1);
   cvCvtColor(carImg, graycar8, CV_BGR2GRAY);
   cvConvertScale(graycar8, graycar, 1.0/255, 0);

   cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);
   cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);

   float *img = (float *) malloc(sizeof(float) * width * height);
   cl_mem d_img = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 sizeof(float) * width * height, 0, &status);
   CHECK_CL_ERROR(status, "clCreateBuffer: d_img");

   float v = 1.0/9;
   int filterSize = 3;
   cl_mem d_filter3x3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       sizeof(float) * filterSize * filterSize, 0, &status);
   CHECK_CL_ERROR(status, "clCreateBuffer: d_filter");
   cl_mem d_filter4x4 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       sizeof(float) * 4 * 4, 0, &status);
   CHECK_CL_ERROR(status, "clCreateBuffer: d_filter");
   cl_mem d_filter5x5 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       sizeof(float) * 5 * 5, 0, &status);
   CHECK_CL_ERROR(status, "clCreateBuffer: d_filter");

   float *imgOut = (float *) malloc(sizeof(float) * width * height);
   cl_mem d_imgOut = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                    sizeof(float) * width * height,
                                    0, &status);
   CHECK_CL_ERROR(status, "clCreateBuffer: d_imgOut");

   for (size_t i = 0; i < width * height; i++)
   {
      // Vertical stripes
      /* img[i] = i/10 % 2; */
      img[i] = cvGetReal2D(graycar, i/width, i%width);
   }


   // --------------------------------------------------------------------------
   // transfer to device
   // --------------------------------------------------------------------------
   CALL_CL_GUARDED(clEnqueueWriteBuffer, (
                      queue, d_img, /*blocking*/ CL_TRUE, /*offset*/ 0,
                      width * height * sizeof(float), img,
                      0, NULL, NULL));

   // --------------------------------------------------------------------------
   // run code on device
   // --------------------------------------------------------------------------

   int nMethods = 0;

   CvMat *blur3x3Box;
   nMethods++;
   {
      float filter[] = {v, v, v,
                        v, v, v,
                        v, v, v};
      applyKernel(queue, knl, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &blur3x3Box);
   }
   // This is the same as bilinear
   CvMat * blur3x3Gaussian;
   nMethods++;
   {
      float filter[] = {1.0/16, 1.0/8, 1.0/16,
                        1.0/8,  1.0/4, 1.0/8,
                        1.0/16, 1.0/8, 1.0/16};
      applyKernel(queue, knl, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &blur3x3Gaussian);
   }
   CvMat * blur5x5Gaussian;
   nMethods++;
   {
      float filter[] = {1,  4,  6,  4, 1,
                        4, 16, 24, 16, 4,
                        6, 24, 36, 24, 6,
                        4, 16, 24, 16, 4,
                        1,  4,  6,  4, 1};
      for (int i = 0; i < 5*5; i++)
         filter[i] /= 256.0;
      applyKernel(queue, knl, d_img, width, height, filter, d_filter5x5, 5, d_imgOut, imgOut, &blur5x5Gaussian);
   }
   CvMat *sharpen3x3;
   nMethods++;
   {
      float filter[] = {0, -1, 0,
                        -1, 5, -1,
                        0, -1, 0};
      applyKernel(queue, knl, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &sharpen3x3);
   }
   CvMat *edge3x3;
   nMethods++;
   {
      float filter[] = {0,  1, 0,
                        1, -4, 1,
                        0,  1, 0};
      applyKernel(queue, knl, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &edge3x3);
   }
   CvMat *cornerEdge3x3;
   nMethods++;
   {
      float filter[] = {-1, -1, -1,
                        -1,  8, -1,
                        -1, -1, -1};
      applyKernel(queue, knl, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &cornerEdge3x3);
   }
   CvMat *cornerEdgePlusOrig3x3;
   nMethods++;
   {
      float filter[] = {-1, -1, -1,
                        -1,  9, -1,
                        -1, -1, -1};
      applyKernel(queue, knl, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &cornerEdgePlusOrig3x3);
   }
   CvMat *unsharpMasking5x5;
   nMethods++;
   {
      float filter[] = {1,  4,    6,  4, 1,
                        4, 16,   24, 16, 4,
                        6, 24, -476, 24, 6,
                        4, 16,   24, 16, 4,
                        1,  4,    6,  4, 1};
      for (int i = 0; i < 5*5; i++)
         filter[i] /= -256.0;

      applyKernel(queue, knl, d_img, width, height, filter, d_filter5x5, 5, d_imgOut, imgOut, &unsharpMasking5x5);
   }

   /* Display the original for comparison */

   int scale = 4;
   {
      CvMat *zoom = cvCreateMat(height*scale, width*scale, CV_32FC1);
      for (size_t i = 0; i < width * height * scale * scale; i++)
      {
         int row = i/(width*scale);
         int col = i%(width*scale);
         int orow = row/scale;
         int ocol = col/scale;
         cvSetReal2D(zoom, row, col, cvGetReal2D(graycar, orow, ocol));
      }

      cvShowImage("Original", zoom);
   }

   CvMat *methods[] = {blur3x3Box,
                       blur3x3Gaussian,
                       blur5x5Gaussian,
                       sharpen3x3,
                       unsharpMasking5x5,
                       cornerEdgePlusOrig3x3,
                       edge3x3,
                       cornerEdge3x3};

   for (int i = 0; ; i = (i+nMethods) % nMethods)
   {
      CvMat *method = methods[i];
      CvMat *zoom = cvCreateMat(height*scale, width*scale, CV_32FC1);
      for (size_t i = 0; i < width * height * scale * scale; i++)
      {
         int row = i/(width*scale);
         int col = i%(width*scale);
         int orow = row/scale;
         int ocol = col/scale;
         cvSetReal2D(zoom, row, col, cvGetReal2D(method, orow, ocol));
      }

      cvShowImage("Output", zoom);

      int key = cvWaitKey(0);
      printf("%i\n", key);
      if (1048689 == key)
         break;
      else if (1113939 == key)
         i++;
      else if (1113937 == key)
         i--;
   }

   /* cvReleaseImage(&carImg); */


   // --------------------------------------------------------------------------
   // clean up
   // --------------------------------------------------------------------------
   CALL_CL_GUARDED(clReleaseMemObject, (d_img));
   CALL_CL_GUARDED(clReleaseMemObject, (d_filter3x3));
   CALL_CL_GUARDED(clReleaseMemObject, (d_filter4x4));
   CALL_CL_GUARDED(clReleaseMemObject, (d_filter5x5));
   CALL_CL_GUARDED(clReleaseMemObject, (d_imgOut));
   CALL_CL_GUARDED(clReleaseKernel, (knl));
   CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
   CALL_CL_GUARDED(clReleaseContext, (ctx));

   return 0;
}
