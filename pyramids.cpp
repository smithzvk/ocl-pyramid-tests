#include "cl-helper.h"
#include <cv.h>
#include <highgui.h>

void enlarge(cl_command_queue queue, cl_kernel knl,
             int scaleFactor, cl_mem d_img, int width, int height,
             float *filter, cl_mem d_filter, int filterSize,
             cl_mem d_out, float *out, CvMat **cvOut)
{
   CALL_CL_GUARDED(clEnqueueWriteBuffer, (
                      queue, d_filter, /*blocking*/ CL_TRUE, /*offset*/ 0,
                      filterSize * filterSize * sizeof(float), filter,
                      0, NULL, NULL));

   SET_7_KERNEL_ARGS(knl, scaleFactor, d_img, width, height, d_filter, filterSize, d_out);
   size_t ldim[] = {1, 1};
   size_t gdim[] = {width * scaleFactor, height * scaleFactor};
   CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                   (queue, knl,
                    /*dimensions*/ 2, NULL, gdim, ldim,
                    0, NULL, NULL));

   // --------------------------------------------------------------------------
   // transfer back & check
   // --------------------------------------------------------------------------
   CALL_CL_GUARDED(clEnqueueReadBuffer, (
                      queue, d_out, /*blocking*/ CL_TRUE, /*offset*/ 0,
                      width * scaleFactor * height * scaleFactor * sizeof(float), out,
                      0, NULL, NULL));

   *cvOut = cvCreateMat(height * scaleFactor, width * scaleFactor, CV_32FC1);

   for (int y = 0; y < height*scaleFactor; y++)
      for (int x = 0; x < width*scaleFactor; x++)
         cvSetReal2D(*cvOut, y, x, out[y*width*scaleFactor + x]);
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
   cl_kernel knl = kernel_from_string(ctx, read_file("../kernels.cl"), "upscale", NULL);

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
   cvShowImage("Output", graycar);
   cvWaitKey(0);

   int scaleFactor = 2;

   CvMat *enlarged = cvCreateMat(height * scaleFactor, width * scaleFactor, CV_32FC1);
   cvPyrUp(graycar, enlarged, CV_GAUSSIAN_5x5);


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

   float *imgOut = (float *) malloc(sizeof(float) * width * scaleFactor * height * scaleFactor);
   cl_mem d_imgOut = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                    sizeof(float) * width * scaleFactor * height * scaleFactor,
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

   CvMat *enlarged3x3Box;

   {
      float filter[] = {v, v, v,
                        v, v, v,
                        v, v, v};
      enlarge(queue, knl, scaleFactor, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &enlarged3x3Box);
   }
   // This is the same as bilinear
   CvMat * enlarged3x3Gaussian;
   {
      float filter[] = {1.0/16, 1.0/8, 1.0/16,
                        1.0/8,  1.0/4, 1.0/8,
                        1.0/16, 1.0/8, 1.0/16};
      enlarge(queue, knl, scaleFactor, d_img, width, height, filter, d_filter3x3, 3, d_imgOut, imgOut, &enlarged3x3Gaussian);
   }
   CvMat * enlarged5x5Gaussian;
   {
      float filter[] = {1,  4,  6,  4, 1,
                        4, 16, 24, 16, 4,
                        6, 24, 36, 24, 6,
                        4, 16, 24, 16, 4,
                        1,  4,  6,  4, 1};
      for (int i = 0; i < 5*5; i++)
         filter[i] /= 256.0;
      enlarge(queue, knl, scaleFactor, d_img, width, height, filter, d_filter5x5, 5, d_imgOut, imgOut, &enlarged5x5Gaussian);
   }
   /* CvMat * enlargedLinear; */
   /* { */
   /*    float filter[] = {1, 2, 1, */
   /*                      2, 4, 2, */
   /*                      1, 2, 1}; */
   /*    for (int i = 0; i < 3*3; i++) */
   /*       filter[i] /= 16; */

   /*    enlarge(queue, knl, scaleFactor, d_img, width, height, filter, d_filter4x4, 4, d_imgOut, imgOut, &enlargedMagic); */
   /* } */
   /* CvMat * enlargedMagic; */
   /* { */
   /*    float filter1d[] = {1.0/8, 3.0/8, 3.0/8, 1.0/8}; */
   /*    float filter[16]; */
   /*    for (int i = 0; i < 4; i++) */
   /*       for (int j = 0; j < 4; j++) */
   /*          filter[i*2 + j] = filter1d[i] * filter1d[j]; */

   /*    enlarge(queue, knl, scaleFactor, d_img, width, height, filter, d_filter4x4, 4, d_imgOut, imgOut, &enlargedMagic); */
   /* } */

   int nMethods = 4;
   CvMat *methods[] = {enlarged, enlarged3x3Box, enlarged3x3Gaussian, enlarged5x5Gaussian};
   for (int i = 0; ; i = (i+1) % nMethods)
   {
      CvMat *method = methods[i];

      cvShowImage("Output", method);

      int key = cvWaitKey(0);
      printf("%i\n", key);
      if (1048689 == key)
         break;
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
