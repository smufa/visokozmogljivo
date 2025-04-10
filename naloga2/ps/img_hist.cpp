#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "FreeImage.h"
#include <omp.h>

#define BINS 256
#define ARGS_LEN (1)

struct histogram
{
	unsigned int *R;
	unsigned int *G;
	unsigned int *B;
};

void drawColumn(unsigned char *image, int width, int height, int i, int value, int max, int offset);
void drawHistogram(histogram H, int argWwidth, int argHeight);
void histogramCPU(unsigned char *imageIn, histogram H, int width, int height);
void printHistogram(histogram H);

int main(int argc, char **argv)
{
	if (argc != ARGS_LEN + 1)
	{
		printf("Invalid arguments ~ Requires image path! \n");
		return -1;
	}

	printf("-------------- Starting CPU compute --------------\n");

	//Load image from file
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, argv[1], 0);
	//Convert it to a 32-bit image
	FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

	//Get image dimensions
	int width = FreeImage_GetWidth(imageBitmap32);
	int height = FreeImage_GetHeight(imageBitmap32);
	int pitch = FreeImage_GetPitch(imageBitmap32);
	//Preapare room for a raw data copy of the image
	unsigned char *imageIn = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));

	//Initalize the histogram
	histogram H;
	H.B = (unsigned int *)calloc(BINS, sizeof(unsigned int));
	H.G = (unsigned int *)calloc(BINS, sizeof(unsigned int));
	H.R = (unsigned int *)calloc(BINS, sizeof(unsigned int));

	//Extract raw data from the image
	FreeImage_ConvertToRawBits(imageIn, imageBitmap, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

	//Free source image data
	FreeImage_Unload(imageBitmap32);
	FreeImage_Unload(imageBitmap);

	double start_t = omp_get_wtime();

	printf("> CPU Start, file '%s'\n", argv[1]);

	//Compute and print the histogram
	histogramCPU(imageIn, H, width, height);

	printf("> CPU compute took:   %lfs \n", omp_get_wtime() - start_t);

	drawHistogram(H, width, height);
	// printHistogram(H);

	return 0;
}

void histogramCPU(unsigned char *imageIn, histogram H, int width, int height)
{

	//Each color channel is 1 byte long, there are 4 channels BLUE, GREEN, RED and ALPHA
	//The order is BLUE|GREEN|RED|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
	for (int i = 0; i < (height); i++)
		for (int j = 0; j < (width); j++)
		{
			H.B[imageIn[(i * width + j) * 4]]++;
			H.G[imageIn[(i * width + j) * 4 + 1]]++;
			H.R[imageIn[(i * width + j) * 4 + 2]]++;
		}
}

void printHistogram(histogram H)
{
	printf("Colour\tNo. Pixels\n");
	for (int i = 0; i < BINS; i++)
	{
		if (H.B[i] > 0)
			printf("%dB\t%d\n", i, H.B[i]);
		if (H.G[i] > 0)
			printf("%dG\t%d\n", i, H.G[i]);
		if (H.R[i] > 0)
			printf("%dR\t%d\n", i, H.R[i]);
	}
}

void drawColumn(unsigned char *image, int width, int height, int i, int value, int max, int offset)
{

	int calc_height = height - (int)(((float)value / (float)max) * 255) - 1;

	for (int j = calc_height; j < height; j++)
	{
		int pos = j * width * 4 + i * 4;

		for (int k = 0; k < 3; k++)
		{
			if (k == offset)
				continue;

			if (j == calc_height)
				image[pos + k] = 0;

			else
				image[pos + k] = image[pos + k] / 2;
		}
	}
}

void drawHistogram(histogram H, int argWidth, int argHeight)
{
	const int width = BINS;
	const int height = BINS;
	int pitch = ((32 * width + 31) / 32) * 4;

	unsigned char *image =
		(unsigned char *)malloc(height * width * sizeof(unsigned char) * 4);

	// set everything to white
	for (int i = 0; i < height * width * 4; i++)
		image[i] = 255;

	int max = 255;

	for (int i = 0; i < BINS; i++)
	{
		if (H.B[i] > max)
			max = H.B[i];
		if (H.G[i] > max)
			max = H.G[i];
		if (H.R[i] > max)
			max = H.R[i];
	}

	for (int i = 0; i < BINS; i++)
	{
		if (H.B[i])
			drawColumn(image, width, height, i, H.B[i], max, 0);

		if (H.R[i])
			drawColumn(image, width, height, i, H.R[i], max, 2);

		if (H.G[i])
			drawColumn(image, width, height, i, H.G[i], max, 1);
	}

	FIBITMAP *dst = FreeImage_ConvertFromRawBits(
		image, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK,
		FI_RGBA_BLUE_MASK, TRUE);

	char f_out_name[256];

	printf("> Saving the Histogram as PNG in 'out/%d-%d-hist.png'\n", argWidth, argHeight);

	snprintf(f_out_name, sizeof(f_out_name), "out/%d-%d-hist.png", argWidth, argHeight);

	FreeImage_Save(FIF_PNG, dst, f_out_name, 0);
}