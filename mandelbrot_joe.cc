/**
 *  \file mandelbrot_joe.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include "render.hh"

#define WIDTH 1000
#define HEIGHT 1000

int
mandelbrot(double x, double y) {
	int maxit = 511;
	double cx = x;
	double cy = y;
	double newx, newy;

	int it = 0;
	for (it = 0; it < maxit && (x*x + y*y) < 4; ++it) {
		newx = x*x - y*y + cx;
		newy = 2*x*y + cy;
		x = newx;
		y = newy;
	}
	return it;
}

int
main (int argc, char* argv[])
{
	double minX = -2.1;
	double maxX = 0.7;
	double minY = -1.25;
	double maxY = 1.25;
	double t_start, t_elapsed;

	int height, width;
	if (argc == 3) {
		height = atoi (argv[1]);
		width = atoi (argv[2]);
		assert (height > 0 && width > 0);
	} else {
		fprintf (stderr, "usage: %s <height> <width>\n", argv[0]);
		fprintf (stderr, "where <height> and <width> are the dimensions of the image.\n");
		return -1;
	}
	double it = (maxY - minY)/height;
	double jt = (maxX - minX)/width;
	double x, y;

	int rank, np;
	MPI_Status status;

	//MPI Start
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* Get process id */
	MPI_Comm_size (MPI_COMM_WORLD, &np);	/*Get number of processes*/

	if(rank == 0){
		t_start = MPI_Wtime();
		printf("Number of MPI processes: %d\r\n", np);
	}
	int N = height / np;    //separate height into np chunks
	int blockSize = N * width;  //the size of each node needs to process
	int sendBuffer[blockSize];

	y = minY + rank*N*it;   //locate the first y of each core
	for (int i = 0; i < N; ++i) {
		x = minX;
		for (int j = 0; j < width; ++j) {
			sendBuffer[(i*width) + j] = mandelbrot(x,y);
			x += jt;
		}
		y += it;
	}

	//create leftOverBuffer and calculate the left over values that is not calculated. The job is done in master thread
	int *leftOverBuffer = NULL;
	int leftOverSize;
	if (rank == 0) {
		leftOverSize = height - (N*np);
		leftOverBuffer = (int *)malloc(sizeof(int)*leftOverSize*width);

		y = minY + N*np*it;
		for (int i = 0; i < leftOverSize; ++i) {
			x = minX;
			for (int j = 0; j < width; ++j) {
				leftOverBuffer[(i*width) + j] = mandelbrot(x,y);
				x += jt;
			}
			y += it;
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);

	int *receiveBuffer = NULL;
	//Master process create the receiveBuffer. the size of the buffer is enough to store every value calculated by every process,which is np * blockSize.
	if (rank == 0) 
	{
		receiveBuffer = (int *)malloc(sizeof(int)*np*blockSize);
	}
	//Gathering datas from all processes to master process
	MPI_Gather(sendBuffer, blockSize, MPI_INT, receiveBuffer, blockSize, MPI_INT, 0, MPI_COMM_WORLD);
	
	if (rank == 0) {

		//Generate Image
		gil::rgb8_image_t img(height, width);
		auto img_view = gil::view(img);

		//render the image calculated by each processr
		for (int k = 0; k < (N*np); ++k) {
			for (int p = 0; p < width; ++p) {
				img_view(p, k) = render(receiveBuffer[ (k * width) + p] / 512.0);
			}
		}
		//render the leftover image
		for (int k = 0; k < leftOverSize; ++k) {
			for (int p = 0; p < width; ++p) {
				img_view(p, k + (N*np) ) = render(leftOverBuffer[ (k * width) + p] / 512.0);
			}
		}

		t_elapsed = MPI_Wtime () - t_start; // compute the overall time taken
		printf("Total time: %f\r\n", t_elapsed);
		gil::png_write_view("mandelbrot-joe.png", const_view(img));

	}
	
	MPI_Finalize();

	return 0;


}

/* eof */
