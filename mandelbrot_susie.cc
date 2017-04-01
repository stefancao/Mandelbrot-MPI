/**
 *  \file mandelbrot_susie.cc
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



	//MPI Start
	MPI_Init (&argc, &argv);
	int rank, np;

	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* Get process id */
	MPI_Comm_size (MPI_COMM_WORLD, &np);	/*Get number of processes*/
	MPI_Barrier (MPI_COMM_WORLD);
	if(rank == 0){
		printf("Number of process: %d\r\n", np);	
		t_start = MPI_Wtime();
	}

	int maxDataPerRow = height / np + 1;    //data of each row
	int blockSize = maxDataPerRow * width;  //the size of block calculated by each process
	int sendBuffer[blockSize];



	y = minY + (it * rank);

	//calculate how many rows a processor needs to go through
	int maxRowsIterations = height / np;
	//the left over rows need to be distribute into processes
	int extraRun = height % np;
	if(rank < extraRun && extraRun != 0){
		maxRowsIterations += 1;
	}
	for (int i = 0; i < maxRowsIterations; ++i) {
		x = minX;
		for (int j = 0; j < width; ++j) {
			sendBuffer[(i*width) + j] = mandelbrot(x,y);
			x += jt;
		}
		y += (it*np);
	}

	int *receiveBuffer = NULL;
	//Master Process creates the receive buffer 
	if (rank == 0) {
		
		receiveBuffer = (int *)malloc(blockSize * np * sizeof(int));
	}
	MPI_Barrier (MPI_COMM_WORLD);
	//Gather all datas from every processes
	MPI_Gather(sendBuffer, blockSize, MPI_INT, receiveBuffer,blockSize, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		//Generate Image
		gil::rgb8_image_t img(height, width);
		auto img_view = gil::view(img);

		int processStart = 0; //index of processors' first data
		int rowGo =0; //index of the row of the processors' data

		for(int i = 0; i < height; ++i){
			processStart = ( i % np) * blockSize; /*find the starting point of the processor*/
			for(int j = 0; j < width; ++j){
				img_view(j, i) = render(receiveBuffer[processStart + (rowGo * width) + j]/512.0); //loop through all the rows that's in the same processor
			}
			rowGo = i / np; 
		}
		//Time Stop and calculate total time
		t_elapsed = MPI_Wtime() - t_start;
		printf("time requires to calculate the data is %f", t_elapsed);
		gil::png_write_view("mandelbrot-susie.png", const_view(img));
	}

	
	MPI_Finalize();
	return 0;
}

/* eof */
