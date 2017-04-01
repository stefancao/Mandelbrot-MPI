/**
 *  \file mandelbrot_ms.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include "render.hh"

#define TAG 0

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

	// MPI Start
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* Get process id */
	MPI_Comm_size (MPI_COMM_WORLD, &np);	/*Get number of processes*/

	// if the number of processors are less than 2, run sequentially
	if (np < 2) {
		t_start = MPI_Wtime();
		
		printf("Number of MPI processes: %d\r\n", np);
		printf("Note: Running sequentially...\n");

		gil::rgb8_image_t img(height, width);
		auto img_view = gil::view(img);

		y = minY;
		for (int i = 0; i < height; ++i) {
			x = minX;
			for (int j = 0; j < width; ++j) {
				img_view(j, i) = render(mandelbrot(x, y)/512.0);
				x += jt;
			}
			y += it;
		}

		t_elapsed = MPI_Wtime () - t_start; // compute the overall time taken
	    printf("Total time: %f\r\n", t_elapsed);
		gil::png_write_view("mandelbrot-ms.png", const_view(img));
	}

	// else execute in master-slave manner
	else {

		// master
		if(rank == 0){

			t_start = MPI_Wtime();
			
			printf("Number of MPI processes: %d\r\n", np);

			// master thread sends a termination flag to the slaves indicating termination
			int terminationFlag = -1;	

			// master thread receives the calculated row from slaves and populates the data to the final array
			int finalArray[width*height];

			// buffer received from slaves
			// adding an extra element to store value of the current row
			int receiveBuff[width + 1];

			// init to 1 because we are excluding master thread
			int currProcessor = 1;

			int currRow = 0;

			// looping through all the rows
			while (currRow < height) {

				// init the send to the slaves
				if (currProcessor < np) {
					MPI_Send(&currRow, 1, MPI_INT, currProcessor, TAG, MPI_COMM_WORLD);
					currProcessor++;	
				}

				// receive data from slaves and send new row to be calculated to slaves
				else {
					MPI_Recv(receiveBuff, width+1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);
					
					// send back to calculate a new row
					MPI_Send(&currRow, 1, MPI_INT, status.MPI_SOURCE, TAG, MPI_COMM_WORLD);
					
					// receiveBuff[width] stores the row number of the receive buff
					memcpy(finalArray + receiveBuff[width]*width, receiveBuff, width*sizeof(int));
				}

				currRow++;	// increment the current row
			}

			// receive data from last rows from slaves and indicate termination
			for (int i = 1; i < np; i++) {
				MPI_Recv(receiveBuff, width+1, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status);

				// send the termination flag to indicate termination
				MPI_Send(&terminationFlag, 1, MPI_INT, status.MPI_SOURCE, TAG, MPI_COMM_WORLD);
				memcpy(finalArray + receiveBuff[width]*width, receiveBuff, width*sizeof(int));
			}

			// Generate Image
			gil::rgb8_image_t img(height, width);
			auto img_view = gil::view(img);

			for (int k = 0; k < height; ++k) {
				for (int p = 0; p < width; ++p) {
					img_view(p, k) = render(receiveBuff[ (k*width) + p] / 512.0);
				}
			}

			t_elapsed = MPI_Wtime () - t_start; // compute the overall time taken
			printf("Total time: %f\r\n", t_elapsed);
			gil::png_write_view("mandelbrot-ms.png", const_view(img));
		}// end master

		// slaves
		else {

			// init var and array
			int currRow;
			int sendBuff[width + 1];

			// continue to loop through until a termination flag has been received
			while(1) {

				// receive which row to calculate from master
				MPI_Recv(&currRow, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &status);
				
				// break if a termination flag has been received
				if (currRow == -1) {
					break;
				}	

				// calculate the mandelbrot for the given row
				y = minY + currRow*it;
				x = minX;
				for (int i = 0; i < width; ++i) {
					sendBuff[i] = mandelbrot(x,y);
					x += jt;
				}

				// set the last element to the current row
				sendBuff[width] = currRow;

				// send the data back to master
				MPI_Send(sendBuff, width+1, MPI_INT, 0, TAG, MPI_COMM_WORLD);
			}
		}// end slaves
	}// end master-slave manner

	// MPI End
	MPI_Finalize();
	return 0;
}

/* eof */
