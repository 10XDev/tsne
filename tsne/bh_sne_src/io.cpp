/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "tsne.h"

namespace TSNE {

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool load_data(const char* dat_file, double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter) {

	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen(dat_file, "r+b")) == NULL) {
		fprintf(stderr,"Error: could not open data file.\n");
		return false;
	}
	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(d, sizeof(int), 1, h);											// original dimensionality
    fread(theta, sizeof(double), 1, h);										// gradient accuracy
	fread(perplexity, sizeof(double), 1, h);								// perplexity
	fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
    fread(max_iter, sizeof(int),1,h);                                       // maximum number of iterations
	*data = (double*) malloc(*d * *n * sizeof(double));
    if(*data == NULL) { fprintf(stderr,"Memory allocation failed!\n"); exit(1); }
    fread(*data, sizeof(double), *n * *d, h);                               // the data
    if(!feof(h)) fread(rand_seed, sizeof(int), 1, h);                       // random seed
	fclose(h);
	fprintf(stderr,"Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

// Function that saves map to a t-SNE file
void save_data(const char* res_file, double* data, int* landmarks, double* costs, int n, int d) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen(res_file, "w+b")) == NULL) {
		fprintf(stderr,"Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
    fwrite(data, sizeof(double), n * d, h);
	fwrite(landmarks, sizeof(int), n, h);
    fwrite(costs, sizeof(double), n, h);
    fclose(h);
	fprintf(stderr,"Wrote the %i x %i data matrix successfully!\n", n, d);
}

void save_csv(const char* csv_file, double* Y, int N, int D) {
    std::ofstream csv(csv_file);

    for (int d = 0; d < D; d++) {
        csv << "TSNE" << d+1 << ",";
    }
    csv << "\n";

    for (int n = 0; n < N; n++) {
        int row_offset = n * D;
        for (int d = 0; d < D; d++) {
            csv << Y[row_offset + d] << ",";
        }
        csv << "\n";
    }

    csv.close();
}

}  // namespace TSNE
