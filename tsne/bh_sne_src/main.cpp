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


#include "tsne.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using std::fprintf;
using std::free;
using std::vector;

// Function that runs the Barnes-Hut implementation of t-SNE
int main(int argc, char *argv[]) {

    // load input and output
    const char *dat_file = "data.dat";
    const char *res_file = "result.dat";
    if (argc > 1) {
        dat_file = argv[1];
        res_file = argv[2];
    }

    // Define some variables
	int origN, N, D, no_dims, max_iter, *landmarks;
	double perc_landmarks;
	double perplexity, theta, *data;
    int rand_seed = -1;

    // Read the parameters and the dataset
	if(TSNE::load_data(dat_file, &data, &origN, &D, &no_dims, &theta, &perplexity, &rand_seed, &max_iter)) {

		// Make dummy landmarks
        N = origN;

		// Now fire up the SNE implementation
		vector<double> Y(N * no_dims);
        TSNE::run(data, N, D, Y.data(), no_dims, perplexity, theta, rand_seed, false, NULL, false, max_iter);

		// Save the results
        vector<int> landmarks(N);
        for(int n = 0; n < N; n++) landmarks[n] = n;
		vector<double> costs(N);
        TSNE::save_data(res_file, Y.data(), landmarks.data(), costs.data(), N, no_dims);

        // Clean up the memory
		free(data); data = NULL;
    }
}
