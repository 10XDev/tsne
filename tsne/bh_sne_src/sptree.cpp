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

#include "sptree.h"

#include <algorithm>  // for move, max
#include <cfloat>     // for DBL_MAX
#include <cstdio>     // for fprintf, stderr, size_t
#include <memory>     // for unique_ptr

using std::max;
using std::max_element;
using std::move;
using std::unique_ptr;

// Constructs cell
template<int NDims>
Cell<NDims>::Cell(typename Cell<NDims>::point_t&& inp_corner,
                  const typename Cell<NDims>::point_t* width) :
    width(width), corner(inp_corner) {}

template<int NDims>
double Cell<NDims>::getCorner(unsigned int d) const {
    return corner[d];
}

template<int NDims>
double Cell<NDims>::getWidth(unsigned int d) const {
    return (*width)[d];
}

template<int NDims>
void Cell<NDims>::setCorner(typename Cell<NDims>::point_t&& val) {
    corner = val;
}

template<int NDims>
void Cell<NDims>::setWidth(const typename Cell<NDims>::point_t* val) {
    width = val;
}

// Checks whether a point lies in a cell
template<int NDims>
bool Cell<NDims>::containsPoint(const double* point) const
{
    for(int d = 0; d < NDims; d++) {
        if(corner[d] - (*width)[d] > point[d]) return false;
        if(corner[d] + (*width)[d] < point[d]) return false;
    }
    return true;
}

template<int NDims>
double Cell<NDims>::maxWidth() const {
    return *max_element(width->begin(), width->end());
}

// Top-node constructor for SPTree -- build tree, too!
template<int NDims>
SPTree<NDims>::SPTree(const double* inp_data, unsigned int N) {
    // Compute mean, width, and height of current map (boundaries of SPTree)
    int nD = 0;
    point_t mean_Y, min_Y, max_Y;
    mean_Y.fill(0.0);
    min_Y.fill(DBL_MAX);
    max_Y.fill(-DBL_MAX);

    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int d = 0; d < NDims; d++) {
            mean_Y[d] += inp_data[n * NDims + d];
            if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
            if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
        }
        nD += NDims;
    }

    for(int d = 0; d < NDims; d++) mean_Y[d] /= (double) N;

    // Construct SPTree
    top_widths = unique_ptr<point_t>(new point_t());
    auto& widths = *top_widths;
    for(int d = 0; d < NDims; d++) {
        widths[d] = max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
    }
    init(nullptr, inp_data, move(mean_Y), top_widths.get());
    fill(N);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
template<int NDims>
SPTree<NDims>::SPTree(const double* inp_data,
                      unsigned int N,
                      typename SPTree<NDims>::point_t&& inp_corner,
                      const typename SPTree<NDims>::point_t* inp_width)
{
    init(nullptr, inp_data, move(inp_corner), inp_width);
    fill(N);
}


// Constructor for SPTree with particular size (do not fill the tree)
template<int NDims>
SPTree<NDims>::SPTree(const double* inp_data,
                      typename SPTree<NDims>::point_t&& inp_corner,
                      const typename SPTree<NDims>::point_t* inp_width)
{
    init(nullptr, inp_data, move(inp_corner), inp_width);
}


// Constructor for SPTree with particular size and parent (do not fill tree)
template<int NDims>
SPTree<NDims>::SPTree(SPTree* inp_parent, const double* inp_data,
                      typename SPTree<NDims>::point_t&& inp_corner,
                      const typename SPTree<NDims>::point_t* inp_width) {
    init(inp_parent, inp_data, move(inp_corner), inp_width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
template<int NDims>
SPTree<NDims>::SPTree(SPTree* inp_parent, const double* inp_data, unsigned int N,
                      typename SPTree<NDims>::point_t&& inp_corner,
                      const typename SPTree<NDims>::point_t* inp_width)
{
    init(inp_parent, inp_data, move(inp_corner), inp_width);
    fill(N);
}


// Main initialization function
template<int NDims>
void SPTree<NDims>::init(SPTree* inp_parent, const double* inp_data,
                         typename SPTree<NDims>::point_t&& inp_corner,
                         const typename SPTree<NDims>::point_t* inp_width)
{
    parent = inp_parent;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = 0;

    boundary.setCorner(move(inp_corner));
    boundary.setWidth(inp_width);

    center_of_mass.fill(0.0);
}


// Update the data underlying this tree
template<int NDims>
void SPTree<NDims>::setData(const double* inp_data)
{
    data = inp_data;
}


// Get the parent of the current tree
template<int NDims>
SPTree<NDims>* SPTree<NDims>::getParent()
{
    return parent;
}


// Insert a point into the SPTree
template<int NDims>
bool SPTree<NDims>::insert(unsigned int new_index)
{
    // Ignore objects which do not belong in this quad tree
    const double* point = data + new_index * NDims;
    if(!boundary.containsPoint(point))
        return false;

    // Online update of cumulative size and center-of-mass
    cum_size++;
    double mult1 = (double) (cum_size - 1) / (double) cum_size;
    double mult2 = 1.0 / (double) cum_size;

    for(unsigned int d = 0; d < NDims; d++) {
      center_of_mass[d] = center_of_mass[d] * mult1 + mult2 * point[d];
    }

    // If there is space in this quad tree and it is a leaf, add the object here
    if(is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }

    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for(unsigned int n = 0; n < size; n++) {
        bool duplicate = true;
        for(unsigned int d = 0; d < NDims; d++) {
            if(point[d] != data[index[n] * NDims + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if(any_duplicate) return true;

    // Otherwise, we need to subdivide the current cell
    if(is_leaf) subdivide();

    // Find out where the point can be inserted
    for(auto& child : children) {
        if(child->insert(new_index)) return true;
    }

    // Otherwise, the point cannot be inserted (this should never happen)
    return false;
}


// Create four children which fully divide this cell into four quads of equal area
template<int NDims>
void SPTree<NDims>::subdivide() {

    // Create new children
    child_widths = unique_ptr<point_t>(new point_t());
    point_t& new_width = *child_widths;
    for(unsigned int d = 0; d < NDims; d++) {
        new_width[d] = .5 * boundary.getWidth(d);
    }

    for(unsigned int i = 0; i < no_children; i++) {
        unsigned int div = 1;
        point_t new_corner;
        for(unsigned int d = 0; d < NDims; d++) {
            if((i / div) % 2 == 1)
                new_corner[d] = boundary.getCorner(d) - new_width[d];
            else
                new_corner[d] = boundary.getCorner(d) + new_width[d];
            div *= 2;
        }
        children[i] = unique_ptr<SPTree>(new SPTree(this, data, move(new_corner), child_widths.get()));
    }

    // Move existing points to correct children
    for(unsigned int i = 0; i < size; i++) {
        for (auto& child : children) {
            if (child->insert(index[i])) {
                break;
            }
        }
    }

    // Empty parent node
    size = 0;
    is_leaf = false;
}


// Build SPTree on dataset
template<int NDims>
void SPTree<NDims>::fill(unsigned int N)
{
    for(unsigned int i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
template<int NDims>
bool SPTree<NDims>::isCorrect() const
{
    for(unsigned int n = 0; n < size; n++) {
        const double* point = data + index[n] * NDims;
        if(!boundary.containsPoint(point)) return false;
    }
    if(!is_leaf) {
        bool correct = true;
        for(const auto& child : children) correct = correct && child->isCorrect();
        return correct;
    }
    else return true;
}



// Build a list of all indices in SPTree
template<int NDims>
void SPTree<NDims>::getAllIndices(unsigned int* indices) const
{
    getAllIndices(indices, 0);
}


// Build a list of all indices in SPTree
template<int NDims>
unsigned int SPTree<NDims>::getAllIndices(unsigned int* indices, unsigned int loc) const
{

    // Gather indices in current quadrant
    for(unsigned int i = 0; i < size; i++) indices[loc + i] = index[i];
    loc += size;

    // Gather indices in children
    if(!is_leaf) {
        for(const auto& child : children) loc = child->getAllIndices(indices, loc);
    }
    return loc;
}

template<int NDims>
unsigned int SPTree<NDims>::getDepth() const {
    if(is_leaf) return 1;
    unsigned int depth = 0;
    for(const auto& child : children) depth = max(depth, child->getDepth());
    return 1u + depth;
}

// Compute non-edge forces using Barnes-Hut algorithm
template<int NDims>
void SPTree<NDims>::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q) {
    double max_width = boundary.maxWidth();
    computeNonEdgeForces(point_index, theta, neg_f, sum_Q, max_width*max_width);
}

// Compute non-edge forces using Barnes-Hut algorithm
template<int NDims>
void SPTree<NDims>::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q, double max_width_squared)
{

    // Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;

    // Compute distance between point and center-of-mass
    double sqdist = .0;
    size_t ind = point_index * NDims;

    for(const auto& cm : center_of_mass) {
        double diff = data[ind] - cm;
        ind += NDims;
        sqdist += diff * diff;
    }

    // Check whether we can use this node as a "summary"
    // max_width / sqrt(sqdist) < theta
    if(is_leaf || max_width_squared < theta * theta * sqdist) {

        // Compute and add t-SNE force between point and current node
        sqdist = 1.0 / (1.0 + sqdist);
        double mult = cum_size * sqdist;
        *sum_Q += mult;
        mult *= sqdist;
        ind = point_index * NDims;
        for(size_t d = 0; d < NDims; ++d) {
            // recompute here rather than storing from before because memory
            // locality matters more than an extra couple of additions and a
            // subtraction.
            double diff = data[ind] - center_of_mass[d];
            ind += NDims;
            neg_f[d] += mult * diff;
        }
    }
    else {

        // Recursively apply Barnes-Hut to children
        max_width_squared /= 4.0;
        for(unsigned int i = 0; i < no_children; i++)
            children[i]->computeNonEdgeForces(
                    point_index, theta, neg_f, sum_Q, max_width_squared);
    }
}


// Computes edge forces
template<int NDims>
void SPTree<NDims>::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f)
{

    // Loop over all edges in the graph
    unsigned int ind1 = 0;
    double sqdist;
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Compute pairwise distance and Q-value
            sqdist = 1.0;
            unsigned int ind2 = col_P[i] * NDims;

            for(unsigned int d = 0; d < NDims; d++) {
                double diff = data[ind1 + d] - data[ind2 + d];
                sqdist += diff * diff;
            }

            sqdist = val_P[i] / sqdist;

            // Sum positive force
            for(unsigned int d = 0; d < NDims; d++) {
                double diff = data[ind1 + d] - data[ind2 + d];
                pos_f[ind1 + d] += sqdist * diff;
            }
        }
        ind1 += NDims;
    }
}


// Print out tree
template<int NDims>
void SPTree<NDims>::print()
{
    if(cum_size == 0) {
        fprintf(stderr,"Empty node\n");
        return;
    }

    if(is_leaf) {
        fprintf(stderr,"Leaf node; data = [");
        for(int i = 0; i < size; i++) {
            const double* point = data + index[i] * NDims;
            for(int d = 0; d < NDims; d++) fprintf(stderr,"%f, ", point[d]);
            fprintf(stderr," (index = %d)", index[i]);
            if(i < size - 1) fprintf(stderr,"\n");
            else fprintf(stderr,"]\n");
        }
    }
    else {
        fprintf(stderr,"Intersection node with center-of-mass = [");
        for(const auto& cm : center_of_mass) fprintf(stderr,"%f, ", cm);
        fprintf(stderr,"]; children are:\n");
        for(int i = 0; i < no_children; i++) children[i]->print();
    }
}

// declare templates explicitly
template class SPTree<2>;
template class SPTree<3>;
