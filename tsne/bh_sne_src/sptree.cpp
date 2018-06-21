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

#include <algorithm>  // for max, max_element
#include <cfloat>     // for DBL_MAX
#include <cstdio>     // for fprintf, stderr, size_t
#include <memory>     // for unique_ptr

using std::array;
using std::max;
using std::max_element;
using std::unique_ptr;
using std::vector;

namespace TSNE {
namespace _sptree_internal {

template<int NDims>
Cell<NDims>::Cell(const typename Cell<NDims>::point_t& p) : corner(p) {}

template<int NDims>
double Cell<NDims>::getCorner(unsigned int d) const {
    return corner[d];
}

template<int NDims>
void Cell<NDims>::setCorner(const typename Cell<NDims>::point_t& val) {
    corner = val;
}

template<int NDims>
void Cell<NDims>::setCorner(unsigned int d, double val) {
    corner[d] = val;
}

// Checks whether a point lies in a cell
template<int NDims>
bool Cell<NDims>::containsPoint(const double* point, const typename Cell<NDims>::point_t& width) const
{
    for(int d = 0; d < NDims; ++d) {
        if(corner[d] - width[d] > point[d]) return false;
        if(corner[d] + width[d] < point[d]) return false;
    }
    return true;
}

// Constructor for SPTreeNode.
template<int NDims>
SPTreeNode<NDims>::SPTreeNode(const typename SPTreeNode<NDims>::point_t& inp_corner) :
    boundary(inp_corner), cum_size(0) {
    center_of_mass.fill(0.0);
}
// Constructor for SPTreeNode.
template<int NDims>
SPTreeNode<NDims>::SPTreeNode() :
    cum_size(0) {
    center_of_mass.fill(0.0);
}

// Update the corner position.
template<int NDims>
void SPTreeNode<NDims>::setCorner(const typename SPTreeNode<NDims>::point_t& inp_corner) {
    boundary.setCorner(inp_corner);
}

template <int NDims>
bool SPTreeNode<NDims>::is_leaf() const {
    return NDims == 0 || cum_size <= QT_NODE_CAPACITY;
}

template<int NDims>
unsigned int SPTreeNode<NDims>::which_child(const double* point) const {
    unsigned int div = 1;
    unsigned int i = 0;
    for (int d = 0; d < NDims; d++) {
        if (boundary.getCorner(d) > point[d]) {
            i |= div;
        }
        div *= 2;
    }
    return i;
}

template<int NDims>
bool SPTreeNode<NDims>::insert(unsigned int new_index, const double* data, vector<point_t>* widths, typename vector<point_t>::size_type depth)
{
    // Ignore objects which do not belong in this quad tree
    const double* point = data + new_index * NDims;
    if(!boundary.containsPoint(point, (*widths)[depth]))
        return false;

    // Online update of cumulative size and center-of-mass
    unsigned int size = cum_size++;
    if (size != 0) {
        double mult2 = 1.0 / (double) cum_size;
        double mult1 = (double) (size) * mult2;

        for(unsigned int d = 0; d < NDims; d++) {
          center_of_mass[d] = center_of_mass[d] * mult1 + mult2 * point[d];
        }
    } else {
        for(unsigned int d = 0; d < NDims; d++) {
          center_of_mass[d] = point[d];
        }
    }

    // If there is space in this quad tree and it is a leaf, add the object here
    if(size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        return true;
    } else if (size == QT_NODE_CAPACITY) {
         // Don't add duplicates for now (this is not very nice)
        for(unsigned int n = 0; n < size; n++) {
            if (__builtin_expect(index[n] == new_index, 0)) {
                cum_size--;
                return true;
            }
            bool duplicate = true;
            for(unsigned int d = 0; d < NDims; d++) {
                if(__builtin_expect(point[d] != data[index[n] * NDims + d], 1)) { duplicate = false; break; }
            }
            if (__builtin_expect(duplicate, 0)) {
                cum_size--;
                return true;
            }
        }       // Otherwise, we need to subdivide the current cell
        subdivide(data, widths, depth);
    }

    // Find out where the point can be inserted
    auto c = which_child(point);
    return children[c]->insert(new_index, data, widths, depth+1);
}

// Create four children which fully divide this cell into four quads of equal area
template<int NDims>
void SPTreeNode<NDims>::subdivide(const double* data, vector<point_t>* widths, typename vector<point_t>::size_type depth) {

    // If nessessary, add to the width.
    if (depth+1 == widths->size()) {
        // extend the list.
        widths->emplace_back();
        point_t& child_widths = widths->back();
        const point_t& width = (*widths)[depth];
        for(unsigned int d = 0; d < NDims; d++) {
            child_widths[d] = .5 * width[d];
        }
    }
    const point_t& new_width = (*widths)[depth+1];

    // Create new children
    for(unsigned int i = 0; i < no_children; i++) {
        children[i].reset(new SPTreeNode());
        Cell<NDims>& new_corner = children[i]->boundary;
        for(unsigned int d = 0; d < NDims; d++) {
            if((i >> d)%2 == 1)
                new_corner.setCorner(d, boundary.getCorner(d) - new_width[d]);
            else
                new_corner.setCorner(d, boundary.getCorner(d) + new_width[d]);
        }
    }

    // Move existing points to correct children
    for(unsigned int i = 0; i < QT_NODE_CAPACITY; i++) {
        auto c = which_child(data+index[i]*NDims);
        children[c]->insert(index[i], data, widths, depth+1);
    }
}

template<int NDims>
bool SPTreeNode<NDims>::isCorrect(const double* data,
                                  typename vector<point_t>::const_iterator width) const {
    if (is_leaf()) {
        for(unsigned int n = 0; n < cum_size; n++) {
            const double* point = data + index[n] * NDims;
            if(!boundary.containsPoint(point, *width)) return false;
        }
    } else {
        ++width;
        for(const auto& child : children) {
            if (!child->isCorrect(data, width)) {
                return false;
            }
        }
    }
    return true;
}

// Build a list of all indices in SPTree
template<int NDims>
unsigned int SPTreeNode<NDims>::getAllIndices(unsigned int* indices, unsigned int loc) const
{

    // Gather indices in current quadrant
    if(is_leaf()) {
        for(unsigned int i = 0; i < cum_size; i++) indices[loc + i] = index[i];
        loc += cum_size;
    } else {
        // Gather indices in children
        for(const auto& child : children)
            loc = child->getAllIndices(indices, loc);
    }
    return loc;
}

template<int NDims>
unsigned int SPTreeNode<NDims>::getDepth() const {
    if(is_leaf()) return 1;
    unsigned int depth = 0;
    for(const auto& child : children) depth = max(depth, child->getDepth());
    return 1u + depth;
}

// Compute non-edge forces using Barnes-Hut algorithm
template<int NDims>
void SPTreeNode<NDims>::computeNonEdgeForces(unsigned int point_index,
    const double* data_point, double theta, double neg_f[], double* sum_Q,
    double max_width_squared) const
{

    // Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 ||
       (cum_size == 1 &&
        __builtin_expect(index[0] == point_index, 0))) return;

    // Compute distance between point and center-of-mass
    double sqdist = .0;

    array<double, NDims> diff;
    for(int i = 0; i < NDims; ++i) {
        diff[i] = data_point[i] - center_of_mass[i];
        sqdist += diff[i] * diff[i];
    }

    // Check whether we can use this node as a "summary"
    // max_width / sqrt(sqdist) < theta
    if(cum_size == 1) {
        sqdist = 1.0 / (1.0 + sqdist);
        *sum_Q += sqdist;
        double mult = sqdist*sqdist;
        for(size_t d = 0; d < NDims; ++d) {
            neg_f[d] += mult * diff[d];
        }
    } else if (max_width_squared < theta * theta * sqdist) {
        // Compute and add t-SNE force between point and current node
        sqdist = 1.0 / (1.0 + sqdist);
        double mult = cum_size * sqdist;
        *sum_Q += mult;
        mult *= sqdist;
        for(size_t d = 0; d < NDims; ++d) {
            neg_f[d] += mult * diff[d];
        }
    } else if (QT_NODE_CAPACITY > 1 && cum_size <= QT_NODE_CAPACITY) {
        // Need to compute forces on a per-point basis.
        auto data = data_point - point_index*NDims;
        for (int j = 0; j < cum_size; ++j) {
            if (index[j] == point_index) {
                continue;
            }
            auto pt = data+index[j]*NDims;
            for(int i = 0; i < NDims; ++i) {
                diff[i] = data_point[i] - pt[i];
                sqdist += diff[i] * diff[i];
            }
            sqdist = 1.0 / (1.0 + sqdist);
            *sum_Q += sqdist;
            double mult = sqdist*sqdist;
            for(size_t d = 0; d < NDims; ++d) {
                neg_f[d] += mult * diff[d];
            }
        }
    } else {

        // Recursively apply Barnes-Hut to children
        max_width_squared /= 4.0;
        for(unsigned int i = 0; i < no_children; i++)
            children[i]->computeNonEdgeForces(point_index, data_point,
                    theta, neg_f, sum_Q, max_width_squared);
    }
}

// Print out tree
template<int NDims>
void SPTreeNode<NDims>::print(const double* data) const
{
    if(cum_size == 0) {
        fprintf(stderr,"Empty node\n");
        return;
    }

    if(is_leaf()) {
        fprintf(stderr,"Leaf node; data = [");
        for(int i = 0; i < cum_size; i++) {
            const double* point = data + index[i] * NDims;
            for(int d = 0; d < NDims; d++) fprintf(stderr,"%f, ", point[d]);
            fprintf(stderr," (index = %d)", index[i]);
            if(i < cum_size - 1) fprintf(stderr,"\n");
            else fprintf(stderr,"]\n");
        }
    }
    else {
        fprintf(stderr,"Intersection node with center-of-mass = [");
        for(const auto& cm : center_of_mass) fprintf(stderr,"%f, ", cm);
        fprintf(stderr,"]; children are:\n");
        for(int i = 0; i < no_children; i++) children[i]->print(data);
    }
}

}  // namespace _sptree_internal

using namespace _sptree_internal;

template<int NDims>
double SPTree<NDims>::maxWidth() const {
    return *max_element(widths[0].begin(), widths[0].end());
}

// Top-node constructor for SPTree -- build tree, too!
template<int NDims>
SPTree<NDims>::SPTree(const double* inp_data, unsigned int N) : data(inp_data) {
    // Compute mean, width, and height of current map (boundaries of SPTree)
    int nD = 0;
    point_t mean_Y, min_Y, max_Y;
    mean_Y.fill(0.0);
    min_Y.fill(DBL_MAX);
    max_Y.fill(-DBL_MAX);

    const double *elem = inp_data;
    for(unsigned int n = 0; n < N; ++n) {
        for(int d = 0; d < NDims; ++d) {
            mean_Y[d] += elem[d];
            if(elem[d] < min_Y[d]) min_Y[d] = elem[d];
            if(elem[d] > max_Y[d]) max_Y[d] = elem[d];
        }
        elem += NDims;
    }

    double dbl_N = static_cast<double>(N);
    for(int d = 0; d < NDims; d++) mean_Y[d] /= dbl_N;

    // Construct SPTree
    widths.emplace_back();
    point_t& width = widths.back();
    for(int d = 0; d < NDims; d++) {
        width[d] = max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
    }
    node.setCorner(mean_Y);
    fill(N);
}

// Insert a point into the SPTree
template<int NDims>
bool SPTree<NDims>::insert(unsigned int new_index) {
    return node.insert(new_index, data, &widths, 0);
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
    return node.isCorrect(data, widths.begin());
}


// Build a list of all indices in SPTree
template<int NDims>
void SPTree<NDims>::getAllIndices(unsigned int* indices) const
{
    node.getAllIndices(indices, 0);
}

template<int NDims>
unsigned int SPTree<NDims>::getDepth() const {
    return node.getDepth();
}

// Compute non-edge forces using Barnes-Hut algorithm
template<int NDims>
void SPTree<NDims>::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q) const {
    double max_width = maxWidth();
    node.computeNonEdgeForces(point_index, data + point_index * NDims, theta, neg_f, sum_Q, max_width*max_width);
}

// Computes edge forces
template<int NDims>
void SPTree<NDims>::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f) const
{

    // Loop over all edges in the graph
    unsigned int ind1 = 0;
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {

            double sqdist;
            // Compute pairwise distance and Q-value
            sqdist = 1.0;
            unsigned int ind2 = col_P[i] * NDims;

            std::array<double, NDims> diffs;
            for(unsigned int d = 0; d < NDims; d++) {
                diffs[d] = data[ind1 + d] - data[ind2 + d];
                sqdist += diffs[d] * diffs[d];
            }

            sqdist = val_P[i] / sqdist;

            // Sum positive force
            for(unsigned int d = 0; d < NDims; d++) {
                pos_f[ind1 + d] += sqdist * diffs[d];
            }
        }
        ind1 += NDims;
    }
}

// Print out tree
template<int NDims>
void SPTree<NDims>::print() const {
    node.print(data);
}

// declare templates explicitly
template class SPTree<2>;
template class SPTree<3>;

}  // namespace TSNE
