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


#ifndef SPTREE_H
#define SPTREE_H

#include <array>
#include <memory>

template <int NDims=2>
class Cell {
 public:
    typedef std::array<double, NDims> point_t;

    Cell() = default;
    Cell(point_t&& inp_corner, const point_t* width);

    double getCorner(unsigned int d) const;
    double getWidth(unsigned int d) const;
    void setCorner(point_t&& inp_corner);
    void setWidth(const point_t* width);
    bool containsPoint(const double* point) const;
    double maxWidth() const;

 private:
    point_t corner;
    const point_t* width;

    // disallow copy
    Cell(const Cell&) = delete;
};

template <int NDims=2>
class SPTree
{
public:
   enum { no_children = 2 * SPTree<NDims-1>::no_children };

   typedef typename Cell<NDims>::point_t point_t;

private:
    // Fixed constants
    static const unsigned int QT_NODE_CAPACITY = 1;

    // Properties of this node in the tree
    SPTree<NDims>* parent;
    unsigned int dimension;
    bool is_leaf;
    unsigned int size;
    unsigned int cum_size;

    // The width for each cell is the same at each level.  The parent
    // node owns the widths array for its children.  The top node must
    // also own the array for itself.
    std::unique_ptr<point_t> top_widths;
    std::unique_ptr<point_t> child_widths;

    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell<NDims> boundary;

    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
    const double* data;
    point_t center_of_mass;
    std::array<unsigned int, QT_NODE_CAPACITY> index;

    // Children
    std::array<std::unique_ptr<SPTree<NDims>>, no_children> children;

    SPTree(const double* inp_data, point_t&& inp_corner, const point_t* inp_width);
    SPTree(const double* inp_data, unsigned int N, point_t&& inp_corner, const point_t* inp_width);
    SPTree(SPTree* inp_parent, const double* inp_data, unsigned int N, point_t&& inp_corner, const point_t* inp_width);
    SPTree(SPTree* inp_parent, const double* inp_data, point_t&& inp_corner, const point_t* inp_width);

    // Disallow copy
    SPTree(const SPTree&) = delete;

public:
    SPTree(const double* inp_data, unsigned int N);

    void setData(const double* inp_data);
    SPTree* getParent();
    void construct(Cell<NDims> boundary);
    bool insert(unsigned int new_index);
    void subdivide();
    bool isCorrect() const;
    void rebuildTree();
    void getAllIndices(unsigned int* indices) const;
    unsigned int getDepth() const;
    void computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q);
    void computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f);
    void print();

private:
    void init(SPTree* inp_parent, const double* inp_data, point_t&& inp_corner, const point_t* inp_width);
    void computeNonEdgeForces(unsigned int point_index,
                              double theta,
                              double neg_f[],
                              double* sum_Q,
                              double max_width_squared);
    void fill(unsigned int N);
    unsigned int getAllIndices(unsigned int* indices, unsigned int loc) const;
    bool isChild(unsigned int test_index, unsigned int start, unsigned int end);
};

template <>
struct SPTree<0>
{
    enum { 	no_children = 1 };
};

#endif
