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
#include <vector>

namespace TSNE {
namespace _sptree_internal {

template <int NDims=2>
class alignas(16) Cell final {
 public:
    typedef std::array<double, NDims> point_t;

    Cell() = default;
    explicit Cell(const point_t&);

    double getCorner(unsigned int d) const;
    void setCorner(const point_t& inp_corner);
    void setCorner(unsigned int d, double val);
    bool containsPoint(const double* point, const point_t& width) const;

 private:
    point_t corner;

    // disallow copy
    Cell(const Cell&) = delete;
};

template <int NDims>
class alignas(16) SPTreeNode final {
public:

    typedef typename Cell<NDims>::point_t point_t;
    enum { no_children = 2 * SPTreeNode<NDims-1>::no_children };

private:
    // Fixed constants
    static constexpr unsigned int QT_NODE_CAPACITY = 1;

    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    point_t center_of_mass;
    Cell<NDims> boundary;

    // Children
    std::unique_ptr<std::array<SPTreeNode<NDims>, no_children>> children;

    // Properties of this node in the tree
    unsigned int cum_size;

    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
    std::array<unsigned int, QT_NODE_CAPACITY> index;

    // Disallow copy
    SPTreeNode(const SPTreeNode&) = delete;

    void subdivide(const double* data, std::vector<point_t>* widths, typename std::vector<point_t>::size_type depth);
    unsigned int which_child(const double* point) const;
    void make_child(unsigned int i, const point_t& width);

    bool is_leaf() const;

public:
    SPTreeNode();
    explicit SPTreeNode(const point_t& corner);

    void setCorner(const point_t& corner);

    bool insert(unsigned int new_index, const double* data, std::vector<point_t>* widths, typename std::vector<point_t>::size_type depth);
    bool isCorrect(const double* data, typename std::vector<point_t>::const_iterator width) const;

    void computeNonEdgeForces(unsigned int point_index,
                              const double* data_point,
                              double neg_f[],
                              double* sum_Q,
                              double max_width_squared) const;
    unsigned int getAllIndices(unsigned int* indices, unsigned int loc) const;
    unsigned int getDepth() const;
    void print(const double* data) const;
};

template <>
class SPTreeNode<0>
{
 public:
    enum { 	no_children = 1 };
};

}  // namespace _sptree_internal

template <int NDims=2>
class SPTree
{
public:
    typedef typename _sptree_internal::SPTreeNode<NDims>::point_t point_t;
    enum { no_children = _sptree_internal::SPTreeNode<NDims>::no_children };

private:
    _sptree_internal::SPTreeNode<NDims> node;

    // The width for each cell is the same at each level.  The top node owns
    // this and the children get references to it.
    std::vector<point_t> widths;
    bool insert(unsigned int new_index);

    const double* data;
    double maxWidth() const;

public:
    SPTree(const double* inp_data, unsigned int N);

    bool isCorrect() const;
    void getAllIndices(unsigned int* indices) const;
    unsigned int getDepth() const;
    void computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q) const;
    void computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f) const;
    void print() const;

private:
    void fill(unsigned int N);
    // Disallow copy
    SPTree(const SPTree&) = delete;
};

}  // namespace TSNE

#endif
