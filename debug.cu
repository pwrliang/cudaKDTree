#include "cukd/spatial-kdtree.h"
#include <iomanip>
#include <thrust/host_vector.h>

using namespace cukd;


thrust::host_vector<float2> read_data_file(const std::string &filename) {
    thrust::host_vector<float2> results;

    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return results;
    }

    float2 temp_pair;
    while (infile >> temp_pair.x >> temp_pair.y) {
        results.push_back(temp_pair);
    }

    infile.close();

    return results;
}


template<typename data_t, typename data_traits>
void checkRec(SpatialKDTree<data_t, data_traits> &tree,
              const cukd::box_t<typename data_traits::point_t> &bounds,
              int nodeID) {
    using point_t = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };

    auto &node = tree.nodes[nodeID];
    if (node.count > 0) {
        for (int i = 0; i < node.count; i++) {
            int primID = tree.primIDs[node.offset + i];
            point_t point = data_traits::get_point(tree.data[primID]);
            if (!bounds.contains(point))
                throw std::runtime_error
                        ("invalid k-d tree - prim " + std::to_string(primID) + " not in parent bounds");
        }
        return;
    }

    const scalar_t curr_s = node.pos;

    cukd::box_t<point_t> lBounds = bounds;
    set_coord(lBounds.upper, node.dim, curr_s);
    cukd::box_t<point_t> rBounds = bounds;
    set_coord(rBounds.lower, node.dim, curr_s);

    checkRec<data_t, data_traits>(tree, lBounds, node.offset + 0);
    checkRec<data_t, data_traits>(tree, rBounds, node.offset + 1);
}

int main(int ac, const char **av) {
    using namespace cukd::common;

    std::string path = av[1];
    thrust::device_vector<float2> d_points = read_data_file(path);
    auto numPoints = d_points.size();
    float2 *p_points = thrust::raw_pointer_cast(d_points.data());

    using kd_tree_t = SpatialKDTree<float2, default_data_traits<float2> >;
    kd_tree_t tree;

    buildTree(tree, p_points, numPoints);
    CUKD_CUDA_SYNC_CHECK();

    auto bounds = tree.bounds;
    thrust::host_vector<typename kd_tree_t::Node> h_nodes =
            thrust::device_vector<typename kd_tree_t::Node>(
                tree.nodes, tree.nodes + tree.numNodes);
    thrust::host_vector<uint32_t> h_primIDs = thrust::device_vector<uint32_t>(
        tree.primIDs, tree.primIDs + tree.numPrims);
    thrust::host_vector<float2> h_data =
            thrust::device_vector<float2>(tree.data,
                                          tree.data + numPoints);
    tree.nodes = h_nodes.data();
    tree.primIDs = h_primIDs.data();
    tree.data = h_data.data();
    checkRec<float2, default_data_traits<float2> >(tree,
                                                   bounds, 0);
    printf("KD tree is valid!\n");
}
