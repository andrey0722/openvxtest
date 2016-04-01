/*
File: ref_GrabCutSegmentation.c
Contains imlementation of GrabCut segmentation method.

Author: Andrey Olkhovsky

Date: 26 March 2016
*/

#include "../ref.h"

/**\brief Finds max flow and min cut in graph.

\details This method implements an experimental max-flow algorithm
by Yuri Boykov and Vladimir Kolmogorov.

\param[in] i_adjGraph An input adjacency matrix (N+2)x(N+2) for the graph
where N is number of graph vertices excluding terminals.
\param[out] o_minCutLabels A byte array of length N to store labels,
indicating, to which part of cut the vertex is assigned.
0 means S set, 1 means T set.
*/
void maxFlow(vx_matrix i_adjGraph, vx_array o_minCutLabels);

vx_status ref_GrabCutSegmentation(const vx_image src_image, vx_image dst_image) {
	const uint32_t src_width = src_image->width;
	const uint32_t src_height = src_image->height;
	const uint32_t dst_width = dst_image->width;
	const uint32_t dst_height = dst_image->height;

	if (src_width != dst_width || src_height != dst_height)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	if (src_image->image_type != VX_DF_IMAGE_RGB) {
		return VX_ERROR_INVALID_PARAMETERS;
	}

	const uint8_t* src_data = src_image->data;
	uint8_t* dst_data = dst_image->data;

	for (uint32_t i = 0; i < src_height * src_width * 3; i++) {
		dst_data[i] = src_data[i];
	}
	return VX_SUCCESS;
}

#pragma warning(disable: 4100)
void maxFlow(vx_matrix i_adjGraph, vx_array o_minCutLabels) {

}
