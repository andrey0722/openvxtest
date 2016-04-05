/*
File: ref_GrabCutSegmentation.c
Contains imlementation of GrabCut segmentation method.

Author: Andrey Olkhovsky

Date: 26 March 2016
*/

#include "../ref.h"

/**  @brief Trimap classes of pixels
     @detailed Each trimap class indicates, to which set: Tbg, Tfg or Tu
	 some pixel is assigned.
*/
enum ETrimapClass {
	TRIMAP_BGD = 1, ///< Background class
	TRIMAP_FGD = 2, ///< Foreground class
	TRIMAP_UNDEF = 4 ///< Undefined class
};

#pragma pack(push,1)

/** @brief The RBG color model
*/
typedef struct _vx_RGB_color {
	vx_uint8 b;
	vx_uint8 g;
	vx_uint8 r;
} vx_RGB_color;

/** @brief Contains all numerical charactiristics 
			of particular GMM component
*/
typedef struct _GmmComponent {
	vx_float64 mean[3];			///< @brief Mean color (BGR)
	vx_float64 inv_cov[3][3];	///< @brief Inverse covariance matrix of colors
	vx_float64 cov_det;			///< @brief The determinant of covariance matrix
	vx_float64 weight;			///< @brief The weight in GMM's weighted sum
} GmmComponent;

/** @brief Contains all GrabCut's data structures and values
*/
typedef struct _param {
	vx_uint32 Npx;				///< @brief The number of pixels in the source image
	vx_RGB_color *px;			///< @brief The source image pixel colors
	vx_uint8 *trimap;			///< @brief Current algorithm's trimap
	vx_uint8 *matte;			///< @brief Current algorithm's matte
	vx_uint32 K;				///< @brief The number of GMM components for each matte class
	vx_uint32 *GMM_index;		///< @brief Contains the index of the component to which each pixel is assigned
	GmmComponent *bgdGMM;		///< @brief The background GMM 
	GmmComponent *fgdGMM;		///< @brief THe foreground GMM
	vx_uint32(*dist)(const vx_RGB_color*, const vx_RGB_color*); ///< @brief The distance function that is being used now
} param;

#pragma pack(pop)

/** @brief Computes euclidian distance between pixels in RGB color space
	@param [in] z1 A pointer to the first pixel
	@param [in] z2 A pointer to the second pixel
	@return Distance between z1 and z2
*/
vx_uint32 euclidian_dist(const vx_RGB_color *z1, const vx_RGB_color *z2);

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

vx_status ref_GrabCutSegmentation(const vx_image src_image, vx_matrix trimap, vx_image dst_image) {
	const vx_uint32 src_width = src_image->width;
	const vx_uint32 src_height = src_image->height;
	const vx_uint32 trimap_width = trimap->width;
	const vx_uint32 trimap_height = trimap->height;

	if (src_width != trimap_width || src_height != trimap_height)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	if (src_image->image_type != VX_DF_IMAGE_RGB) {
		return VX_ERROR_INVALID_PARAMETERS;
	}

	vx_uint32 N = src_width * src_height;
	vx_uint32 K = 5;
	param p;
	p.Npx = N;
	p.px = (vx_RGB_color*)src_image->data;
	p.trimap = (vx_uint8*)trimap->data;
	p.matte = (vx_uint8*)calloc(N, sizeof(vx_uint32));
	p.K = K;
	p.GMM_index = (vx_uint32*)calloc(N, sizeof(vx_uint32));
	p.bgdGMM = (GmmComponent*)calloc(K, sizeof(GmmComponent));
	p.fgdGMM = (GmmComponent*)calloc(K, sizeof(GmmComponent));
	p.dist = euclidian_dist;

	vx_RGB_color* dst_data = (vx_RGB_color*)dst_image->data;
	for (vx_uint32 i = 0; i < N; i++) {
		dst_data[i] = p.px[i]; // Just copy
	}

	free(p.bgdGMM);
	free(p.fgdGMM);
	free(p.matte);
	free(p.GMM_index);

	return VX_SUCCESS;
}

#pragma warning(disable: 4100)
void maxFlow(vx_matrix i_adjGraph, vx_array o_minCutLabels) {

}

vx_uint32 euclidian_dist(const vx_RGB_color *z1, const vx_RGB_color *z2) {
	vx_uint8 d1 = (z1->r - z2->r > 0) ? z1->r - z2->r : z2->r - z1->r;
	vx_uint8 d2 = (z1->g - z2->g > 0) ? z1->g - z2->g : z2->g - z1->g;
	vx_uint8 d3 = (z1->b - z2->b > 0) ? z1->b - z2->b : z2->b - z1->b;
	return d1*d1 + d2*d2 + d3*d3;
}