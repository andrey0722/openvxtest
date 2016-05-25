/*
File: ref_GrabCutSegmentation.c
Contains imlementation of GrabCut segmentation method.

Author: Andrey Olkhovsky

Date: 26 March 2016
*/

#include "../ref.h"

#define _USE_MATH_DEFINES
#include <math.h>

#pragma pack(push,1)

const vx_float64 EPS = +1.0e-12; //< An accuracy of comparison of floating point values

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

/** @brief Represents an adjacent edge to some vertex of the graph
*/
typedef struct _vx_GC_neighbour {
	vx_uint32 edge;		///< Index of the edge weight in edges array
	vx_uint32 px;		///< Index of the adjacent vertex, connected by this edge
} vx_GC_neighbour;

/** @brief Stores two adjacent edges to some vertex of the graph.
		One is outcoming, other is incoming
*/
typedef struct _vx_GC_couple {
	vx_GC_neighbour in;		///< @brief Incoming edge
	vx_GC_neighbour out;	///< @brief Outcoming edge
} vx_GC_couple;

/** @brief The graph adjacency, represented by N-by-N sparse matrix using Compressed Sparse Row (CSR) format.
		Stores P edges and all adjacent edges for each vertex.
*/
typedef struct _vx_GC_graph {
	vx_float64 *edges;	///< @brief Potentially non-zero edges of the graph, 1-by-P array
	vx_uint32 *nz;		///< @brief The number of elements in previous rows, 1-by-N array
	vx_GC_couple *nbr;	///< @brief Adjacent edges for each vertex, 1-by-P array
} vx_GC_graph;

#pragma pack(pop)

// Predicates that check whether exist specific neighbour of pixel or not
// Unificates the check process

vx_bool has_top_left	(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);
vx_bool has_top			(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);
vx_bool has_top_right	(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);
vx_bool has_left		(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);
vx_bool has_right		(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);
vx_bool has_bottom_left	(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);
vx_bool has_bottom		(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);
vx_bool has_bottom_right(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols);

// An array, consisting of predicates
vx_bool(*check[])(vx_uint32, vx_uint32, vx_uint32, vx_uint32) = { 
	has_top_left,
	has_top,
	has_top_right,
	has_left,
	has_right,
	has_bottom_left,
	has_bottom,
	has_bottom_right };

// Functions than return index of specific neighbour of given pixel

vx_uint32 get_top_left		(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);
vx_uint32 get_top			(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);
vx_uint32 get_top_right		(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);
vx_uint32 get_left			(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);
vx_uint32 get_right			(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);
vx_uint32 get_bottom_left	(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);
vx_uint32 get_bottom		(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);
vx_uint32 get_bottom_right	(vx_uint32 i, vx_uint32 rows, vx_uint32 cols);

// An array, consisting of these functions
vx_uint32(*get_neighbour[])(vx_uint32, vx_uint32, vx_uint32) = {
	get_top_left,
	get_top,
	get_top_right,
	get_left,
	get_right,
	get_bottom_left,
	get_bottom,
	get_bottom_right };

/** @brief Computes euclidian distance between integer pixels in RGB color space
	@param [in] z1 A pointer to the first pixel, integer
	@param [in] z2 A pointer to the second pixel, integer
	@return Squared distance between z1 and z2
*/
vx_uint32 euclidian_dist_ii(const vx_uint8 *z1, const vx_uint8 *z2);

/** @brief Computes euclidian distance between integer and floating pixels in RGB color space
    @param [in] z1 A pointer to the first pixel, integer
    @param [in] z2 A pointer to the second pixel, floating
    @return Squared distance between z1 and z2
*/
vx_float64 euclidian_dist_if(const vx_uint8 *z1, const vx_float64 *z2);

/** @brief Computes maximal eigenvalue and it's eigenvector
		of symmetric matrix M 3-by-3
	@param [in] M The source matrix 3-by-3
	@param [out] eigVal The poiner where maximal eigenvalue is written
	@param [out] eigVec An array where eigenvector is written
*/
void eigen(vx_float64 M[3][3], vx_float64 *eigVal, vx_float64 eigVec[3]);

/** @brief Initializes GMMs from matte.
	@detailed Uses a binary tree quantization algorithm 
		described by Orchard and Bouman.
	@param [in] N The number of pixels
	@param [in] K The number of GMM components for each GMM
	@param [in] px Source pixels, 1-by-N array
	@param [out] gmm_index GMM components indexes, assigned to each pixel, 1-by-N array
	@param [in] mask Algorithm's mask, 1-by-N array
	@param [in] maskClass A mask class to initialize corresponding GMM
*/
void initGmmComponents(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px,
					   vx_uint32 *gmm_index, const vx_uint8 *mask, vx_uint8 maskClass);

/** @brief Separates all pixels to GMM components basing on the previous partition.
	@param [in] N The number of pixels
	@param [in] K The number of GMM components for each GMM
	@param [in] px Source pixels, 1-by-N array
	@param [in,out] gmm_index GMM components indexes, assigned to each pixel, 1-by-N array
	@param [in] gmm The GMM component that is being reassigned, 1-by-K array
	@param [in] mask Algorithm's matte, 1-by-N array
	@param [in] maskClass A mask class to initialize corresponding GMM
*/
void assignGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px, vx_uint32 *gmm_index,
				GmmComponent *bgdGMM, GmmComponent *fgdGMM, const vx_uint8 *mask);

/** @brief Computes all required numerical characteristics of the GMM components.
		Does process only GMM, corresponding to given matte class
	@param [in] N The number of pixels
	@param [in] K The number of GMM components for each GMM
	@param [in] px Source pixels, 1-by-N array
	@param [in] gmm_index GMM components indexes, assigned to each pixel, 1-by-N array
	@param [out] gmm GMM component, whose characteristics are to be computed, 1-by-K array
	@param [in] mask Algorithm's matte, 1-by-N array
	@param [in] maskClass A mask class to learn corresponding GMM parameters
*/
void learnGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px,
			   const vx_uint32 *gmm_index, GmmComponent *bgdGMM,
			   GmmComponent *fgdGMM, const vx_uint8 *mask);

/** @brief Computes the 'beta' parameter of the algorithm.
	@param [in] data Image's pixel colors, height-by-width array
	@param [in] width The width of image
	@param [in] height The height of image
	@return Returns the value of beta.
*/
vx_float64 computeBeta(const vx_RGB_color *data, vx_uint32 width, vx_uint32 height);

/** @brief Computes the maximum weight of the graph edge,
		that is obviously not less than any another.
	@param [in] N The number of pixels
	@param [in] adj The graph adjacency matrix
	@return Returns value of max-weight.
*/
vx_float64 computeMaxWeight(vx_uint32 N, const vx_GC_graph *adj);

/** @brief Helps to calculate position of edge, adjacent to 'cur' and 'other' vertices
		and outcoming from 'cur'. Contains a lot of magic numbers. Boosts performance.
	@param [in] cur The vertex, from which needed edge outcomes
	@param [in] other The vertex, to which needed edge incomes
	@param [in] width Width of the image
	@param [in] height Height of the image
	@return Returns offset of the edge in adjacency list of 'cur'
*/
vx_uint32 getEdgeOffset(vx_uint32 cur, vx_uint32 other, vx_uint32 width, vx_uint32 height);

/** @brief Sets N-links in the given graph for the source image.
	@detailed N-links are the links between neighbouring pixels in image.
			8-neighbourhood scheme is used.
	@param [in] data Image's pixel colors, 1-by-(width*height) array
	@param [in] width The width of image
	@param [in] height The height of image
	@param [in] mask Algorithm's mask, 1-by-N array
	@param [in] gamma Parameter of the algorithm
	@param [in] beta Parameter of the algorithm
	@param [in,out] adj A graph adjacency matrix (N+2)-by-(N+2)
*/
void setGraphNLinks(const vx_RGB_color *data, vx_uint32 width,
					vx_uint32 height, const vx_uint8 *mask,
					vx_float64 gamma, vx_float64 beta, vx_GC_graph *adj);

/** @brief Allocates required memory for the graph
	@param [in,out] adj An empty graph
	@param [in] NNZ The number of edges in graph
	@param [in] N The number of vertices
*/
void buildGraph(vx_GC_graph *adj, vx_uint32 NNZ, vx_uint32 N);

/** @brief Deallocates memory of given graph
	@param [in,out] adj A non-empty graph
*/
void destructGraph(vx_GC_graph *adj);

/** @brief Copies the content of graph to another. Another
        graph must already be allocated enough memory.
    @param [in] src Source graph
    @param [out] dst Destination graph
    @param [in] N The number of vertices in source graph
*/
void copyGraph(const vx_GC_graph *src, vx_GC_graph *dst, vx_uint32 N);

/** @brief Computes data term for given GMM component.
    @details Computes an expression \f$-\log{\pi_n}+\frac{1}{2}\log{\det{\Sigma_n}}+\frac{1}{2}\left[z_n-
        \mu_n\right]^T\Sigma_n^{-1}\left[z_n-\mu_n\right]\f$ for given GMM component.
    @param [in] comp GMM component
    @param [out] color Given color
    @return The value determined by formula
*/
vx_float64 computeGmmComponentDataTerm(const GmmComponent *comp, const vx_RGB_color *color);

/** @brief Computes value that defines likelihood of
		belonging given color to particular GMM
	@param [in] K The number of components in given GMM
	@param [in] gmm The GMM, consisting of K components
	@param [in] color A pointer to RGB color
	@return Returns described value
*/
vx_float64 computeGmmDataTerm(vx_uint32 K, const GmmComponent *gmm, const vx_RGB_color *color);

/** @brief Sets N-links in the given graph for the source image.
	@detailed T-links are links between pixels and
		terminals - source and sink of the network.
	@param [in] N The number of pixels
	@param [in] K The number of GMM components for each GMM
	@param [in] data Image's pixel colors, 1-by-N array
	@param [in] bgdGMM The background GMM
	@param [in] fgdGMM The foreground GMM
	@param [in] mask The algorithm's mask
	@param [in] maxWeight Parameter of the algorithm, pretty large
	@param [in,out] adj A graph adjacency matrix (N+2)-by-(N+2)
*/
void setGraphTLinks(vx_uint32 N, vx_uint32 K, const vx_RGB_color *data,
						   const GmmComponent *bgdGMM, const GmmComponent *fgdGMM,
						   const vx_uint8 *mask, vx_float64 maxWeight, vx_GC_graph *adj);

/** @brief Finds the max-flow through the network and the corresponding min-cut
	@param [in] N The number of pixels (non-terminal vertices)
	@param [in,out] adj An adjacency matrix of network
	@param [in] mask Algorithm's mask, 1-by-N array, generates by min-cut
*/
void maxFlow(vx_uint32 N, vx_GC_graph *adj, vx_uint8 *mask);

vx_status ref_GrabCutSegmentation(const vx_image src_image, vx_matrix mask,
								  vx_rectangle_t rect, vx_uint32 iterCount, vx_uint8 mode) {
	const vx_uint32 src_width = src_image->width;
	const vx_uint32 src_height = src_image->height;
	const vx_uint32 mask_width = mask->width;
	const vx_uint32 mask_height = mask->height;

	if (src_width != mask_width || src_height != mask_height)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
	if (src_image->image_type != VX_DF_IMAGE_RGB) {
		return VX_ERROR_INVALID_PARAMETERS;
	}
	if (mask->array_type != VX_TYPE_UINT8) {
		return VX_ERROR_INVALID_PARAMETERS;
	}

	// The number of pixels
	vx_uint32 N = src_width * src_height;
	// The number of GMM components for each GMM
	vx_uint32 K = 5;
	// Pixels' colors
	vx_RGB_color *px = (vx_RGB_color*)src_image->data;
	// The mask, indicates pixels separation
	vx_uint8 *mask_data = (vx_uint8*)mask->data;
	// GMM components indexes, assigned to each pixel
	vx_uint32 *GMM_index = (vx_uint32*)calloc(N, sizeof(vx_uint32));
	// Background GMM
	GmmComponent *bgdGMM = (GmmComponent*)calloc(K, sizeof(GmmComponent));
	// Foreground GMM
	GmmComponent *fgdGMM = (GmmComponent*)calloc(K, sizeof(GmmComponent));
	// The image's neighbourhood graph adjacency matrix
	vx_GC_graph adj_graph;
	// The adacency matrix for the rest graph (the same size)
	vx_GC_graph adj_rest;

	if (mode == VX_GC_CONTINUE) {
		// Initialize GMMs from input (perspective)
	}
	else {
		if (mode == VX_GC_INIT_WITH_RECT) {
			if (rect.end_x > rect.start_x + mask->width) {
				rect.end_x = rect.start_x + mask->width;
			}
			if (rect.end_y > rect.start_y + mask->height) {
				rect.end_y = rect.start_y + mask->height;
			}
			memset(mask_data, VX_GC_BGD, N * sizeof(uint8_t));
			for (vx_uint32 y = rect.start_y; y < rect.end_y; y++) {
				for (vx_uint32 x = rect.start_x; x < rect.end_x; x++) {
					mask_data[y * mask->width + x] = VX_GC_FGD | VX_GC_UNDEF;
				}
			}
		}
		initGmmComponents(N, K, px, GMM_index, mask_data, VX_GC_BGD);
		initGmmComponents(N, K, px, GMM_index, mask_data, VX_GC_FGD);
		learnGMMs(N, K, px, GMM_index, bgdGMM, fgdGMM, mask_data);
	}

	// The number of all non-zero elements in the graph adjacency matrix
	vx_uint32 NNZ_total = (4 * N - 3 * (src_width + src_height) + 2) * 2 + 4 * N;
	buildGraph(&adj_graph, NNZ_total, N + 2);
	buildGraph(&adj_rest, NNZ_total, N + 2);
	vx_float64 gamma = 50;
	vx_float64 beta = computeBeta(px, src_width, src_height);
	setGraphNLinks(px, src_width, src_height, mask_data, gamma, beta, &adj_graph);
	vx_float64 maxWeight = computeMaxWeight(N, &adj_graph);

	for (vx_uint32 iter = 0; iter < iterCount; iter++) {
		assignGMMs(N, K, px, GMM_index, bgdGMM, fgdGMM, mask_data);
		learnGMMs(N, K, px, GMM_index, bgdGMM, fgdGMM, mask_data);
		setGraphTLinks(N, K, px, bgdGMM, fgdGMM, mask_data, maxWeight, &adj_graph);
		copyGraph(&adj_graph, &adj_rest, N + 2);
		maxFlow(N, &adj_rest, mask_data);
	}

	destructGraph(&adj_graph);
	destructGraph(&adj_rest);
	free(bgdGMM);
	free(fgdGMM);
	free(GMM_index);

	return VX_SUCCESS;
}

void eigen(vx_float64 M[3][3], vx_float64 *eigVal, vx_float64 eigVec[3]) {
	// Modified trigonometric formula of Viet is used to solve cubic equation 
	// x^3 - x^2 * trace(M) - x * (trace(M^2)-trace(M)^2)/2 - det(M) = 0
	// to determine eigenvalues

	// M = p*B + q*E, where E is the unit matrix
	// Matrix B has similar eigenvalues and eigenvectors as M

	vx_float64 trace = M[0][0] + M[1][1] + M[2][2];
	vx_float64 q = trace / 3;
	vx_float64 p2 = (M[0][0] - q) * (M[0][0] - q) +
		(M[1][1] - q) * (M[1][1] - q) + (M[2][2] - q) * (M[2][2] - q) +
		2 * (M[0][1] * M[0][1] + M[0][2] * M[0][2] + M[1][2] * M[1][2]);
	vx_float64 p = sqrt(p2 / 6);
	vx_float64 B[3][3];		// M = p*B + q*E;  B = (M - q*E) / p
	for (vx_uint32 i = 0; i < 3; i++) {
		for (vx_uint32 j = 0; j < 3; j++) {
			B[i][j] = (M[i][j] - (i == j ? q : 0)) / p;
		}
	}
	vx_float64 detB = B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) +
		B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0]) -
		B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]);

	vx_float64 phi = acos(detB / 2) / 3; // Angle of first equation's root

	// Eigenvalues eig1 >= eig2 >= eig3
	vx_float64 eig1 = q + 2 * p * cos(phi);
	vx_float64 eig3 = q + 2 * p * cos(phi + (2 * M_PI / 3));
	vx_float64 eig2 = trace - eig1 - eig3; // by the Viet's theorem

	*eigVal = eig1;

	// Computation of eigenvector, corresponding to eig1
	// (M - eig2*E) * (M - eig3*E) = A
	// where A is matrix whose columns are eigenvectors,
	// corresponding to eig1. There could be a null vector so
	// the sum of these vectors is taken.
	// Quite sophisticated way to do it is obtained
	// after rewriting the matrix A

	memset(eigVec, 0, 3 * sizeof(vx_float64));
	for (vx_uint32 i = 0; i < 3; i++) {
		for (vx_uint32 j = 0; j < 3; j++) {
			eigVec[i] += M[i][j] * (M[j][0] + M[j][1] + M[j][2]);
		}
		eigVec[i] += eig2 * eig3 - (eig2 + eig3) * (M[i][0] + M[i][1] + M[i][2]);
	}
}

void initGmmComponents(vx_uint32 N, vx_uint32 K,
					   const vx_RGB_color *px, vx_uint32 *gmm_index,
					   const vx_uint8 *mask, vx_uint8 maskClass) {

	// Stores numbers of pixels, assigned to each component
	vx_uint32 *pxCount = (vx_uint32*)calloc(K, sizeof(vx_uint32));
	// Stores sums of each component's pixel colors
	vx_uint32(*pxSum)[3] = (vx_uint32(*)[3])calloc(K, 3 * sizeof(vx_uint32));
	// Stores all productions of each component's pixel colors (for covariance)
	vx_uint32(*pxProd)[3][3] = (vx_uint32(*)[3][3])calloc(K, 9 * sizeof(vx_uint32));
	// Stores eigenvalues of each component's covariance matrix
	vx_float64 *eigenVal = (vx_float64*)calloc(K, sizeof(vx_float64));
	// Stores eigenvectors of each component's covariance matrix
	vx_float64 (*eigenVec)[3] = (vx_float64(*)[3])calloc(K, 3 * sizeof(vx_float64));

	vx_float64 cov[3][3];	// Covariance matrix

	// Add all pixels of matteClass to the 0-th component
	for (vx_uint32 p = 0; p < N; p++) {
		if (mask[p] & maskClass) {
			gmm_index[p] = 0;		// Assign to component
			vx_uint8 *color = (vx_uint8*)(px + p);
			for (vx_uint32 i = 0; i < 3; i++) {
				pxSum[0][i] += color[i];
				for (vx_uint32 j = 0; j < 3; j++) {
					pxProd[0][i][j] += color[i] * color[j];
				}
			}
			pxCount[0]++;
		}
	}

	// Do until number of components is reached K
	for (vx_uint32 k = 1; k < K; k++) {

		// Compute eigenvalues and eigenvectors of last built component
		for (vx_uint32 i = 0; i < 3; i++) {
			for (vx_uint32 j = 0; j < 3; j++) {
				vx_uint32 d = pxProd[k - 1][i][j] - pxSum[k - 1][i] * pxSum[k - 1][j];
				cov[i][j] = (vx_float64)d / pxCount[k - 1];
			}
		}
		eigen(cov, eigenVal + k - 1, eigenVec[k - 1]);

		// Search for maximal eigenvalue of component
		vx_uint32 maxV = 0;		// The component with max eighenvalue
		vx_float64 maxEigenVal = eigenVal[0];
		for (vx_uint32 i = 1; i < K; i++) {
			if (eigenVal[i] > maxEigenVal) {
				maxEigenVal = eigenVal[i];
				maxV = i;
			}
		}

		// limit = eigenvector * mean[maxV]
		vx_float64 limit = eigenVec[maxV][0] * pxSum[maxV][0] / pxCount[maxV] +
			eigenVec[maxV][1] * pxSum[maxV][1] / pxCount[maxV] +
			eigenVec[maxV][2] * pxSum[maxV][2] / pxCount[maxV];

		pxCount[k] = 0;

		// Add all pixels from maxV-th component that satisfy
		// eigenvector * color <= eigenvector * mean[maxV]
		//  to the new k-th component
		for (vx_uint32 p = 0; p < N; p++) {
			if (mask[p] & maskClass) {
				vx_uint32 *color = (vx_uint32*)(px + p);
				// value = eigenvector * color
				vx_float64 value = eigenVec[maxV][0] * color[0] +
					eigenVec[maxV][1] * color[1] +
					eigenVec[maxV][2] * color[2];
				if (value <= limit) {		// if satisfies
					gmm_index[p] = k;		// Assign to the k-th component
					for (vx_uint32 i = 0; i < 3; i++) {
						pxSum[k][i] += color[i];	// Include in k-th
						pxSum[maxV][i] -= color[i];	// Exclude from maxV-th
						for (vx_uint32 j = 0; j < 3; j++) {
							pxProd[k][i][j] += color[i] * color[j];		// Include in k-th
							pxProd[maxV][i][j] -= color[i] * color[j];	// Exclude from maxV-th
						}
					}
					pxCount[k]++;		// Include in k-th
					pxCount[maxV]--;	// Exclude from maxV-th
				}
			}
		}

		// Compute eigenvalues and eigenvectors of the new k-th component
		for (vx_uint32 i = 0; i < 3; i++) {
			for (vx_uint32 j = 0; j < 3; j++) {
				vx_uint32 d = pxProd[maxV][i][j] - pxSum[maxV][i] * pxSum[maxV][j];
				cov[i][j] = (vx_float64)d / pxCount[maxV];
			}
		}
		eigen(cov, eigenVal + maxV, eigenVec[maxV]);
	}

	free(pxCount);
	free(pxSum);
	free(pxProd);
	free(eigenVal);
	free(eigenVec);
}

#define IND(A) (((A) & (VX_GC_BGD | VX_GC_FGD)) - 1)

void assignGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px, vx_uint32 *gmm_index,
				GmmComponent *bgdGMM, GmmComponent *fgdGMM, const vx_uint8 *mask) {

	GmmComponent *gmm[2];
	gmm[IND(VX_GC_FGD)] = fgdGMM;
	gmm[IND(VX_GC_BGD)] = bgdGMM;

	for (vx_uint32 i = 0; i < N; i++) {
		vx_uint8 ind = IND(mask[i]);
		const vx_RGB_color *color = px + i;
		vx_uint32 min_comp = 0;
		vx_float64 min = computeGmmComponentDataTerm(gmm[ind], color);
		for (vx_uint32 j = 0; j < K; j++) {
			vx_float64 D = computeGmmComponentDataTerm(gmm[ind] + j, color);
			if (D < min) {
				min = D;
				min_comp = j;
			}
		}
		gmm_index[i] = min_comp;
	}
}

void learnGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px,
			   const vx_uint32 *gmm_index, GmmComponent *bgdGMM,
			   GmmComponent *fgdGMM, const vx_uint8 *mask) {

	// Stores sums of color components in every GMM component
	vx_uint32 (*sums)[2] = (vx_uint32(*)[2])calloc(K * 3 * 2, sizeof(vx_uint32));
	// Stores sums of productions of all pairs of color components in every GMM component
	vx_uint32 (*prods)[2] = (vx_uint32(*)[2])calloc(K * 9 * 2, sizeof(vx_uint32));
	// Stores the number of pixels in each GMM component
	vx_uint32 (*counts)[2] = (vx_uint32(*)[2])calloc(K * 2, sizeof(vx_uint32));
	// Stores total number of pixels in this GMM
	vx_uint32 counts_total[2];

	memset(sums, 0, K * 3 * 2 * sizeof(vx_uint32));
	memset(prods, 0, K * 9 * 2 * sizeof(vx_uint32));
	memset(counts, 0, K * 2 * sizeof(vx_uint32));
	memset(counts_total, 0, 2 * sizeof(vx_uint32));

	GmmComponent *gmm[2]; 
	gmm[IND(VX_GC_FGD)] = fgdGMM;
	gmm[IND(VX_GC_BGD)] = bgdGMM;

	// Accumulating
	for (vx_uint32 k = 0; k < N; k++) {
		vx_uint8 ind = IND(mask[k]);
		vx_uint32 comp = gmm_index[k];
		vx_uint8 *color = (vx_uint8*)(px + k);
		for (vx_uint8 i = 0; i < 3; i++) {
			sums[comp * 3 + i][ind] += color[i];
		}
		for (vx_uint32 i = 0; i < 3; i++) {
			for (vx_uint32 j = 0; j < 3; j++) {
				prods[(comp * 9) + (i * 3) + j][ind] += color[i] * color[j];
			}
		}
		counts[comp][ind]++;
		counts_total[ind]++;
	}

	vx_float64 cov[3][3];	// covariance matrix, just local

	// Computing parameters
	for (vx_uint8 ind = 0; ind < 2; ind++) {
		for (vx_uint32 comp = 0; comp < K; comp++) {
			GmmComponent *gc = gmm[ind] + comp;
			for (vx_uint32 i = 0; i < 3; i++) {		// mean colors
				gc->mean[i] = (vx_float64)sums[comp * 3 + i][ind] / counts[comp][ind];
			}
			for (vx_uint32 i = 0; i < 3; i++) {		// covariance matrix
				for (vx_uint32 j = 0; j < 3; j++) {
					cov[i][j] = (vx_float64)prods[(comp * 9) + (i * 3) + j][ind] / counts[comp][ind];
					cov[i][j] -= gc->mean[i] * gc->mean[j];
				}
			}

			// Determinant
			gc->cov_det = cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1]);
			gc->cov_det -= cov[0][1] * (cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0]);
			gc->cov_det += cov[0][2] * (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0]);

			// Inverse covariance matrix
			gc->inv_cov[0][0] = (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1]) / gc->cov_det;
			gc->inv_cov[0][1] = -(cov[0][1] * cov[2][2] - cov[0][2] * cov[2][1]) / gc->cov_det;
			gc->inv_cov[0][2] = (cov[0][1] * cov[1][2] - cov[0][2] * cov[1][1]) / gc->cov_det;

			gc->inv_cov[1][0] = -(cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0]) / gc->cov_det;
			gc->inv_cov[1][1] = (cov[0][0] * cov[2][2] - cov[0][2] * cov[2][0]) / gc->cov_det;
			gc->inv_cov[1][2] = -(cov[0][0] * cov[1][2] - cov[0][2] * cov[1][0]) / gc->cov_det;

			gc->inv_cov[2][0] = (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0]) / gc->cov_det;
			gc->inv_cov[2][1] = -(cov[0][0] * cov[2][1] - cov[0][1] * cov[2][0]) / gc->cov_det;
			gc->inv_cov[2][2] = (cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]) / gc->cov_det;

			gc->weight = (vx_float64)counts[comp][ind] / counts_total[ind]; // component weight (pi)
		}
	}

	free(sums);
	free(prods);
	free(counts);
}

vx_uint32 euclidian_dist_ii(const vx_uint8 *z1, const vx_uint8 *z2) {
    vx_uint32 result = 0;
    for (vx_uint32 i = 0; i < 3; i++) {
        result += (vx_uint32)((z1[0] - z2[0]) * (z1[0] - z2[0]));
    }
    return result;
}

vx_float64 euclidian_dist_if(const vx_uint8 *z1, const vx_float64 *z2) {
    vx_float64 result = 0;
    for (vx_uint32 i = 0; i < 3; i++) {
        result += (z1[0] - z2[0]) * (z1[0] - z2[0]);
    }
    return result;
}

#pragma warning(disable: 4100)
vx_bool has_top_left(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return (i > 0) && (j > 0);
}

vx_bool has_top(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return i > 0;
}

vx_bool has_top_right(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return (i > 0) && (j < cols - 1);
}

vx_bool has_left(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return j > 0;
}

vx_bool has_right(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return j < cols - 1;
}

vx_bool has_bottom_left(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return (i < rows - 1) && (j > 0);
}

vx_bool has_bottom(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return i < rows - 1;
}

vx_bool has_bottom_right(vx_uint32 i, vx_uint32 j, vx_uint32 rows, vx_uint32 cols) {
	return (i < rows - 1) && (j < cols - 1);
}

vx_uint32 get_top_left(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i - cols - 1;
}

vx_uint32 get_top(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i - cols;
}

vx_uint32 get_top_right(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i - cols + 1;
}

vx_uint32 get_left(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i - 1;
}

vx_uint32 get_right(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i + 1;
}

vx_uint32 get_bottom_left(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i + cols - 1;
}

vx_uint32 get_bottom(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i + cols;
}

vx_uint32 get_bottom_right(vx_uint32 i, vx_uint32 rows, vx_uint32 cols) {
	return i + cols + 1;
}

#pragma warning(default: 4100)

vx_float64 computeBeta(const vx_RGB_color *data, vx_uint32 width, vx_uint32 height) {
	vx_uint32 sum = 0;
	vx_uint32 count = 0;
	for (vx_uint32 i = 0; i < height; i++) {
		for (vx_uint32 j = 0; j < width; j++) {
            const vx_uint8 *current = (const vx_uint8*)(data + i * width + j);
			for (vx_uint32 p = 4; p < 8; p++) {				// process all neighbours
				if (check[p](i, j, height, width)) {		// in bottom-right direction
					vx_uint32 other_offset = get_neighbour[p](i * width + j, height, width);
					const vx_uint8 *other = (const vx_uint8*)(data + other_offset);
					sum += euclidian_dist_ii(current, other);
					count++;
				}
			}
		}
	}
	return 1 / ((vx_float64)sum / count * 2);
}

void buildGraph(vx_GC_graph *adj, vx_uint32 NNZ, vx_uint32 N) {
	adj->edges = (vx_float64*)calloc(NNZ, sizeof(vx_float64));
	adj->nz = (vx_uint32*)calloc(N + 1, sizeof(vx_uint32));
	adj->nbr = (vx_GC_couple*)calloc(NNZ, sizeof(vx_GC_couple));

	memset(adj->edges, 0, NNZ * sizeof(vx_float64));
	memset(adj->nz, 0, (N + 1) * sizeof(vx_uint32));
	memset(adj->nbr, 0, NNZ * sizeof(vx_GC_couple));
}

vx_uint32 getEdgeOffset(vx_uint32 cur, vx_uint32 other, vx_uint32 width, vx_uint32 height) {
	vx_uint32 cur_col = cur % width;
	vx_uint32 cur_row = cur / width;
	vx_uint32 other_col = other % width;
	vx_uint32 other_row = other / width;
	vx_int32 a = other_row - cur_row;
	vx_int32 b = other_col - cur_col;

	/* This function calculates position of "cur --> other" edge
	   There are nine possible locations of 'cur' (on schema):
	                    ________________________________________
	   top-left case ->|_|_____________top case_______________|_|<- top-right case
	                   | |                                    | |
	                   |l|                                    |r|
	                   |e|                                    |i|
	                   |f|                                    |g|
	                   |t|                                    |h|
	                   | |              MIDDLE                |t|
	                   | |               CASE                 | |
	                   |c|                                    |c|
	                   |a|                                    |a|
	                   |s|                                    |s|
	                   |e|                                    |e|
	                   | |____________________________________| |
	bottom-left case ->|_|____________bottom case_____________|_|<- bottom-right case

	Depending on the case different formulas are used, quite sophisticated, but effective.

	*/

	if (cur_row == 0) {
		if (cur_col == 0) {
			return a * 2 + b - 1;							// top-left case
		}
		else if (cur_col == width - 1) {
			return (a * 2 + (b + 1) + 1) / 2;				// top-right case
		}
		else {
			return a * 2 + (b + 2) / 2 + (a == b ? 1 : 0);	// top case
		}
	}
	else if (cur_row == height - 1) {
		if (cur_col == 0) {
			return ((a + 1) * 2 + b + 1) / 2;				// bottom-left case
		}
		else if (cur_col == width - 1) {
			return 3 - (-a * 2 - b);						// bottom-right case
		}
		else {
			return 4 - (-a * 2 + (-b + 2) / 2 + (a == b ? 1 : 0));  // bottom case
		}
	}
	else {
		if (cur_col == 0) {
			vx_uint32 x = (a + 1) * 2 + (b + 1) / 2;
			return x - (x + 2) / 4;							// left case
		}
		else if (cur_col == width - 1) {
			vx_uint32 x = (-a + 1) * 2 + (-b + 1) / 2;
			return 4 - (x - (x + 2) / 4);					// right case
		}
		else {
			return a + b + 2 + (a ? ((a + 1) / 2) * 3 : (3 - b) / 2); // middle case
		}
	}
}

void setGraphNLinks(const vx_RGB_color *data, vx_uint32 width,
					vx_uint32 height, const vx_uint8 *mask,
					vx_float64 gamma, vx_float64 beta, vx_GC_graph *adj) {

	vx_float64 sqrt_2 = sqrt(2);
	vx_uint32 N = width * height;
	vx_uint32 source = N;
	vx_uint32 sink = N + 1;

	vx_uint32 NNZ_cur = 0;

	for (vx_uint32 i = 0; i < N; i++) {
		vx_uint32 row = i / width;
        vx_uint32 col = i % width;
        const vx_uint8 *cur_data = (const vx_uint8*)(data + i);

		for (vx_uint32 j = 0; j < 8; j++) {			// process all the neighbours
			if (check[j](row, col, height, width)) {
				vx_uint32 other = get_neighbour[j](i, height, width);
				vx_uint32 other_row = other / width;
				vx_uint32 other_col = other % width;
				vx_bool same = ((mask[i] ^ mask[other]) & (VX_GC_BGD | VX_GC_FGD)) == 0;
				vx_bool diagonal = ((col == other_col) ^ (row == other_row));
				const vx_uint8 *otherData = (const vx_uint8*)(data + other);
				vx_uint32 d = euclidian_dist_ii(cur_data, otherData);
				vx_float64 weight = same ? gamma * exp(-beta * d) / (diagonal ? sqrt_2 : 1) : 0;
				adj->edges[NNZ_cur] = weight;
				adj->nbr[NNZ_cur].out.edge = NNZ_cur;
				adj->nbr[NNZ_cur].out.px = other;
				NNZ_cur++;
			}
		}

		adj->nbr[NNZ_cur].out.edge = NNZ_cur;
		adj->nbr[NNZ_cur].out.px = source;	// init t-link to source
		NNZ_cur++;

		adj->nbr[NNZ_cur].out.edge = NNZ_cur;
		adj->nbr[NNZ_cur].out.px = sink;		// init t-link to sink
		NNZ_cur++;

		adj->nz[i + 1] = NNZ_cur;
	}

	for (vx_uint32 i = source; i <= sink; i++) {
		for (vx_uint32 j = 0; j < N; j++) {
			adj->nbr[NNZ_cur].out.edge = NNZ_cur;			// outcoming from terminal
			vx_uint32 pos = adj->nz[j + 1] - (sink - i + 1);
			adj->nbr[NNZ_cur].in.edge = adj->nbr[pos].out.edge; // incoming from terminal
			adj->nbr[NNZ_cur].out.px = adj->nbr[NNZ_cur].in.px = j;
			NNZ_cur++;
		}
		adj->nz[i + 1] = NNZ_cur;
	}

	for (vx_uint32 cur = 0; cur < N; cur++) {
		for (vx_uint32 i = adj->nz[cur]; i < adj->nz[cur + 1]; i++) {
			vx_uint32 other = adj->nbr[i].out.px;
			vx_uint32 offset = other < N ? getEdgeOffset(other, cur, width, height) : cur;
			vx_uint32 j = adj->nz[other] + offset;
			adj->nbr[i].in.px = other;
			adj->nbr[i].in.edge = j;
		}
	}
}

void destructGraph(vx_GC_graph *adj) {
	free(adj->edges);
	free(adj->nz);
	free(adj->nbr);
}

void copyGraph(const vx_GC_graph *src, vx_GC_graph *dst, vx_uint32 N) {
    for (vx_uint32 i = 0; i < N + 1; i++) {
        dst->nz[i] = src->nz[i];
    }
    for (vx_uint32 i = 0; i < src->nz[N]; i++) {
		dst->edges[i] = src->edges[i];
        dst->nbr[i] = src->nbr[i];
    }
}

vx_float64 computeMaxWeight(vx_uint32 N, const vx_GC_graph *adj) {
	vx_float64 maxSum = 0;
	for (vx_uint32 i = 0; i < N; i++) { // Search for max links sum in row
		vx_float64 sum = 0;
		for (vx_uint32 j = adj->nz[i]; j < adj->nz[i + 1]; j++) {
			sum += adj->edges[j];
		}
		if (sum > maxSum) {
			maxSum = sum;
		}
	}
	return 1 + maxSum;
}

vx_float64 computeGmmComponentDataTerm(const GmmComponent *comp, const vx_RGB_color *color) {
    vx_uint8 *clr = (vx_uint8*)color;
    vx_float64 prod = 0.0;
    for (vx_uint32 i = 0; i < 3; i++) {
        for (vx_uint32 j = 0; j < 3; j++) {
            prod += (clr[i] - comp->mean[i]) * (clr[j] - comp->mean[j]) * comp->inv_cov[i][j];
        }
    }
    return -(log(comp->weight) - 0.5*log(comp->cov_det) - 0.5 * prod);
}

vx_float64 computeGmmDataTerm(vx_uint32 K, const GmmComponent *gmm, const vx_RGB_color *color) {
    vx_float64 result = 0.0;
    for (const GmmComponent *comp = gmm; comp < gmm + K; comp++) {
		if (comp->weight > +0.0 && comp->cov_det > EPS) {
			result += computeGmmComponentDataTerm(comp, color);
		}
    }
    return result;
}

void setGraphTLinks(vx_uint32 N, vx_uint32 K, const vx_RGB_color *data, const GmmComponent *bgdGMM,
					const GmmComponent *fgdGMM, const vx_uint8 *mask,
					vx_float64 maxWeight, vx_GC_graph *adj) {

	vx_uint32 source = N;
	vx_uint32 sink = N + 1;

	for (vx_uint32 i = 0; i < N; i++) {
		vx_float64 fromSouce = 0;
		vx_float64 toSink = 0;
		if (mask[i] & VX_GC_UNDEF) {		// Undefined
			fromSouce = computeGmmDataTerm(K, bgdGMM, data + i);
			toSink = computeGmmDataTerm(K, fgdGMM, data + i);
		}
		else {
			if (mask[i] & VX_GC_BGD) {		// Background
				fromSouce = 0;
				toSink = maxWeight;
			}
			else {							// Foreground
				fromSouce = maxWeight;
				toSink = 0;
			}
		}

		vx_uint32 sourcePos = adj->nbr[adj->nz[source] + i].out.edge;
		vx_uint32 sinkPos = adj->nbr[adj->nz[sink] + i].in.edge;

		adj->edges[sourcePos] = fromSouce;
		adj->edges[sinkPos] = toSink;
	}
}

// Gets the neighbouring edge by index i in adj_nbr[]. For TREE_S - outcoming, for TREE_T - incoming.
#define GET_NBR_OUT(TREE, i) ((TREE) == TREE_S ? &adj->nbr[(i)].out : &adj->nbr[(i)].in)
// Gets the neighbouring edge by index i in adj_nbr[]. For TREE_S - incoming, for TREE_T - outcoming.
#define GET_NBR_IN(TREE, i) ((TREE) == TREE_S ? &adj->nbr[(i)].in : &adj->nbr[(i)].out)

void maxFlow(vx_uint32 N, vx_GC_graph *adj, vx_uint8 *mask) {
	const vx_uint32 SOURCE = N;		// The source terminal
	const vx_uint32 SINK = N + 1;	// The sink terminal
	const vx_uint8 TREE_S = 0;		// Pixel from source tree
	const vx_uint8 TREE_T = 1;		// Pixel from sink tree
	const vx_uint8 TREE_FREE = 2;	// Free pixel

	vx_uint32 TERMINAL[2];
	TERMINAL[TREE_S] = SOURCE;
	TERMINAL[TREE_T] = SINK;

	// Stores the labels of the tree to which belongs each pixel
	vx_uint8 *tree = (vx_uint8*)calloc(N + 2, sizeof(vx_uint8));
	// Stores the parent of each pixel in trees (if parent(n) == n then n has no parent)
	vx_uint32 *parent = (vx_uint32*)calloc(N + 2, sizeof(vx_uint32));
	// Linked list of all active nodes, for FIFO (if active_next(n) == n then n is last)
	vx_uint32 *active_next = (vx_uint32*)calloc(N + 2, sizeof(vx_uint32));

	for (vx_uint32 i = 0; i < N + 2; i++) {
		parent[i] = i;			// initially there is no parent for each
		tree[i] = TREE_FREE;	// initially each is free
		active_next[i] = i;		// initially there are no actives
	}

	tree[SOURCE] = TREE_S;		// initial source tree
	tree[SINK] = TREE_T;		// initial sink tree
	vx_uint32 active_first = SOURCE;	// The first active node (beginning of the list)
	vx_uint32 active_last = SINK;		// The last active node (end of the list)
	active_next[SOURCE] = SINK;			// First two actives

	// Stores orphan pixels, which require further handle
	vx_uint32 *orphan = (vx_uint32*)calloc(N + 2, sizeof(vx_uint32));
	// Ponts to ghost element after the last orphan
	vx_uint32 orphan_end = 0;

	vx_bool done = vx_false_e;
	while (!done) {
		vx_uint32 bound_edge = 0;			// Boundary edge, by which trees do touch
		vx_uint32 bound_nodes[2];			// Nodes of the boundary edge

		//////////////////////
		//// Growth stage
		//////////////////////

		vx_bool pathFound = vx_false_e;
		vx_uint32 p = active_first;			// Current active pixel
		while (!pathFound) {
			if (tree[p] != TREE_FREE) {
				// The bypass of edges, adjacent to p
				for (vx_uint32 i = adj->nz[p]; i < adj->nz[p + 1] && !pathFound; i++) {
					vx_GC_neighbour *nghbr = GET_NBR_OUT(tree[p], i); // Adjacent adge
					if (adj->edges[nghbr->edge] > EPS) {
						vx_uint32 q = nghbr->px;		// Neighbouring pixel/terminal
						if (tree[q] == TREE_FREE) {
							tree[q] = tree[p];			// If free, then add to the tree
							parent[q] = p;
							if (active_next[q] == q && q != active_last) {	// and make it active
								active_next[active_last] = q;
								active_last = q;
							}
						}
						else if (tree[q] != tree[p]) {	// Trees have met
							bound_nodes[tree[p]] = p;	// Remember the boundary edge
							bound_nodes[tree[q]] = q;
							bound_edge = nghbr->edge;
							pathFound = vx_true_e;
						}
					}
				}
			}
			if (!pathFound) {
				if (p != active_last) {				// If current is not last active
					active_first = active_next[p];	// Throw away current active, take the next one
					active_next[p] = p;
					p = active_first;
				}
				else {				// If current is last active
					break;			// End of the growth stage
				}
			}
		}
		if (!pathFound) {			// There is no more non-saturated path,
			done = vx_true_e;		// end of the algorithm
		}
		else {

			//////////////////////
			//// Saturation stage
			//////////////////////

			// The max possible flow through whe found path (so-called "bottleneck")
			vx_float64 bottleneck = adj->edges[bound_edge];
			for (vx_uint8 cur_tree = TREE_S; cur_tree <= TREE_T; cur_tree++) {
				vx_uint32 p_i = bound_nodes[cur_tree];		// Going from the boundary edge
				while (p_i != TERMINAL[cur_tree]) {			// to the both terminals
					vx_uint32 p_parent = parent[p_i];
					// The bypass of edges, adjacent to p_i
					for (vx_uint32 i = adj->nz[p_i]; i < adj->nz[p_i + 1]; i++) {
						vx_GC_neighbour *nghbr = GET_NBR_IN(cur_tree, i);
						if (nghbr->px == p_parent) {
							if (adj->edges[nghbr->edge] < bottleneck - EPS) {
								bottleneck = adj->edges[nghbr->edge];	// Search for min non-zero
							}
							break;
						}
					}
					p_i = p_parent;
				}
			}

			// Bottleneck is found, now we should push this flow through the path

			adj->edges[bound_edge] -= bottleneck;	// Push

			for (vx_uint8 cur_tree = TREE_S; cur_tree <= TREE_T; cur_tree++) {
				vx_uint32 p_i = bound_nodes[cur_tree];		// Going from the boundary edge
				while (p_i != TERMINAL[cur_tree]) {			// to the both terminals
					vx_uint32 p_parent = parent[p_i];
					// The bypass of edges, adjacent to p_i
					for (vx_uint32 i = adj->nz[p_i]; i < adj->nz[p_i + 1]; i++) {
						vx_GC_neighbour *nghbr = GET_NBR_IN(TREE_S, i);
						if (nghbr->px == p_parent) {
							adj->edges[nghbr->edge] -= bottleneck;	// Push
							if (adj->edges[nghbr->edge] < EPS) {	// If edge becomes saturated
								parent[p_i] = p_i;					// the node is cut off
								orphan[orphan_end] = p_i;			// and becomes 'orphan'
								orphan_end++;
							}
							break;
						}
					}
					p_i = p_parent;
				}
			}

			//////////////////////
			//// Adoption stage
			//////////////////////

			for (vx_uint32 orph_cur = 0; orph_cur < orphan_end; orph_cur++) {
				vx_uint32 p = orphan[orph_cur];					// Current orphan
				vx_bool parentFound = vx_false_e;
				// Search for new parent among the neighbours
				for (vx_uint32 i = adj->nz[p]; i < adj->nz[p + 1] && !parentFound; i++) {
					vx_GC_neighbour *nghbr = GET_NBR_IN(tree[p], i);
					if (adj->edges[nghbr->edge] > EPS) {	// linked by non-saturated edge
						vx_uint32 q = nghbr->px;
						if (tree[q] == tree[p]) {	// in the same tree
							vx_uint32 q_i = q;	// Check if neighbour's origin is this tree's terminal
							while (q_i != TERMINAL[tree[p]] && parent[q_i] != q_i) {
								q_i = parent[q_i];
							}
							if (q_i == TERMINAL[tree[p]]) {
								parent[p] = q;			// New parent is found
								parentFound = vx_true_e;
							}
						}
					}
				}
				if (!parentFound) {		// If no parent is found in original tree
					for (vx_uint32 i = adj->nz[p]; i < adj->nz[p + 1]; i++) {
						vx_GC_neighbour *nghbr = GET_NBR_OUT(tree[p], i);
						vx_uint32 q = nghbr->px;
						if (adj->edges[nghbr->edge] > EPS && tree[nghbr->px] != TREE_FREE) {
							if (active_next[q] == q && q != active_last) {	// 
								active_next[active_last] = q;	// All neighbours linked by
								active_last = q;		// non-saturated edges will make active
							}
						}
						if (parent[q] == p) {
							parent[q] = q;				// All child nodes
							orphan[orphan_end] = q;		// will make 'orphans'
							orphan_end++;
						}
					}
					tree[p] = TREE_FREE;				// Current orphan becomes free
				}
			}
			orphan_end = 0;
		}
	}

	//////////////////////
	//// Cut stage
	//////////////////////

	for (vx_uint32 i = 0; i < N; i++)
	{
		if (mask[i] & VX_GC_UNDEF) {
			mask[i] = (tree[i] == TREE_S ? VX_GC_FGD : VX_GC_BGD) | VX_GC_UNDEF;
		}
	}

	free(tree);
	free(parent);
	free(active_next);
	free(orphan);
}
