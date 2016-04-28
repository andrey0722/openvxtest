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

/** @brief Provides strict segmentation
*/
enum EMatteClass {
	MATTE_BGD, ///< Background
	MATTE_FGD  ///< Foreground
};

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

/** @brief Represents M-by-N sparse matrix using Compressed Sparse Row (CSR) format.
		Stores P non-zero elements of type vx_float64
*/
typedef struct _vx_sparse_matrix {
	vx_float64 *data;	///< @brief Non-zero elements of the sparse matrix, 1-by-P array
	vx_uint32 *nz;		///< @brief The number of non-zero elements in previous rows, 1-by-M array
	vx_uint32 *col;		///< @brief Column indexes of corresponding elements from 'data', 1-by-P array
} vx_sparse_matrix;

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
	has_bottom_left,
	has_bottom,
	has_bottom_right,
	has_right };

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
	get_bottom_left,
	get_bottom,
	get_bottom_right,
	get_right };

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

/** @brief Initializes matte from the trimap.
	@param [in] N The number of elements
	@param [in] trimap Agorithm's trimap
	@param [out] matte Algorithm's matte

*/
void initMatte(vx_uint32 N, const vx_uint8 *trimap, vx_uint8 *matte);

/** @brief Initialize random generator specifically with source image and user input
	@detailed This function set random generator seed without using time() function, so
		so the seed will be different on the different calls with similar input.
	@param [in] N The number of pixels
	@param [in] data Source pixel colors, 1-by-N array
	@param [in] matte Algorithm's matte, 1-by-N array
*/
void initRnd(vx_uint32 N, const vx_RGB_color *data, const vx_uint8 *matte);

/** @brief Initializes GMMs from matte
	@detailed Uses k-means clustering method to divide pixels into K components
		for given matte class well enough. Initial centroids are being selected
		with k-means++ algorithm.
	@param [in] N The number of pixels
	@param [in] K The number of GMM components for each GMM
	@param [in] px Source pixels, 1-by-N array
	@param [out] gmm_index GMM components indexes, assigned to each pixel, 1-by-N array
	@param [in] matte Algorithm's matte, 1-by-N array
	@param [in] matteClass A matte class to initialize corresponding GMM
*/
void initGmmComponents(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px,
					   vx_uint32 *gmm_index, const vx_uint8 *matte, vx_uint8 matteClass);

/** @brief Separates all pixels to GMM components basing on the previous partition.
	@param [in] N The number of pixels
	@param [in] K The number of GMM components for each GMM
	@param [in] px Source pixels, 1-by-N array
	@param [in,out] gmm_index GMM components indexes, assigned to each pixel, 1-by-N array
	@param [in] gmm The GMM component that is being reassigned, 1-by-K array
	@param [in] matte Algorithm's matte, 1-by-N array
	@param [in] matteClass A matte class to initialize corresponding GMM
*/
void assignGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px, vx_uint32 *gmm_index,
                const GmmComponent *gmm, const vx_uint8 *matte, vx_uint8 matteClass);

/** @brief Computes all required numerical characteristics of the GMM components.
		Does process only GMM, corresponding to given matte class
	@param [in] N The number of pixels
	@param [in] K The number of GMM components for each GMM
	@param [in] px Source pixels, 1-by-N array
	@param [in] gmm_index GMM components indexes, assigned to each pixel, 1-by-N array
	@param [out] gmm GMM component, whose characteristics are to be computed, 1-by-K array
	@param [in] matte Algorithm's matte, 1-by-N array
	@param [in] matteClass A matte class to learn corresponding GMM parameters
*/
void learnGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px,
			   const vx_uint32 *gmm_index, GmmComponent *gmm,
			   const vx_uint8 *matte, vx_uint8 matteClass);

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
	@param [in] adj_graph The sparse adjacency matrix for graph
	@return Returns value of max-weight.
*/
vx_float64 computeMaxWeight(vx_uint32 N, const vx_sparse_matrix *adj_graph);

/** @brief Sets N-links in the given graph for the source image.
	@detailed N-links are the links between neighbouring pixels in image.
			8-neighbourhood scheme is used.
	@param [in] data Image's pixel colors, 1-by-(width*height) array
	@param [in] width The width of image
	@param [in] height The height of image
	@param [in] matte Algorithm's matte, 1-by-N array
	@param [in] gamma Parameter of the algorithm
	@param [in] beta Parameter of the algorithm
	@param [in,out] adj_graph An adjacency matrix (N+2)-by-(N+2)
*/
void setGraphNLinks(const vx_RGB_color *data, vx_uint32 width,
					vx_uint32 height, const vx_uint8 *matte,
					vx_float64 gamma, vx_float64 beta, vx_sparse_matrix *adj_graph);

/** @brief Allocates required memory for the non-zero elements of the original matrix
	@param [in,out] mat An empty sparse matrix
	@param [in] NNZ The number of non-zero elements in matrix
	@param [in] N The number of rows in the original matrix
*/
void buildSparse(vx_sparse_matrix *mat, vx_uint32 NNZ, vx_uint32 N);

/** @brief Deallocates memory of given sparse matrix
	@param [in,out] mat A non-empty sparse matrix
*/
void destructSparse(vx_sparse_matrix *mat);

/** @brief Copies the content of sparse marix to another. Another
        matrix must already be allocated enough memory.
    @param [in] src Source sparse matrix
    @param [out] dst Destination sparse matrix
    @param [in] rows The number of rows in source matrix
*/
void copySparse(const vx_sparse_matrix *src, vx_sparse_matrix *dst, vx_uint32 rows);

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
	@param [in] trimap The algorithm's trimap
	@param [in] maxWeight Parameter of the algorithm, pretty large
	@param [in,out] adj_graph An adjacency matrix (N+2)-by-(N+2)
*/
void setGraphTLinks(vx_uint32 N, vx_uint32 K, const vx_RGB_color *data,
						   const GmmComponent *bgdGMM, const GmmComponent *fgdGMM,
						   const vx_uint8 *trimap, vx_float64 maxWeight, vx_sparse_matrix *adj_graph);

/** @brief Finds the max-flow through the network and the corresponding min-cut
	@param [in] N The number of pixels (non-terminal vertices)
	@param [in,out] adj An adjacency matrix of network
	@param [in] matte Algorithm's matte, 1-by-N array, generates by min-cut
*/
void maxFlow(vx_uint32 N, vx_sparse_matrix *adj, vx_uint8 *matte);

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

	// The number of pixels
	vx_uint32 N = src_width * src_height;
	// The number of GMM components for each GMM
	vx_uint32 K = 5;
	// Pixels' colors
	vx_RGB_color *px = (vx_RGB_color*)src_image->data;
	// The trimap, indicates pixels separation
	vx_uint8 *trimap_data = (vx_uint8*)trimap->data;
	// The matte, indicates current segmentation
	vx_uint8 *matte = (vx_uint8*)calloc(N, sizeof(vx_uint8));
	// GMM components indexes, assigned to each pixel
	vx_uint32 *GMM_index = (vx_uint32*)calloc(N, sizeof(vx_uint32));
	// Background GMM
	GmmComponent *bgdGMM = (GmmComponent*)calloc(K, sizeof(GmmComponent));
	// Foreground GMM
	GmmComponent *fgdGMM = (GmmComponent*)calloc(K, sizeof(GmmComponent));
	// The image's neighbourhood graph adjacency matrix
	vx_sparse_matrix adj_graph;
	// The adacency matrix for the rest graph (the same size)
	vx_sparse_matrix adj_rest;

	// The number of all non-zero elements in the graph adjacency matrix
	vx_uint32 NNZ_total = (4 * N - 3 * (src_width + src_height) - 1) * 2 + 2 * N;
	buildSparse(&adj_graph, NNZ_total, N + 1);
	buildSparse(&adj_rest, NNZ_total, N + 1);

	initRnd(N, px, matte);
	initMatte(N, trimap_data, matte);
	vx_float64 gamma = 50;
	vx_float64 beta = computeBeta(px, src_width, src_height);
	setGraphNLinks(px, src_width, src_height, matte, gamma, beta, &adj_graph);
	vx_float64 maxWeight = computeMaxWeight(N, &adj_graph);
	initGmmComponents(N, K, px, GMM_index, matte, MATTE_BGD);
	initGmmComponents(N, K, px, GMM_index, matte, MATTE_FGD);
	learnGMMs(N, K, px, GMM_index, bgdGMM, matte, MATTE_BGD);
	learnGMMs(N, K, px, GMM_index, fgdGMM, matte, MATTE_FGD);

    assignGMMs(N, K, px, GMM_index, bgdGMM, matte, MATTE_BGD);
    assignGMMs(N, K, px, GMM_index, fgdGMM, matte, MATTE_FGD);
	learnGMMs(N, K, px, GMM_index, bgdGMM, matte, MATTE_BGD);
	learnGMMs(N, K, px, GMM_index, fgdGMM, matte, MATTE_FGD);
	setGraphTLinks(N, K, px, bgdGMM, fgdGMM, trimap_data, maxWeight, &adj_graph);
	copySparse(&adj_graph, &adj_rest, N + 1);
	maxFlow(N, &adj_rest, matte);

	vx_RGB_color* dst_data = (vx_RGB_color*)dst_image->data;
	memset(dst_data, 0, N * sizeof(vx_RGB_color));
	for (vx_uint32 i = 0; i < N; i++) {
		if (matte[i] == MATTE_FGD) {
			dst_data[i] = px[i]; // Copy foreground pixels
		}
	}

	destructSparse(&adj_graph);
	destructSparse(&adj_rest);
	free(bgdGMM);
	free(fgdGMM);
	free(matte);
	free(GMM_index);

	return VX_SUCCESS;
}

void initMatte(vx_uint32 N, const vx_uint8 *trimap, vx_uint8 *matte) {
	for (vx_uint32 i = 0; i < N; i++) {
		matte[i] = (trimap[i] == TRIMAP_BGD) ? MATTE_BGD : MATTE_FGD;
	}
}

void initRnd(vx_uint32 N, const vx_RGB_color *data, const vx_uint8 *matte) {
	vx_uint32 seed = 0;
	for (vx_uint32 i = 0; i < N; i++) {
		if (matte[i] == MATTE_FGD) {
			seed += data[i].b;
			seed += data[i].g;
			seed += data[i].r;
		}
	}
	srand(seed);
}

void initGmmComponents(vx_uint32 N, vx_uint32 K,
					   const vx_RGB_color *px, vx_uint32 *gmm_index,
					   const vx_uint8 *matte, vx_uint8 matteClass) {

	////////////////////////////////
	/////////// k-means++ (Initial centroids selection)
	////////////////////////////////

	// Stores distances from each pixel to the closest centroid
	vx_float64 *dists = (vx_float64*)calloc(N, sizeof(vx_float64));
	// Stores coordinates of centroids
	vx_float64 *centroids = (vx_float64*)calloc(K, sizeof(vx_float64) * 3);

	vx_uint32 rndFirst = rand() % N; // first centroid is random
	centroids[0] = px[rndFirst].b;
	centroids[1] = px[rndFirst].g;
	centroids[2] = px[rndFirst].r;
	for (vx_uint32 i = 1; i < K; i++) {
		vx_float64 sum = 0;		// Total sum of distances
		for (vx_uint32 j = 0; j < N; j++) {
			if (matte[j] == matteClass) {
                const vx_uint8 *cur_px = (const vx_uint8*)(px + j);
                dists[j] = sqrt(euclidian_dist_if(cur_px, centroids)); // search for minimal distance
				for (vx_uint32 m = 1; m < i; m++) {
                    vx_float64 d = sqrt(euclidian_dist_if(cur_px, centroids + m * 3));
					if (d < dists[j]) {
						dists[j] = d;
					}
				}
				sum += dists[j];
			}
		}
		// Some pixel will be the next centroid with probability
		// proportional to it's distance from 'dists' array
		vx_float64 rnd = (vx_float64)rand() / RAND_MAX * sum; // Continious uniform distribution on [0 sum2)
		vx_float64 nsum = 0; // Current sq sum accumulator
		vx_uint32 j = 0;
		for (; nsum < rnd; j++) {
			if (matte[j] == matteClass) {
				nsum += dists[j];
			}
		}
		// Here j is that random pixel
        centroids[i * 3 + 0] = px[j].b;
        centroids[i * 3 + 1] = px[j].g;
        centroids[i * 3 + 2] = px[j].r;
	}

	////////////////////////////////
	/////////// k-means
	////////////////////////////////

	// Stores numbers of pixels, assigned to each centroid
	vx_uint32 *pxCount = (vx_uint32*)calloc(K, sizeof(vx_uint32));
	// Stores sums of pixels, assigned to each centroid
	vx_uint32 *pxSum = (vx_uint32*)calloc(K, sizeof(vx_uint32) * 3);

	// The amount of k-means iterations. 10 is enough for good start.
	const vx_uint32 iterLimit = 10;

	for (vx_uint32 iter = 0; iter < iterLimit; iter++) {
		memset(pxCount, 0, sizeof(vx_uint32) * K);
		memset(pxSum, 0, sizeof(vx_uint32) * 3 * K);
		for (vx_uint32 i = 0; i < N; i++) {
			if (matte[i] == matteClass) {
				vx_uint32 bestCluster = 0; // The closest
                const vx_uint8 *cur_px = (const vx_uint8*)(px + i);
                vx_float64 minDist = sqrt(euclidian_dist_if(cur_px, centroids));
				for (vx_uint32 j = 1; j < K; j++) {		// Search for the best cluster
					vx_float64 d = sqrt(euclidian_dist_if(cur_px, centroids + j * 3));
					if (d < minDist) {
						bestCluster = j;
						minDist = d;
					}
				}
				gmm_index[i] = bestCluster;
				pxSum[bestCluster * 3 + 0] += px[i].b;
				pxSum[bestCluster * 3 + 1] += px[i].g;
				pxSum[bestCluster * 3 + 2] += px[i].r;
				pxCount[bestCluster]++;
			}
		}

        // Looking for empty clusters
        for (vx_uint32 i = 0; i < K; i++) {
            if (pxCount[i] > 0) {
                continue;
            }

            vx_uint32 maxCluster = 0;
            vx_uint32 maxCnt = 0;
            for (vx_uint32 j = 0; j < K; j++) {     // Search for the most fat cluster
                if (pxCount[j] > maxCnt) {
                    maxCnt = pxCount[j];
                    maxCluster = j;
                }
            }

            vx_uint32 farthestPx = 0;
            vx_float64 maxDist = 0;
            for (vx_uint32 j = 0; j < N; j++) {         // And move the farthest pixel to the empty
                if (matte[j] == matteClass && gmm_index[j] == maxCluster) {
                    vx_float64 d = euclidian_dist_if((vx_uint8*)(px + j), centroids + maxCluster * 3);
                    if (d > maxDist) {
                        maxDist = d;
                        farthestPx = j;
                    }
                }
            }

            pxCount[maxCluster]--;
            pxCount[i]++;
            gmm_index[farthestPx] = i;
        }

		for (vx_uint32 i = 0; i < K; i++) {
			// Move centroids to the mass center of clusters
            centroids[i * 3 + 0] = (vx_float64)pxSum[i * 3 + 0] / pxCount[i];
            centroids[i * 3 + 1] = (vx_float64)pxSum[i * 3 + 1] / pxCount[i];
            centroids[i * 3 + 2] = (vx_float64)pxSum[i * 3 + 2] / pxCount[i];
		}
	}

	free(dists);
	free(pxCount);
	free(pxSum);
	free(centroids);
}

void assignGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px, vx_uint32 *gmm_index,
                const GmmComponent *gmm, const vx_uint8 *matte, vx_uint8 matteClass) {
    vx_uint32 *cnt = (vx_uint32*)calloc(K, sizeof(vx_uint32));
    memset(cnt, 0, K * sizeof(vx_uint32));
    for (vx_uint32 i = 0; i < N; i++) {
        if (matte[i] & matteClass) {
            const vx_RGB_color *color = px + i;
            vx_uint32 min_comp = 0;
            vx_float64 min = computeGmmComponentDataTerm(gmm, color);
            for (vx_uint32 j = 0; j < K; j++) {
                vx_float64 D = computeGmmComponentDataTerm(gmm + j, color);
                if (D < min) {
                    min = D;
                    min_comp = j;
                }
            }
            gmm_index[i] = min_comp;
            cnt[min_comp]++;
        }
    }
    for (vx_uint32 i = 0; i < K; i++) {
        if (cnt[i] > 0) {
            continue;
        }

        vx_uint32 maxComp = 0;
        vx_uint32 maxCnt = 0;
        for (vx_uint32 j = 0; j < K; j++) {
            if (cnt[j] > maxCnt) {
                maxCnt = cnt[j];
                maxComp = j;
            }
        }

        vx_uint32 farthestPx = 0;
        vx_float64 maxVal = 0;
        for (vx_uint32 j = 0; j < N; j++) {
            if (matte[j] == matteClass && gmm_index[j] == maxComp) {
                vx_float64 D = computeGmmComponentDataTerm(gmm + maxComp, px + j);
                if (D > maxVal) {
                    maxVal = D;
                    farthestPx = j;
                }
            }
        }

        cnt[maxComp]--;
        cnt[i]++;
        gmm_index[farthestPx] = i;

    }
}

void learnGMMs(vx_uint32 N, vx_uint32 K, const vx_RGB_color *px,
			   const vx_uint32 *gmm_index, GmmComponent *gmm,
			   const vx_uint8 *matte, vx_uint8 matteClass) {

	// Stores sums of color components in every GMM component
	vx_uint32 *sums = (vx_uint32*)calloc(K * 3, sizeof(vx_uint32));
	// Stores sums of productions of all pairs of color components in every GMM component
	vx_uint32 *prods = (vx_uint32*)calloc(K * 9, sizeof(vx_uint32));
	// Stores the number of pixels in each GMM component
	vx_uint32 *counts = (vx_uint32*)calloc(K, sizeof(vx_uint32));
	// Stores total number of pixels in this GMM
	vx_uint32 counts_total;

	memset(sums, 0, K * 3 * sizeof(vx_uint32));
	memset(prods, 0, K * 9 * sizeof(vx_uint32));
	memset(counts, 0, K * sizeof(vx_uint32));
	counts_total = 0;

	// Accumulating
	for (vx_uint32 k = 0; k < N; k++) {
		if (matte[k] != matteClass) {
			continue;		// Only given matte class
		}
		vx_uint32 comp = gmm_index[k];
		vx_uint8 *color = (vx_uint8*)(px + k);
		for (vx_uint8 i = 0; i < 3; i++) {
			sums[comp * 3 + i] += color[i];
		}
		for (vx_uint32 i = 0; i < 3; i++) {
			for (vx_uint32 j = 0; j < 3; j++) {
				prods[(comp * 9) + (i * 3) + j] += color[i] * color[j];
			}
		}
		counts[comp]++;
		counts_total++;
	}

	// Computing parameters
	for (vx_uint32 comp = 0; comp < K; comp++) {
		GmmComponent *gc = gmm + comp;
		vx_float64 cov[3][3];	// covariance matrix, just local
		for (vx_uint32 i = 0; i < 3; i++) {		// mean colors
			gc->mean[i] = (vx_float64)sums[comp * 3 + i] / counts[comp];
		}
		for (vx_uint32 i = 0; i < 3; i++) {		// covariance matrix
			for (vx_uint32 j = 0; j < 3; j++) {
				cov[i][j] = (vx_float64)prods[(comp * 9) + (i * 3) + j] / counts[comp];
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

		gc->weight = (vx_float64)counts[comp] / counts_total; // component weight (pi)
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

void buildSparse(vx_sparse_matrix *mat, vx_uint32 NNZ, vx_uint32 N) {
	mat->data = (vx_float64*)calloc(NNZ, sizeof(vx_float64));
	mat->nz = (vx_uint32*)calloc(N + 1, sizeof(vx_uint32));
	mat->col = (vx_uint32*)calloc(NNZ, sizeof(vx_uint32));

	memset(mat->data, 0, NNZ * sizeof(vx_float64));
	memset(mat->nz, 0, (N + 1) * sizeof(vx_uint32));
	memset(mat->col, 0, NNZ * sizeof(vx_uint32));
}

void setGraphNLinks(const vx_RGB_color *data, vx_uint32 width,
					vx_uint32 height, const vx_uint8 *matte,
					vx_float64 gamma, vx_float64 beta, vx_sparse_matrix *adj_graph) {

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
				if (matte[i] == matte[other]) {
					const vx_uint8 *otherData = (const vx_uint8*)(data + other);
					vx_uint32 d = euclidian_dist_ii(cur_data, otherData);
					vx_float64 weight = gamma * exp(-beta * d) / (j & 1 ? 1 : sqrt_2);
					adj_graph->data[NNZ_cur] = weight;
					adj_graph->col[NNZ_cur] = other;
					NNZ_cur++;
				}
			}
		}

		adj_graph->col[NNZ_cur] = sink;		// init t-link to sink 
		NNZ_cur++;

		adj_graph->nz[i + 1] = NNZ_cur;
	}

	// init t-links from source
	adj_graph->nz[source + 1] = NNZ_cur + N;
	for (vx_uint32 i = 0; i < N; i++) {
		adj_graph->col[NNZ_cur + i] = i;
	}
}

void destructSparse(vx_sparse_matrix *mat) {
	free(mat->data);
	free(mat->nz);
	free(mat->col);
}

void copySparse(const vx_sparse_matrix *src, vx_sparse_matrix *dst, vx_uint32 rows) {
    for (vx_uint32 i = 0; i < rows + 1; i++) {
        dst->nz[i] = src->nz[i];
    }
    for (vx_uint32 i = 0; i < src->nz[rows]; i++) {
        dst->data[i] = src->data[i];
        dst->col[i] = src->col[i];
    }
}

vx_float64 computeMaxWeight(vx_uint32 N, const vx_sparse_matrix *adj_graph) {
	vx_float64 maxSum = 0;
	for (vx_uint32 i = 0; i < N; i++) { // Search for max links sum in row
		vx_float64 sum = 0;
		for (vx_uint32 j = adj_graph->nz[i]; j < adj_graph->nz[i + 1]; j++) {
			sum += adj_graph->data[j];
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
        result += computeGmmComponentDataTerm(comp, color);
    }
    return result;
}

void setGraphTLinks(vx_uint32 N, vx_uint32 K, const vx_RGB_color *data, const GmmComponent *bgdGMM,
					const GmmComponent *fgdGMM, const vx_uint8 *trimap,
					vx_float64 maxWeight, vx_sparse_matrix *adj_graph) {

	vx_uint32 source = N;
	vx_uint32 sink = N + 1;

	for (vx_uint32 i = 0; i < N; i++) {
		vx_float64 fromSouce = 0;
		vx_float64 toSink = 0;
		switch (trimap[i]) {
		case TRIMAP_BGD:		// Background
			fromSouce = 0;
			toSink = maxWeight;
			break;
		case TRIMAP_FGD:		// Foreground
			fromSouce = maxWeight;
			toSink = 0;
			break;
		case TRIMAP_UNDEF:		// Undefined
			fromSouce = computeGmmDataTerm(K, bgdGMM, data + i);
			toSink = computeGmmDataTerm(K, fgdGMM, data + i);
			break;
		}

		vx_uint32 sourcePos = adj_graph->nz[source] + i;
		vx_uint32 sinkPos = adj_graph->nz[i];
		while (adj_graph->col[sinkPos] != sink) {
			sinkPos++;
		}

		adj_graph->data[sourcePos] = fromSouce;
		adj_graph->data[sinkPos] = toSink;
	}
}

void maxFlow(vx_uint32 N, vx_sparse_matrix *adj, vx_uint8 *matte) {
	const vx_uint32 SOURCE = N;		// The source terminal
	const vx_uint32 SINK = N + 1;	// The sink terminal
	const vx_uint8 TREE_FREE = 0;	// Free pixel
	const vx_uint8 TREE_S = 1;		// Pixel from source tree
	const vx_uint8 TREE_T = 5;		// Pixel from sink tree

	// Stores the labels of the tree to which belongs each pixel
	vx_uint8 *tree = (vx_uint8*)calloc(N + 2, sizeof(vx_uint8));
	// Stores the parent of each pixel in trees (if parent(n) == n then n has no parent)
	vx_uint32 *parent = (vx_uint32*)calloc(N + 2, sizeof(vx_uint32));

	for (vx_uint32 i = 0; i < N + 2; i++) {
		parent[i] = i;			// initially there is no parent for each
		tree[i] = TREE_FREE;	// initially each is free
	}

	tree[SOURCE] = TREE_S;		// initial source tree
	tree[SINK] = TREE_T;		// initial sink tree

	// Indicates whether pixel is active or not
	vx_bool *is_active = (vx_bool*)calloc(N + 2, sizeof(vx_bool));
	memset(is_active, vx_false_e, (N + 2) * sizeof(vx_bool));
	is_active[SOURCE] = vx_true_e;
	is_active[SINK] = vx_true_e;

	// Stores orphan pixels, which require further handle
	vx_uint32 *orphan = (vx_uint32*)calloc(N + 2, sizeof(vx_uint32));
	// Ponts to ghost element after the last in 'orphan' array
	vx_uint32 orphan_end = 0;
	// Stores active pixels, from which the tree will grow
	vx_uint32 *active = (vx_uint32*)calloc((N + 2) * 2, sizeof(vx_uint32));
	active[0] = SOURCE;
	active[1] = SINK;
	// Ponts to ghost element after the last in 'active' array
	vx_uint32 active_end = 2;

	vx_bool done = vx_false_e;
	while (!done) {
		vx_uint32 st_edge = 0;				// Corresponds to the edge, by which trees do intersect
		vx_uint32 s_bound = SOURCE;			// The pixel of this edge on the source side
		vx_uint32 t_bound = SINK;			// The pixel of this edge on the sink side

		//////////////////////
		//// Growth stage
		//////////////////////

		vx_bool pathFound = vx_false_e;
		vx_uint32 cur_a = 0;
		while (cur_a < active_end && !pathFound) {
			vx_uint32 p = active[cur_a];	// Current active pixel
			if (tree[p] == TREE_S) {
				for (vx_uint32 i = adj->nz[p]; i < adj->nz[p + 1] && !pathFound; i++) {
					if (adj->data[i] > EPS) {
						vx_uint32 q = adj->col[i];
						if (tree[q] == TREE_FREE) {
							tree[q] = tree[p];
							parent[q] = p;
							if (!is_active[q]) {
								active[active_end] = q;
								active_end++;
								is_active[q] = vx_true_e;
							}
						}
						else if (tree[q] != tree[p]) {
							s_bound = p;
							t_bound = q;
							st_edge = i;
							pathFound = vx_true_e;
						}
					}
				}
			}
			else if (tree[p] == TREE_T) {
				for (vx_uint32 i = 0; i < N && !pathFound; i++) {
					for (vx_uint32 j = adj->nz[i]; j < adj->nz[i + 1] && !pathFound; j++) {
						if (adj->col[j] == p && adj->data[j] > EPS) {
							vx_uint32 q = i;
							if (tree[q] == TREE_FREE) {
								tree[q] = tree[p];
								parent[q] = p;
								if (!is_active[q]) {
									active[active_end] = q;
									active_end++;
									is_active[q] = vx_true_e;
								}
							}
							else if (tree[q] != tree[p]) {
								s_bound = q;
								t_bound = p;
								st_edge = j;
								pathFound = vx_true_e;
							}
						}
					}
				}
			}
			if (!pathFound) {
				cur_a++;
				is_active[p] = vx_false_e;
			}
		}
		if (!pathFound) {
			done = vx_true_e;
		}
		else {
			vx_uint32 i = 0;
			for (; cur_a < active_end; cur_a++) {
				active[i] = active[cur_a];
				i++;
			}
			active_end = i;

			//////////////////////
			//// Saturation stage
			//////////////////////

			// The max possible flow through whe found path
			vx_float64 bottleneck = adj->data[st_edge];
			vx_uint32 s_i = s_bound;
			while (s_i != SOURCE) {
				vx_uint32 s_parent = parent[s_i];
				for (vx_uint32 i = adj->nz[s_parent]; i < adj->nz[s_parent + 1]; i++) {
					if (adj->col[i] == s_i) {
						if (adj->data[i] < bottleneck - EPS) {
							bottleneck = adj->data[i];
						}
						break;
					}
				}
				s_i = s_parent;
			}
			vx_uint32 t_i = t_bound;
			while (t_i != SINK) {
				vx_uint32 t_parent = parent[t_i];
				for (vx_uint32 i = adj->nz[t_i]; i < adj->nz[t_i + 1]; i++) {
					if (adj->col[i] == t_parent) {
						if (adj->data[i] < bottleneck - EPS) {
							bottleneck = adj->data[i];
						}
						break;
					}
				}
				t_i = t_parent;
			}

			// Bottleneck is found now we should push this flow through the path

			adj->data[st_edge] -= bottleneck;

			s_i = s_bound;
			while (s_i != SOURCE) {
				vx_uint32 s_parent = parent[s_i];
				for (vx_uint32 i = adj->nz[s_parent]; i < adj->nz[s_parent + 1]; i++) {
					if (adj->col[i] == s_i) {
						adj->data[i] -= bottleneck;
						if (adj->data[i] < EPS) {
							parent[s_i] = s_i;
							orphan[orphan_end] = s_i;
							orphan_end++;
						}
						break;
					}
				}
				s_i = s_parent;
			}
			t_i = t_bound;
			while (t_i != SINK) {
				vx_uint32 t_parent = parent[t_i];
				for (vx_uint32 i = adj->nz[t_i]; i < adj->nz[t_i + 1]; i++) {
					if (adj->col[i] == t_parent) {
						adj->data[i] -= bottleneck;
						if (adj->data[i] < EPS) {
							parent[t_i] = t_i;
							orphan[orphan_end] = t_i;
							orphan_end++;
						}
						break;
					}
				}
				t_i = t_parent;
			}

			//////////////////////
			//// Adoption stage
			//////////////////////

			for (vx_uint32 orph_cur = 0; orph_cur < orphan_end; orph_cur++) {
				vx_uint32 p = orphan[orph_cur];					// Current orphan
				if (tree[p] == TREE_S) {
					vx_bool parentFound = vx_false_e;
					for (vx_uint32 i = 0; i < N && !parentFound; i++) {
						for (vx_uint32 j = adj->nz[i]; j < adj->nz[i + 1] && !parentFound; j++) {
							if (adj->col[j] == p && adj->data[j] > EPS) {
								vx_uint32 q = i;
								if (tree[q] == tree[p]) {
									vx_uint32 q_i = q;
									while (q_i != SOURCE && parent[q_i] != q_i) {
										q_i = parent[q_i];
									}
									if (q_i == SOURCE) {
										parent[p] = q;
										parentFound = vx_true_e;
									}
								}
							}
						}
					}
					if (!parentFound) {		// No parent is found in original tree
						for (vx_uint32 i = 0; i < N; i++) {
							for (vx_uint32 j = adj->nz[i]; j < adj->nz[i + 1]; j++) {
								if (adj->col[j] == p) {
									vx_uint32 q = i;
									if (adj->data[j] > EPS) {
										if (!is_active[q]) {
											active[active_end] = q;
											active_end++;
											is_active[q] = vx_true_e;
										}
									}
									if (parent[q] == p) {
										parent[q] = q;
										orphan[orphan_end] = q;
										orphan_end++;
									}
								}
							}
						}
						tree[p] = TREE_FREE;
					}
				}
				else if (tree[p] == TREE_T) {
					vx_bool parentFound = vx_false_e;
					for (vx_uint32 i = adj->nz[p]; i < adj->nz[p + 1] && !parentFound; i++) {
						if (adj->data[i] > EPS) {
							vx_uint32 q = adj->col[i];
							if (tree[q] == tree[p]) {
								vx_uint32 q_i = q;
								while (q_i != SINK && parent[q_i] != q_i) {
									q_i = parent[q_i];
								}
								if (q_i == SINK) {
									parent[p] = q;
									parentFound = vx_true_e;
								}
							}
						}
					}
					if (!parentFound) {			// No parent is found in original tree
						for (vx_uint32 i = adj->nz[p]; i < adj->nz[p + 1]; i++) {
							vx_uint32 q = adj->col[i];
							if (adj->data[i] > EPS) {
								if (!is_active[q]) {
									active[active_end] = q;
									active_end++;
									is_active[q] = vx_true_e;
								}
							}
							if (parent[q] == p) {
								parent[q] = q;
								orphan[orphan_end] = q;
								orphan_end++;
							}
						}
						tree[p] = TREE_FREE;
					}
				}
			}
			orphan_end = 0;
		}
	}

	//////////////////////
	//// Cut stage
	//////////////////////

	// Trying to get as far as possible from source

	memset(matte, MATTE_BGD, N * sizeof(vx_uint8));
	vx_uint32 *stack = (vx_uint32*)calloc(N + 2, sizeof(vx_uint32));
	stack[0] = SOURCE;
	vx_uint32 stack_top = 1;

	while (stack_top) {
		stack_top--;
		vx_uint32 node = stack[stack_top];
		if (node != SOURCE) {
			matte[node] = MATTE_FGD;	// Achievable pixel is foreground
		}
		for (vx_uint32 i = adj->nz[node + 1] - 1; i >= adj->nz[node]; i--) {
			vx_uint32 p = adj->col[i];
			if (parent[p] == node) {
				stack[stack_top] = p;
				stack_top++;
			}
		}
	}

	free(stack);
	free(tree);
	free(parent);
	free(active);
	free(orphan);
}
