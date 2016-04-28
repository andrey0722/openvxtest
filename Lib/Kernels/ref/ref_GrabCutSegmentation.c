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
	vx_sparse_matrix adj_graph;

	initRnd(N, px, matte);
	initMatte(N, trimap_data, matte);
	vx_float64 gamma = 50;
	vx_float64 beta = computeBeta(px, src_width, src_height);
	setGraphNLinks(px, src_width, src_height, matte, gamma, beta, &adj_graph);
	vx_float64 maxWeight = computeMaxWeight(N, &adj_graph);
	initGmmComponents(N, K, px, GMM_index, matte, MATTE_BGD);
	initGmmComponents(N, K, px, GMM_index, matte, MATTE_FGD);

    assignGMMs(N, K, px, GMM_index, bgdGMM, matte, MATTE_BGD);
    assignGMMs(N, K, px, GMM_index, fgdGMM, matte, MATTE_FGD);
	learnGMMs(N, K, px, GMM_index, bgdGMM, matte, MATTE_BGD);
	learnGMMs(N, K, px, GMM_index, fgdGMM, matte, MATTE_FGD);
	setGraphTLinks(N, K, px, bgdGMM, fgdGMM, trimap_data, maxWeight, &adj_graph);

	vx_RGB_color* dst_data = (vx_RGB_color*)dst_image->data;
	for (vx_uint32 i = 0; i < N; i++) {
		dst_data[i] = px[i]; // Just copy
	}

	destructSparse(&adj_graph);
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
		// proportional to it's squared distance from 'dists' array
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

	// The amount of k-means iterations. 5 is enough for good start.
	const vx_uint32 iterLimit = 5;

	for (vx_uint32 iter = 0; iter < iterLimit; iter++) {
		memset(pxCount, 0, sizeof(vx_uint32) * K);
		memset(pxSum, 0, sizeof(vx_uint32) * 3 * K);
		for (vx_uint32 i = 0; i < N; i++) {
			if (matte[i] == matteClass) {
				vx_uint32 bestCluster = 0; // The closest
                const vx_uint8 *cur_px = (const vx_uint8*)(px + i);
                vx_float64 minDist = sqrt(euclidian_dist_if(cur_px, centroids));
				for (vx_uint32 j = 1; j < K; j++) {		// Search for the best cluster
					vx_float64 d = sqrt(euclidian_dist_if(cur_px, centroids + j));
					if (d < minDist) {
						bestCluster = j;
						minDist = d;
					}
				}
				gmm_index[i] = bestCluster;
				pxSum[bestCluster * 3 + 0] += px[i].r;
				pxSum[bestCluster * 3 + 1] += px[i].g;
				pxSum[bestCluster * 3 + 2] += px[i].b;
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
				cov[i][j] = prods[(comp * 9) + (i * 3) + j] / counts[comp];
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

vx_float64 computeBeta(const vx_RGB_color *data, vx_uint32 width, vx_uint32 height) {
	vx_uint32 sum = 0;
	vx_uint32 count = 0;
	for (vx_uint32 i = 0; i < height; i++) {
		for (vx_uint32 j = 0; j < width; j++) {
            const vx_uint8 *current = (const vx_uint8*)(data + i * width + j);
            if (j < width - 1) {
                sum += euclidian_dist_ii(current, current + 1 * 3); // right
                count++;
            }
            if (i < height - 1) {
                if (j > 0) {
                    sum += euclidian_dist_ii(current, current + (width - 1) * 3); // bottom-left
                    count++;
                }
                sum += euclidian_dist_ii(current, current + width * 3); // bottom
                count++;
                if (j < width - 1) {
                    sum += euclidian_dist_ii(current, current + (width + 1) * 3); // bottom-right
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

	vx_uint32 NNZ_total = (4 * width * height - 3 * (width + height) - 1) * 2 + 2 * N;
	buildSparse(adj_graph, NNZ_total, N + 1);

	vx_uint32 NNZ_cur = 0;
	for (vx_uint32 i = 0; i < N; i++) {
		vx_uint32 row = i / width;
        vx_uint32 col = i % width;
        const vx_uint8 *cur_data = (const vx_uint8*)(data + i);

        if (row > 0) {		// top side
            if (col > 0) {
                vx_uint32 other = i - width - 1;	// top-left
                if (matte[i] == matte[other]) {
                    vx_uint32 other_pos = 0;
                    while (adj_graph->col[adj_graph->nz[other] + other_pos] != i) {
                        other_pos++;
                    }
                    vx_float64 weight = adj_graph->data[adj_graph->nz[other] + other_pos];
                    adj_graph->data[NNZ_cur] = weight;
                    adj_graph->col[NNZ_cur] = other;
                    NNZ_cur++;
                }
            }

            vx_uint32 other = i - width;	// top
            if (matte[i] == matte[other]) {
                vx_uint32 other_pos = 0;
                while (adj_graph->col[adj_graph->nz[other] + other_pos] != i) {
                    other_pos++;
                }
                vx_float64 weight = adj_graph->data[adj_graph->nz[other] + other_pos];
                adj_graph->data[NNZ_cur] = weight;
                adj_graph->col[NNZ_cur] = other;
                NNZ_cur++;
            }

            if (col < width - 1) {
                vx_uint32 other = i - width + 1;	// top-right
                if (matte[i] == matte[other]) {
                    vx_uint32 other_pos = 0;
                    while (adj_graph->col[adj_graph->nz[other] + other_pos] != i) {
                        other_pos++;
                    }
                    vx_float64 weight = adj_graph->data[adj_graph->nz[other] + other_pos];
                    adj_graph->data[NNZ_cur] = weight;
                    adj_graph->col[NNZ_cur] = other;
                    NNZ_cur++;
                }
            }
        }
        if (col > 0) {
            vx_uint32 other = i - 1;	// left
            if (matte[i] == matte[other]) {
                vx_uint32 other_pos = 0;
                while (adj_graph->col[adj_graph->nz[other] + other_pos] != i) {
                    other_pos++;
                }
                vx_float64 weight = adj_graph->data[adj_graph->nz[other] + other_pos];
                adj_graph->data[NNZ_cur] = weight;
                adj_graph->col[NNZ_cur] = other;
                NNZ_cur++;
            }
        }

        if (col < width - 1) {
            vx_uint32 other = i + 1;	// right
            if (matte[i] == matte[other]) {
                const vx_uint8 *otherData = (const vx_uint8*)(data + other);
                vx_float64 weight = gamma * exp(-beta * euclidian_dist_ii(cur_data, otherData));
                adj_graph->data[NNZ_cur] = weight;
                adj_graph->col[NNZ_cur] = other;
                NNZ_cur++;
            }
        }
        if (row < height - 1) {		// bottom side
            if (col > 0) {
                vx_uint32 other = i + width - 1;	// bottom-left
                if (matte[i] == matte[other]) {
                    const vx_uint8 *otherData = (const vx_uint8*)(data + other);
                    vx_float64 weight = gamma * exp(-beta * euclidian_dist_ii(cur_data, otherData)) / sqrt_2;
                    adj_graph->data[NNZ_cur] = weight;
                    adj_graph->col[NNZ_cur] = other;
                    NNZ_cur++;
                }
            }

            vx_uint32 other = i + width;	// bottom
            if (matte[i] == matte[other]) {
                const vx_uint8 *otherData = (const vx_uint8*)(data + other);
                vx_float64 weight = gamma * exp(-beta * euclidian_dist_ii(cur_data, otherData));
                adj_graph->data[NNZ_cur] = weight;
                adj_graph->col[NNZ_cur] = other;
                NNZ_cur++;
            }

            if (col < width - 1) {
                vx_uint32 other = i + width + 1;	// bottom-right
                if (matte[i] == matte[other]) {
                    const vx_uint8 *otherData = (const vx_uint8*)(data + other);
                    vx_float64 weight = gamma * exp(-beta * euclidian_dist_ii(cur_data, otherData)) / sqrt_2;
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