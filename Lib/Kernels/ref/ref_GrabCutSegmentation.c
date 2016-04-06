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

/** @brief Initializes matte from the trimap.
	@param [in,out]	p A pointer to all data

*/
void initMatte(param *p);

/** @brief Initialize random generator specifically with source image and user input
	@detailed This function set random generator seed without using time() function, so
		so the seed will be different on the different calls with similar input
	@param [in] p A pointer to all data
*/
void initRnd(const param *p);

/** @brief Initializes GMMs from matte
	@detailed Uses k-means clustering method to divide pixels into K components
		for given matte class well enough. Initial centroids are being selected
		with k-means++ algorithm.
	@param [in,out] p A pointer to all data
	@param [in] matteClass A matte class to initialize corresponding GMM
*/
void initGmmComponents(param *p, int matteClass);

/** @brief Computes all required numerical characteristics of the GMM components.
		Does process only GMM, corresponding to given matte class
	@param [in,out] p A pointer to all data
	@param [in] matteClass A matte class to learn corresponding GMM parameters
*/
void learnGMMs(param *p, int matteClass);

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

	initRnd(&p);
	initMatte(&p);
	initGmmComponents(&p, MATTE_BGD);
	initGmmComponents(&p, MATTE_FGD);

	learnGMMs(&p, MATTE_BGD);
	learnGMMs(&p, MATTE_FGD);

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

void initMatte(param *p) {
	for (vx_uint32 i = 0; i < p->Npx; i++) {
		p->matte[i] = (p->trimap[i] == TRIMAP_BGD) ? MATTE_BGD : MATTE_FGD;
	}
}

void initRnd(const param *p) {
	vx_uint8 *data = (vx_uint8*)p->px;
	vx_uint32 seed = 0;
	for (vx_uint32 i = 0; i < p->Npx; i++) {
		if (p->matte[i] == MATTE_FGD) {
			seed += data[i];
		}
	}
	srand(seed);
}

void initGmmComponents(param *p, int matteClass) {

	////////////////////////////////
	/////////// k-means++ (Initial centroids selection)
	////////////////////////////////

	// Stores squares of distance from each pixel to the closest centroid
	vx_uint32 *dists2 = (vx_uint32*)calloc(p->Npx, sizeof(vx_uint32));
	// Stores coordinates of centroids
	vx_RGB_color *centroids = (vx_RGB_color*)calloc(p->K, sizeof(vx_RGB_color));

	centroids[0] = p->px[rand() % p->Npx]; // first centroid is random
	for (vx_uint32 i = 1; i < p->K; i++) {
		vx_uint32 sum2 = 0;		// Total sum of squared distances
		for (vx_uint32 j = 0; j < p->Npx; j++) {
			if (p->matte[j] == matteClass) {
				dists2[j] = p->dist(p->px + j, centroids); // search for minimal distance
				for (vx_uint32 m = 1; m < i; m++) {
					vx_uint32 d = p->dist(p->px + j, centroids + m);
					if (d < dists2[j]) {
						dists2[j] = d;
					}
				}
				sum2 += dists2[j];
			}
		}
		// Some pixel will be the next centroid with probability
		// proportional to it's squared distance from 'dists' array
		vx_float64 rnd = (vx_float64)rand() / RAND_MAX * sum2; // Continious uniform distribution on [0 sum2)
		vx_float64 nsum = 0; // Current sq sum accumulator
		vx_uint32 j = 0;
		for (; nsum < rnd; j++) {
			if (p->matte[j] == matteClass) {
				nsum += dists2[j];
			}
		}
		// Here j is that random pixel
		centroids[i] = p->px[j];
	}

	////////////////////////////////
	/////////// k-means
	////////////////////////////////

	// Stores numbers of pixels, assigned to each centroid
	vx_uint32 *pxCount = (vx_uint32*)calloc(p->K, sizeof(vx_uint32));
	// Stores sums of pixels, assigned to each centroid
	vx_uint32 *pxSum = (vx_uint32*)calloc(p->K, sizeof(vx_uint32) * 3);

	// The amount of k-means iterations. 5 is enough for good start.
	const vx_uint32 iterLimit = 5;

	for (vx_uint32 iter = 0; iter < iterLimit; iter++) {
		memset(pxCount, 0, sizeof(vx_uint32) * p->K);
		memset(pxSum, 0, sizeof(vx_uint32) * 3 * p->K);
		for (vx_uint32 i = 0; i < p->Npx; i++) {
			if (p->matte[i] == matteClass) {
				vx_uint32 bestCluster = 0; // The closest
				vx_uint32 minDist = p->dist(p->px + i, centroids);
				for (vx_uint32 j = 1; j < p->K; j++) {		// Search for the best cluster
					vx_uint32 d = p->dist(p->px + i, centroids + j);
					if (d < minDist) {
						bestCluster = j;
						minDist = d;
					}
				}
				p->GMM_index[i] = bestCluster;
				pxSum[bestCluster * 3 + 0] += p->px[i].r;
				pxSum[bestCluster * 3 + 1] += p->px[i].g;
				pxSum[bestCluster * 3 + 2] += p->px[i].b;
				pxCount[bestCluster]++;
			}
		}
		for (vx_uint32 i = 0; i < p->K; i++) {
			// Move centroids to the mass center of clusters
			centroids[i].r = (vx_uint8)(pxSum[i * 3 + 0] / pxCount[i]);
			centroids[i].g = (vx_uint8)(pxSum[i * 3 + 1] / pxCount[i]);
			centroids[i].b = (vx_uint8)(pxSum[i * 3 + 2] / pxCount[i]);
		}
	}

	free(dists2);
	free(pxCount);
	free(pxSum);
	free(centroids);
}

void learnGMMs(param *p, int matteClass) {

	// Stores sums of color components in every GMM component
	vx_uint32 *sums = (vx_uint32*)calloc(p->K * 3, sizeof(vx_uint32));
	// Stores sums of productions of all pairs of color components in every GMM component
	vx_uint32 *prods = (vx_uint32*)calloc(p->K * 9, sizeof(vx_uint32));
	// Stores the number of pixels in each GMM component
	vx_uint32 *counts = (vx_uint32*)calloc(p->K, sizeof(vx_uint32));
	// Stores total number of pixels in this GMM
	vx_uint32 counts_total;

	memset(sums, 0, p->K * 3 * sizeof(vx_uint32));
	memset(prods, 0, p->K * 9 * sizeof(vx_uint32));
	memset(counts, 0, p->K * sizeof(vx_uint32));
	counts_total = 0;

	// Choose corresponding GMM
	GmmComponent *gmm = (matteClass == MATTE_BGD) ? p->bgdGMM : p->fgdGMM;

	// Accumulating
	for (vx_uint32 k = 0; k < p->Npx; k++) {
		if (p->matte[k] != matteClass) {
			continue;		// Only given matte class
		}
		vx_uint32 comp = p->GMM_index[k];
		vx_uint8 *color = (vx_uint8*)(p->px + k);
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
	for (vx_uint32 comp = 0; comp < p->K; comp++) {
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

vx_uint32 euclidian_dist(const vx_RGB_color *z1, const vx_RGB_color *z2) {
	vx_uint8 d1 = (z1->r - z2->r > 0) ? z1->r - z2->r : z2->r - z1->r;
	vx_uint8 d2 = (z1->g - z2->g > 0) ? z1->g - z2->g : z2->g - z1->g;
	vx_uint8 d3 = (z1->b - z2->b > 0) ? z1->b - z2->b : z2->b - z1->b;
	return d1*d1 + d2*d2 + d3*d3;
}