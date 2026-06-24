# Self-Optimal Clustering (SOC) for Image Segmentation

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red)
![Status](https://img.shields.io/badge/Status-Active-success)

# Self-Optimal Clustering (SOC) for Image Segmentation

A Python implementation of the **Self-Optimal Clustering (SOC)** algorithm, an advanced mathematically optimized version of the **Improved Mountain Clustering (IMC)** technique. This repository translates the original MATLAB implementation into a modular, high-performance Python pipeline using **NumPy**, **OpenCV**, and **Scikit-learn**.

The algorithm is primarily designed for robust **color image segmentation**. It dynamically computes the optimal threshold function for clustering using **Lagrange's interpolation polynomial**, maximizing the **Global Silhouette Index (GSI)** to achieve superior cluster compactness and separation compared to traditional **K-Means**, **FCM**, and standard **IMC** methods.

---

## Mathematical Background

Unlike heuristic-based thresholding, SOC analytically optimizes the neighborhood threshold value $\delta_m$ for each cluster.

The potential value of each data point is computed using the **Mountain Function**:
```math
P_{mr}
=
\sum_{j=1}^{n}
\exp\left(
-\frac{d^2(\bar{x}_r,\bar{x}_j)}
{\delta_m^2}
\right)
```
where:

* $P_{mr}$ is the potential associated with data point $r$
* $d(\bar{x}_r,\bar{x}_j)$ is the Euclidean distance between points $r$ and $j$
* $\delta_m$ is the neighborhood threshold parameter

To optimize $\delta_m$, SOC employs **Lagrange interpolation** to model the relationship between threshold values and the resulting cluster quality measured by the **Silhouette Index**.

The interpolation polynomial is defined as:

$$
S_t
===

\sum_{m=1}^{M}
S_m
\prod_{\substack{k=1 \ k \neq m}}^{M}
\frac{(\delta_t-\delta_k)}
{(\delta_m-\delta_k)}
$$

where:

* $S_t$ is the estimated silhouette value at threshold $\delta_t$
* $S_m$ represents the silhouette score corresponding to threshold $\delta_m$
* $M$ is the number of interpolation points

By computing the roots of the derivative of the interpolation polynomial, the algorithm obtains an optimal scaling factor $\beta_m$ that iteratively updates the threshold value toward the maximum achievable **Global Silhouette Index (GSI)**.

This optimization process enables SOC to automatically discover cluster structures with high compactness and strong inter-cluster separation, eliminating the need for manually selected heuristic thresholds.


---

## Repository Structure

The implementation is organized into modular components for maintainability and algorithmic clarity:

### `main.py`

Driver script that:

* Loads images from a local path or URL
* Reshapes image matrices for clustering
* Executes SOC and IMC pipelines
* Visualizes segmentation results using Matplotlib

### `soc.py`

Core implementation of the Self-Optimal Clustering algorithm:

* Spatial normalization
* Mountain function computation
* Cluster center extraction
* Threshold optimization integration

### `factorcal.py`

Optimization engine responsible for:

* Iterative threshold refinement
* Lagrange polynomial fitting
* Scaling factor computation
* Convergence toward optimal cluster quality

### `imc2.py`

Implementation of the baseline **Improved Mountain Clustering (IMC)** algorithm used for benchmarking SOC performance.

### `lagrangepoly.py`

Mathematical utility for:

* Lagrange polynomial construction
* Derivative evaluation
* Root extraction for threshold optimization

### `slht.py`

Computes:

* Individual cluster silhouette values
* Global Silhouette Index (GSI)
* Cluster quality assessment metrics

---

## Prerequisites

Ensure Python 3.7 or later is installed.

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn opencv-python requests
```

---

## Usage

Run the main script:

```bash
python main.py
```

### Execution Flow

1. Enter an image URL or local image path.
2. Specify the desired number of clusters (**M**).
3. The script sequentially executes:

   * SOC
   * IMC1
   * IMC2
4. Global Silhouette Scores (GSI) are reported.
5. Segmentation results are displayed alongside the original image.

Example:

```text
Enter image URL: https://example.com/image.jpg
Enter number of clusters: 5
```

---

## Performance Note

Mountain Clustering has computational complexity:

[
O(N^2)
]

where (N) is the number of data points.

For high-resolution images, consider downsampling before clustering:

```python
image = cv2.resize(image, (width, height))
```

to reduce execution time and memory requirements.

---

## Results

SOC consistently achieves higher cluster compactness and separation by optimizing threshold values through interpolation-based search rather than relying on fixed heuristic parameters.

Performance is evaluated using the **Global Silhouette Index (GSI)** and compared against:

* K-Means
* Fuzzy C-Means (FCM)
* Improved Mountain Clustering (IMC)

---

## Reference

This implementation is based on:

> Verma, N. K., & Roy, A. (2014). *Self-Optimal Clustering Technique Using Optimized Threshold Function*. IEEE Systems Journal, 8(4), 1213–1226.

---

## Author

**Manasvi Gangwar**
Indian Institute of Technology Kanpur (IIT Kanpur)
