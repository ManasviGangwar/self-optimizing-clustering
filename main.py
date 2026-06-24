import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import silhouette_samples

from factorcal import factorcal
from soc import soc
from imc2 import imc2
from slht import slht

def main():
    image_path = input("Enter image path (e.g., 'sample.jpg'): ")
    try:
        # Read image and convert BGR to RGB
        a = cv2.imread(image_path)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    f, h, k = a.shape
    nk = int(input("No. of clusters required: "))
    n = f * h
    
    # Reshape image to (n x 3) array
    x = a.reshape((n, 3)).astype(float)
    
    print("Running SOC...")
    fac = factorcal(x, nk, iter_max=10)
    result = soc(x, nk, fac)
    
    print("Running IMC1...")
    res1 = imc2(x, nk, 1.0)
    
    print("Running IMC2...")
    res2 = imc2(x, nk, nk / (nk + 1.0))
    
    print("Calculating metrics...")
    s = silhouette_samples(x, result['idx'])
    S, GSS = slht(s, result['idx'], result['n'], result['m'], nk)
    print(f"GSS (SOC): {GSS}")
    
    s1 = silhouette_samples(x, res1['idx'])
    S1, GSI1 = slht(s1, res1['idx'], res1['n'], res1['m'], nk)
    print(f"GSI1 (IMC1): {GSI1}")
    
    s2 = silhouette_samples(x, res2['idx'])
    S2, GSI2 = slht(s2, res2['idx'], res2['n'], res2['m'], nk)
    print(f"GSI2 (IMC2): {GSI2}")
    
    metrics = [GSS, GSI1, GSI2]
    methods = ['SOC', 'IMC1', 'IMC2']
    results = [result, res1, res2]
    
    plt.figure(figsize=(15, 10))
    
    for m_idx in range(3):
        res = results[m_idx]
        labels = res['idx']
        
        # Compute cluster means
        cmap = np.zeros((nk, 3))
        for v in range(nk):
            pts = x[labels == v]
            if len(pts) > 0:
                cmap[v, :] = np.mean(pts, axis=0)
                
        # Rebuild segmented image
        seg_rgb = cmap[labels]
        seg_img = seg_rgb.reshape((f, h, 3)).astype(np.uint8)
        
        # Plot Original
        plt.subplot(2, 3, m_idx + 1)
        plt.imshow(a)
        plt.title(f"{methods[m_idx]} - Original")
        plt.axis('off')
        
        # Plot Segmented
        plt.subplot(2, 3, m_idx + 4)
        plt.imshow(seg_img)
        plt.title(f"{methods[m_idx]} - Segmented")
        plt.text(10, 20, f"GSI = {metrics[m_idx]:.3f}", 
                 color='yellow', fontsize=12, fontweight='bold', 
                 backgroundcolor='black')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
