#### Background Content

"The deep learning based object detection has been widely used. The adversarial attack algorithms reduce the accuracy of the recognition model by adding perturbation noise to the target image to achieve the purpose of hiding sensitive objects. Existing adversarial attack algorithms cannot balance the aggression and imperceptibility issues of adversarial perturbations. To address this problem, an Attention maps guided Adversarial Attack Method for object detection (AAAM) is proposed. This method first designs an initial perturbation region positioning strategy based on the Class Activation Mapping (CAM) algorithm and then uses the RPAttack’s gradient based contribution judgment perturbation region refinement strategy during the adversarial perturbation iteration generation phase, thereby achieving stronger attacks with a smaller perturbation region."

"Attention determines the degree of human focus on external information and processing ability. In the field of computer vision, the attention map can help the model to better understand the image content and focus on the important regions. Adversarial attacks against object detection need to accurately localize the perturbed region, while the CAM algorithm is introduced to extract the attention map of the attacked model or the alternative model [16]. According to the set attention threshold to obtain the perturbed region mask, which can be combined with the localization frame obtained by the model’s prediction on the original image to obtain the initial adversarial perturbed region mask."

"In detail, in phase (1), obtaining the model’s localization frame and attention maps based mask is mainly done by calculating the concatenation set to obtain the initial perturbed region mask. A suitable CAM algorithm can be selected for each object detection model. According to pytorch-gradcam’s [15] suggestion, AblationCAM is selected for the Faster RCNN model and EigenCAM for YOLOv4 [17]. For two configurations, the attention map binarization thresholds can be set to 0.2 and 0.3, respectively. In phase (2), for each Ak iterations, a new patch region is added based on the gradient at that time. When the number of localization frames reaches a minimum of Dk iterations, unimportant perturbed pixels are removed and the increase or decrease is based on the accumulation of gradients on the image. Here Ak and Dk need to be set manually. In this paper we set 30 and 8, respectively. The perturbation area mask strictly limits the distribution area of the adversarial perturbation on the image. The detail algorithm is described in Algorithm 1. There are two main improvements: (1) The extract function is utilized at the beginning of the iteration to extract the localization frame region and the CAM attention map binarization mask as the initial perturbation region, respectively, from the object detection results; (2) The perturbation region of RPAttack is used in the iteration process selection and refinement strategy function refining to dynamically adjust the perturbation region."


#### Algorithm 1 AAAM Algorithm
**Input**: Object detection model function OD; Original image set X; Maximum number of iterations MAXX_ITER; Extraction of the binarized mask function based on the results extract; CAM algorithm function CCCCAM; RPAttack regional refinement strategy refining; Clip function clip
**Output**: Adversarial sample sets Xadv_set 

```
Xadv_set = {};

# Iterate over each original image in the image set x
for each x in X:
    i = 0; p = 0; Xadv = x + p; mask = 0;

    # Extract bounding box and class activation masks
    bmask = extract(OD(x)); 
    cmask = extract(CAM(OD(x))); 
    mask = bmask ∨ cmask;

    while i < MAX_ITER do:
        bn = OD(Xadv);
        
        if bn == 0 then:
            break;
        end

        # Calculate disappearance loss L
        noise = ∇Xadv(L);
        noise = bn × noise;
        p = sign(noise);
        mask = refining(noise, mask);

        # Update adversarial example
        Xadv = Xadv + mask ⨀ p;
        Xadv = clip(Xadv, 0, 255);

        i = i + 1;
    end

    # Add the adversarial example to the set
    Xadv_set = Xadv_set ∪ {Xadv};
end

return Xadv_set;
```


Here's the rewritten algorithm in mathematical form:

---

### Adversarial Point Cloud Generation Algorithm

1. **Initialization**:  
   Define the adversarial set:  
   \[
   \mathcal{X}_{adv} = \{\}
   \]  

2. **Iterate Over Input Point Clouds**:  
   For each input point cloud \( X \in \mathcal{X} \):  
   \[
   P = 0, \quad X_{adv} = X + P
   \]  
   Compute point-wise attention scores using Grad-CAM:  
   \[
   \text{scores} = \text{GradCAM}(M, X)
   \]  
   Normalize the attention scores to create the mask:  
   \[
   \text{mask} = \text{normalize}(\text{scores})
   \]  

3. **Adversarial Iterations**:  
   For \( i = 1 \) to \( \text{MAX\_ITER} \):  
   - Compute classification logits using the model:  
     \[
     \text{logits} = M(X_{adv})
     \]  
   - Check termination condition:  
     If the logits for the target class reach the desired threshold, exit the loop.  

   - Compute the loss:  
     \[
     \mathcal{L} = \text{target\_logit} - \text{source\_logit}
     \]  

   - Compute gradients of the loss with respect to the adversarial point cloud:  
     \[
     \nabla X_{adv} = \frac{\partial \mathcal{L}}{\partial X_{adv}}
     \]  

   - Apply the gradients scaled by the mask:  
     \[
     \Delta P = \text{mask} \odot \nabla X_{adv}
     \]  

   - Refine the perturbation using the 3D RPAttack strategy:  
     \[
     \Delta P = \text{refining}(\Delta P, X_{adv})
     \]  

   - Update the adversarial point cloud:  
     \[
     X_{adv} = X_{adv} + \Delta P
     \]  

   - Clip the adversarial point cloud to valid bounds:  
     \[
     X_{adv} = \text{clip}(X_{adv}, \text{min}, \text{max})
     \]  

4. **Update the Adversarial Set**:  
   Add the adversarial point cloud to the set:  
   \[
   \mathcal{X}_{adv} = \mathcal{X}_{adv} \cup \{ X_{adv} \}
   \]  

5. **Return the Result**:  
   \[
   \text{return } \mathcal{X}_{adv}
   \]

---

This format maintains a mathematical flow while encapsulating the logic of the algorithm. Let me know if you need further clarification!    




### **Pseudocode for the Class Activation Map-based Point Attack 3D (CAMPA-3D)**

---

**Input:**  
- **Model Function:** \( M \) (PointNet)  
- **Point Cloud Set:** \( \mathcal{X} = \{ X_i \} \), \( X_i \in \mathbb{R}^{N \times 3} \) (where \( N \) is the number of points, each with 3D coordinates)  
- **Max Iterations:** \( \text{MAX\_ITER} \)  
- **Attention Map Function:** \( \text{GradCAM} \)  
- **Refinement Strategy Function:** \( \text{refining} \)  
- **Clip Bounds:** \( \text{min}, \text{max} \)  

**Output:**  
- **Adversarial Point Cloud Set:** \( \mathcal{X}_{adv} \)

---

```python
# Initialize adversarial point cloud set
X_adv_set = {}

# Iterate over each point cloud in the dataset
for X in X_set:
    # Initialize variables
    P = 0               # Initial perturbation
    X_adv = X + P       # Adversarial point cloud (starts as the original)
    
    # Generate point-wise attention scores
    scores = GradCAM(M, X)  # Attention map scores for each point
    mask = normalize(scores)  # Normalize attention scores to create a perturbation mask

    # Iteratively apply adversarial perturbations
    for i in range(MAX_ITER):
        # Compute classification logits for the adversarial point cloud
        logits = M(X_adv)
        
        # If target class logits reach the threshold, terminate the attack
        if check_target_class(logits):  # Define this to stop when target class is fooled
            break

        # Compute the loss for the attack (targeting a specific class)
        loss = compute_loss(logits, target_class)  # Example: Cross-entropy loss

        # Backpropagate gradients with respect to point cloud coordinates
        gradients = compute_gradients(loss, X_adv)

        # Compute perturbations scaled by attention mask
        delta_P = mask * gradients  # Element-wise multiplication of mask and gradients

        # Refine perturbations using 3D RPAttack strategy
        delta_P = refining(delta_P, X_adv)

        # Update the adversarial point cloud with perturbations
        X_adv = X_adv + delta_P

        # Clip points to ensure they remain within valid bounds
        X_adv = clip(X_adv, min, max)

    # Add the final adversarial point cloud to the set
    X_adv_set.add(X_adv)

# Return the adversarial point cloud set
return X_adv_set
```

---

---

### Notes

- **Normalization of Scores:**  
  \( \text{mask}_i = \frac{\text{score}_i}{\max(\text{scores})} \) ensures that the mask values are between 0 and 1.  

- **Refinement:**  
  The \( \text{refining} \) function removes perturbations from points with minimal impact, ensuring imperceptibility and efficient perturbation.  

- **Clip Function:**  
  The \( \text{clip} \) function ensures that points do not exceed physical or logical bounds (e.g., LiDAR point range).

This pseudocode provides a structured pipeline for the 3D adversarial attack, adapting the AAAM methodology to point cloud-based models.