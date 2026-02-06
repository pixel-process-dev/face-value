# Face Value: Methodoligcal Approach

## Motivation & Goals
Face Value aims to demonstrate an end-to-pipeline for collecting images with minimal manual curation for classification with ecological validity. Classification of emotional expressions often struggles to bridge the gap between controlled datasets and "in the wild" expressions. Reported metrics here inform how this approach performs in multiple scenarios. 

### Multiple Validation
- **Ecological**: facial expressions from well known movies are extracted and classified. Although accuracy on individual faces is not assessed here, clear patterns support that the approach generalizes well. 
- **Standardized**: evaluation against two established datasets of emotional facial expressions (FER 2013, CK+) are used to provide comparison with established metrics. 
- **Performance**: classification metrics, confusion matrices, and performance metrics demonstrate that this process yields stable signal (with minimal curation) for images collected. 

## Design Decisions & Generalizability

### Why Face Detection Was Critical

This approach relied on MediaPipe's face detector for several reasons:

1. **Consistent framing:** Stock photos contain full scenes; face detection 
   isolates the relevant emotional signal
2. **Domain alignment:** Training crops must match inference crops
3. **Noise reduction:** Filters out images with no faces, multiple faces, etc.

**Implication for generalization:** This approach is specific to facial
emotion recognition. Extending to full-body emotions, activities, or 
object states would require different detectors and likely different
keyword strategies.

### Why This Approach May Generalize

The core principle—multi-keyword weak supervision matched to validation
domain—likely transfers to:
- Hand gestures (planned next project)
- Activity recognition
- Object state detection

The validation methodology (temporal patterns in narrative content) could
extend to any domain with interpretable temporal structure.