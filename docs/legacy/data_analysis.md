# COCO 2017 Validation Dataset Analysis

## Dataset Overview

The COCO 2017 validation split contains:

- 5,000 images
- 80 object categories
- 36,781 object annotations
- 25,014 captions

The dataset was validated against the local image files and annotation
metadata before analysis. No missing images, invalid image references, or
invalid category references were detected.

## Object Distribution

The object annotations are strongly imbalanced.

The most frequent category is `person` with 11,004 annotations, while
`toaster` contains only 9 annotations. This produces a maximum-to-minimum
category frequency ratio of approximately 1222.67:1.

Other frequent categories include `car`, `chair`, `book`, `bottle`, and
`cup`.

This imbalance should be considered when interpreting downstream
category-level evaluation results.

## Scene Complexity

Images contain an average of 7.36 object annotations, with a median of 4.

The distribution is strongly right-skewed:

- P25: 2 objects
- P75: 10 objects
- P95: 23 objects
- Maximum: 63 objects

Manual inspection showed that the maximum-complexity example is a collage
containing many separate food scenes. Therefore, extreme annotation counts
do not necessarily represent ordinary photographic scene complexity.

48 images contain no COCO instance annotations. These images may still
contain visually meaningful content that falls outside the instance
annotation ontology.

## Caption Characteristics

The dataset contains approximately five captions per image.

Caption length statistics:

- Mean: 10.44 words
- Median: 10 words
- P95: 15 words
- Maximum: 50 words

Most captions are therefore short natural-language descriptions rather
than detailed scene descriptions.

## Image Geometry

The dataset contains:

- 3,588 landscape images
- 1,207 portrait images
- 205 approximately square images

The median aspect ratio is 1.33, while observed aspect ratios range from
0.38 to 4.03.

This variation is important for image preprocessing because resizing all
images directly to a fixed square resolution may introduce geometric
distortion.

## Qualitative Findings

Manual inspection identified several important characteristics.

1. Caption content and instance annotations are not equivalent. Captions
   may describe visually important concepts that are not represented by
   COCO object annotations.

2. Images with zero instance annotations can still contain meaningful
   visual content.

3. Extremely high object counts can result from unusual image structures,
   such as collages, rather than naturally dense scenes.

4. Multiple captions for the same image may emphasize different objects
   or interpretations of the scene.

5. Large variation in image aspect ratio must be considered when designing
   preprocessing and batching strategies.

## Implications for the Text-to-Image Pipeline

The analysis suggests that preprocessing should preserve image geometry
where possible rather than applying uncontrolled stretching.

Caption supervision should be treated independently from object
annotations because the two annotation sources encode different aspects
of image content.

Category imbalance and scene-complexity variation should also be considered
when constructing evaluation subsets and interpreting downstream results.