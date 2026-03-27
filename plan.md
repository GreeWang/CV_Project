# Project 3 Implementation Plan

## 1. Project Goal

The goal of this project is to build a complete video object removal and inpainting pipeline.  
Given a video containing dynamic objects such as pedestrians, bicycles, or vehicles, the system should:

1. Detect and segment dynamic objects automatically.
2. Distinguish truly moving objects from static objects.
3. Generate temporally consistent masks across frames.
4. Remove the selected objects and restore the background.
5. Evaluate the method with quantitative metrics and qualitative comparisons.

This project should be organized into three parts required by the assignment:

1. Part 1: Classical baseline pipeline
2. Part 2: SOTA reproduction pipeline
3. Part 3: One optimization or extension attempt

---

## 2. Deliverables

The final submission must include:

1. `Public GitHub repository`
   The repository should contain source code, dependency instructions, usage examples, model weights information, and visualization results in the README.
2. `PDF report`
   Use the CVPR LaTeX template. The report should be 6 to 8 pages excluding references.
3. `Demo videos`
   Processed videos for all mandatory datasets must be exported and packed into `videos.zip`.

---

## 3. Recommended Overall Strategy

To reduce risk and keep the project manageable, the implementation should follow a progressive three-stage strategy:

1. `Part 1`
   Build a classical baseline using object segmentation, motion analysis, and traditional inpainting.
2. `Part 2`
   Replace the classical modules with stronger video segmentation and video inpainting methods.
3. `Part 3`
   Add one focused improvement on mask quality, temporal consistency, or difficult background reconstruction.

Recommended practical combination:

1. `Part 1`
   `YOLOv8-Seg + Lucas-Kanade sparse optical flow + mask dilation + cv2.inpaint + temporal background copy`
2. `Part 2`
   `SAM 2 + ProPainter`
3. `Part 3`
   `Mask refinement and temporal consistency optimization`

Why this is recommended:

1. YOLOv8-Seg is relatively easy to deploy and is suitable for a fast baseline.
2. SAM 2 is a more realistic reproduction target than very recent methods like VGGT4D or SAM 3.
3. ProPainter is explicitly recommended in the project description and is strong for video inpainting.
4. Mask refinement is a safer Part 3 direction than a full generative pipeline.

---

## 4. System Decomposition

The project should be divided into the following modules:

1. `Data preparation`
   Organize videos, extract frames, resize data, and manage input/output folders.
2. `Segmentation`
   Generate candidate object masks for each frame.
3. `Motion analysis`
   Judge whether an object is dynamic or static.
4. `Mask post-processing`
   Apply dilation, hole filling, temporal smoothing, and consistency refinement.
5. `Inpainting`
   Perform object removal and background restoration.
6. `Evaluation`
   Compute JM, JR, PSNR, and SSIM when ground truth is available.
7. `Visualization`
   Save side-by-side results, mask overlays, and selected frame comparisons.
8. `Experiment management`
   Keep configs, logs, outputs, and checkpoints organized.

---

## 5. Suggested Project Structure

```text
CV_project/
├─ plan.md
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ part1.yaml
│  ├─ part2_sam2_propainter.yaml
│  └─ part3_explore.yaml
├─ data/
│  ├─ raw/
│  │  ├─ wild_video/
│  │  ├─ bmx-trees/
│  │  ├─ tennis/
│  │  └─ davis/
│  ├─ frames/
│  ├─ masks/
│  └─ outputs/
├─ checkpoints/
├─ scripts/
│  ├─ prepare_video.py
│  ├─ run_part1.py
│  ├─ run_part2.py
│  ├─ run_part3.py
│  ├─ evaluate_masks.py
│  ├─ evaluate_video.py
│  └─ make_visualizations.py
├─ src/
│  ├─ data/
│  ├─ segmentation/
│  ├─ motion/
│  ├─ inpainting/
│  ├─ evaluation/
│  └─ utils/
└─ results/
   ├─ part1/
   ├─ part2/
   ├─ part3/
   └─ figures/
```

---

## 6. Part 1 Plan: Classical Baseline

### 6.1 Objective

Build a full object removal pipeline using classical computer vision logic:

1. detect candidate objects
2. determine whether they are dynamic
3. generate masks
4. remove them with traditional inpainting

### 6.2 Segmentation Module

Recommended model:

1. `YOLOv8-Seg`

Alternative:

1. `Mask R-CNN`

Suggested dynamic object classes:

1. person
2. bicycle
3. motorcycle
4. car
5. bus
6. truck

Output per frame:

1. bounding boxes
2. instance masks
3. class labels
4. confidence scores

### 6.3 Dynamic Object Decision

Detection alone is not enough because some objects may be present but static.  
To avoid removing static objects, add a motion-based filtering stage.

Implementation:

1. Detect feature points inside each instance mask.
2. Track them between neighboring frames using Lucas-Kanade sparse optical flow.
3. Compute motion magnitude statistics such as mean displacement or median displacement.
4. Keep only objects whose motion exceeds a threshold.

Recommended improvements:

1. `Global motion compensation`
   If the camera moves, estimate background motion and measure relative object motion.
2. `Temporal persistence`
   Require the object to be dynamic for several consecutive frames before removal.

### 6.4 Mask Post-Processing

Apply post-processing to make masks usable for inpainting:

1. binary dilation
2. hole filling
3. removal of tiny connected components
4. temporal smoothing across frames

Purpose:

1. cover motion blur and mask boundary errors
2. reduce temporal flicker
3. improve inpainting quality

### 6.5 Inpainting Module

Base implementation:

1. `cv2.inpaint` with Telea
2. `cv2.inpaint` with Navier-Stokes

Improved implementation:

1. first search neighboring frames for clean background pixels at the same location
2. if usable temporal information exists, copy or aggregate those pixels
3. if no valid temporal background is available, fall back to spatial inpainting

Suggested strategy:

1. `Temporal copy first`
2. `Spatial inpainting fallback`

### 6.6 Expected Outcome of Part 1

This part should work reasonably well on simple scenes with:

1. limited object motion
2. relatively static background
3. moderate mask size

Likely weaknesses:

1. blurry background reconstruction
2. poor handling of large occlusions
3. unstable masks in crowded scenes
4. failure under camera motion

---

## 7. Part 2 Plan: SOTA Reproduction

### 7.1 Objective

Replace the classical baseline with a modern AI-driven pipeline that produces:

1. higher-quality masks
2. better temporal consistency
3. better inpainting on large masked areas

### 7.2 Recommended Method Choice

The project requires one dynamic mask extraction method from the provided options.

Recommended choice:

1. `SAM 2`

Backup choice:

1. `Track Anything`

Why SAM 2 is a good choice:

1. it is strong and widely discussed
2. it supports video segmentation
3. it is easier to justify in the report
4. it is more practical than newer and less mature methods

### 7.3 Mask Generation with SAM 2

Implementation steps:

1. set up the official environment and verify that the demo runs
2. prepare video frames in the required format
3. initialize objects using prompts
4. propagate segmentation through the full video
5. export binary masks for each frame

Prompt options:

1. `Manual prompt`
   Start with manual box or point initialization to guarantee a working baseline.
2. `Automatic prompt`
   Use YOLO detections to generate prompt boxes automatically.

Recommended plan:

1. first complete a manual-prompt version
2. if time permits, add automatic prompt generation as an engineering improvement

### 7.4 Inpainting with ProPainter

Implementation steps:

1. set up the official environment
2. run the official demo first
3. inspect required input structure
4. convert your masks into the required format
5. run ProPainter on your own video and masks

Expected input:

1. video frames
2. frame-aligned binary masks

Expected output:

1. restored frames
2. final inpainted video

### 7.5 Part 2 Success Criteria

Part 2 should clearly outperform Part 1 in:

1. mask quality
2. temporal consistency
3. restoration quality in large occluded regions
4. preservation of background textures

---

## 8. Part 3 Plan: Optimization and Extension

### 8.1 Objective

Identify one limitation in the baseline or SOTA pipeline and attempt an improvement.

### 8.2 Recommended Direction

The safest and most practical option is:

1. `Mask refinement and temporal consistency enhancement`

This is recommended because it:

1. has lower engineering risk
2. integrates naturally with Part 2
3. can produce visible qualitative gains
4. is easy to analyze in the report

### 8.3 Possible Implementation Options

#### Option A: Mask refinement

1. refine coarse edges
2. enlarge unstable boundaries
3. smooth masks over time using neighboring frames
4. fuse masks from multiple adjacent frames

#### Option B: Better prompt generation

1. use YOLO boxes as automatic prompts for SAM 2
2. combine motion cues with detection prompts
3. improve initialization for long videos

#### Option C: Generative repair for hard regions

1. identify areas where the background never appears
2. inpaint selected keyframes using a generative image model
3. propagate repaired content to neighboring frames

This is more ambitious but also much riskier.

### 8.4 Minimum Acceptable Outcome for Part 3

Even if the improvement is weak, Part 3 should still include:

1. a clearly stated optimization idea
2. an implementation attempt
3. before-and-after comparisons
4. analysis of why it helped or failed

---

## 9. Dataset Plan

### 9.1 Mandatory Datasets

The project must include:

1. `Wild Video`
   A self-recorded video or generated video containing dynamic objects.
2. `bmx-trees`
3. `tennis`

### 9.2 Optional but Recommended Dataset

1. `DAVIS`

DAVIS is useful because:

1. it is standard in research
2. it provides better evaluation support
3. it helps produce stronger quantitative results

### 9.3 Wild Video Collection Suggestions

To reduce risk, record at least two videos:

1. `Easy scene`
   one moving object, stable background, short duration
2. `Hard scene`
   multiple moving objects or heavier occlusion

Practical recommendations:

1. keep duration around 5 to 10 seconds
2. avoid aggressive camera motion
3. keep object count moderate
4. use 720p or lower if computation is limited

---

## 10. Evaluation Plan

### 10.1 Mask Metrics

Required by the assignment:

1. `JM`
   mean IoU
2. `JR`
   IoU recall

Apply these to:

1. Part 1 masks
2. Part 2 masks
3. Part 3 masks if improved

### 10.2 Video Quality Metrics

Required by the assignment:

1. `PSNR`
2. `SSIM`

Important note:

1. these can only be computed when ground truth clean frames are available
2. for wild videos without ground truth, use qualitative comparison only

### 10.3 Qualitative Evaluation

Prepare visual comparisons for:

1. input frame
2. predicted mask
3. Part 1 result
4. Part 2 result
5. Part 3 result
6. local zoom-in comparisons
7. failure cases

For each scene, select at least:

1. one frame where the object enters
2. one frame with major occlusion
3. one frame after the object leaves

---

## 11. Experiment Plan

### 11.1 Main Experiments

At minimum:

1. `Exp-1`
   Full Part 1 pipeline
2. `Exp-2`
   Full Part 2 pipeline
3. `Exp-3`
   Part 3 improved pipeline

### 11.2 Ablation Studies

Recommended lightweight ablations:

1. with vs without motion-based dynamic filtering
2. with vs without mask dilation and temporal smoothing
3. different mask sources fed into ProPainter

### 11.3 Parameter Study

If time permits, test:

1. motion threshold
2. dilation kernel size
3. input resolution
4. important ProPainter inference settings

Keep this limited. The main goal is to complete the full pipeline, not excessive tuning.

---

## 12. Implementation Order

The project should be built in the following order.

### Stage 1: Build a minimal Part 1 pipeline

1. prepare one short sample video
2. run YOLOv8-Seg
3. export masks
4. run `cv2.inpaint`
5. save the first baseline video

Goal:

1. obtain a visible end-to-end result as early as possible

### Stage 2: Add motion filtering and mask refinement

1. add Lucas-Kanade optical flow
2. filter static objects
3. add mask dilation and smoothing
4. produce the final Part 1 baseline

### Stage 3: Reproduce the SOTA segmentation method

1. install SAM 2 or Track Anything
2. run on a short video first
3. export stable masks

### Stage 4: Integrate ProPainter

1. run official demo
2. adapt your own frame and mask format
3. test one short clip
4. batch process mandatory datasets

### Stage 5: Add Part 3 improvement

1. choose one focused optimization
2. implement it on top of Part 2
3. compare before and after

### Stage 6: Evaluation and visualization

1. compute metrics
2. export tables
3. generate comparison figures

### Stage 7: Report and final packaging

1. write the report
2. finalize README
3. package output videos
4. verify submission completeness

---

## 13. Timeline

### Week 1

1. read the listed papers
2. finalize technical choices
3. set up environments
4. organize mandatory datasets
5. complete a minimal Part 1 result

### Week 2

1. finalize Part 1
2. run SAM 2 or Track Anything
3. run ProPainter demo
4. integrate Part 2

### Week 3

1. implement Part 3
2. compute metrics
3. prepare visualizations
4. write Method and Experiments sections

### Week 4

1. polish the report
2. improve figures and tables
3. clean the repository
4. package the final videos and submission files

---

## 14. Two-Person Work Division

Since the assignment is for two students per group, a practical split is:

### Student A

1. data preparation
2. Part 1 implementation
3. metric computation
4. experiment section of the report

### Student B

1. SAM 2 or Track Anything setup
2. ProPainter integration
3. Part 3 improvement
4. method and related work sections of the report

### Shared tasks

1. wild video collection
2. result selection
3. figure preparation
4. README and final submission checking

---

## 15. Report Writing Plan

### Abstract

Should include:

1. problem statement
2. overall three-part strategy
3. key result summary
4. GitHub repository link

### Introduction

Should explain:

1. why video object removal matters
2. what makes the task difficult
3. why both classical and modern methods are worth comparing
4. what your project contributes

### Related Work

Should cover all references listed in the assignment, grouped into:

1. object detection and segmentation
2. video segmentation and tracking
3. video inpainting

### Method

Recommended structure:

1. Part 1 baseline pipeline
2. Part 2 SOTA pipeline
3. Part 3 optimization

### Experiments

Should contain:

1. datasets
2. implementation details
3. quantitative results
4. qualitative comparisons
5. ablations
6. failure cases

### Conclusion

Should summarize:

1. what worked
2. what still failed
3. future improvement directions

---

## 16. Planned Figures and Tables

Recommended minimum set:

1. `Figure 1`
   Overall pipeline diagram
2. `Figure 2`
   Part 1 versus Part 2 architecture comparison
3. `Figure 3`
   Qualitative comparison across multiple scenes
4. `Figure 4`
   Part 3 before-versus-after comparison
5. `Table 1`
   JM and JR results
6. `Table 2`
   PSNR and SSIM results
7. `Table 3`
   Ablation study

---

## 17. Risks and Backup Plans

### Main Risks

1. SOTA environments may be hard to set up.
2. Some datasets may not provide suitable ground truth for all metrics.
3. Wild videos may be too complex and unstable.
4. Part 3 may consume too much time.

### Backup Plans

If `SAM 2` is too difficult:

1. switch to `Track Anything`
2. or keep a stronger engineering pipeline using YOLO-based prompts and tracking

If `ProPainter` is too difficult:

1. switch to `E2FGVI`
2. keep Part 1 fully completed so the project remains complete

If `Part 3` gives weak results:

1. keep the implementation attempt
2. emphasize failure analysis and limitations
3. explain why the improvement did not generalize

If `ground truth` is insufficient:

1. use DAVIS for standard quantitative experiments
2. use mandatory datasets mainly for qualitative results

---

## 18. Minimum Viable Completion Line

If time becomes tight, the minimum acceptable completion should still include:

1. a working full Part 1 pipeline
2. a working Part 2 pipeline on at least one or two mandatory datasets
3. one implemented Part 3 attempt
4. processed output videos for all mandatory datasets
5. report figures, tables, and failure analysis

This is the minimum line that should not be compromised.

---

## 19. Immediate Next Steps

The next actions should be:

1. confirm the Part 2 choice as `SAM 2 + ProPainter`
2. create the project folder structure
3. prepare `bmx-trees`, `tennis`, and one easy wild video
4. build the minimal Part 1 pipeline first
5. verify one short clip end to end before moving to larger models

---

## 20. Suggested One-Sentence Project Summary

You can use the following sentence in the report or README:

`We implement a progressive video object removal pipeline consisting of a classical motion-aware baseline, a SOTA segmentation-plus-video-inpainting pipeline, and one exploratory extension for improving mask or restoration quality.`
