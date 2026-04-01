# SUDFace Labels and Annotations

This document summarizes the labels, annotations, and FLAME exports currently available under the local `SUDFace/` directory.

It combines information from:

- `SUDFace/s13428-022-01951-z.pdf`
- `SUDFace/Facial Features and Measurements.xlsx`
- `SUDFace/Expression Changes.xlsx`
- `SUDFace/SUDFace_7_11s_dataset_(Validation_Experiment1-Begining).xlsx`
- `SUDFace/SUDFace_28_32s_dataset_(Validation_Experiment2-Middle).xlsx`
- `SUDFace/video_details.md`
- exported FLAME frame files under `SUDFace/flame/*/*.npz`

## 1. Dataset structure

The SUDFace dataset is organized around videos of the form:

`mov_subj{ID}_{CONDITION}`

Examples:

- `mov_subj1_FS`
- `mov_subj1_NA`
- `mov_subj1_SY`

Condition codes observed in the local dataset:

- `FS`: Free Speech
- `NA`: Natural/Neutral
- `SY`: Synchronized or scripted/read speech

Notes:

- The paper describes three speech conditions for each subject.
- The local `video_details.md` uses the names above for the condition codes.
- The paper text describes two scripted speeches plus one free speech.

## 2. Annotation levels

The available labels fall into three practical levels:

- `video-level`: one record per video
- `clip-level`: one record per short temporal segment of a video
- `frame-level`: one record per frame, from FLAME export

There is also one event-style annotation file for visible expression changes over a temporal interval.

## 3. Video-level metadata

Video-level identifiers and metadata come from the file naming convention and dataset notes.

### Core identifiers

- `subject_id`
- `video_name`
- `condition`

### Technical video metadata

From `SUDFace/video_details.md`:

- `resolution`: 1920 x 1080
- `fps`: about 59.94
- `codec`: H.264
- `pixel_format`: YUV420p, BT.709, 8-bit
- `duration_seconds`: about 60
- `frames_per_video`: about 3600-3620
- `audio`: AAC stereo, 48 kHz
- `recording_date`: 2020-04-27

These are dataset/video descriptors rather than subjective annotations, but they are part of the usable label space.

## 4. Facial feature measurements

Source file:

- `SUDFace/Facial Features and Measurements.xlsx`

This file contains objective face measurements and derived ratios. It is effectively a video-level or subject-video-level annotation table.

### Key columns

- `Subject ID`
- `Video name`

### Raw geometric measurements

- `Face length1`
- `Face lenth2`
- `Nose width1`
- `Nose width2`
- `Nose length1`
- `Nose length2`
- `Forehead length1`
- `Forehead Length2`
- `Chin length1`
- `Chin length2`
- `Eye height1`
- `Eye height2`
- `Eye width1`
- `Eye width2`
- `Face width at mouth 1`
- `Face width at mouth 2`

### Derived ratios / shape descriptors

- `Nose shape`
- `Nose shape2`
- `Chin size 1`
- `Chin size2`
- `Eye shape1`
- `Eye shape2`
- `Eye size 1`
- `Eye size2`
- `Face roundness`
- `Face roundness` (a second column with the same display name)
- `Face All 1`
- `Face All 2`

### Annotator / calculator duplication

The spreadsheet indicates two calculators:

- `RA1`
- `RA2`

In practice, many measurement fields appear as paired columns, one per rater/calculator.

### Interpretation notes

- The spreadsheet includes description rows defining each metric.
- Several labels are duplicated with `1` and `2` suffixes because two raters measured them separately.
- `Face lenth2` is spelled this way in the file and should be treated as the original column name.

## 5. Validation annotations from the paper

The paper states that participants evaluated short clips from each 60-second video on several psychological dimensions.

### Clip windows described in the paper

- `Beginning`: 7-11 seconds
- `Middle`: 28-32 seconds
- `End`: 56-60 seconds

### Validation dimensions described in the paper

- `Neutralness`
- `Naturalness`
- `Valence`
- `Perceived mental state`

### Mental state categories described in the paper

- `Proud`
- `Confused`
- `Bored`
- `Relaxed`
- `Concentrated`
- `Thinking`
- `Stressed`

These are the main subjective annotations for expression quality and perceived state.

## 6. Validation spreadsheets present locally

The local folder currently contains:

- `SUDFace_7_11s_dataset_(Validation_Experiment1-Begining).xlsx`
- `SUDFace_28_32s_dataset_(Validation_Experiment2-Middle).xlsx`

The paper also mentions an end segment, but no corresponding local `.xlsx` file was found for the `56-60s` clip during inspection.

### What is stored in these spreadsheets

These files are Qualtrics-style exports. They contain:

- participant metadata
- consent and demographic fields
- per-stimulus ratings

### Participant / session fields

Examples of metadata columns observed:

- `StartDate`
- `EndDate`
- `Progress`
- `Duration (in seconds)`
- `Finished`
- `RecordedDate`
- `RecipientLastName`
- `RecipientFirstName`
- `RecipientEmail`
- `ExternalReference`
- `DistributionChannel`
- `UserLanguage`
- `Consent`
- browser / OS / resolution metadata
- participant ID field
- gender
- age
- first language

These are useful if you want rater-level analyses, but they are not direct facial labels.

### Per-stimulus annotation schema

For each shown clip, the files repeat the same set of core expression annotations:

- `How neutral is the expression?`
- `Which mental state is the most prominent?`
- `How natural is the expression?`
- `How positive or negative is the expression?`

Operationally, each clip has four fields:

- `neutralness`
- `mental_state`
- `naturalness`
- `valence`

### Rating types

- `neutralness`: Likert-style ordinal rating
- `naturalness`: Likert-style ordinal rating
- `valence`: signed rating with negative to positive range
- `mental_state`: one categorical choice from the seven-state list

### Practical summary

If you aggregate these spreadsheets, the clip-level annotation target space is:

- `clip_id`
- `video_name`
- `clip_window`
- `neutralness`
- `naturalness`
- `valence`
- `mental_state`
- optionally `rater_id` and participant metadata

## 7. Expression change annotations

Source file:

- `SUDFace/Expression Changes.xlsx`

This file is event-based and marks intervals where annotators observed expression changes.

### Columns

- `Rater`
- `MovieID`
- `Onset`
- `Offset`
- `Direction`

### Meaning

- `Rater`: the annotator who marked the event
- `MovieID`: the video identifier, such as `subj1 SY`
- `Onset`: start time or frame index of the change interval
- `Offset`: end time or frame index of the change interval
- `Direction`: direction of the expression change, for example `positive`

### Practical use

This file can be treated as:

- interval annotations over time
- weak supervision for non-neutral expression drift
- a source for filtering frames or clips that depart from strict neutralness

## 8. FLAME frame-level exports

Source:

- `SUDFace/flame/*/*.npz`

Each `.npz` file stores frame-level FLAME or tracking outputs. A sampled archive contains the following arrays.

### FLAME parameter fields

- `shape`
- `exp`
- `head_pose`
- `jaw_pose`
- `eye_pose`
- `tex`
- `light`
- `cam`
- `fov`
- `K`

### Landmark / auxiliary fields

- `lmks_68`
- `lmks_ears`
- `lmks_eyes`
- `blendshape_scores`

### Expected semantics

- `shape`: canonical FLAME identity shape coefficients
- `exp`: FLAME expression coefficients
- `head_pose`: global head rotation / pose parameters
- `jaw_pose`: jaw articulation parameters
- `eye_pose`: eyeball pose parameters
- `tex`: FLAME texture coefficients
- `light`: lighting coefficients
- `cam`: camera parameters
- `fov`: camera field of view
- `K`: camera intrinsic matrix
- `lmks_68`: 68 face landmarks
- `lmks_ears`: ear landmarks
- `lmks_eyes`: eye landmarks
- `blendshape_scores`: Mediapipe-style blendshape scores

### Annotation level

These are frame-level labels rather than human annotations.

They are especially useful when building:

- temporal tracking datasets
- expression regression targets
- pose estimation targets
- multimodal alignment between human ratings and per-frame geometry

## 9. Complete label inventory

Below is the practical combined inventory of labels available from the current local dataset.

### A. Identity and video labels

- `subject_id`
- `video_name`
- `condition`
- `clip_window`
- `frame_id`

### B. Technical metadata

- `resolution`
- `fps`
- `codec`
- `pixel_format`
- `duration_seconds`
- `frames_per_video`
- `audio_format`
- `recording_date`

### C. Objective face measurement labels

- `face_length`
- `nose_width`
- `nose_length`
- `nose_shape`
- `forehead_length`
- `chin_length`
- `chin_size`
- `eye_height`
- `eye_width`
- `eye_shape`
- `eye_size`
- `face_width_at_mouth`
- `face_roundness`
- `face_all`

In the original spreadsheet, these are often duplicated as rater-specific columns such as `...1` and `...2`.

### D. Subjective clip-level expression labels

- `neutralness`
- `naturalness`
- `valence`
- `mental_state`

### E. Mental state class labels

- `proud`
- `confused`
- `bored`
- `relaxed`
- `concentrated`
- `thinking`
- `stressed`

### F. Event-level temporal labels

- `rater`
- `movie_id`
- `onset`
- `offset`
- `direction`

### G. Frame-level FLAME labels

- `shape`
- `exp`
- `head_pose`
- `jaw_pose`
- `eye_pose`
- `tex`
- `light`
- `cam`
- `fov`
- `K`
- `lmks_68`
- `lmks_ears`
- `lmks_eyes`
- `blendshape_scores`

## 10. Recommended unified schema

If you want to merge everything into a single coherent dataset design, the cleanest split is:

### `videos.csv`

One row per video:

- `subject_id`
- `video_name`
- `condition`
- technical metadata
- static facial measurements

### `clip_annotations.csv`

One row per clip or per rater-per-clip:

- `video_name`
- `clip_window`
- `clip_start_sec`
- `clip_end_sec`
- `rater_id`
- `neutralness`
- `naturalness`
- `valence`
- `mental_state`

### `expression_events.csv`

One row per marked interval:

- `movie_id`
- `rater`
- `onset`
- `offset`
- `direction`

### `frame_annotations.parquet` or frame folders

One row per frame or one file per frame:

- `video_name`
- `frame_id`
- all FLAME arrays / derived frame-level features

## 11. Important caveats

- The paper describes three validation clip windows, but only beginning and middle annotation spreadsheets were found locally.
- The local dataset currently contains FLAME exports for a subset of videos, not necessarily all 150 SUDFace videos.
- Some naming conventions differ slightly between files, for example `mov_subj1_NA` versus `subj1 NA`.
- The facial measurement sheet contains duplicated columns per calculator, so downstream processing should preserve rater identity instead of averaging too early.
- The validation spreadsheets are raw survey exports; they will likely need cleaning before direct use in modeling.

## 12. Short summary

From the current local SUDFace folder, you effectively have:

- static face measurement labels
- subjective clip-level expression annotations
- temporal expression-change interval annotations
- frame-level FLAME parameters and landmarks
- video metadata and condition labels

This gives you a combined annotation space spanning morphology, perception, temporal events, and 3D parametric face tracking.
