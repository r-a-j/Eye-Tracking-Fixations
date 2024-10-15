# Eye-Tracking Case Study

**Winter Semester 2024-2025**  
*Department of Statistics, TU Dortmund*  

---

## Project Overview

Eye-tracking is a fascinating technique used to measure where and how long people (or animals) focus their gaze within their visual environment. By tracking eye movements, we can gain insights into human cognition, behavior, and decision-making processes, which have applications in a wide variety of fields including medicine, marketing, education, and more.

In this case study, we will use a publicly available dataset from the Nemo Science Museum (Amsterdam, Netherlands) to explore key research questions regarding how people’s eye movements may vary based on age and how these patterns evolve over time during the experiment.

## Goals

Our analysis focuses on answering two primary research questions:

1. **Age and Eye Movements**: Do fixation durations (time spent looking at a specific point) decrease with age during childhood, then increase later in life?
   
2. **Fixation Patterns Over Time**: Are there any systematic patterns in fixation durations throughout the experiment? Do fixations get shorter or longer, or follow a more complex pattern?

## Data

The data used in this project were collected from participants at the Nemo Science Museum, who viewed a feature-rich image for 10 seconds while their gaze was recorded. The dataset contains over 2,600 participants and is available [here](https://osf.io/sk4fr/).

### Data Structure

The dataset is divided into three main types:

- **Pre-processed data**: Gaze positions (x, y coordinates), pupil size, timestamps, and experimental event messages.
  
- **Processed data**: Information on eye fixations (average x, y coordinates and duration of each fixation).

- **Demographics data**: Participant information such as gender and year of birth.

## Methodology

Used statistical analysis methods to assess the research questions. The scope of analysis ranged from simple descriptive statistics to more complex inferential methods.

The analysis consists of:

1. **Data Preprocessing**: Converted the pre-processed data into fixation data. This is crucial, as fixations are our main unit of analysis.
   
2. **Statistical Analysis**: Used appropriate methods to investigate the two main research questions.

3. **Bonus - Robustness Analysis**:  
   As an additional challenge, we investigate the robustness of results by modifying the data processing pipeline. For example:
   - Change the hyperparameters in the fixation detection algorithm.
   - Implement a new processing pipeline.
   - Add noise or simulate missing data to test the stability of the results.

## References

- **De Haas et al. (2019)**: Individual differences in visual salience vary along semantic dimensions.
- **Duchowski (2017)**: Eye tracking methodology: Theory and practice.
- **Hessels et al. (2020)**: Task-related gaze control in human crowd navigation.
- **Holmqvist et al. (2011)**: Eye tracking: A comprehensive guide to methods and measures.
- **Nuthmann et al. (2010)**: A computational model of fixation durations in scene viewing.
- **Salvucci & Goldberg (2000)**: Identifying fixations and saccades in eye-tracking protocols.
