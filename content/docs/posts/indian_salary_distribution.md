+++
title = "Distribution for SDE Salaries in India"
description = "Weekend project exploring the distribution of Indian SDE salaries using a GMM-based classification."
tags = [
    "weekend projects"
]
date = "2025-02-22"
categories = [
    "plotnine"
]
menu = "main"
+++

## **Overview**

I recently was having discussions with junior engineers about salary expectations for Software Development Engineers (SDEs) in India, especially how it changes with years of experience. Inspired by [The Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/trimodal-nature-of-tech-compensation) and [@deedydas’ tweet](https://x.com/deedydas/status/1797078188390289892), I decided to examine a dataset of Indian salaries.

My goal was to parse the data, clean it, cluster salary ranges by experience, and visualize how salaries distribute across different “tiers” of companies.

## **Methodology**

1. **Data Collection**  
   - Parse the [raw excel sheet](https://docs.google.com/spreadsheets/d/e/2PACX-1vQO5OJ__99jG2ekwHh_HrLcrgzfZy9x6uOuuW4v1JOtj0607pQbK4Cr8pDC08dVBBRguIP_jxB56Lt-/pubhtml#) into a pandas dataframe using `BeautifulSoup`. 
   Each record contains:
     - Relevant Experience (years)
     - Base Salary
     - Variable Bonus
     - Stock Components

1. **Data Cleaning**  
   - Filtered out missing or non-sensical values (“NULL” or negative).
   - Grouped records by integer years of experience.
   - Computed `totalSalary` as base + (bonus + stocks) for those with 4+ years of experience.  
   - Removed outliers within each experience group by cutting off the lower 4% and upper 4% of salaries.

2. **Categorizing Companies**  
   - For each experience bracket, compute a tri-modal **Gaussian Mixture Model** (GMM) to cluster salaries into **“Low”**, **“Medium”**, and **“High”** categories.

3. **Plotting and Summaries**  
   - Using [plotnine](https://plotnine.org/) (a Python port of **ggplot2**), plot the faceted histogram by years of experience.  
   - Added vertical dashed lines showing mean salaries for each cluster within each experience group.  
   - Labeled each cluster with the sample size for clarity.

## **Output**
![Salary Histogram](https://raw.githubusercontent.com/viig99/viig99.github.io/main/assets/images/salary_histogram.png)
