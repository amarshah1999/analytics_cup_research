# SkillCorner X PySport Analytics Cup
This repository contains the submission template for the SkillCorner X PySport Analytics Cup **Research Track**. 
Your submission for the **Research Track** should be on the `main` branch of your own fork of this repository.

Find the Analytics Cup [**dataset**](https://github.com/SkillCorner/opendata/tree/master/data) and [**tutorials**](https://github.com/SkillCorner/opendata/tree/master/resources) on the [**SkillCorner Open Data Repository**](https://github.com/SkillCorner/opendata).

## Submitting
Make sure your `main` branch contains:
1. A single Jupyter Notebook in the root of this repository called `submission.ipynb`
    - This Juypter Notebook can not contain more than 2000 words.
    - All other code should also be contained in this repository, but should be imported into the notebook from the `src` folder.
2. An abstract of maximum 500 words that follows the **Research Track Abstract Template**.
    - The abstract can contain a maximum of 2 figures, 2 tables or 1 figure and 1 table.
3. Submit your GitHub repository on the [Analytics Cup Pretalx page](https://pretalx.pysport.org)

Finally:
- Make sure your GitHub repository does **not** contain big data files. The tracking data should be loaded directly from the [Analytics Cup Data GitHub Repository](https://github.com/SkillCorner/opendata).For more information on how to load the data directly from GitHub please see this [Jupyter Notebook](https://github.com/SkillCorner/opendata/blob/master/resources/getting-started-skc-tracking-kloppy.ipynb).
- Make sure the `submission.ipynb` notebook runs on a clean environment.

_⚠️ Not adhering to these submission rules and the [**Analytics Cup Rules**](https://pysport.org/analytics-cup/rules) may result in a point deduction or disqualification._

---

## Abstract

#### Introduction

This research looks to give a mathematical, objective approach to defensive positioning. There are numerous surfaces freely available such as Pitch Control, Dangerous Accessible Space, Expected Threat that can be computed for an entire pitch to associate a “score” with a defensive setup. There are not, to my knowledge, efficient techniques for optimising player positioning to maximise one, or multiple of these objectives. 

This research aims to put together a completely generalisable framework for finding the optimal defensive player positions for any arbitrary complex surface, provided it can be computed from a tracking frame. Additionally, this framework allows a coach to input a tactic, such as prioritising pressure on the opposition attackers and risking space in behind, and see where their players should be in order to objectively optimise that risk-reward tradeoff.

#### Methods

Machine Learning methods are effective at learning patterns from large samples of football (tracking) data, but suffer from two practical limitations.

1. They require large amounts of clean, labelled data
2. They’re difficult to interpret from a tactical point of view

In this competition, given we only have access to ten games worth of tracking data, we instead will use Mathematical Optimisation based methods, as these work with no training data, and are not only interpretable but controllable such that a tactical blueprint can be an input to any model.

Since the surfaces we would be looking to optimise are geospatial and non-linear (Pitch Control, Pressure, Center-Back distances, Dangerous Accessible Space), an exact method such as Linear Programming would not be possible. 

Instead, we opt to use Simulated Annealing, an optimisation method that involves randomly perturbing the system and measuring the change in objective function. In practical terms, we move a player by a small amount (0.5 yards, say), and measure how the entire team’s “score” changes in response to that move. After a few thousand iterations of this, we would have moved the entire team to maximise some objective. This can effectively capture the impact of multi-player movements that human coaches may not think of.

#### Results

We were able to build a system capable of performing simulated annealing and randomly moving players in order to improve 2 objectives simultaneously - xT Weighted Pitch Control and Average Pressure. This system is completely generalisable to any metric, or combination of metrics, and therefore has great potential for a number of applications as part of a team's data toolbox.

#### Conclusion

This research shows promise - we were able to use an optimisation based approach that required zero training data to create a powerful engine that can improve any arbitrary metric, as well as allow trade-offs between metrics, but further work is needed. The main limitation to this approach is that we have not yet found a metric that truly captures everything about defensive position, and minimizing DAS could be a good next step. Additionally, there may be other suitable applications for optimisation based approaches, such as attacking positioning.
<figure>
  <img width="2472" height="504" alt="image" src="https://github.com/user-attachments/assets/84cafca6-ee2a-4b4f-bf7e-b5391e8712cc" />
  <figcaption>Fig. 1 - From Left to Right: The original position of the Red Team defending, the position optimised for xT Weighted Pitch Control (xTWPC), and the position optimised for both xTWPC and Pressure</figcaption>
</figure>


