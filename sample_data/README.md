# Synthetic student-score data

`student_scores.csv` contains 32 hand-created observations with a roughly
linear trend plus small variations. Hours means study hours per day; Scores
means exam points out of 100. No real students or personal records are involved.

The dataset is for learning regression. Its deliberately simple relationship
produces optimistic evaluation results and does not establish real-world
predictive accuracy or a causal effect of studying. It has not been confirmed
as the dataset used in the teacher's videos.

To use your teacher's data, provide a CSV with exactly `Hours,Scores` and run
the training command with `--data`. Inputs need at least 10 rows, five distinct
hour values, varying scores, no missing/nonfinite values and no exact duplicate
rows. Hours must be 0-24 and Scores 0-100.
