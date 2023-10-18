# Neural-Rule-Based-model-for-Text-to-SQL
Combination of linguistic rules with pre-trained embedding models for a Text-to-SQL task


## Reproduce Results
 To reproduce our results you must enter the "Run Experiments folder" run the files with different embbeding models which are identified in the paper. It is necessary to change the path to the data so for reproducing the train (pre-selected databases) results the user must change in line 4746  the directory. The output of each run will be a csv file with the "database, question, predicted_SQL, 1 or 0 (Correct or Incorrect).
 Our results are clearly shown in the "Results" folder you can find each result for each similarity metric and if the experiment were conducted on the whole benchmakr or on the pre-selected databases.

## Roadmap
At the momment the text-to-SQl model is not completly optimized for inference. However, in the near future we will create a specific interface and optimize it to work on any database or csv file.
We welcome any suggestions of improving.