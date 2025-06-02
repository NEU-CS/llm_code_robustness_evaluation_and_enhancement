# Perturbation_pipeline
## we use functions in pipeline.py to transform code snippets
Different transformations have different application conditions: for example, renaming a identifier requires the presence of a variable in a piece of code (which most code satisfies), while converting a for loop to a while loop requires the presence of a for loop in the code (fewer pieces of code may satisfy this).

We categorize all perturbations into 5 groups:  
1. Reformatting  
2. Rename  
3. Expression substitution  
4. Statement substitution  
5. Insertion  

You can see the inner transformation methods of every group in pipeline.py