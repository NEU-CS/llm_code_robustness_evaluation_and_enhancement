# MBPP datasets
Here is our test dataset MBPP, which includes the original MBPP test dataset (Python) and our manually translated datasets (C++, Java, JavaScript). Each programming language contains a total of 500 entries, with an average of 3 test cases per entry. As we mentioned earlier, different code segments may be suitable for different transformations. As seen in the experimental results directory, statement-based transformations have the most stringent adaptation requirements—only 30+ out of 500 code entries may be suitable—while methods like variable renaming and dead code insertion can generally be adapted to almost all.

# Robust Code Transformation Counts by Programming Language

The following table shows the number of successfully transformed code samples for each programming language and transformation type in the MBPP dataset.

## Summary Table

| Programming Language | Transformation Type               | Count |
|----------------------|-----------------------------------|-------|
| **C++**              | Expression Exchange               | 443   |
|                      | Statement Exchange                | 34    |
|                      | Code Style                        | 500   |
|                      | Dead Code Insertion               | 500   |
|                      | Variable Renaming                 | 499   |
| **Java**             | Expression Exchange               | 417   |
|                      | Statement Exchange                | 188   |
|                      | Code Style                        | 500   |
|                      | Dead Code Insertion               | 500   |
|                      | Variable Renaming                 | 500   |
| **JavaScript**       | Expression Exchange               | 279   |
|                      | Statement Exchange                | 178   |
|                      | Code Style                        | 500   |
|                      | Dead Code Insertion               | 500   |
|                      | Variable Renaming                 | 500   |
| **Python**           | Expression Exchange               | 218   |
|                      | Statement Exchange                | 10    |
|                      | Code Style                        | 500   |
|                      | Dead Code Insertion               | 500   |
|                      | Variable Renaming                 | 500   |

## Observations
- **Statement Exchange** is the most restrictive transformation, especially in Python (only 10/500 adaptable).  
- **Code Style**, **Dead Code Insertion**, and **Variable Renaming** achieve nearly 100% adaptability across all languages.  
- **Expression Exchange** varies significantly by language (C++: 443, Python: 218).  
- Java and JavaScript show moderate adaptability for **Statement Exchange** (188 and 178, respectively).  