

# Commonly highlighted issues:
1. The pycache and .virtual_documents/reports are not properly named or .gitignore them.The data folder is getting the unnecessary folder, such as .ipynb_checkpoints folder.
2. The code chunks in the READme.md are normal texts so they are hard to distinguish from normal text. The paragraphs arent well indented. The local run instruction is not so clear. it just says "After activating the environment, you can run the analysis script or Jupyter notebook called bank_marketing_analysis.ipynb" which is ambigious.
3. The function documentation does not suffice. It doesn't have the meaning of parameters passed in. There is no main() function for some of the scripts
4. Reproducibility: I wasn't able to run the docker container with the instructions provided (see error message below):Few minor fixes I've identified, when I tried to run the analysis locally: (1) in creating the environment step, a wrong path to the file is specified. (2) In addition, conda env create and conda activate commands are written on the same line, which can be confusing, so I would recommend separating them out. (3) step 3 about running .ipynb file is no longer applicable, and can be removed (4) command lines for running scripts don't work, I think wrong paths were specified to the script files
5. There is no contribution to the third party software that being used in the analysis, such as python, pandas, etc. in the report.
6. https://github.com/UBC-MDS/BankMarketingPreditions-/blob/main/reports/bank_marketing_analysis.qmd The file indicated that report with the number hardcode, which is not reproducible if the code rerun. For example, the following screen captures shown the Discussion sections with all hardcode results.

# Fixes for Highlighted issues

1. Git ignore created. 