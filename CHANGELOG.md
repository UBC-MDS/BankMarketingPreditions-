
# Commonly Highlighted Issues

### 1. `.pycache` and `.virtual_documents/reports` are not properly named or ignored in `.gitignore`. 
   The `data` folder is receiving unnecessary folders, such as the `.ipynb_checkpoints` folder.
   - **URL**: [GitHub - .gitignore](https://github.com/UBC-MDS/BankMarketingPreditions-/blob/main/.gitignore)
   - **Fix**: A `.gitignore` template was used, and additional files were added based on the project requirements.

### 2. Code chunks in the `README.md` are presented as normal text, making them hard to distinguish from regular text. Additionally, the paragraphs are not well indented. The local run instructions are unclear — for example, the line "After activating the environment, you can run the analysis script or Jupyter notebook called `bank_marketing_analysis.ipynb`" is ambiguous.
   - **URL**: [GitHub - README.md](https://github.com/UBC-MDS/BankMarketingPreditions-/blob/main/README.md)
   - **Fix**: Indentation and paragraphing were improved for better clarity.

### 3. Function documentation is insufficient. It lacks descriptions for the parameters being passed, and some scripts do not have a `main()` function.
   - **URL**: [GitHub - Source Code](https://github.com/UBC-MDS/BankMarketingPreditions-/tree/main/src)
   - **Fix**: A `main()` function was added to all scripts, and the documentation for the models was updated to include parameter descriptions.

### 4. Reproducibility issue: I was unable to run the Docker container using the provided instructions (see error message below). A few minor fixes identified during the local run: 
   1. The path to the environment creation file is incorrect.
   2. The `conda env create` and `conda activate` commands are written on the same line, which is confusing. I recommend separating them.
   3. Step 3, which refers to running the `.ipynb` file, is no longer applicable and should be removed.
   - **Fix**: The Docker Compose file was updated to address these issues. As previously mentioned a git ignore was created. And the environment was updated. 

### 5. There is no mention of third-party software used in the analysis (e.g., Python, Pandas) in the report.
   - **URL**: [GitHub - References.bib](https://github.com/UBC-MDS/BankMarketingPreditions-/blob/main/reports/references.bib)
   - **Fix**: The references were updated to include third-party software references

### 6. The report file [bank_marketing_analysis.qmd](https://github.com/UBC-MDS/BankMarketingPreditions-/blob/main/reports/bank_marketing_analysis.qmd) contains hardcoded values (e.g., report numbers), which prevent the code from being reproducible if rerun. For example, the following screen captures show the Discussion sections with hardcoded results.
   - **Fix**: Refactoring of the code is currently in progress to make it more reproducible.

### 7. General
   - **Fix**:Tests: Are there automated tests or manual steps described so that the function of the software can be verified? Are they of sufficient quality to ensure software robsutness?
   #### Tests were written : https://github.com/UBC-MDS/BankMarketingPreditions-/tree/main/test
   #### Data : https://github.com/UBC-MDS/BankMarketingPreditions-/tree/main/data
   Data was made accessible as part of the repository.
Chat gpt was used to modify the markdown code. 