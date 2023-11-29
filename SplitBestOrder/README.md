Usage & Code Convention
=======================
1. All your code should be written in the src  directory.
2. As a convenience, we automatically dumped 50 rows of your data.
3. If you want to change the amount of data, you must dump it again. Open a new terminal and write: `download_data --help` to get details of that command, and e.g. `download_data --limit 100` to download a sample limiting it to 100 rows per table. If you want to download all of your data, just type `download_data`, with no arguments.
4. We will automatically pass Jedi (AI Code Assistant) your table Schema and File information. You can ask Jedi information about them. For example "Give me my table Schema".
5. To manipulate your data you will need some basic knowledge in [Pandas](https://pandas.pydata.org/pandas-docs/version/1.4.3/) or [Dask](https://docs.dask.org/en/stable/).
6. Head to main.py in the src directory for an example file with helper functions.
7. Once you finish writing your code you will need to push your code to git and make sure you have an output table by using one of our helper functions in the main.py example.
8. We automatically activated Jedi (AI Code Assistant) in your IDE. Just select the Yoda icon in the IDE menu and start asking it questions.
More Information
================
This file is far from complete in terms of documentation. For a full documentation of the usage of this operator and its associated IDE, take a look at [this page](https://proud-botany-7dd.notion.site/Code-Operator-IDE-3c5978b7435040ad9609eb93ba639992?pvs=4).