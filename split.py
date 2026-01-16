import splitfolders
splitfolders.ratio("/home/brook/VSCode/MakeUofT/Software/Dataset/garbage_classification", output="garbage-big", seed=1337, ratio=(0.8, 0.2))
#Switch the path to the dataset path, and run this program to split it into test and val subcategories