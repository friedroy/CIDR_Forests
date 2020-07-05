https://github.cs.huji.ac.il/cidr-center/leads/issues/126
# To Do List 
- [ ] Visualize spatial/temporal correlation in and outside of blocks (this will help decide how much we actually need to "pad" the blocks to remove autocorrelation)
    - [ ] Pearson correlation between the features showed that there is a lot of spatial correlation. Instead, I will try to use [Moran's I](https://en.wikipedia.org/wiki/Moran%27s_I#Defining_weights_matrices)
- [ ] Hyper-parameter search for Decision Trees/Random Forests to reduce overfitting
- [ ] Look into using LIME/SHAP values for interpretability of less classical models



# Pipeline Breakdown
The suggested pipeline for learning the time series can be broken down into the following stages:
1. Read the time series as well as the constant data into two different ```pandas.DataFrame```s; the time series is 
stored in ```df``` and the fixed data is stored in ```fdf```
2. The data will be full of "holes" where the clouds blocked the satellites' view. We chose **not** to interpolate the 
values in the spatial domain, but to do so in the temporal domain. What this means is that if we are missing a value at
some specific time, we insert the last known value in there
3. Aggregate the attributes according to what was discussed (which can be found 
[here](https://docs.google.com/spreadsheets/d/188OjODdWSf7AR1he4f3eu2v0kSG8JEa1_swgaAjGCxQ/edit#gid=0))
4. Create time series tensors which will be easier to break up into feature vectors later. There are created using 
`numpy` and will have the shapes of `[#att, #years, x, y]` for the time series data and shape `[#att, x, y]` for the
fixed data
5. Build feature vectors and target values that can be used to fit the ML models (also saving the names of 
each feature so they can be used for feature extraction later on in the process)
6. Since the data has temporal and spatial auto-correlation we can't (unfortunately) simply use a regression to fit the 
data as is. Instead, the data will be split into temporal and spatial "blocks", separated by some margin. This will 
will make sure that each of these blocks are isolated, which will help give unbiased estimates of the models' accuracy
later down the line
7. Training a model:
    1. Separating these blocks into _k_ folds for model validation while leaving out a portion for test data
    2. Choosing a model using validation on the _k_ folds
    3. Training on full training data and testing on the left out samples
8. Extracting feature importance from the trained model and visualizing this information


*Although this seems like quite a lot of steps, note that steps 1-6 are already finished...*  

### Finished (old) tasks from the To Do List 
- [x] Check how the train/test score of the models is affected by adding more history/spatial information
- [x] Check how the train/test score of the models is affected by choosing different boundary and block sizes - it could that small boundaries/small blocks negatively affect the models
- [x] Load image data and organize according to time series
- [x] Visualize general information about the data - for instance, what is the average change in each of the attributes 
as a function of time (this will give us a general idea of how the data "feels")
- [x] Create k-fold blocks separated with a margin between different times/spatial locations (according 
to [this](https://onlinelibrary.wiley.com/doi/10.1111/ecog.02881))
- [x] Fully document code 
- [x] Create code for k-folds of blocks (for model validation)
- [x] Train a Decision Tree (DF) model on the blocks that are not auto-correlated
- [x] Train a Random Forest (RF) model on the blocks that are not auto-correlated
- [x] Visualize in some manner the feature significance learned by the DF\RF 


Links:
https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
https://en.wikipedia.org/wiki/Kriging
https://christophm.github.io/interpretable-ml-book/extend-lm.html

