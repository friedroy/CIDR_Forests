https://github.cs.huji.ac.il/cidr-center/leads/issues/126
# To Do List 
- [ ] Train and validate different models on full data (both standard CV and blocked CV)
- [ ] Validate different amounts of history given to the models (both standard CV and blocked)
- [ ] Model hyper parameter search
- [ ] Add spatial blocking option
- [x] Look into using LIME/SHAP values for interpretability of less classical models
- [x] Since there are a lot of features, we should probably use permutation tests for feature importance in the DTs and RFs
- [x] Rewrite feature importance for the DTs and RFs
- [x] Remove duplicates of static features when creating the (X, y) training pair (duplicates are created for each lookback year)


# Pipeline Breakdown
The suggested pipeline for learning the time series can be broken down into the following stages:
1. The data is now saved as a big CSV with rows as data points and columns for different features at different dates. The first part of the pipeline is to read this CSV and convert it into a format that is slightly easier to use. I loaded the CSV into a big tensor of shape (# data points, # years, # features + ndvi), where the months of the different features were aggregated according to [our previously chosen aggregation functions](https://docs.google.com/spreadsheets/d/188OjODdWSf7AR1he4f3eu2v0kSG8JEa1_swgaAjGCxQ/edit#gid=0)). Since the new data points are not on a regular spatial grid, we chose not to use the coordinates for now, which may change in the future
2. After loading this tensor, we have to reshape it to accomodate typical SKLearn models. Since we are trying to see what features affect the ndvi, I chose not to add the previous years ndvi to the features the model sees, as it is most correlated with the current year's ndvi (but this can be easily changed). In addition, I added a feature to add more years into the history the model sees for it's prediction. After this stage, the machine learning `X` and `y` equivalents are built
3. Validation options:
    1. Since we are not taking into account the spatial coordinates, the first cross-validation technique I used was to train on previous years and test on the next year each time (this is essentially what the time series is trying to achieve); this is basically the [Time Series Split cross validation technique provided by SKlearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
    2. The problem with the above split is that what we are trying to show is not to predict the ndvi the following year but rather to generalize to new points. A better way to handle this is by breaking the data into spatio-temporal blocks, then using typical k-folds validation on different blocks (that are somewhat separated from each other). I will add this method of validation to the code, as it is probably better
4. After choosing the model, we would like to extract feature importances from it. We can do this to any model using either simple [permutation tests](https://scikit-learn.org/stable/modules/permutation_importance.html), which are simple but effective, or the more complicated [SHAP values](https://christophm.github.io/interpretable-ml-book/shap.html), which are more accurate and we can check how the features push the model output


# Image Based (Old) Pipeline Breakdown
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
- [x] Changing the order of the features before the flatten (load_data.py, line 85-86) drastically changes performance - so that's probably a bug I need to fix



Links:
https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

https://en.wikipedia.org/wiki/Kriging

https://christophm.github.io/interpretable-ml-book/extend-lm.html

https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1

https://spacetimewithr.org/

https://scikit-learn.org/stable/modules/permutation_importance.html
